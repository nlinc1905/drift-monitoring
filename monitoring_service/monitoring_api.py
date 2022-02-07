import os
import logging
import pandas as pd
from datetime import datetime, date
from fastapi import FastAPI, Response, Body, Depends
from fastapi.encoders import jsonable_encoder
from typing import Optional, List
from pydantic import BaseModel, create_model
from sklearn.metrics import f1_score
from scipy.stats import wasserstein_distance
from sqlalchemy.orm import Session

from monitoring_service import pydantic_models
from monitoring_service.metrics_instrumentation import instrumentator
from monitoring_service.stat_tests import chi_square_test, ks_test, bayesian_a_b_test

from sql_db import crud, db_models
from sql_db.database import SessionLocal, engine


MIN_SAMPLE_SIZE = int(os.environ.get("MIN_SAMPLE_SIZE", 30))
RESAMPLE_FOR_HYPOTHESIS_TEST = bool(os.environ.get("RESAMPLE_FOR_HYPOTHESIS_TEST", True))

logging.basicConfig(level=logging.INFO)

# Bind the DB engine to db_models and set up the DB as a dependency for this app
db_models.Base.metadata.create_all(bind=engine)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


app = FastAPI()
instrumentator.instrument(app).expose(app)

# start app with: uvicorn monitoring_service.monitoring_api:app --reload
# to view the app: http://localhost:8000
# to view metrics:  http://localhost:8000/metrics
# when running with Docker, configure a job for this app in prometheus.yml, and test at http://localhost:9090/targets


@app.get("/")
def root():
    return {"message": "API working"}


@app.get("/prediction_history/{model_id}")
def get_predictions(model_id: str, db: Session = Depends(get_db)):
    predictions, columns = crud.get_all_predictions_for_model(db, model_id=model_id)
    return pd.DataFrame.from_records(predictions, columns=columns).to_dict()


@app.post("/metrics")
async def compute_metrics(
        response: Response,
        inputdatamodel: pydantic_models.InputDataModelCreate = Body(..., examples=pydantic_models.input_examples),
        db: Session = Depends(get_db)
):
    """
    Updates the model monitoring database with the incoming new data (JSON), performs
    statistical tests to detect drift, and saves the results as response headers for the
    Prometheus instrumentator to send to Prometheus.
    """
    # update the database
    crud.create_or_update_model(
        db=db,
        model_id=inputdatamodel.model_id,
        model_type=inputdatamodel.model_type,
        last_training_time=inputdatamodel.last_training_time
    )
    for pred in inputdatamodel.predictions:
        pred_dict = pred.dict()
        pred_dict['parent_model_id'] = inputdatamodel.model_id
        crud.create_prediction(db=db, prediction=pred_dict)

    # get reference data
    records, columns = crud.get_reference_predictions_for_model(db=db, model_id=inputdatamodel.model_id)
    reference_df = pd.DataFrame.from_records(records, columns=columns)

    # get data to be compared
    records, columns = crud.get_current_predictions_for_model(db=db, model_id=inputdatamodel.model_id)
    request_df = pd.DataFrame.from_records(records, columns=columns)

    nbr_ref_samples = len(reference_df)
    nbr_comparison_samples = len(request_df)
    total_samples = nbr_ref_samples + nbr_comparison_samples

    # if there are not enough comparison samples, or if there are not enough samples to form a reference set, return
    if (
            (total_samples < 2 * MIN_SAMPLE_SIZE)
            or (total_samples >= 2 * MIN_SAMPLE_SIZE and nbr_comparison_samples < MIN_SAMPLE_SIZE)
    ):
        message = {"message": f"Not enough samples for model {inputdatamodel.model_id}.  Waiting for more..."}
        logging.warn(message)
        return message

    # if there are not enough reference samples, but the comparison set has enough to lend
    if total_samples >= 2 * MIN_SAMPLE_SIZE and nbr_ref_samples < 30:
        nbr_samples_needed = MIN_SAMPLE_SIZE - nbr_ref_samples
        loaned_samples = request_df.sort_values('prediction_time').head(nbr_samples_needed)
        reference_df = reference_df.append(loaned_samples, ignore_index=True)
        request_df.drop(loaned_samples.index, inplace=True)
        logging.warn(
            f"Insufficient reference samples for model {inputdatamodel.model_id}. "
            f"Moved {nbr_samples_needed} oldest comparison samples to reference set."
        )

    # perform statistical tests and distance computations
    if inputdatamodel.model_type == "classifier":

        concept_a = reference_df.apply(lambda x: int(x['prediction'] == x['ground_truth_label']), axis=1)
        concept_b = request_df.apply(lambda x: int(x['prediction'] == x['ground_truth_label']), axis=1)
        concept_drift = chi_square_test(
            reference_data=concept_a,
            current_data=concept_b,
            resample=RESAMPLE_FOR_HYPOTHESIS_TEST,
        )
        concept_emd = wasserstein_distance(concept_a.values, concept_b.values)

        prediction_drift = chi_square_test(
            reference_data=reference_df['prediction'],
            current_data=request_df['prediction'],
            resample=RESAMPLE_FOR_HYPOTHESIS_TEST,
        )
        prediction_emd = wasserstein_distance(reference_df['prediction'].values, request_df['prediction'].values)

        prediction_prob_drift = ks_test(
            reference_data=reference_df['prediction_probability'],
            current_data=request_df['prediction_probability'],
            resample=RESAMPLE_FOR_HYPOTHESIS_TEST,
        )
        prediction_prob_emd = wasserstein_distance(
            reference_df['prediction_probability'].values,
            request_df['prediction_probability'].values
        )

        prior_drift = chi_square_test(
            reference_data=reference_df['ground_truth_label'],
            current_data=request_df['ground_truth_label'],
            resample=RESAMPLE_FOR_HYPOTHESIS_TEST,
        )
        prior_emd = wasserstein_distance(
            reference_df['ground_truth_label'].values,
            request_df['ground_truth_label'].values
        )

    else:

        concept_a = np.abs(reference_df['prediction'] - reference_df['ground_truth_label'])
        concept_b = np.abs(request_df['prediction'] - request_df['ground_truth_label'])
        concept_drift = ks_test(
            reference_data=concept_a,
            current_data=concept_b,
            resample=RESAMPLE_FOR_HYPOTHESIS_TEST,
        )
        concept_emd = wasserstein_distance(concept_a.values, concept_b.values)

        prediction_drift = ks_test(
            reference_data=reference_df['prediction'],
            current_data=request_df['prediction'],
            resample=RESAMPLE_FOR_HYPOTHESIS_TEST,
        )
        prediction_emd = wasserstein_distance(reference_df['prediction'].values, request_df['prediction'].values)

        prediction_prob_drift = ""
        prediction_prob_emd = ""

        prior_drift = ks_test(
            reference_data=reference_df['ground_truth_label'],
            current_data=request_df['ground_truth_label'],
            resample=RESAMPLE_FOR_HYPOTHESIS_TEST,
        )
        prior_emd = wasserstein_distance(
            reference_df['ground_truth_label'].values,
            request_df['ground_truth_label'].values
        )

    # save the metrics to be tracked for each model to the response headers
    # the instrumentator in metrics_instrumentation.py will read from these headers
    response.headers["X-model_id"] = str(inputdatamodel.model_id)
    response.headers["x-concept_drift"] = str(concept_drift)
    response.headers["x-concept_drift_emd"] = str(concept_emd)
    response.headers["X-prediction_drift"] = str(prediction_drift)
    response.headers["x-prediction_emd"] = str(prediction_emd)
    response.headers["X-prediction_prob_drift"] = str(prediction_prob_drift)
    response.headers["X-prediction_prob_emd"] = str(prediction_prob_emd)
    response.headers["X-prior_drift"] = str(prior_drift)
    response.headers["X-prior_emd"] = str(prior_emd)

    return {"message": response.headers}
