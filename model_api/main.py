import os
import pickle
import json
import pandas as pd
import datetime as dt
from datetime import datetime
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel


ARTIFACTS_DIR = "data/artifacts/"

app = FastAPI()


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


class Article(BaseModel):
    model_id: int
    body: str
    target: int

    class Config:
        # this will be used as the example in Swagger docs
        schema_extra = {
            "example": {
                "model_id": 1,
                "body": "This is an article about clothing.",
                "target": 1,
            }
        }


def get_preds_from_model(json_data):
    """
    Gets predictions for an article

    :param json_data: JSON data for the article

    :return: prediction as JSON object
    """
    # convert JSON data to dict
    payload_dict = {k: [v] for k, v in jsonable_encoder(json_data).items()}

    # retrieve the model artifacts, model_id determines which model to use
    with open(f"{ARTIFACTS_DIR}data_processor_{payload_dict['model_id'][0]}.pkl", "rb") as pfile:
        bow_model = pickle.load(pfile)
    with open(f"{ARTIFACTS_DIR}model_{payload_dict['model_id'][0]}.pkl", "rb") as pfile:
        model = pickle.load(pfile)
    with open(f"{ARTIFACTS_DIR}sampled_features_{payload_dict['model_id'][0]}.pkl", "rb") as pfile:
        sampled_features = pickle.load(pfile)
    with open(f"{ARTIFACTS_DIR}sampled_feature_names_{payload_dict['model_id'][0]}.pkl", "rb") as pfile:
        sampled_feature_names = pickle.load(pfile)

    # transform the test data and make predictions
    test_vect = bow_model.transform(payload_dict["body"])
    test_preds = model.predict(test_vect)
    test_preds_prob = model.predict_proba(test_vect)

    # create JSON output
    output = {
        "model_id": str(payload_dict["model_id"][0]),
        "model_type": "classifier",  # hardcoded for this example
        "last_training_time": datetime.fromtimestamp(
            os.path.getmtime(f"{ARTIFACTS_DIR}model_{payload_dict['model_id'][0]}.pkl")
        ),
        "predictions": [
            {
                "prediction_time": datetime.now(),
                "record_id": i,
                "prediction": test_preds[i].tolist(),
                "prediction_probability": test_preds_prob[i, 1].tolist(),
                "ground_truth_label": payload_dict["target"][i]
            } for i in range(len(test_preds))
        ]
    }

    return json.dumps(output, default=json_serial)


@app.get("/")
def root():
    return {"message": "API working"}


@app.post("/predict")
async def predict_article_class(article: Article):
    return get_preds_from_model(json_data=article)
