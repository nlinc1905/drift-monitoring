import pandas as pd
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, create_model
from os import path

from metrics_instrumentation import instrumentator
from utils import create_config, Monitor


MONITOR_TRACKER = {}


# create a dynamic Pydantic model with optional fields, bc each sample word will vary by client_id
# set them to floats with default of None to make them optional
DynamicDataModel = create_model(
    'DynamicDataModel',
    target_=(int, 1),
    predicted_=(int, 1),
    date_=(int, 1000),
    client_id_=(int, 1),
    sample_feature_1=(float, None),
    sample_feature_2=(float, None),
    sample_feature_3=(float, None),
    sample_feature_4=(float, None),
    sample_feature_5=(float, None),
    sample_feature_6=(float, None),
    sample_feature_7=(float, None),
    sample_feature_8=(float, None),
    sample_feature_9=(float, None),
    sample_feature_10=(float, None),
)


app = FastAPI()
instrumentator.instrument(app).expose(app)

# start app with: uvicorn monitoring_api:app --reload
# to view the app: http://localhost:8000
# to view metrics:  http://localhost:8000/metrics
# when running with Docker, configure a job for this app in prometheus.yml, and test at http://localhost:9090/targets


@app.get("/")
def root():
    return {"message": "API working"}


@app.post("/iterate")
async def iterate(response: Response, dynamicdatamodel: DynamicDataModel):
    """
    Routes incoming new data (JSON) to the monitoring service to be appended to
    the current dataset (monitoring_service.iterate is called with new_rows argument).
    """
    request_data = jsonable_encoder(dynamicdatamodel)
    request_data_df = pd.DataFrame(request_data, index=[0])

    # create a new config file if the client_id has never been seen before
    client_id = str(request_data_df["client_id_"][0])
    if not path.exists(f"config/monitoring/monitoring_config{('_' + client_id) or '_1'}.yaml"):
        create_config(
            feature_names=[
                c for c in request_data_df.columns
                if c not in ["client_id_", "target_", "predicted_", "date_"]
            ],
            filename_suffix=('_' + client_id)
        )

    # perform statistical tests
    if client_id not in MONITOR_TRACKER.keys():
        MONITOR_TRACKER[client_id] = Monitor(client_id=client_id, reference_data=None)
    updated = MONITOR_TRACKER[client_id].iterate(new_rows=request_data_df)
    if updated:
        # save the desired metrics to be tracked for each model to the response headers
        # the instrumentator in metrics_instrumentation.py will read from these headers
        response.headers["X-target"] = str(MONITOR_TRACKER[client_id].metrics['target_chi_square_p_value'])
        response.headers["X-predicted"] = str(MONITOR_TRACKER[client_id].metrics['predicted_chi_square_p_value'])
        response.headers["X-client_id"] = str(request_data_df["client_id_"][0])
        sampled_features = [
            c for c in request_data_df.columns
            if c not in ["target_", "predicted_", "client_id_", "date_"]
        ]
        for sf_id, sf in enumerate(sampled_features):
            response.headers[f"X-{sf}"] = str(
                MONITOR_TRACKER[client_id].metrics[f'{sf}_p_value']
            )

        return "ok"
    else:
        return f"Not enough data for client {client_id} comparison.  Waiting for more requests..."
