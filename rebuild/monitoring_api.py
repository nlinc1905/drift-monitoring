import pandas as pd
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from pydantic import create_model
from os import path

from metrics_instrumentation import instrumentator
from utils import create_config


# create a dynamic Pydantic model with optional fields, bc each sample word will vary by client_id
# set them to floats with default of None to make them optional
DynamicDataModel = create_model(
    'DynamicDataModel',
    target_=(int, 1),
    predicted_=(int, 1),
    date_=(int, 1000),
    client_id_=(int, 1),
    sample_word1=(float, None),
    sample_word2=(float, None),
    sample_word3=(float, None),
    sample_word4=(float, None),
    sample_word5=(float, None),
    sample_word6=(float, None),
    sample_word7=(float, None),
    sample_word8=(float, None),
    sample_word9=(float, None),
    sample_word10=(float, None),
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

    # save the desired metrics to be tracked for each model to the response headers
    # the instrumentator in metrics_instrumentation.py will read from these headers
    response.headers["X-target"] = str(request_data_df["target_"][0])
    response.headers["X-predicted"] = str(request_data_df["predicted_"][0])
    response.headers["X-client_id"] = str(request_data_df["client_id_"][0])
    # sampled_features = [
    #     c for c in request_data_df.columns
    #     if c not in ["target_", "predicted_", "client_id_", "date_"]
    # ]
    # for sf in sampled_features:
    #     response.headers[f"X-{sf}"] = str(request_data_df[sf][0])

    return "ok"
