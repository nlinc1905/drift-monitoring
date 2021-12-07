import pandas as pd
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from pydantic import create_model
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from os import walk

from metrics_instrumentation import instrumentator


# create a dynamic model with optional fields, bc each sample word will vary by client_id
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


# request count for each client ID
config_files = next(walk("config/monitoring"), (None, None, []))[2]


def create_config_if_not_exist():
    """
    For an incoming request, check if there is a config.yml file for the client_id
    in config/monitoring.  If there is, do nothing.  If there is not, create one.
    """
    # TODO: find out what the request will look like by running example_run_request, then finish this
    pass


def create_metrics_for_new_config():
    """
    When a new config is created, it needs metrics.  This function sets up metrics for the
    new config and appends them to the existing set of metrics.  This function will only run
    if a new config.yml is created.
    """
    new_metric = Counter(
        'request_count', 'Request Count for Client ID',
        ['app_name', 'method', 'endpoint', 'http_status']
    )
    return new_metric


def update_metric(metric):
    """
    pass
    :return:
    """
    metric.labels(
        'test_app', 'my_method', 'my_path', 200,  # request.method, request.path, response.status_code
    ).inc()


app = FastAPI()
# Instrumentator().instrument(app).expose(app)
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

    return request_data
