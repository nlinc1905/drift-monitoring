# Drift Monitoring Service

Drift monitoring for machine learning models with Prometheus & Grafana

## Contents
1. [Running the Example](#running-the-example)
   1. [Testing Different Types of Drift](#testing-drift-types)
   2. [Testing Multiple Clients or Models](#testing-multiple-models)
2. [Unit Tests](#unit-tests)
3. [Helpful Notes for Developing & Testing](#developer-notes)
   1. [Drift Monitoring Service API](#monitoring-serivce)
   2. [Prometheus](#prometheus)
   3. [Grafana](#grafana)
   4. [The Model API](#model-api)

<a name="running-the-example"></a>
## Running the Example

The first time you run the example, run `docker-compose up --build`.  This only has to be done once. 

To run the example, `docker-compose up` to start the services.  Get the container ID by running `docker ps` and finding 
the ID associated with drift_monitoring_serivce.  Then prepare the example data:
```
docker exec <container_id> python3 monitoring_service/examples/prepare_datasets.py
```
This will create the necessary test data.  There are options you can run to perform different tests with this example.  
See the next section.

After preparing the test data, you can 'stream' data from `data/production.csv` by running:
```
docker exec <container_id> python3 monitoring_service/examples/example_run_request.py
```
This will sample from the production data, 1 row every 3 seconds, and send requests to the drift monitoring 
service.  (Actually, the requests are sent to the model_api service first, then to the drift monitoring service.)

You can view the drift monitoring dashboard by going to Grafana at [http://localhost:3000](http://localhost:3000).  Log 
in with username = admin, password = admin, and skip the password update.

<a name="testing-drift-types"></a>
### Testing Different Types of Drift

To test that the drift monitoring service can find different types of drift, you can use the `-dtt` or 
`--drift_test_type` argument for prepare_datasets.py.  The default is None, but if you run
```
docker exec <container_id> python3 monitoring_service/examples/prepare_datasets.py -dtt data_drift
```
you will run a test for data drift (when the distribution of the features change).  This test is carried out by 
setting up the training data for news articles related to space exploration, and setting up the test data for news 
articles related to christianity.  The difference in topics will cause different feature distributions for the bag of 
words models.  

There are 3 types of drift you can test for:
* **data_drift**, or when the distributions of the features (covariates, or X) vary between the training set and 
current conditions
* **prior_drift**, or when the distribution of the target variable changes between the training set and current conditions
* **concept_drift**, or when the relationship between the target, y, and the features, x, changes between the training 
set and current conditions

The docstrings in `monitoring_service/examples/prepare_datasets.py` describe how the tests were engineered.  

After preparing the datasets, run 
```
docker exec <container_id> python3 monitoring_service/examples/example_run_request.py 
```

When you look at the drift monitoring dashboard in Grafana, you should see what you might expect for each type of drift.

<a name="testing-multiple-models"></a>
### Testing Multiple Client IDs or Models

We will have many client IDs/models.  To test this functionality, you can use the `-nc` or `--nbr_clients` argument 
for prepare_datasets.py.  The default is 1, but if you run
```
docker exec <container_id> python3 monitoring_service/examples/prepare_datasets.py -nc 30
```
you will run a test for 30 clients/models.  The client IDs are randomly assigned for testing, so if you run a test for 
multiple clients, you will not be able to test for different types of drift simultaneously.  The example will coerce 
your arguments to prevent error.  All you have to remember is that if you want to test a particular type of drift, do 
not use the `-nc` argument, as that will always take priority.

After preparing the datasets, run 
```
docker exec <container_id> python3 monitoring_service/examples/example_run_request.py 
```

<a name="unit-tests"></a>
## Unit Tests

To run the unit tests from the root directory:

```
docker exec <container_id> python3 -m pytest
```

<a name="developer-notes"></a>
## Helpful Notes for Developing & Testing

<a name="monitoring-api"></a>
### Drift Monitoring Service API

This API's swagger documentation can be found by going to [localhost:5000/docs](localhost:8000/docs) after the service 
is up and running.  To get it up and running, run `docker-compose up`.  The swagger documentation shows example 
requests and responses.

You can also get this API up and running by itself, just to test the example requests shown in the Swagger 
documentation.  To do that, you can run `uvicorn monitoring_api:app --reload`.  Running the API by itself might be 
useful for testing quick changes while developing.

<a name="prometheus"></a>
### Prometheus

Testing Prometheus is easiest to do in isolation of Grafana, so these instructions assume that is how you want to do it.

To test Prometheus, run `docker-compose up`.  Prometheus is defined as a service in the docker-compose file.  It mounts 
a volume to store time series metrics that it periodically scrapes.  Scraping behavior and other configuration can be 
found in config/prometheus.yml.  

Once up, go to [http://localhost:9090/targets](http://localhost:9090/targets).  You should see the 
drift-monitoring-service running on port 5000, and localhost:9090/metrics running for Prometheus.  If the status for 
each of these is a green UP button, they are working.  

#### Adding a Custom Metric

FastAPI has a Prometheus Instrumentator library for Python that is being used to integrate the drift monitoring service 
with Prometheus.  To create a new metric, first add it to the response header in `monitoring_service/monitoring_api.py`.  
For example, this is the response header that was added to track model predictions:
```
response.headers["X-predicted"] = str(request_data_df["predicted_"][0])
```
The header is added to the `/iterate` endpoint.  

Next, define your new metric in `monitoring_service/metrics_instrumenation.py`.  You can copy the model_metric 
function that is there now to get started.  Just replace the METRIC with whatever you want.  Note that there are 
[4 types of metrics](https://prometheus.io/docs/concepts/metric_types/) that Prometheus can track.  See how to create 
them with the Python Prometheus client here: 
[https://github.com/prometheus/client_python#instrumenting](https://github.com/prometheus/client_python#instrumenting).

Finally, add your metric to the instrumentator.  Here is an example for adding the predicted metric:
```
instrumentator.add(
    model_metric(
        metric_name="predicted",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
```

#### Testing Custom Metrics

If you click on the graph link at the top of the [targets page](http://localhost:9090/targets), you will be able to 
test Prometheus queries.  For example, if you want to check that your custom metric is being scraped by Prometheus, 
simply search for it.  Note that the namespace and subsystem should come first in the metric name, as defined by the 
environment variables in the Dockerfile.  So for example: `drift_monitoring_api_model_predicted` is the metric for 
the predictions made by the model_api (or CisionAI) that have hit the /iterate endpoint in the drift monitoring API.  

If the metric does not appear, you can run a sample request using the Swagger documentation for the drift monitoring 
API.  Do this by going to [http://localhost:5000/docs](http://localhost:5000/docs) and sending a sample request.  When 
you look back to the Prometheus graph page on localhost:9090/targets, you should see the time series update.  

##### Metrics Per Client

Since there is 1 model per client ID to be monitored, there needs to be a separate time series for each model.  This 
functionality is handled by the metric label.  It is not best practice to have high cardinality in metric labels, but 
Prometheus can handle a few million time series.  See [this post on StackOverflow](https://stackoverflow.com/questions/46373442/how-dangerous-are-high-cardinality-labels-in-prometheus?rq=1).

To test that your custom metric is working for each client.  Go to the [graph](http://localhost:9090/graph) and run a 
PromQL query.  For example, you could query for the model predicted value for client 1:
<br/>
`drift_monitoring_api_model_predicted{client_id="1"}`
<br/>
You can go to the [drift monitoring service API Swagger docs](http://localhost:5000/docs) to run a test query, and then 
re-run your PromQL query to see the time series update.  

<a name="grafana"></a>
### Grafana

Grafana can be accessed at [http://localhost:3000](http://localhost:3000).  Log in with username = admin and password = 
admin, and skip the password update. 

#### Creating or Editing Dashboards

Grafana loads pre-build dashboards from /dashboards, but you can create your own too.  To create one, use the Grafana 
interface and either 
1) save it, which will save it to the Grafana volume, or 
2) export it to JSON and put it in the /dashboards folder for future use.

The 2nd option is preferred.

**Make sure you copy/paste the JSON WITHOUT checking the export for sharing externally option.**  See 
this [bug](https://github.com/grafana/grafana/issues/11018).  The easiest way to do this is to click the save icon and 
copy the JSON from there.  

To edit an existing dashboard, make your changes in the Grafana interface and copy or export the JSON to overwrite the 
file in /dashboards.

See [Grafana's docs](https://grafana.com/docs/) for anything else.

<a name="model-api"></a>
### The Model API

The model API will be replaced by CisionAI in production.  However, it was created here to test what might come out of 
a model service and need to be monitored.  Running the example will hit this API.  You will notice in 
`monitoring_service/examples/examples_run_request.py` that the data is first sent to the model API, then to the drift monitoring API.  
To test that the data is in the format necessary, you can compare what comes back from the model API to what is expected 
by the Pydantic model defined in monitoring_api.py.  Additionally, you can run 
`monitoring_service/examples/test-reading_sample_response_from_model_api.py`, which was created solely for the 
purpose of testing this integration.  
