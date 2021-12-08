# Stuff

# Helpful Notes for Developing & Testing

## Prometheus

Testing Prometheus is easiest to do in isolation of Grafana, so these instructions assume that is how you want to do it.

To test Prometheus, run `docker-compose up`.  Prometheus is defined as a service in the docker-compose file.  It mounts 
a volume to store time series metrics that it periodically scrapes.  Scraping behavior and other configuration can be 
found in config/prometheus.yml.  

Once up, go to [http://localhost:9090/targets](http://localhost:9090/targets).  You should see the 
drift-monitoring-service running on port 5000, and localhost:9090/metrics running for Prometheus.  If the status for 
each of these is a green UP button, they are working.  

### Adding a Custom Metric

FastAPI has a Prometheus Instrumentator library for Python that is being used to integrate the drift monitoring service 
with Prometheus.  To create a new metric, first add it to the response header in `monitoring_api.py`.  For example, this 
is the response header that was added to track model predictions:
<br/>
`response.headers["X-predicted"] = str(request_data_df["predicted_"][0])`
<br/>
The header is added to the `/iterate` endpoint.  

Next, define your new metric in `metrics_instrumenation.py`.  You can copy the model_metric function that is there now 
to get started.  Just replace the METRIC with whatever you want.  Note that there are 
[4 types of metrics](https://prometheus.io/docs/concepts/metric_types/) that Prometheus can track.  

Finally, add your metric to the instrumentator.  Here is an example for adding the predicted metric:
<br/>
```
instrumentator.add(
    model_metric(
        metric_name="predicted",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
```

### Testing Custom Metrics

If you click on the graph link at the top of the [targets page](http://localhost:9090/targets), you will be able to 
test Prometheus queries.  For example, if you want to check that your custom metric is being scraped by Prometheus, 
simply search for it.  Note that the namespace and subsystem should come first in the metric name, as defined by the 
environment variables in the Dockerfile.  So for example: `drift_monitoring_api_model_predicted` is the metric for 
the predictions made by the model_api (or CisionAI) that have hit the /iterate endpoint in the drift monitoring API.  

If the metric does not appear, you can run a sample request using the Swagger documentation for the drift monitoring 
API.  Do this by going to [http://localhost:5000/docs](http://localhost:5000/docs) and sending a sample request.  When 
you look back to the Prometheus graph page on localhost:9090/targets, you should see the time series update.  

#### Metrics Per Client

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
