# drift-monitoring
Model drift monitoring

# What's Going on Here?

This app pulls down the evidently fork for model drift monitoring.  It then builds with the required materials to run 3 services: 
* Prometheus on port 9090
* Grafana on port 3000
* Evidently on port 5000

The evidently service is a Flask API defined by app.py.  The monitoring_service folder in the evidently repo fork contains the code for serving model drift monitoring metrics.  The evidently Flask API pulls from the code in the monitoring_service folder to serve metrics that Prometheus scrapes on a periodic basis (10 seconds by default).

To change Prometheus' scraping behavior, change config/prometheus.yml 

The Grafana dashboard is defined in dashboards/drift_dashboard.json.  You could modify the JSON directly, but it would be easier to run the example, modify or build new dashboards in the Grafana GUI, then click save to export the generated JSON, and put it in the dashboards folder.  The next time you run the app, the new dashboards will show up in Grafana.

# How to Run the Example

The example allows you to experiment with different types of drift.  You can prepare the test datasets (the 20 news groups dataset from scikit-learn), which are conveniently categorized, allowing you to train on one category and test on another, for instance.  By setting up these datasets and running the example, you can check for covariate/data drift, prior drift, and concept/prediction drift.  

For this example, the predictions are being served via FastAPI to mimic what would happen in production.  

To run the example monitoring service, start the container (add `--build`, if you have not built it yet).

```
docker-compose up
```

Then get the container id and run the scripts required for the example.  If you run `docker ps`, you will see 3 containers running.  You want to use the ID for the one running the drift-monitoring_evidently_service, NOT the ones running Prometheus or Grafana.  The example below shows how you would test concept drift.  You could test another type of drift, or pass no argument at all to run the default example with no drift.

```
docker exec <container_id> python3 prepare_datasets.py --drift_test_type concept_drift
docker exec <container_id> python3 example_run_request.py
```

Open localhost:3000 to see the Grafana dashboard.
