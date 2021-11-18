# drift-monitoring
Model drift monitoring

# What's Going on Here?

This app pulls down the evidently fork for model drift monitoring.  It then builds with the required materials to run 3 services: 
* Prometheus on port 9090
* Grafana on port 3000
* Evidently on port 5000
The evidently service is a Flask API defined by app.py.  The monitoring_service folder in the evidently repo fork contains the code for serving model drift monitoring metrics.  The evidently Flask API pulls from the code in the monitoring_service folder to serve metrics that Prometheus scrapes on a periodic basis (15 seconds by default).

To change Prometheus' scraping behavior, change config/prometheus.yml 

The Grafana dashboard is defined in dashboards/drift_dashboard.json.  You could modify the JSON directly, but it would be easier to run the example, modify or build new dashboards in the Grafana GUI, then click save to export the generated JSON, and put it in the dashboards folder.  The next time you run the app, the new dashboards will show up in Grafana.

# How to Run the Example

To run the example monitoring service, start the container (add `--build`, if you have not built it yet).

```
docker-compose up
```

Then get the container id and run the scripts required for the example.  If you run `docker ps`, you will see 3 containers running.  You want to use the ID for the one running the drift-monitoring_evidently_service, NOT the ones running Prometheus or Grafana.

```
docker exec <container_id> python3 prepare_datasets.py
docker exec <container_id> python3 example_run_request.py
```

Open localhost:3000 to see the Grafana dashboard.
