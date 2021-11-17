# drift-monitoring
Model drift monitoring

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
