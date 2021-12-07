import os
import numpy as np
from typing import Callable
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info


NAMESPACE = os.environ.get("PROMETHEUS_METRICS_NAMESPACE", "drift_monitoring_api")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "client_id")


instrumentator = Instrumentator()


def model_metric(
    metric_name: str = "predicted",
    metric_doc: str = "The metric of a model to be monitored.",
    metric_namespace: str = "",
    metric_subsystem: str = "",
) -> Callable[[Info], None]:
    """
    A custom metric for a model to be monitored by Prometheus.

    :param metric_name: the metric name that will appear in Prometheus
    :param metric_doc: text description of the metric
    :param metric_namespace: the namespace for Prometheus that the metric will appear in
    :param metric_subsystem: subsystem of the namespace for Prometheus -> this will be client_id

    :return: callable function
    """
    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/iterate":
            metric_value = info.response.headers.get(f"X-{metric_name}")
            client_id = info.response.headers.get("X-client_id")
            if metric_value:
                METRIC = Gauge(
                    metric_name,
                    metric_doc,
                    namespace=metric_namespace,
                    subsystem=str(client_id),
                )
                METRIC.set(float(metric_value))

    return instrumentation


# metric to track the number of requests to the monitoring_api
instrumentator.add(
    metrics.requests(
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

# metric to track the request latency to the monitoring_api
instrumentator.add(
    metrics.latency(
        buckets=(1, 2, 3,),
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

# metric to track the predictions for each client_id/model
instrumentator.add(
    model_metric(
        metric_name="predicted",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

# metric to track the target variable for each client_id/model
instrumentator.add(
    model_metric(
        metric_name="target",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
