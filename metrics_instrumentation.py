import os
from typing import Callable
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from prometheus_fastapi_instrumentator.metrics import Info


NAMESPACE = os.environ.get("PROMETHEUS_METRICS_NAMESPACE", "drift_monitoring_api")
SUBSYSTEM = os.environ.get("METRICS_SUBSYSTEM", "model")


instrumentator = Instrumentator()


def model_metric(
    metric_name: str = "predicted",
    metric_doc: str = "The metric of a model to be monitored.",
    metric_namespace: str = "",
    metric_subsystem: str = "",
) -> Callable[[Info], None]:
    """
    A custom metric for a model to be monitored by Prometheus.  Notice that there is
    a metric for each client_id/model.  The differentiation is handled by the label
    'client_id'.  When querying for these metrics in PromQL, you can specify the
    query as follows:
        drift_monitoring_api_client_id_predicted{client_id="1"}
    You can also omit the filter to show all requests, which would be useful for
    aggregation.

    Since there is 1 label for each client_id, and we plan to have thousands of clients/models,
    there will be thousands of time series for Prometheus to monitor.  This is not best practice.
    See: https://stackoverflow.com/questions/46373442/how-dangerous-are-high-cardinality-labels-in-prometheus?rq=1
    But the alternative is to only report aggregated metrics, which will not tell us which specific models
    need to be re-trained (unless we do that separately for V2 and only report the aggregated stats
    in Prometheus)

    :param metric_name: the metric name that will appear in Prometheus
    :param metric_doc: text description of the metric
    :param metric_namespace: the namespace for Prometheus that the metric will appear in
    :param metric_subsystem: subsystem of the namespace for Prometheus

    :return: callable function
    """
    METRIC = Gauge(
        metric_name,
        metric_doc,
        namespace=metric_namespace,
        subsystem=metric_subsystem,
        labelnames=("client_id",),
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/iterate":
            client_id = str(info.response.headers.get("X-client_id"))
            metric_value = info.response.headers.get(f"X-{metric_name}")
            if metric_value:
                METRIC.labels(client_id).set(float(metric_value))

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

# metric to track the sample features for each client_id/model
for i in range(10):
    instrumentator.add(
        model_metric(
            metric_name=f"sample_feature_{int(i + 1)}",
            metric_namespace=NAMESPACE,
            metric_subsystem=SUBSYSTEM,
        )
    )
