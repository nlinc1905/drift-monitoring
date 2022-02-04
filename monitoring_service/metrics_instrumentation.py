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
    a metric for each model_id/model.  The differentiation is handled by the label
    'model_id'.  When querying for these metrics in PromQL, you can specify the
    query as follows:
        drift_monitoring_api_model_id_predicted{model_id="1"}
    You can also omit the filter to show all requests, which would be useful for
    aggregation.

    Since there is 1 label for each model_id, and we plan to have thousands of clients/models,
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
        labelnames=("model_id",),
    )

    def instrumentation(info: Info) -> None:
        if info.modified_handler == "/metrics":
            model_id = str(info.response.headers.get("X-model_id"))
            metric_value = info.response.headers.get(f"X-{metric_name}")
            # -100 used as the value to return for stat test failures
            if metric_value and metric_value != -100:
                METRIC.labels(model_id).set(float(metric_value))

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

instrumentator.add(
    model_metric(
        metric_name="concept_drift",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

instrumentator.add(
    model_metric(
        metric_name="prediction_drift",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

instrumentator.add(
    model_metric(
        metric_name="prediction_prob_drift",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)

instrumentator.add(
    model_metric(
        metric_name="prior_drift",
        metric_namespace=NAMESPACE,
        metric_subsystem=SUBSYSTEM,
    )
)
