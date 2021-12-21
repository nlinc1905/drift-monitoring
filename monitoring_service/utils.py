from ruamel import yaml


DATA_DIR = "data/"


def create_config(feature_names, filename_suffix=None):
    """
    Create a new config file for the client_id in config/monitoring.

    :feature_names: (list of strings) names of the features, or the words for a BoW model
    :filename_suffix: (str) optional string to append to the config file name, such as when there
        is 1 config per model/client
    """
    # TODO: update this after seeing what prod will look like
    output = dict(
        data_format=dict(
            separator=",",
            header=True,
            date_column="date_",
        ),
        column_mapping=dict(
            target="target_",
            prediction="predicted_",
            datetime="date_",
            numerical_features=feature_names,
            categorical_features=[]
        ),
        pretty_print=True,
        service=dict(
            reference_path=f"{DATA_DIR}reference{filename_suffix or '_1'}.csv",
            min_reference_size=30,
            use_reference=True,
            moving_reference=False,
            window_size=30,
            calculation_period_sec=10,
            monitors=["data_drift", "concept_drift", "regression_performance"],
        ),
    )

    with open(f"config/monitoring/monitoring_config{filename_suffix or '_1'}.yaml", "w") as outfile:
        yaml.dump(output, outfile, default_flow_style=False)
