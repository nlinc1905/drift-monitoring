import argparse
import logging
import random
import pickle
import pandas as pd
import numpy as np
from ruamel import yaml
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections.abc import Iterable
from sklearn.svm import SVC
from datetime import datetime, timedelta


SEED = 14
random.seed(SEED)
logging.basicConfig(level=logging.INFO)


def train_bow_model(text_input, sklearn_vectorizer):
    """
    Fits a bag of words (BoW) model to the text

    :param text_input: list of documents
    :param sklearn_vectorizer: scikit-learn vectorizer object

    :return: vectorizer
    """
    assert isinstance(text_input, Iterable) and not isinstance(text_input, str)
    vectorizer = sklearn_vectorizer(
        stop_words="english",
        strip_accents="ascii",
        min_df=50,
        max_df=1.0
    )
    vectorizer.fit(text_input)
    return vectorizer


def write_config_yaml(feature_names):
    """
    The config.yaml file requires specifying variable names, but since we are randomly sampling
    which features to use, they will not be known in advance.  Therefore, we generate a new
    yaml file here and overwrite the existing config.yaml

    :feature_names: (list of strings) names of the features, or the words for a BoW model
    """
    output = dict(
        data_format = dict(
            separator = ",",
            header = True,
            date_column = "__date__",
        ),
        column_mapping = dict(
            target = "__target__",
            prediction = "__predicted__",
            datetime = "__date__",
            numerical_features = feature_names,
            categorical_features = []
        ),
        pretty_print = True,
        service = dict(
            reference_path = "data/reference.csv",
            min_reference_size = 30,
            use_reference = True,
            moving_reference = False,
            window_size = 30,
            calculation_period_sec = 10,
            monitors = ["data_drift", "concept_drift", "regression_performance"],
        ),
    )

    with open('config.yaml', 'w') as outfile:
        yaml.dump(output, outfile, default_flow_style=False)


def prepare_reference_dataset(drift_test_type=None):
    """
    Prepares a selection of news articles as the reference dataset (a.k.k. the training dataset, as
    the model is trained on this data).

    :param drift_test_type: (str) can be 1 of [data_drift, prior_drift, concept_drift]
        for data drift test (alter the features only, leave labels alone):
          train features on space data, train model on mixed data
          test on hockey data (model assumes the labels are still for space/religion)
        for prior drift test (alter the labels only, leave features alone):
          train features on mixed data, train model on mixed data
          test on space data
        for concept drift test (alter both the features and labels):
          train features on space data, train model on mixed data
          test on religion data
    """
    train = fetch_20newsgroups(
        subset="train",
        categories=["sci.space", "soc.religion.christian"],
        shuffle=True,
        random_state=SEED
    )

    # map the targets to what they mean
    target_map = {
        0: "space",
        1: "christian",
    }

    # balance the dataset
    minority_class_nbr_samples = pd.DataFrame(train.target).value_counts().min()
    train_indices_to_keep = pd.DataFrame(train.target).groupby(0)[0].apply(
        lambda x: x.sample(minority_class_nbr_samples, random_state=SEED)
    ).droplevel(0).index.tolist()
    train_data = [i for idx, i in enumerate(train.data) if idx in train_indices_to_keep]
    train_target = [i for idx, i in enumerate(train.target) if idx in train_indices_to_keep]

    # apply filters determined by drift_test_type
    space_indices = np.where(np.array(train_target)==0)[0].tolist()

    if drift_test_type is not None and (drift_test_type == "data_drift" or drift_test_type == "concept_drift"):
        # train features for space (bow_model will train on train_data_filtered)
        # train model on mixed (model will train on train_data, train_target)

        train_data_filtered = [i for idx, i in enumerate(train_data) if idx in space_indices]
        bow_model = train_bow_model(text_input=train_data_filtered, sklearn_vectorizer=CountVectorizer)
        features = bow_model.get_feature_names()  # 1 feature per word in vocabulary
        train_vect = bow_model.transform(train_data)

        model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            # if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma
            probability=False,  # set to True to enable predict_proba
            tol=0.001,
            random_state=SEED
        )

        model.fit(train_vect, train_target)
        train_preds = model.predict(train_vect)

    elif drift_test_type is None or (drift_test_type is not None and drift_test_type == "prior_drift"):
        # train features for mixed (bow_model will train on train_data)
        # train model on mixed (model will train on train_data, train_target)

        bow_model = train_bow_model(text_input=train_data, sklearn_vectorizer=CountVectorizer)
        features = bow_model.get_feature_names()  # 1 feature per word in vocabulary
        train_vect = bow_model.transform(train_data)

        model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            # if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma
            probability=False,  # set to True to enable predict_proba
            tol=0.001,
            random_state=SEED
        )

        model.fit(train_vect, train_target)
        train_preds = model.predict(train_vect)

    # bow_model (the vectorizer) is essentially the entire feature engineering pipeline
    # save it to the artifacts folder so that it can be used to process new data
    with open("data/artifacts/data_processor.pkl", "wb") as pfile:
        pickle.dump(bow_model, pfile)

    # save the trained model to artifacts, since it will be required to process new data
    with open("data/artifacts/model.pkl", "wb") as pfile:
        pickle.dump(model, pfile)

    # randomly sample from the features, and put everything together in a Pandas df for Evidently to read
    nbr_features_to_sample = 10
    sampled_features = random.sample([i for i in range(len(features))], k=nbr_features_to_sample)
    sampled_feature_names = [f for f_idx, f in enumerate(features) if f_idx in sampled_features]

    # save the sampled features as artifacts, so they can be used to sample the features for test data
    with open("data/artifacts/sampled_features.pkl", "wb") as pfile:
        pickle.dump(sampled_features, pfile)
    with open("data/artifacts/sampled_feature_names.pkl", "wb") as pfile:
        pickle.dump(sampled_feature_names, pfile)

    train_df = pd.DataFrame(
        train_vect[:, sampled_features].todense(),
        columns=sampled_feature_names
    )
    train_df["__target__"] = train_target
    train_df["__predicted__"] = train_preds
    train_df["__date__"] = datetime.today() - timedelta(days=1)

    train_df.to_csv("data/reference.csv", index=False)

    logging.info(f"Reference dataset created with {train_df.shape[0]} rows")

    # overwrite config.yaml
    write_config_yaml(feature_names=sampled_feature_names)


def prepare_production_dataset(drift_test_type=None):
    """
    Prepares a dataset to be used as the production data.  This data will be passed to the model API
    for predictions, and will be compared to the reference dataset for drift.

    :param drift_test_type: (str) can be 1 of [data_drift, prior_drift, concept_drift]
        for data drift test (alter the features only, leave labels alone):
          train features on space data, train model on mixed data
          test on hockey data (model assumes the labels are still for space/religion)
        for prior drift test (alter the labels only, leave features alone):
          train features on mixed data, train model on mixed data
          test on space data
        for concept drift test (alter both the features and labels):
          train features on space data, train model on mixed data
          test on religion data
    """
    if drift_test_type is not None and drift_test_type == "data_drift":
        categories = ["rec.sport.hockey"]
    elif drift_test_type is not None and drift_test_type == "prior_drift":
        categories = ["sci.space"]
    elif drift_test_type is not None and drift_test_type == "concept_drift":
        categories = ["soc.religion.christian"]
    else:
        categories = ["sci.space", "soc.religion.christian"]

    test = fetch_20newsgroups(
        subset="test",
        categories=categories,
        shuffle=True,
        random_state=SEED
    )

    # balance the dataset if more than 1 class
    if len(set(test.target)) > 1:
        minority_class_nbr_samples = pd.DataFrame(test.target).value_counts().min()
        test_indices_to_keep = pd.DataFrame(test.target).groupby(0)[0].apply(
            lambda x: x.sample(minority_class_nbr_samples, random_state=SEED)
        ).droplevel(0).index.tolist()
        test_data = [i for idx, i in enumerate(test.data) if idx in test_indices_to_keep]
        test_target = [i for idx, i in enumerate(test.target) if idx in test_indices_to_keep]
    else:
        test_data = test.data
        test_target = test.target

    # test_df should contain the fields to be sent to the API
    test_df = pd.DataFrame({
        "client_id": [1] * len(test_target),  # dummy
        "body": test_data,
        "target": test_target,
    })

    test_df.to_csv("data/production.csv", index=False)

    logging.info(f"Production dataset created with {test_df.shape[0]} rows")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare datasets for testing different types of drift detection.')
    parser.add_argument(
        '-dtt', '--drift_test_type',
        choices=['data_drift', 'prior_drift', 'concept_drift'],
        help='Enter 1 of [data_drift, prior_drift, concept_drift].  If none specified, defaults to None',
        required=False
    )
    args = vars(parser.parse_args())

    prepare_reference_dataset(drift_test_type=args['drift_test_type'])
    prepare_production_dataset(drift_test_type=args['drift_test_type'])
