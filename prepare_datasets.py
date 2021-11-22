import logging
import random
import pickle
import pandas as pd
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
    vectorizer = sklearn_vectorizer(stop_words="english")
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
            numerical_features = [],
            categorical_features = feature_names
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


def prepare_reference_dataset():
    """
    Prepares a selection of news articles as the reference dataset (a.k.k. the training dataset, as
    the model is trained on this data).
    """
    train = fetch_20newsgroups(
        subset="train",
        categories=["sci.space", "soc.religion.christian"],
        shuffle=True,
        random_state=SEED
    )

    # create features for model - ensure to train BoW only on training data
    bow_model = train_bow_model(text_input=train.data, sklearn_vectorizer=CountVectorizer)
    features = bow_model.get_feature_names()  # 1 feature per word in vocabulary
    train_vect = bow_model.transform(train.data)

    # bow_model (the vectorizer) is essentially the entire feature engineering pipeline
    # save it to the artifacts folder so that it can be used to process new data
    with open("data/artifacts/data_processor.pkl", "wb") as pfile:
        pickle.dump(bow_model, pfile)

    # map the targets to what they mean
    target_map = {
        0: "space",
        1: "christian",
    }

    model = SVC(
        C=1.0,
        kernel='rbf',
        gamma='scale',  # if gamma='scale' (default) is passed then it uses 1 / (n_features * X.var()) as value of gamma
        probability=False,  # set to True to enable predict_proba
        tol=0.001,
        random_state=SEED
    )

    model.fit(train_vect, train.target)
    train_preds = model.predict(train_vect)

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
    train_df["__target__"] = train.target
    train_df["__predicted__"] = train_preds
    train_df["__date__"] = datetime.today() - timedelta(days=1)

    train_df.to_csv("data/reference.csv", index=False)

    logging.info(f"Reference dataset created with {train_df.shape[0]} rows")

    # overwrite config.yaml
    write_config_yaml(feature_names=sampled_feature_names)


def prepare_production_dataset():
    """
    Prepares a dataset to be used as the production data.  This data will be passed to the model API
    for predictions, and will be compared to the reference dataset for drift.
    """
    test = fetch_20newsgroups(
        subset="test",
        categories=["sci.space", "soc.religion.christian"],
        shuffle=True,
        random_state=SEED
    )

    # test_df should contain the fields to be sent to the API
    test_df = pd.DataFrame({
        "client_id": [1] * len(test.target),  # dummy
        "body": test.data,
        "target": test.target,
    })

    test_df.to_csv("data/production.csv", index=False)

    logging.info(f"Production dataset created with {test_df.shape[0]} rows")


if __name__ == '__main__':
    prepare_reference_dataset()
    prepare_production_dataset()
