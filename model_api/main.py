import pickle
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel


app = FastAPI()


class Article(BaseModel):
    client_id: int
    body: str
    target: int


def get_preds_from_model(json_data):
    """
    Gets predictions for an article

    :param json_data: JSON data for the article

    :return: prediction as JSON object
    """
    # retrieve the model artifacts
    # do this here in the function bc the articles could vary by model in the future
    with open("data/artifacts/data_processor.pkl", "rb") as pfile:
        bow_model = pickle.load(pfile)
    with open("data/artifacts/model.pkl", "rb") as pfile:
        model = pickle.load(pfile)
    with open("data/artifacts/sampled_features.pkl", "rb") as pfile:
        sampled_features = pickle.load(pfile)
    with open("data/artifacts/sampled_feature_names.pkl", "rb") as pfile:
        sampled_feature_names = pickle.load(pfile)

    # convert JSON data to dict
    payload_dict = {k:[v] for k,v in jsonable_encoder(json_data).items()}

    # transform the test data and make predictions
    test_vect = bow_model.transform(payload_dict["body"])
    test_preds = model.predict(test_vect)

    # limit the data to the sampled features (for use in drift monitoring dashboard)
    # make sure this sampling is done AFTER model prediction
    test_df = pd.DataFrame(
        test_vect[:, sampled_features].todense(),
        columns=sampled_feature_names
    )
    test_df["__target__"] = payload_dict["target"]
    test_df["__predicted__"] = test_preds
    test_df["__date__"] = datetime.today()

    return test_df.to_json(orient="index")


@app.get("/")
def root():
    return {"message": "API working"}


@app.post("/predict")
async def predict_article_class(article: Article):
    return get_preds_from_model(json_data=article)
