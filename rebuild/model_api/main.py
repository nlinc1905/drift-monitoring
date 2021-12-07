import pickle
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel


ARTIFACTS_DIR = "../data/artifacts/"


app = FastAPI()


class Article(BaseModel):
    client_id: int
    body: str
    target: int

    class Config:
        # this will be used as the example in Swagger docs
        schema_extra = {
            "example": {
                "client_id": 1,
                "body": "This is an article about clothing.",
                "target": 1,
            }
        }


def get_preds_from_model(json_data):
    """
    Gets predictions for an article

    :param json_data: JSON data for the article

    :return: prediction as JSON object
    """
    # convert JSON data to dict
    payload_dict = {k: [v] for k, v in jsonable_encoder(json_data).items()}

    # retrieve the model artifacts, client_id determines which model to use
    with open(f"{ARTIFACTS_DIR}data_processor_{payload_dict['client_id'][0]}.pkl", "rb") as pfile:
        bow_model = pickle.load(pfile)
    with open(f"{ARTIFACTS_DIR}model_{payload_dict['client_id'][0]}.pkl", "rb") as pfile:
        model = pickle.load(pfile)
    with open(f"{ARTIFACTS_DIR}sampled_features_{payload_dict['client_id'][0]}.pkl", "rb") as pfile:
        sampled_features = pickle.load(pfile)
    with open(f"{ARTIFACTS_DIR}sampled_feature_names_{payload_dict['client_id'][0]}.pkl", "rb") as pfile:
        sampled_feature_names = pickle.load(pfile)

    # transform the test data and make predictions
    test_vect = bow_model.transform(payload_dict["body"])
    test_preds = model.predict(test_vect)

    # limit the data to the sampled features (for use in drift monitoring dashboard)
    # make sure this sampling is done AFTER model prediction
    test_df = pd.DataFrame(
        test_vect[:, sampled_features].todense(),
        columns=sampled_feature_names
    )
    test_df["target_"] = payload_dict["target"]
    test_df["predicted_"] = test_preds
    test_df["date_"] = datetime.today()
    test_df["client_id_"] = payload_dict["client_id"][0]

    return test_df.to_json(orient="index")


@app.get("/")
def root():
    return {"message": "API working"}


@app.post("/predict")
async def predict_article_class(article: Article):
    return get_preds_from_model(json_data=article)
