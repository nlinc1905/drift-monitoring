from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta


input_examples = {
    "classifier": {
        "summary": "Example of a post from a classifier",
        "value": {
            "model_id": "C123",
            "model_type": "classifier",
            "last_training_time": datetime.now(),
            "predictions": [
                {
                    "prediction_time": datetime.now() - timedelta(days=1),
                    "record_id": 101,
                    "prediction": 1,
                    "prediction_probability": 0.8,
                    "ground_truth_label": 1,
                },
                {
                    "prediction_time": datetime.now() + timedelta(days=1),
                    "record_id": 102,
                    "prediction": 0,
                    "prediction_probability": 0.2,
                    "ground_truth_label": 1,
                },
            ],
        },
    },
    "regression": {
        "summary": "Example of a post from a regression",
        "value": {
            "model_id": "C123",
            "model_type": "classifier",
            "last_training_time": datetime.now(),
            "predictions": [
                {
                    "prediction_time": datetime.now() - timedelta(days=1),
                    "record_id": 101,
                    "prediction": 38.63,
                    "prediction_probability": None,
                    "ground_truth_label": 40.01,
                },
                {
                    "prediction_time": datetime.now() + timedelta(days=1),
                    "record_id": 102,
                    "prediction": 45.90,
                    "prediction_probability": None,
                    "ground_truth_label": 41.49,
                },
            ],
        },
    },
}


class PredictionDataModel(BaseModel):
    """
    Pydantic data model that defines the structure of the expected input data
    """
    prediction_time: datetime
    record_id: int
    prediction: float
    prediction_probability: Optional[float] = None
    ground_truth_label: Optional[float] = None


class InputDataModelBase(BaseModel):
    """
    Pydantic data model that defines the structure of the expected input data
    """
    model_id: str
    model_type: str
    last_training_time: Optional[datetime] = datetime.now()
    predictions: List[PredictionDataModel]


class InputDataModelCreate(InputDataModelBase):
    """
    Attributes required for creation but that should not be returned when read, like passwords.
    """
    pass


class InputDataModel(InputDataModelBase):
    """
    Attributes not in base model that are also returned when read, like ID/primary key.
    """
    id: int

    class Config:
        orm_mode = True
