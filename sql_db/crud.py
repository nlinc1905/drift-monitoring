import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime

from sql_db import db_models


logging.basicConfig(level=logging.INFO)


def get_all_predictions_for_model(db: Session, model_id: str):
    """
    Gets all predictions for a given modl ID
    """
    column_names = (
            ["db_models.Models." + c.name for c in db_models.Models.__table__.columns] +
            ["db_models.Predictions." + c.name for c in db_models.Predictions.__table__.columns]
    )
    columns = [eval(cname) for cname in column_names]
    return db.query(*columns).join(
        db_models.Predictions,
        db_models.Models.model_id == db_models.Predictions.parent_model_id
    ).all(), [c[c.rindex(".") + 1:] for c in column_names]


def get_reference_predictions_for_model(db: Session, model_id: str):
    """
    Gets all predictions for a given model ID up until the last training time
    """
    column_names = (
            ["db_models.Models." + c.name for c in db_models.Models.__table__.columns] +
            ["db_models.Predictions." + c.name for c in db_models.Predictions.__table__.columns]
    )
    columns = [eval(cname) for cname in column_names]
    return db.query(*columns).join(
        db_models.Predictions,
        db_models.Models.model_id == db_models.Predictions.parent_model_id
    ).filter(
        and_(
            db_models.Models.model_id == model_id,
            db_models.Predictions.prediction_time <= db_models.Models.last_training_time
        )
    ).all(), [c[c.rindex(".") + 1:] for c in column_names]


def get_current_predictions_for_model(db: Session, model_id: str):
    """
    Gets all predictions for a given model ID since the last training time
    """
    column_names = (
            ["db_models.Models." + c.name for c in db_models.Models.__table__.columns] +
            ["db_models.Predictions." + c.name for c in db_models.Predictions.__table__.columns]
    )
    columns = [eval(cname) for cname in column_names]
    return db.query(*columns).join(
        db_models.Predictions,
        db_models.Models.model_id == db_models.Predictions.parent_model_id
    ).filter(
        and_(
            db_models.Models.model_id == model_id,
            db_models.Predictions.prediction_time > db_models.Models.last_training_time
        )
    ).all(), [c[c.rindex(".") + 1:] for c in column_names]


def create_or_update_model(db: Session, model_id: str, model_type: str, last_training_time: datetime):
    existing_record = db.query(db_models.Models).filter(db_models.Models.model_id == model_id).one_or_none()
    if existing_record is not None:
        existing_record.model_type = model_type
        existing_record.last_training_time = last_training_time
        db.commit()
        return logging.info(f"Model {model_id} updated.")
    else:
        model_dict = {"model_id": model_id, "model_type": model_type, "last_training_time": last_training_time}
        model = db_models.Models(**model_dict)
        db.add(model)
        db.commit()
        return logging.info(f"Model {model_id} added.")


def create_prediction(db: Session, prediction: dict):
    pred = db_models.Predictions(**prediction)
    db.add(pred)
    db.commit()
    return logging.info(f"Prediction added for model {prediction['parent_model_id']}.")
