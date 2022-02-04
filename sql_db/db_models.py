from sqlalchemy import Column, ForeignKey, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from sql_db.database import Base


class Models(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True)
    model_type = Column(String)
    last_training_time = Column(DateTime, server_default=func.now())

    predictions = relationship("Predictions", back_populates="parent_model")


class Predictions(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_time = Column(DateTime, server_default=func.now())
    record_id = Column(Integer, index=True)
    prediction = Column(Float)
    prediction_probability = Column(Float, nullable=True)
    ground_truth_label = Column(Float, nullable=True)
    parent_model_id = Column(String, ForeignKey("models.model_id"))

    parent_model = relationship("Models", back_populates="predictions")
