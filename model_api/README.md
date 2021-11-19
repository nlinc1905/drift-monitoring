# Model API

This API service is built with [FastAPI](https://fastapi.tiangolo.com/).  Its purpose is to receive requests for text 
classification, and return responses for use in the drift monitoring service.  In production, this API is replaced by 
CisionAI's API.  
<br/>
An important difference between CisionAI and this API is that this API separates feature engineering from modeling, 
so that it is possible to get feature information from the model, for the purpose of data drift detection.  CisionAI 
packages the model as a scikit-learn pipeline, meaning the model and its features are versioned together.  

## Swagger Documentation

This API's swagger documentation can be found by going to [localhost:8000/docs](localhost:8000/docs) after the service 
is up and running.  To get it up and running, run `docker-compose up`.  The swagger documentation shows example 
requests and responses.
