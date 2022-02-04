# Model API

**This API is only used by the examples.  In production, your model's API will replace this model API.**

This API service is built with [FastAPI](https://fastapi.tiangolo.com/).  Its purpose is to receive requests for text 
classification, and return responses for use in the drift monitoring service.  In production, this API is replaced by 
your model's API.  

## Swagger Documentation

This API's swagger documentation can be found by going to [localhost:8000/docs](localhost:8000/docs) after the service 
is up and running.  To get it up and running, run `docker-compose up`.  The swagger documentation shows example 
requests and responses.

You can also get this API up and running by itself, just to test the example requests shown in the Swagger 
documentation.  To do that, you can `cd model_api` and run `uvicorn main:app --reload`.  Note that if you do this, you 
will need to update the ARTIFACTS_DIR in main.py first, since it is a relative path that is written to run from the 
root directory.  Change it to: `ARTIFACTS_DIR = "../data/artifacts/"`.  Running the API by itself might be useful for 
testing quick changes while developing.
