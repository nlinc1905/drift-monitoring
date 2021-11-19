from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API working"}
