import os
import uvicorn
import boto3
import json

from fastapi import FastAPI
from mangum import Mangum
from pydantic import BaseModel
from src.main import QueryResponse
from src.main import query_req

app = FastAPI()

class SubmitQueryRequest(BaseModel):
    user_input: str
 
@app.get("/")
def index():
    return {"Hello": "World"}


@app.post("/submit_query")
def submit_query_endpoint(request: SubmitQueryRequest) -> QueryResponse:
    query_response = query_req(request.user_input)
    return query_response

if __name__ == "__main__":
    # Run this as a server directly.
    port = 8000
    print(f"Running the FastAPI server on port {port}.")
    uvicorn.run("app_api_handler:app", host="localhost", port=port)