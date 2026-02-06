from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()


class SolveRequest(BaseModel):
    names: List[str]
    dates: List[str]


class SolveResponse(BaseModel):
    names: List[str]
    dates: List[str]
    matrix: List[List[str]]


@app.get("/")
def root():
    return {"status": "ok", "service": "lajk-taxi-scheduler"}


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    # PLACEHOLDER – wszyscy OFF
    matrix = [["OFF" for _ in req.dates] for _ in req.names]

    return SolveResponse(
        names=req.names,
        dates=req.dates,
        matrix=matrix
    )
