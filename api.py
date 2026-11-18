"""HTTP API compatible with the /v1/systemone request format.

    uvicorn api:app --port 8000
    curl -s localhost:8000/v1/systemone -H 'Content-Type: application/json' -d @examples/request.json
"""
from typing import Literal, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, model_validator

from decision_model import DecisionModel


class Question(BaseModel):
    type: Literal["noul", "choice", "score"]
    instructions: str
    criteria: Union[dict[str, str], list[str], None] = None

    @model_validator(mode="after")
    def check(self):
        if self.type != "noul" and not self.criteria:
            raise ValueError(f"'{self.type}' question needs non-empty 'criteria'")
        if self.type != "noul" and len(self.criteria) < 2:
            raise ValueError("at least two criteria are required")
        return self


class Request(BaseModel):
    model: str = "dm-latest"
    state: str
    questions: dict[str, Question]


app = FastAPI(title="Decision model (Jev-like)")
_model: DecisionModel | None = None


def model() -> DecisionModel:
    global _model
    if _model is None:
        _model = DecisionModel("artifacts/heads.joblib")
    return _model


@app.post("/v1/systemone")
def systemone(req: Request):
    if not req.questions:
        raise HTTPException(400, "questions must not be empty")
    qs = {k: q.model_dump(exclude_none=True) for k, q in req.questions.items()}
    return model().predict(req.state, qs)


@app.get("/health")
def health():
    return {"ok": True}
