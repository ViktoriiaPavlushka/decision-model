"""Decision model: frozen encoder + logistic-regression heads + calibration.

Question types (same as the /v1/systemone API):
    noul   - yes/no question, output P(yes)
    choice - pick one of the named criteria, output distribution + confidence
    score  - ordinal scale 0..K-1, output distribution, expected score, confidence

Every question is turned into one or more (state, hypothesis) pairs:
    noul   -> 1 pair,   hypothesis = instructions
    choice -> K pairs,  hypothesis = instructions + " -> " + key + ": " + description
    score  -> K pairs,  hypothesis = instructions + " -> " + level + ": " + description
Pair features come from Encoder; two logistic regressions score them:
    head "noul"   : P(answer is yes | pair)
    head "option" : logit that this option is the right one (shared by choice & score)
Options of one question are normalised with a softmax whose temperature is fitted
on held-out data (temperature scaling).  For noul a Platt scaling (a*z + b) is
fitted the same way.  No text is generated, so the output format is guaranteed
and output_tokens is always 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_softmax, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = "dm-0.1.0"


# --------------------------------------------------------------------------- pairs
def normalize_options(qtype: str, criteria) -> list[tuple[str, str]]:
    """Return list of (key, description)."""
    if qtype == "choice":
        if isinstance(criteria, dict):
            return [(str(k), str(v or "")) for k, v in criteria.items()]
        return [(str(k), "") for k in criteria]
    if qtype == "score":
        if isinstance(criteria, dict):
            return [(str(k), str(v)) for k, v in criteria.items()]
        return [(str(i), str(v)) for i, v in enumerate(criteria)]
    raise ValueError(qtype)


def option_hypothesis(instructions: str, key: str, desc: str, qtype: str) -> str:
    if qtype == "score":
        return f"{instructions} -> {key}: {desc}"
    return f"{instructions} -> {key}: {desc}" if desc else f"{instructions} -> {key}"


def question_pairs(state: str, q: dict) -> list[tuple[str, str]]:
    qtype = q["type"]
    if qtype == "noul":
        return [(state, q["instructions"])]
    opts = normalize_options(qtype, q["criteria"])
    return [(state, option_hypothesis(q["instructions"], k, d, qtype)) for k, d in opts]


# --------------------------------------------------------------------------- calibration
def fit_temperature(groups_logits: list[np.ndarray], labels: list[int]) -> float:
    def nll(logt):
        t = np.exp(logt[0])
        return -np.mean([log_softmax(z / t)[y] for z, y in zip(groups_logits, labels)])
    res = minimize(nll, x0=[0.0], method="Nelder-Mead")
    return float(np.exp(res.x[0]))


def fit_platt(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    def nll(p):
        a, b = p
        pr = np.clip(expit(a * z + b), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr))
    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


# --------------------------------------------------------------------------- heads
def make_lr(C: float):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))


@dataclass
class Heads:
    encoder_name: str
    noul: object = None
    option: object = None
    platt: tuple = (1.0, 0.0)
    temperature: dict = field(default_factory=lambda: {"choice": 1.0, "score": 1.0})

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "Heads":
        return joblib.load(path)

    # raw scores ---------------------------------------------------------
    def noul_logit(self, X: np.ndarray) -> np.ndarray:
        return self.noul.decision_function(X)

    def option_logit(self, X: np.ndarray) -> np.ndarray:
        return self.option.decision_function(X)

    # calibrated ---------------------------------------------------------
    def p_yes(self, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
        z = self.noul_logit(X)
        if calibrated:
            a, b = self.platt
            return expit(a * z + b)
        return expit(z)

    def option_probs(self, X: np.ndarray, qtype: str, calibrated: bool = True) -> np.ndarray:
        z = self.option_logit(X)
        t = self.temperature.get(qtype, 1.0) if calibrated else 1.0
        return softmax(z / t)


# --------------------------------------------------------------------------- public model
class DecisionModel:
    def __init__(self, heads_path: str, encoder=None):
        from .encoder import Encoder
        self.heads = Heads.load(heads_path)
        self.encoder = encoder or Encoder(self.heads.encoder_name)

    def predict(self, state: str, questions: dict) -> dict:
        # encode all pairs of all questions in one batch
        spans, pairs = {}, []
        for name, q in questions.items():
            qp = question_pairs(state, q)
            spans[name] = (len(pairs), len(pairs) + len(qp))
            pairs.extend(qp)
        X = self.encoder.featurize(pairs) if pairs else np.zeros((0, 1))
        answers = {}
        for name, q in questions.items():
            s, e = spans[name]
            qtype = q["type"]
            if qtype == "noul":
                answers[name] = {"type": "noul", "noul": round(float(self.heads.p_yes(X[s:e])[0]), 4)}
                continue
            opts = normalize_options(qtype, q["criteria"])
            p = self.heads.option_probs(X[s:e], qtype)
            if qtype == "choice":
                ranked = sorted(zip([k for k, _ in opts], p), key=lambda kv: -kv[1])
                answers[name] = {
                    "type": "choice",
                    "choice": ranked[0][0],
                    "probabilities": {k: round(float(v), 4) for k, v in ranked},
                    "confidence": round(float(ranked[0][1]), 4),
                }
            else:
                answers[name] = {
                    "type": "score",
                    "score": round(float(np.dot(np.arange(len(p)), p)), 4),
                    "confidence": round(float(p.max()), 4),
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "legend": {str(i): d for i, (_, d) in enumerate(opts)},
                }
        return {
            "model": VERSION,
            "answers": answers,
            "usage": {"input_tokens": self.encoder.count_tokens(pairs), "output_tokens": 0},
        }
