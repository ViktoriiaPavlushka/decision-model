"""Step 2: train logistic-regression heads on cached features and calibrate them.

Runs on CPU in about a minute, no neural network is trained.
    * 5-fold cross-validation grouped by example (all options of one question are
      in the same fold) selects the regularisation strength C: the strongest
      regularisation whose CV accuracy is within TOL of the best one (simpler
      model -> better transfer to unseen tasks);
    * out-of-fold logits of the selected C fit the calibration
      (Platt scaling for noul, temperature scaling for choice / score),
      so calibration is fitted on predictions for data the model has not seen;
    * the final heads are refit on all training data.
"""
import argparse
import json

import numpy as np
from scipy.special import expit, log_softmax

from decision_model.model import Heads, fit_platt, fit_temperature, make_lr
from featurize import load_split

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="artifacts/features")
ap.add_argument("--out", default="artifacts/heads.joblib")
ap.add_argument("--folds", type=int, default=5)
args = ap.parse_args()

X, owner, exs, meta = load_split("train", args.features)
rows_of = {}
for r, o in enumerate(owner):
    rows_of.setdefault(int(o), []).append(r)
rng = np.random.default_rng(0)
fold_of = rng.integers(0, args.folds, size=len(exs))       # fold per example
Cs = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
TOL = 0.005


def select(results):
    """results: list of (acc, C, *payload) in increasing C -> pick by the tolerance rule."""
    top = max(r[0] for r in results)
    return next(r for r in results if r[0] >= top - TOL)


def design(kind):
    """kind 'noul' | 'option' -> (rows, y, example id per row)."""
    idx, y, ex_id = [], [], []
    for i, ex in enumerate(exs):
        if kind == "noul" and ex["type"] == "noul":
            idx += rows_of[i]; y.append(ex["label"]); ex_id.append(i)
        elif kind == "option" and ex["type"] != "noul":
            rr = rows_of[i]
            idx += rr; y += [int(k == ex["label"]) for k in range(len(rr))]; ex_id += [i] * len(rr)
    return np.array(idx), np.array(y), np.array(ex_id)


def oof_logits(kind, C):
    idx, y, ex_id = design(kind)
    z = np.zeros(len(idx))
    folds = fold_of[ex_id]
    for f in range(args.folds):
        tr, te = folds != f, folds == f
        z[te] = make_lr(C).fit(X[idx[tr]], y[tr]).decision_function(X[idx[te]])
    return idx, y, ex_id, z


def option_groups(ex_id, z, qtype):
    out = []
    for i in dict.fromkeys(ex_id):
        if exs[i]["type"] == qtype:
            out.append((z[ex_id == i], exs[i]["label"]))
    return out


heads = Heads(encoder_name=meta["encoder"])
log = {"encoder": meta["encoder"], "n_examples": len(exs), "folds": args.folds, "cv_accuracy": {}}

# ---------------- noul head
res = []
for C in Cs:
    idx, y, ex_id, z = oof_logits("noul", C)
    acc = float(((z > 0) == y).mean())
    log["cv_accuracy"][f"noul_C={C}"] = acc
    print(f"noul   C={C:<6} cv_acc={acc:.4f}", flush=True)
    res.append((acc, C, y, z))
acc, C, y, z = select(res)
heads.platt = fit_platt(z, y)
a, b = heads.platt
nll = lambda p: float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
log["noul"] = {"C": C, "cv_acc": acc, "platt": heads.platt,
               "oof_nll_before_after": [nll(np.clip(expit(z), 1e-7, 1 - 1e-7)),
                                        nll(np.clip(expit(a * z + b), 1e-7, 1 - 1e-7))]}
idx, y, _ = design("noul")
heads.noul = make_lr(C).fit(X[idx], y)
print("noul:", log["noul"])

# ---------------- option head (choice + score share it)
res = []
for C in Cs:
    idx, y, ex_id, z = oof_logits("option", C)
    corr = [int(np.argmax(g) == lab) for qt in ("choice", "score") for g, lab in option_groups(ex_id, z, qt)]
    acc = float(np.mean(corr))
    log["cv_accuracy"][f"option_C={C}"] = acc
    print(f"option C={C:<6} cv_acc={acc:.4f}", flush=True)
    res.append((acc, C, ex_id, z))
acc, C, ex_id, z = select(res)
log["option"] = {"C": C, "cv_acc": acc}
for qt in ("choice", "score"):
    gl = option_groups(ex_id, z, qt)
    t = fit_temperature([g for g, _ in gl], [lab for _, lab in gl])
    heads.temperature[qt] = t
    log["option"][f"T_{qt}"] = t
    log["option"][f"oof_nll_{qt}_before_after"] = [
        float(-np.mean([log_softmax(g)[lab] for g, lab in gl])),
        float(-np.mean([log_softmax(g / t)[lab] for g, lab in gl]))]
idx, y, _ = design("option")
heads.option = make_lr(C).fit(X[idx], y)
print("option:", log["option"])

heads.save(args.out)
json.dump(log, open("results/train_log.json", "w"), indent=2, ensure_ascii=False)
print("saved", args.out)
