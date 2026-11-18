"""Step 3: evaluate on the test split (in-domain + held-out tasks).

Compares three systems that share the same frozen encoder:
    nli_zero_shot  - raw NLI entailment, no training (approach 1, baseline)
    dm_uncal       - our logistic-regression heads, no calibration
    dm             - our heads + Platt / temperature calibration (final model)
Metrics: accuracy, ECE (15 bins), Brier, NLL, MAE for score; output tokens = 0.
"""
import json
import time

import matplotlib
import numpy as np
from scipy.special import expit, softmax

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from decision_model.model import Heads  # noqa: E402
from featurize import load_split  # noqa: E402

FEAT = "artifacts/features"
heads = Heads.load("artifacts/heads.joblib")
X, owner, exs, meta = load_split("test", FEAT)
rows_of = {}
for r, o in enumerate(owner):
    rows_of.setdefault(int(o), []).append(r)
NLI = slice(X.shape[1] - 3, X.shape[1])          # entailment, neutral, contradiction
ENT, CON = 0, 2


def predict(ex, rr, system):
    """Return probability vector over answers (noul: [P(no), P(yes)])."""
    x = X[rr]
    if ex["type"] == "noul":
        if system == "nli_zero_shot":
            l = x[0, NLI]
            p = softmax([l[CON], l[ENT]])[1]
        else:
            p = heads.p_yes(x, calibrated=(system == "dm"))[0]
        return np.array([1 - p, p])
    if system == "nli_zero_shot":
        return softmax(x[:, NLI][:, ENT])
    return heads.option_probs(x, ex["type"], calibrated=(system == "dm"))


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


def metrics(recs):
    conf = [r["conf"] for r in recs]
    corr = [r["correct"] for r in recs]
    out = {
        "n": len(recs),
        "accuracy": float(np.mean(corr)),
        "ece": ece(conf, corr),
        "brier": float(np.mean([r["brier"] for r in recs])),
        "nll": float(np.mean([r["nll"] for r in recs])),
        "mean_conf": float(np.mean(conf)),
    }
    sc = [r for r in recs if r["type"] == "score"]
    if sc:
        out["mae_expected_score"] = float(np.mean([abs(r["exp"] - r["label"]) for r in sc]))
    return out


SYSTEMS = ["nli_zero_shot", "dm_uncal", "dm"]
records = {s: [] for s in SYSTEMS}
for i, ex in enumerate(exs):
    for s in SYSTEMS:
        p = predict(ex, rows_of[i], s)
        y = ex["label"]
        onehot = np.eye(len(p))[y]
        records[s].append(dict(
            task=ex["task"], type=ex["type"], lang=ex["lang"], held_out=ex["held_out"], label=y,
            conf=float(p.max()), correct=int(np.argmax(p) == y),
            brier=float(np.sum((p - onehot) ** 2)), nll=float(-np.log(max(p[y], 1e-9))),
            exp=float(np.dot(np.arange(len(p)), p)), probs=p.tolist()))

res = {"by_task": {}, "by_type": {}, "overall": {}}
tasks = list(dict.fromkeys(ex["task"] for ex in exs))
for s in SYSTEMS:
    R = records[s]
    res["overall"][s] = {
        "in_domain": metrics([r for r in R if not r["held_out"]]),
        "held_out": metrics([r for r in R if r["held_out"]]),
    }
    for t in tasks:
        res["by_task"].setdefault(t, {})[s] = metrics([r for r in R if r["task"] == t])
    for ty in ["noul", "choice", "score"]:
        for part, ho in [("in_domain", False), ("held_out", True)]:
            sub = [r for r in R if r["type"] == ty and r["held_out"] == ho]
            res["by_type"].setdefault(f"{ty}/{part}", {})[s] = metrics(sub)
for t in tasks:
    ex0 = next(e for e in exs if e["task"] == t)
    res["by_task"][t]["meta"] = {"type": ex0["type"], "lang": ex0["lang"], "held_out": ex0["held_out"]}

# --------------------------------------------------------------- reliability diagrams
COL = {"nli_zero_shot": "#1baf7a", "dm_uncal": "#eb6834", "dm": "#2a78d6"}
LBL = {"nli_zero_shot": "NLI zero-shot (без навчання)", "dm_uncal": "LR без калібрування",
       "dm": "LR + калібрування (фінальна)"}


def reliability(ax, R_by_sys, title, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#9a9893", label="ідеальна калібрація")
    for s in SYSTEMS:
        conf = np.array([r["conf"] for r in R_by_sys[s]])
        corr = np.array([r["correct"] for r in R_by_sys[s]], float)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean()); ys.append(corr[m].mean())
        e = ece(conf, corr)
        ax.plot(xs, ys, marker="o", ms=5, lw=2, color=COL[s], label=f"{LBL[s]} (ECE={e:.3f})")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("впевненість моделі"); ax.set_ylabel("частка правильних")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, loc="upper left", frameon=False)


for part, ho in [("in_domain", False), ("held_out", True)]:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, ty, name in zip(axes, ["noul", "choice", "score"], ["Noul (так/ні)", "Choice", "Score"]):
        reliability(ax, {s: [r for r in records[s] if r["type"] == ty and r["held_out"] == ho]
                         for s in SYSTEMS}, f"{name} — {'знайомі задачі' if not ho else 'нові задачі'}")
    fig.tight_layout()
    fig.savefig(f"results/reliability_{part}.png", dpi=160)
    plt.close(fig)

# accuracy bar chart per task (final model vs baseline)
fig, ax = plt.subplots(figsize=(12, 4.2))
w = 0.38
xs = np.arange(len(tasks))
for k, s in enumerate(["nli_zero_shot", "dm"]):
    v = [res["by_task"][t][s]["accuracy"] for t in tasks]
    ax.bar(xs + (k - 0.5) * w, v, w - 0.03, color=COL[s], label=LBL[s])
ax.set_xticks(xs)
ax.set_xticklabels([t + ("*" if res["by_task"][t]["meta"]["held_out"] else "") for t in tasks],
                   rotation=30, ha="right", fontsize=9)
ax.set_ylabel("accuracy"); ax.set_ylim(0, 1)
ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=9)
ax.set_title("Accuracy по задачах (* — задача не бачена під час навчання)", fontsize=11)
fig.tight_layout(); fig.savefig("results/accuracy_by_task.png", dpi=160); plt.close(fig)

# --------------------------------------------------------------- speed of the full model
from decision_model import DecisionModel  # noqa: E402

EXAMPLE = json.load(open("examples/request.json"))
dm = DecisionModel("artifacts/heads.joblib")
dm.predict(EXAMPLE["state"], EXAMPLE["questions"])
t = time.time()
for _ in range(5):
    out = dm.predict(EXAMPLE["state"], EXAMPLE["questions"])
res["speed"] = {"sec_per_request": (time.time() - t) / 5, "device": dm.encoder.device,
                "encoder": dm.encoder.name, "usage": out["usage"],
                "featurize_sec_train": load_split("train", FEAT)[3]["sec"],
                "featurize_sec_test": meta["sec"]}
json.dump(out, open("examples/response.json", "w"), indent=2, ensure_ascii=False)
json.dump(res, open("results/metrics.json", "w"), indent=2, ensure_ascii=False)

# --------------------------------------------------------------- console summary
print(f"{'task':18} {'type':6} {'lang':4} {'held':5} | {'acc0':>6} {'acc':>6} | {'ece0':>6} {'eceU':>6} {'ece':>6}")
for t in tasks:
    m = res["by_task"][t]
    print(f"{t:18} {m['meta']['type']:6} {m['meta']['lang']:4} {str(m['meta']['held_out']):5} | "
          f"{m['nli_zero_shot']['accuracy']:6.3f} {m['dm']['accuracy']:6.3f} | "
          f"{m['nli_zero_shot']['ece']:6.3f} {m['dm_uncal']['ece']:6.3f} {m['dm']['ece']:6.3f}")
for k, v in res["by_type"].items():
    print(k, {s: (round(v[s]["accuracy"], 3), round(v[s]["ece"], 3)) for s in SYSTEMS})
print("speed:", res["speed"])
