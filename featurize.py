"""Step 1: build examples and run the frozen encoder once; cache features per task.

    python featurize.py                 # mDeBERTa-v3-base (default, ~280M, inference only)
    python featurize.py --lite          # MiniLM-L6 (~107M) for very weak machines
Features are cached in artifacts/features/<split>/<task>.npz, so adding a new
task only encodes that task.
"""
import argparse, glob, json, os, time
import numpy as np


def load_split(split, root="artifacts/features"):
    """Concatenate cached tasks -> X, owner (example index per row), examples, meta."""
    Xs, owners, exs, sec, enc = [], [], [], 0.0, None
    for meta_path in sorted(glob.glob(f"{root}/{split}/*.json")):
        meta = json.load(open(meta_path))
        z = np.load(meta_path[:-5] + ".npz")
        Xs.append(z["X"]); owners.append(z["owner"] + len(exs))
        exs += meta["examples"]; sec += meta["sec"]; enc = meta["encoder"]
    return np.concatenate(Xs), np.concatenate(owners), exs, {"encoder": enc, "sec": sec}


if __name__ == "__main__":
    import data
    from decision_model.encoder import DEFAULT_ENCODER, LITE_ENCODER, Encoder
    from decision_model.model import question_pairs

    ap = argparse.ArgumentParser()
    ap.add_argument("--lite", action="store_true")
    ap.add_argument("--out", default="artifacts/features")
    args = ap.parse_args()
    enc = None
    for split in ["train", "test"]:
        os.makedirs(f"{args.out}/{split}", exist_ok=True)
        by_task = {}
        for ex in data.build(split):
            by_task.setdefault(ex["task"], []).append(ex)
        for task, exs in by_task.items():
            path = f"{args.out}/{split}/{task}"
            if os.path.exists(path + ".npz"):
                continue
            if enc is None:
                enc = Encoder(LITE_ENCODER if args.lite else DEFAULT_ENCODER)
                print("encoder:", enc.name, "device:", enc.device)
            pairs, owner = [], []
            for i, ex in enumerate(exs):
                for p in question_pairs(ex["state"], ex["q"]):
                    pairs.append(p); owner.append(i)
            t = time.time()
            X = enc.featurize(pairs)
            dt = time.time() - t
            print(f"{split}/{task}: {len(exs)} examples, {len(pairs)} pairs, {dt:.0f}s", flush=True)
            np.savez_compressed(path + ".npz", X=X, owner=np.array(owner))
            json.dump({"encoder": enc.name, "examples": exs, "sec": dt}, open(path + ".json", "w"),
                      ensure_ascii=False)
