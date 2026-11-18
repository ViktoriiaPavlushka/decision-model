"""Frozen encoder that turns (state, question/option) pairs into feature vectors.

Approach 4 from the assignment (encoder-only): the network itself is never
trained.  We run a pretrained multilingual NLI cross-encoder over the pair
    [CLS] state [SEP] question + option [SEP]
and take
    * the mean of the last hidden states over the *option/question* tokens
      (the second segment) - "average of vectors that correspond to option X";
    * the [CLS] vector;
    * the three NLI logits (entailment / neutral / contradiction).
These features are fed into a logistic regression (see model.py).
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_ENCODER = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
LITE_ENCODER = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Encoder:
    def __init__(self, name: str = DEFAULT_ENCODER, device: str | None = None,
                 max_length: int = 256, batch_size: int = 16):
        self.name = name
        self.device = device or pick_device()
        self.max_length = max_length
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval().to(self.device)
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail_id = labels.get("entailment", 0)
        self.contra_id = labels.get("contradiction", 2)

    @property
    def dim(self) -> int:
        return 2 * self.model.config.hidden_size + self.model.config.num_labels

    def count_tokens(self, pairs: list[tuple[str, str]]) -> int:
        n = 0
        for a, b in pairs:
            n += len(self.tok(a, b, truncation="only_first", max_length=self.max_length)["input_ids"])
        return n

    @torch.no_grad()
    def featurize(self, pairs: list[tuple[str, str]], progress: bool = False) -> np.ndarray:
        """Return array [len(pairs), dim]."""
        out = np.zeros((len(pairs), self.dim), dtype=np.float32)
        # sort by length -> less padding -> faster on 8 GB MacBook
        order = np.argsort([len(a) + len(b) for a, b in pairs])
        it = range(0, len(pairs), self.batch_size)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="encode", unit="batch")
        for s in it:
            idx = order[s:s + self.batch_size]
            prem = [pairs[i][0] for i in idx]
            hyp = [pairs[i][1] for i in idx]
            enc = self.tok(prem, hyp, truncation="only_first", max_length=self.max_length,
                           padding=True, return_tensors="pt")
            # mask of tokens belonging to the second segment (question + option)
            seg = torch.zeros_like(enc["input_ids"], dtype=torch.float32)
            for r in range(len(idx)):
                for c, sid in enumerate(enc.sequence_ids(r)):
                    if sid == 1:
                        seg[r, c] = 1.0
            enc = {k: v.to(self.device) for k, v in enc.items()}
            res = self.model(**enc, output_hidden_states=True)
            h = res.hidden_states[-1]                      # [B, T, H]
            m = seg.to(self.device).unsqueeze(-1)
            mean_opt = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            cls = h[:, 0]
            feats = torch.cat([mean_opt, cls, res.logits], dim=-1).float().cpu().numpy()
            out[idx] = feats
        return out

    def nli_slice(self) -> slice:
        """Columns holding the raw NLI logits (used by the zero-shot baseline)."""
        n = self.model.config.num_labels
        return slice(self.dim - n, self.dim)
