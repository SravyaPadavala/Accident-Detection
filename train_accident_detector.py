"""Train the BiGRU-attention classifier on synthetic kinematic sequences.
Run:  python train_accident_detector.py
Prints per-epoch validation accuracy / precision / recall / F1 and saves the
best checkpoint (by F1) to models/accident_bigru.pt
"""
import os
import numpy as np
import torch
import torch.nn as nn
from detector.accident_model import AccidentBiGRU, MODEL_PATH
from detector.synthetic import make_dataset


def _metrics(logits, y):
    pred = (torch.sigmoid(logits) >= 0.5).float()
    tp = float(((pred == 1) & (y == 1)).sum()); fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum()); tn = float(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": (tp + tn) / len(y), "precision": prec, "recall": rec, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def train(epochs=20, n_samples=1600, out_path=MODEL_PATH, seed=42, verbose=True):
    torch.manual_seed(seed)
    X, y = make_dataset(n_samples, seed)
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int(0.8 * len(X))                                    # 80 / 20 split
    Xtr, ytr = torch.tensor(X[idx[:cut]]), torch.tensor(y[idx[:cut]])
    Xva, yva = torch.tensor(X[idx[cut:]]), torch.tensor(y[idx[cut:]])
    model = AccidentBiGRU()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best = {"f1": -1}
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            m = _metrics(model(Xva), yva)
        if verbose:
            print(f"epoch {ep:02d}  loss {loss.item():.4f}  val_acc {m['acc']:.3f}  "
                  f"precision {m['precision']:.3f}  recall {m['recall']:.3f}  F1 {m['f1']:.3f}")
        if m["f1"] > best["f1"]:
            best = m
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "hidden": 32}, out_path)
    if verbose:
        print(f"\nBest validation F1 = {best['f1']:.3f}   saved -> {out_path}")
        print(f"Confusion matrix (best epoch): TP={best['tp']} FP={best['fp']} FN={best['fn']} TN={best['tn']}")
        print("NOTE: this is accuracy on SYNTHETIC validation data, not real-world accuracy.")
    return best


if __name__ == "__main__":
    train()
