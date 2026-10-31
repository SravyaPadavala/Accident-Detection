"""BiGRU + temporal-attention sequence classifier (PyTorch)."""
import os
import numpy as np
from .tracker import SEQ_LEN, NUM_FEATURES

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "accident_bigru.pt")

try:
    import torch
    import torch.nn as nn
except (ImportError, OSError):   # lets the rest of the app start without PyTorch
    torch = None
    nn = None

if nn is not None:
    class AccidentBiGRU(nn.Module):
        """input (B, 16, 6) -> logit (B,)"""

        def __init__(self, in_dim=NUM_FEATURES, hidden=32):
            super().__init__()
            self.gru = nn.GRU(in_dim, hidden, batch_first=True, bidirectional=True)
            self.attn = nn.Linear(2 * hidden, 1)                  # one score per frame
            self.head = nn.Sequential(nn.Linear(2 * hidden, 16), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(16, 1))

        def forward(self, x, return_attention=False):
            out, _ = self.gru(x)                                  # (B, T, 2H)
            w = torch.softmax(self.attn(out).squeeze(-1), dim=1)  # (B, T) attention weights
            ctx = (out * w.unsqueeze(-1)).sum(dim=1)              # weighted context vector
            logit = self.head(ctx).squeeze(-1)
            return (logit, w) if return_attention else logit


class AccidentClassifier:
    """Loads a checkpoint and returns P(accident) for a (16, 6) window."""

    def __init__(self, model):
        self.model = model.eval()

    def predict(self, window):
        x = torch.as_tensor(np.asarray(window, np.float32)).unsqueeze(0)
        with torch.no_grad():
            return float(torch.sigmoid(self.model(x))[0])


def load_classifier(path=MODEL_PATH):
    """Return an AccidentClassifier, or None if PyTorch / checkpoint is missing."""
    if torch is None or not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu")
    model = AccidentBiGRU(hidden=ckpt.get("hidden", 32))
    model.load_state_dict(ckpt["state_dict"])
    return AccidentClassifier(model)
