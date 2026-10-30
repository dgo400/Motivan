"""How well the trained model reconstructs and predicts, and what to do about it.

'evaluate' (pipeline.py) runs the model on random time windows of the training
and test sets and saves, per set, the input windows, their reconstruction, the
frames that really came next and the model's prediction of them. This module
reads that file and scores it; no VAME or torch needed.

Scores are the share of the signal's variance the model explains (R²):
1 is perfect, 0 is no better than a flat line at the mean, negative is worse.
They do not depend on the units of the data, unlike VAME's summed losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Thresholds for the advice. Predicting the future is harder than reconstructing
# the past, so prediction gets lower ones.
GOOD_RECONSTRUCTION, POOR_RECONSTRUCTION = 0.8, 0.5
GOOD_PREDICTION, POOR_PREDICTION = 0.5, 0.2
OVERFIT_GAP = 0.15  # training data scoring this much better than test data


@dataclass
class Scores:
    reconstruction: float  # test set
    prediction: float | None  # None: the model has no prediction decoder
    train_reconstruction: float
    train_prediction: float | None


@dataclass
class Advice:
    level: str  # "ok" or "warning"
    text: str


def explained_variance(original: np.ndarray, estimate: np.ndarray) -> float:
    """R² of ``estimate`` over all windows, frames and features (arrays: windows x frames x features)."""
    mean = original.mean(axis=(0, 1), keepdims=True)  # per feature
    total = ((original - mean) ** 2).sum()
    return float(1 - ((original - estimate) ** 2).sum() / total) if total else 0.0


def load(path: Path) -> dict[str, np.ndarray] | None:
    return dict(np.load(path)) if path.exists() else None


def scores(data: dict[str, np.ndarray]) -> Scores:
    def score(kind: str, split: str) -> float | None:
        if f"{split}_{kind}" not in data:
            return None
        original = data[f"{split}_input" if kind == "reconstruction" else f"{split}_future"]
        return explained_variance(original, data[f"{split}_{kind}"])

    return Scores(score("reconstruction", "test"), score("prediction", "test"),
                  score("reconstruction", "train"), score("prediction", "train"))


def quality(value: float | None, good: float, poor: float) -> str:
    if value is None:
        return "-"
    return "good" if value >= good else "poor" if value < poor else "fair"


def advice(s: Scores, zdims: int) -> list[Advice]:
    """What the scores suggest doing, most important first."""
    rec = quality(s.reconstruction, GOOD_RECONSTRUCTION, POOR_RECONSTRUCTION)
    pred = quality(s.prediction, GOOD_PREDICTION, POOR_PREDICTION)
    retrain = "then tick Force and Run all to retrain."
    fewer_zdims = f"fewer latent dimensions (Advanced → zdims, now {zdims}; try {max(zdims // 2, 2)})"
    out = []

    if rec == "good" and pred == "poor":
        out.append(Advice("warning", (
            "Overfitted: the model reconstructs what it saw well but cannot predict what comes next, so it "
            "memorises each window instead of learning how the movement goes on. Its motifs may follow posture "
            f"more than movement. Try {fewer_zdims}, or improve the data quality: exclude poorly tracked body parts "
            "(see 'validate') or raise the minimum confidence; " + retrain)))
    pairs = [(s.train_reconstruction, s.reconstruction, "reconstructs")]
    if s.prediction is not None and s.train_prediction is not None:
        pairs.append((s.train_prediction, s.prediction, "predicts"))
    for train, test, verb in pairs:
        if train - test > OVERFIT_GAP:
            out.append(Advice("warning", (
                f"Overfitted: the model {verb} the training data much better than unseen test data "
                f"({train:.2f} vs {test:.2f}). Try {fewer_zdims}, or train on more recordings; " + retrain)))
    if rec == "poor":
        out.append(Advice("warning", (
            "Poor reconstruction: the model has not learned the poses (underfit), or the data are too noisy. "
            "If the loss was still going down when training stopped, train it further with 'train-more'. "
            "Otherwise check the tracking ('validate' warnings) and exclude poorly tracked body parts, "
            f"or try more latent dimensions (zdims, now {zdims}); " + retrain)))
    elif rec == "fair":
        out.append(Advice("warning", (
            "Fair reconstruction: the model could still improve. Training it further with 'train-more' "
            "often helps; if it does not, check the tracking quality.")))
    if rec == "fair" and pred == "poor":
        out.append(Advice("warning", (
            "Poor prediction: the model has not learned how the movement goes on. Train it further "
            "('train-more') or check the tracking quality.")))
    if not out:
        out.append(Advice("ok", (
            f"No problems found: reconstruction is {rec} and prediction is {pred} on unseen test data, "
            "and about as good as on the training data. Go on with 'segment' (or Run all).")))
    return out
