"""Scores and advice from the model's evaluation (no VAME run needed)."""
import numpy as np
import pytest

from vame_motifs.evaluation import Scores, advice, explained_variance, scores


def test_explained_variance_is_one_when_perfect_and_zero_at_the_mean():
    original = np.random.default_rng(0).normal(size=(4, 10, 3))
    assert explained_variance(original, original) == pytest.approx(1)
    assert explained_variance(original, np.broadcast_to(original.mean(axis=(0, 1)), original.shape)) == pytest.approx(0)
    assert explained_variance(original, -original) < 0  # worse than the mean


def test_scores_compare_each_output_with_its_data():
    data = np.random.default_rng(0).normal(size=(4, 10, 3))
    future = np.random.default_rng(1).normal(size=(4, 5, 3))
    saved = {"test_input": data, "test_reconstruction": data, "test_future": future, "test_prediction": 0 * future,
             "train_input": data, "train_reconstruction": data, "train_future": future, "train_prediction": future}
    s = scores(saved)
    assert s.reconstruction == pytest.approx(1) and s.train_prediction == pytest.approx(1)
    assert s.prediction < 0.1


def test_no_prediction_decoder_scores_reconstruction_only():
    data = np.ones((2, 10, 3)) * np.arange(10)[None, :, None]
    s = scores({"test_input": data, "test_reconstruction": data, "train_input": data, "train_reconstruction": data})
    assert s.prediction is None and s.reconstruction == pytest.approx(1)
    assert advice(s, zdims=30)[0].level == "ok"


def _levels_and_text(s: Scores) -> tuple[list[str], str]:
    items = advice(s, zdims=30)
    return [a.level for a in items], " ".join(a.text for a in items)


def test_good_reconstruction_but_poor_prediction_is_overfitting():
    levels, text = _levels_and_text(Scores(0.95, 0.1, 0.95, 0.1))
    assert levels == ["warning"]
    assert "Overfitted" in text and "zdims, now 30; try 15" in text and "data quality" in text


def test_much_better_on_training_data_is_overfitting():
    levels, text = _levels_and_text(Scores(0.97, 0.57, 0.97, 0.84))
    assert levels == ["warning"]
    assert "predicts the training data much better" in text and "0.84 vs 0.57" in text


def test_poor_reconstruction_suggests_training_more_or_checking_the_data():
    levels, text = _levels_and_text(Scores(0.3, 0.1, 0.3, 0.1))
    assert levels == ["warning"]
    assert "underfit" in text and "train-more" in text and "validate" in text


def test_a_good_model_gets_the_go_ahead():
    levels, text = _levels_and_text(Scores(0.9, 0.6, 0.92, 0.65))
    assert levels == ["ok"] and "segment" in text
