import pytest

from vame_motifs.cli import main


def test_validate_ok(experiment, capsys):
    assert main(["validate", "-c", str(experiment)]) == 0
    assert "2 file(s) valid" in capsys.readouterr().out


def test_user_error_exit_code_and_stderr(tmp_path, capsys):
    assert main(["validate", "-c", str(tmp_path / "missing.yaml")]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("Error:") and "Traceback" not in captured.err


def test_status_before_anything_ran(experiment, capsys):
    assert main(["status", "-c", str(experiment)]) == 0
    assert "not created" in capsys.readouterr().out


def test_gif_options_are_parsed():
    from vame_motifs.cli import build_parser

    args = build_parser().parse_args(["gif", "-c", "x.yaml", "--session", "rat01", "--start", "10", "--label", "motif"])
    assert (args.session, args.start, args.length, args.label, args.subtract_background) == ("rat01", 10, 500, "motif", False)


def test_umap_without_segmentation_says_what_to_run(experiment, capsys):
    assert main(["umap", "-c", str(experiment)]) == 1
    assert "Run 'segment' first" in capsys.readouterr().err


def test_train_more_requires_epochs(experiment, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["train-more", "-c", str(experiment)])
    assert exc.value.code == 2
    assert "--epochs" in capsys.readouterr().err


def test_evaluate_prints_scores_and_suggestions(experiment, capsys, monkeypatch):
    import numpy as np

    from vame_motifs.pipeline import VamePipeline

    def fake_evaluate(self):  # a model that reconstructs perfectly and predicts nothing
        data = np.random.default_rng(0).normal(size=(4, 10, 3))
        future = np.random.default_rng(1).normal(size=(4, 5, 3))
        self.evaluation_path.parent.mkdir(parents=True)
        np.savez(self.evaluation_path, **{f"{split}_{key}": value for split in ("train", "test") for key, value in
                                          [("input", data), ("reconstruction", data),
                                           ("future", future), ("prediction", 0 * future)]})

    monkeypatch.setattr(VamePipeline, "evaluate", fake_evaluate)
    assert main(["evaluate", "-c", str(experiment)]) == 0
    out = capsys.readouterr().out
    assert "reconstruction: 1.00" in out and "Suggestion: Overfitted" in out


def test_status_shows_the_settings_of_the_model_in_the_folder(experiment, capsys):
    from conftest import write_yaml

    from vame_motifs.config import load_config
    from vame_motifs.pipeline import VamePipeline

    pipeline = VamePipeline(load_config(experiment))
    pipeline.project_path.mkdir(parents=True)
    pipeline._write_model_settings()  # as 'prepare' leaves it
    assert main(["status", "-c", str(experiment)]) == 0
    out = capsys.readouterr().out
    assert "settings : min_confidence=0.9, align_center=snout" in out and "zdims=30" in out
    assert "WARNING" not in out

    write_yaml(experiment.parent, exclude="[left_ear]")
    assert main(["status", "-c", str(experiment)]) == 0
    assert "differs (exclude: [] -> ['left_ear'])" in capsys.readouterr().out
