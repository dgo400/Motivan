"""The GUI's form <-> YAML conversion; the window itself is not opened."""
import yaml

from vame_motifs.cli import COMMANDS
from vame_motifs.config import load_config
from vame_motifs.gui import DEFAULTS, STEP_COMMANDS, form_to_yaml, read_bodyparts, yaml_to_form


def test_yaml_round_trip_is_read_by_the_cli(experiment):
    values = yaml_to_form(yaml.safe_load(experiment.read_text()))
    assert values["pose_files"] == "pose/" and values["n_clusters"] == "5"

    values["n_clusters"] = "7"
    values["exclude"] = ["left_ear"]
    experiment.write_text(yaml.safe_dump(form_to_yaml(values), sort_keys=False))

    cfg = load_config(experiment)
    assert cfg.n_clusters == 7 and cfg.exclude == ["left_ear"]
    assert cfg.time_window == 30 and cfg.videos == []


def test_every_step_button_is_a_cli_command():
    assert "train-more" in STEP_COMMANDS
    assert set(STEP_COMMANDS) <= set(COMMANDS)


def test_missing_sections_keep_defaults():
    assert yaml_to_form({}) == DEFAULTS


def test_read_bodyparts(experiment):
    pose = min((experiment.parent / "pose").glob("*.csv"))
    assert read_bodyparts(pose) == ["snout", "left_ear", "right_ear", "centre", "tailbase"]


def test_trained_model_is_found_before_training_again(experiment):
    from vame_motifs.gui import trained_model
    from vame_motifs.pipeline import VamePipeline

    assert trained_model(experiment) is None  # nothing trained yet: no question asked

    model_path = VamePipeline(load_config(experiment)).model_path
    model_path.parent.mkdir(parents=True)
    model_path.touch()
    assert trained_model(experiment) == model_path

    experiment.write_text("input: [not, a, section]\n")  # invalid experiment: the command reports it
    assert trained_model(experiment) is None


def test_bouts_are_runs_of_frames():
    import numpy as np

    from vame_motifs.viewer import bouts

    assert bouts(np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)).tolist() == [[1, 2], [5, 5], [7, 9]]
    assert bouts(np.zeros(4, dtype=bool)).tolist() == []


def test_short_names_are_the_last_digits_of_the_video_name():
    from vame_motifs.viewer import short_names

    long = "01022023_001_16236-{}DLC_Resnet50_Test-ExampleSep16shuffle1_snapshot_best-84_filtered"
    assert short_names([long.format(62755), long.format(62764)]) == {
        long.format(62755): "62755", long.format(62764): "62764"}
    # No digits, or the same digits twice: the full name.
    assert short_names(["mouseDLC_x", "a_7DLC", "b_7DLC", "rat01DLC"]) == {
        "mouseDLC_x": "mouseDLC_x", "a_7DLC": "a_7DLC", "b_7DLC": "b_7DLC", "rat01DLC": "01"}


def test_renamed_recordings_are_saved_and_read_back(experiment):
    import pytest

    from vame_motifs.viewer import Results

    cfg = load_config(experiment)  # rat01DLC_resnet50, rat02DLC_resnet50
    results = Results(cfg, "hmm")
    assert results.display == {"rat01DLC_resnet50": "01", "rat02DLC_resnet50": "02"}

    results.rename("rat02DLC_resnet50", "  control  ")
    assert Results(cfg, "hmm").display["rat02DLC_resnet50"] == "control"  # saved in the output folder
    with pytest.raises(ValueError, match="already the name"):
        results.rename("rat01DLC_resnet50", "control")
    results.rename("rat02DLC_resnet50", "")  # empty: back to the short name
    assert Results(cfg, "hmm").display["rat02DLC_resnet50"] == "02"


def test_usage_is_per_recording(experiment):
    import numpy as np

    from vame_motifs.pipeline import VamePipeline
    from vame_motifs.viewer import Results

    cfg = load_config(experiment)
    for pose_path, motif in zip(cfg.pose_files, (0, 3)):
        label_file = VamePipeline(cfg).label_file(pose_path.stem, "hmm")
        label_file.parent.mkdir(parents=True)
        np.save(label_file, np.full(100, motif))

    results = Results(cfg, "hmm")
    assert results.usage("rat01DLC_resnet50").tolist() == [1, 0, 0, 0, 0]
    assert results.usage("rat02DLC_resnet50").tolist() == [0, 0, 0, 1, 0]
    assert results.usage().tolist() == [0.5, 0, 0, 0.5, 0]  # all recordings together


def test_saving_keeps_what_the_form_does_not_show(experiment):
    original = yaml.safe_load(experiment.read_text())
    original["project_name"] = "pilot"
    original["notes"] = {"author": "me"}
    original["advanced"]["beta"] = 2
    values = yaml_to_form(original)
    values["zdims"], values["videos"] = "10", ""

    saved = form_to_yaml(values, original)

    assert saved["project_name"] == "pilot" and saved["notes"] == {"author": "me"}
    assert saved["advanced"]["beta"] == 2 and saved["advanced"]["zdims"] == 10  # the form's field wins
    assert original["advanced"] == {"time_window": 30, "beta": 2}  # the opened YAML is not changed
    assert list(saved)[:2] == list(original)[:2]  # sections stay where they were


def test_emptied_videos_are_removed():
    original = {"input": {"pose_files": "pose/", "videos": "videos/", "fps": 30}}
    values = yaml_to_form(original)
    values["videos"] = ""
    assert "videos" not in form_to_yaml(values, original)["input"]


def test_write_yaml_replaces_the_file_whole(tmp_path, monkeypatch):
    import pytest

    from vame_motifs.gui import write_yaml

    path = tmp_path / "experiment.yaml"
    write_yaml(path, {"output": "outputs"})
    assert yaml.safe_load(path.read_text()) == {"output": "outputs"}

    def broken_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(yaml, "safe_dump", broken_dump)
    with pytest.raises(OSError):
        write_yaml(path, {"output": "other"})
    assert yaml.safe_load(path.read_text()) == {"output": "outputs"}  # the old file is intact
    assert [p.name for p in tmp_path.iterdir()] == ["experiment.yaml"]  # no temporary file left


def test_experiment_name_is_in_the_form(experiment):
    original = yaml.safe_load(experiment.read_text())
    assert yaml_to_form(original)["project_name"] == "vame_project"  # not in the YAML: the default

    values = yaml_to_form(original | {"project_name": "z30"})
    assert values["project_name"] == "z30"
    values["project_name"] = "  z10 "
    experiment.write_text(yaml.safe_dump(form_to_yaml(values, original), sort_keys=False))
    assert load_config(experiment).project_name == "z10"
