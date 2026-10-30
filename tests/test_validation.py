import pytest
from conftest import write_dlc_csv, write_video, write_yaml

from vame_motifs.config import ConfigError, load_config
from vame_motifs.io import PoseFileError
from vame_motifs.pipeline import vame_settings
from vame_motifs.validation import validate


def test_valid_experiment(experiment):
    _, reports = validate(load_config(experiment))
    assert [r.n_frames for r in reports] == [200, 150]
    assert all(not r.warnings for r in reports)


def test_warns_on_poor_tracking(experiment):
    write_dlc_csv(experiment.parent / "pose" / "rat01DLC_resnet50.csv", bad="left_ear")
    _, reports = validate(load_config(experiment))
    assert any("left_ear" in w for w in reports[0].warnings)


def test_unknown_bodypart_lists_available(experiment):
    write_yaml(experiment.parent, align_center="nose")
    with pytest.raises(ConfigError, match="Unknown body part 'nose'. Available: snout"):
        validate(load_config(experiment))


def test_mismatched_bodyparts(experiment):
    write_dlc_csv(experiment.parent / "pose" / "rat03.csv", bodyparts=["snout", "centre", "tailbase"])
    with pytest.raises(PoseFileError, match="All files must match"):
        validate(load_config(experiment))


def test_config_rejects_bad_values(experiment):
    write_yaml(experiment.parent, method="spectral")
    with pytest.raises(ConfigError, match="motifs.method"):
        load_config(experiment)
    write_yaml(experiment.parent, n_clusters='"many"')
    with pytest.raises(ConfigError, match="wrong type"):
        load_config(experiment)


def test_patience_defaults_to_vame_and_reaches_its_config(experiment):
    cfg = load_config(experiment)
    assert cfg.patience == 50
    assert vame_settings(cfg)["model_convergence"] == 50


def test_config_paths_are_relative_to_yaml(experiment):
    cfg = load_config(experiment)
    assert cfg.output == experiment.parent / "outputs"
    assert [p.name for p in cfg.pose_files] == ["rat01DLC_resnet50.csv", "rat02DLC_resnet50.csv"]


def test_videos_are_optional(experiment):
    assert load_config(experiment).videos == []


def test_videos_paired_by_dlc_naming_convention(experiment):
    videos = experiment.parent / "videos"
    videos.mkdir()
    write_video(videos / "rat02.mp4")
    write_video(videos / "rat01.avi")
    write_yaml(experiment.parent, videos="videos/")

    cfg = load_config(experiment)
    # Paired positionally with pose_files (rat01, then rat02), not folder listing order.
    assert [v.name for v in cfg.videos] == ["rat01.avi", "rat02.mp4"]


def test_single_video_file(tmp_path):
    pose = tmp_path / "pose"
    pose.mkdir()
    write_dlc_csv(pose / "rat01DLC_resnet50.csv", seed=1)
    write_video(tmp_path / "rat01.mp4")
    yaml_path = write_yaml(tmp_path, videos="rat01.mp4")
    cfg = load_config(yaml_path)
    assert [p.name for p in cfg.pose_files] == ["rat01DLC_resnet50.csv"]
    assert [v.name for v in cfg.videos] == ["rat01.mp4"]


def test_videos_missing_match_lists_pose_file(experiment):
    videos = experiment.parent / "videos"
    videos.mkdir()
    write_video(videos / "rat01.mp4")  # rat02 has no matching video
    write_yaml(experiment.parent, videos="videos/")
    with pytest.raises(ConfigError, match="no video matches pose file 'rat02DLC_resnet50.csv'"):
        load_config(experiment)


def test_videos_unsupported_format_in_folder_is_ignored(experiment):
    videos = experiment.parent / "videos"
    videos.mkdir()
    write_video(videos / "rat01.mp4")
    write_video(videos / "rat02.mov")  # unsupported extension: not picked up
    write_yaml(experiment.parent, videos="videos/")
    with pytest.raises(ConfigError, match="no video matches pose file 'rat02DLC_resnet50.csv'"):
        load_config(experiment)


def test_videos_path_does_not_exist(experiment):
    write_yaml(experiment.parent, videos="nope/")
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(experiment)


def test_videos_raw_preferred_over_labeled(experiment):
    # A typical DeepLabCut folder: raw and labeled videos together, plus the pose CSVs.
    pose = experiment.parent / "pose"
    for rat in ("rat01", "rat02"):
        write_video(pose / f"{rat}.mp4")
        write_video(pose / f"{rat}DLC_resnet50_labeled.mp4")
    write_yaml(experiment.parent, videos="pose/")
    assert [v.name for v in load_config(experiment).videos] == ["rat01.mp4", "rat02.mp4"]


def test_videos_labeled_used_when_no_raw(experiment):
    videos = experiment.parent / "videos"
    videos.mkdir()
    write_video(videos / "rat01DLC_resnet50_labeled.mp4")
    write_video(videos / "rat02DLC_resnet50_labeled.mp4")
    write_yaml(experiment.parent, videos="videos/")
    assert [v.name for v in load_config(experiment).videos] == [
        "rat01DLC_resnet50_labeled.mp4", "rat02DLC_resnet50_labeled.mp4"]


def test_videos_longest_raw_prefix_wins(tmp_path):
    # rat1.mp4 is also a prefix of rat10DLC_...: the more specific rat10.mp4 must win.
    pose = tmp_path / "pose"
    pose.mkdir()
    write_dlc_csv(pose / "rat1DLC_resnet50.csv", seed=1)
    write_dlc_csv(pose / "rat10DLC_resnet50.csv", seed=2)
    write_video(pose / "rat1.mp4")
    write_video(pose / "rat10.mp4")
    cfg = load_config(write_yaml(tmp_path, videos="pose/"))
    assert [(p.name, v.name) for p, v in zip(cfg.pose_files, cfg.videos)] == [
        ("rat10DLC_resnet50.csv", "rat10.mp4"), ("rat1DLC_resnet50.csv", "rat1.mp4")]


def test_raw_and_filtered_pose_files_together_rejected(experiment):
    write_dlc_csv(experiment.parent / "pose" / "rat01DLC_resnet50_filtered.csv", seed=1)
    with pytest.raises(ConfigError, match=r"rat01DLC_resnet50\.csv / rat01DLC_resnet50_filtered\.csv"):
        load_config(experiment)


def test_only_filtered_pose_files_accepted(experiment):
    pose = experiment.parent / "pose"
    for rat in ("rat01", "rat02"):
        (pose / f"{rat}DLC_resnet50.csv").rename(pose / f"{rat}DLC_resnet50_filtered.csv")
    assert [p.name for p in load_config(experiment).pose_files] == [
        "rat01DLC_resnet50_filtered.csv", "rat02DLC_resnet50_filtered.csv"]


def test_min_confidence_has_one_default(experiment):
    from dataclasses import fields

    from vame_motifs.config import ExperimentConfig

    dataclass_default = next(f.default for f in fields(ExperimentConfig) if f.name == "min_confidence")
    assert load_config(experiment).min_confidence == dataclass_default == 0.9  # the YAML has no 'cleaning'


def test_pose_files_with_upper_case_extension_are_found(experiment):
    pose = experiment.parent / "pose"
    (pose / "rat02DLC_resnet50.csv").rename(pose / "rat02DLC_resnet50.CSV")
    (pose / "notes").mkdir()  # folders are not pose files
    assert [p.name for p in load_config(experiment).pose_files] == ["rat01DLC_resnet50.csv", "rat02DLC_resnet50.CSV"]


@pytest.mark.parametrize("name", ["runs/z30", "..", "", " z30", "a\\b", "z 30", "a:b", "a?b"])
def test_project_name_must_be_a_folder_name(experiment, name):
    experiment.write_text(experiment.read_text() + f"project_name: {name!r}\n")
    with pytest.raises(ConfigError, match="project_name"):
        load_config(experiment)


def test_project_name_names_the_project_folder(experiment):
    experiment.write_text(experiment.read_text() + "project_name: z30\n")
    cfg = load_config(experiment)
    assert cfg.project_name == "z30"
