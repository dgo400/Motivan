"""Loading and checking the experiment YAML.

One YAML file describes one experiment. It only holds the decisions that
belong to the researcher (which files, frame rate, reference body parts,
number of motifs). Everything VAME can work out by itself (body part names,
session names, number of features) is never asked for becasue they are detected automatically. 

All paths in the YAML are relative to the YAML file itself, so the command
works from any folder.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml


# Default parameters
MIN_EPOCHS = 10
METHODS = {"hmm": ["hmm"], "kmeans": ["kmeans"], "both": ["hmm", "kmeans"]}
VIDEO_SUFFIXES = (".mp4", ".avi")  # the formats VAME's own video step accepts
DEFAULT_MIN_CONFIDENCE = 0.6  # the dataclass and load_config must agree
DEFAULT_PROJECT_NAME = "Project"  # the experiment name: its folder in the output folder
FORBIDDEN_IN_NAME = '/\\:*?"<>| '  # characters a folder name cannot hold (Windows, and the GPU server)
FILTERED_SUFFIX = "_filtered"  # DeepLabCut's filter_predictions writes <name>_filtered.csv next to <name>.csv


class ConfigError(ValueError):
    """The experiment file is missing something or has an invalid value."""


@dataclass
class ExperimentConfig:
    # store all the paramteres of the experimental config into a dataclass
    yaml_path: Path
    pose_files: list[Path]
    fps: float
    align_center: str
    align_direction: str
    n_clusters: int 
    method: str = "hmm"
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    exclude: list[str] = field(default_factory=list)
    videos: list[Path] = field(default_factory=list)
    output: Path = Path("outputs")
    project_name: str = DEFAULT_PROJECT_NAME

    # advanced parameters with sensible defaults (most researchers don't need to change them)
    max_epochs: int = 100
    patience: int = 50  # epochs without a better test loss before training stops early
    time_window: int = 30
    zdims: int = 30
    seed: int = 42 # reproducibility
    community_cut_tree: int = 3

    @property 
    def algorithms(self) -> list[str]:
        """VAME segmentation algorithms for the chosen method (hmm and/or kmeans)"""
        return METHODS[self.method]

    @property
    def motifs_dir(self) -> Path:
        return self.output / "motifs"


def _require(section: dict, key: str, where: str):
    # search if some required fields of the experiment file are missing
    if key not in section or section[key] is None: 
        raise ConfigError(f"Missing required field '{where}.{key}'")
    return section[key]


def _section(raw: dict, name: str) -> dict:
    # read sections by their name and demands for a dict
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a section (key: value lines), got {value!r}")
    return value


def check_project_name(name: str) -> None:
    """Raise if 'name' cannot be the experiment's folder (output/<project_name>).

    Spaces are rejected too: the folder name is reused on the GPU server, where a
    path with spaces is refused, and FORBIDDEN_IN_NAME is what Windows cannot hold.
    """
    if name in ("", ".", "..") or name != name.strip() or any(c in name for c in FORBIDDEN_IN_NAME):
        raise ConfigError(
            f"project_name must be a plain folder name: should not include any spaces nor any of {FORBIDDEN_IN_NAME} "
            f"(got {name!r}), e.g. z30"
        )


def _find_pose_files(pose_path: Path) -> list[Path]:
    if not pose_path.exists():
        raise ConfigError(f"input.pose_files: {pose_path} does not exist")
    if pose_path.is_file():
        return [pose_path] # file option
    files = sorted(p for p in pose_path.iterdir() if p.is_file() and p.suffix.lower() == ".csv") # search .csv files
    if not files:
        raise ConfigError(f"No .csv files found in {pose_path}")
    # Raw and filtered output of the same recording side by side would be
    # analysed as two recordings: the same data twice. Ask which one to use.
    stems = {p.stem for p in files}
    both = sorted(s for s in stems if s.endswith(FILTERED_SUFFIX) and s.removesuffix(FILTERED_SUFFIX) in stems)
    if both:
        names = ", ".join(f"{s.removesuffix(FILTERED_SUFFIX)}.csv / {s}.csv" for s in both)
        raise ConfigError(
            f"input.pose_files: {pose_path} has both the raw and the filtered DeepLabCut output "
            f"of the same recording ({names}). Keep only one version of each recording in the folder."
        )
    return files

def _is_raw_video_of(video_stem: str, pose_stem: str) -> bool:
    """True if the video's name is the recording's name in the pose file (rat1 is not rat10DLC_)."""
    if not pose_stem.startswith(video_stem):
        return False
    rest = pose_stem[len(video_stem):]
    return rest == "" or rest.startswith("DLC")

def _string_list(value, where: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigError(f"'{where}' must be a list, e.g. [tail_end] (got {value!r})")
    return [str(item) for item in value]

def _find_videos(video_path: Path, pose_files: list[Path]) -> list[Path]:
    """Videos paired to 'pose_files', one per file, in the same order.

    Two DeepLabCut naming patterns are supported:
    - raw video: the video name is a prefix of the pose CSV's stem
      (``session1.mp4`` -> ``session1DLC_resnet50_....csv``).

    - labeled video: the pose CSV's stem is a prefix of the video name,
      with DLC's own suffix appended
      (``session1DLC_resnet50_..._filtered.csv`` ->
      ``session1DLC_resnet50_..._filtered_p60_labeled.mp4``).

    The raw video is preferred: when a folder holds both, the labeled one
    (with DeepLabCut's points drawn on it) is only used if there is no raw
    video. Within each kind, the most specific match wins: the longest raw
    name (``rat10.mp4`` over ``rat1.mp4`` for ``rat10DLC_...``), or the
    shortest labeled name.
    """
    # check if the path of the video exist
    if not video_path.exists():
        raise ConfigError(f"input.videos: {video_path} does not exist")
    if video_path.is_file():
        if video_path.suffix.lower() not in VIDEO_SUFFIXES: # mp4 avi
            raise ConfigError(f"input.videos: {video_path.name} is not a supported video format ({', '.join(VIDEO_SUFFIXES)})")
        candidates = [video_path]
    else:
        candidates = sorted(p for p in video_path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        if not candidates:
            raise ConfigError(f"input.videos: no video files ({', '.join(VIDEO_SUFFIXES)}) found in {video_path}")

    videos = []
    for pose_path in pose_files:
        raw = [v for v in candidates if _is_raw_video_of(v.stem, pose_path.stem)]
        labeled = [v for v in candidates if v.stem.startswith(pose_path.stem)]
        if not raw and not labeled:
            raise ConfigError(
                f"input.videos: no video matches pose file '{pose_path.name}' "
                f"(expected either the raw video DeepLabCut analysed, whose file name "
                f"is a prefix of '{pose_path.stem}', or a labeled video whose file name "
                f"starts with '{pose_path.stem}')"
            )
        # we select shortest videos in labeled
        # max/min keep the first file on ties, as the sorted() above makes that deterministic.
        if raw:
            videos.append(max(raw, key=lambda v: len(v.stem)))
        else:
            videos.append(min(labeled, key=lambda v: len(v.stem)))

    reused = {v for v in videos if videos.count(v) > 1}
    if reused:
        raise ConfigError(f"input.videos: {', '.join(sorted(v.name for v in reused))} each match more than one pose file")
    return videos


def load_config(path: str | Path) -> ExperimentConfig:
    """Read the experiment YAML into a checked ExperimentConfig."""
    path = Path(path).expanduser().resolve() 
    if not path.is_file():
        raise ConfigError(f"Experiment file not found: {path}")
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}  # safe_load: never runs code embedded in the file
    except yaml.YAMLError as e:
        raise ConfigError(f"{path.name} is not valid YAML: {e}") from e
    base = path.parent # folder where YAML resides
    if not isinstance(raw, dict):
        raise ConfigError(f"{path.name} must contain sections (key: value lines), got {type(raw).__name__}")

    # check and define the sections
    inp, kp = _section(raw, "input"), _section(raw, "keypoints") 
    motifs, adv = _section(raw, "motifs"), _section(raw, "advanced")

    try:
        # find the pose files (check if they are already at the path indicated by the experimental yaml)
        pose_files = _find_pose_files(base / _require(inp, "pose_files", "input"))
        # we use require to check if the sections are properly filled
        cfg = ExperimentConfig(
            yaml_path=path,
            pose_files=pose_files,
            fps=float(_require(inp, "fps", "input")),
            align_center=str(_require(kp, "align_center", "keypoints")),
            align_direction=str(_require(kp, "align_direction", "keypoints")),
            n_clusters=int(_require(motifs, "n_clusters", "motifs")),
            method=str(motifs.get("method", "hmm")),
            min_confidence=float(_section(raw, "cleaning").get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
            exclude=_string_list(kp.get("exclude") or [], "keypoints.exclude"),
            videos=_find_videos(base / inp["videos"], pose_files) if inp.get("videos") else [],
            output=base / raw.get("output", "outputs"),
            project_name=str(raw.get("project_name", DEFAULT_PROJECT_NAME)),
            max_epochs=int(adv.get("max_epochs", 100)),
            patience=int(adv.get("patience", 50)),
            time_window=int(adv.get("time_window", 30)),
            zdims=int(adv.get("zdims", 30)),
            seed=int(adv.get("seed", 42)),
            community_cut_tree=int(adv.get("community_cut_tree", 3)),
        )
    except ConfigError:
        raise
    except (TypeError, ValueError) as e: 
        raise ConfigError(f"{path.name}: a value has the wrong type ({e})") from e

    # Value checks: fail now, not minutes into training.
    check_project_name(cfg.project_name)
    if cfg.method not in METHODS:
        raise ConfigError(f"method must be one of {', '.join(METHODS)} (got '{cfg.method}')")
    if cfg.fps <= 0:
        raise ConfigError("fps must be positive")
    if not 0 <= cfg.min_confidence <= 1:
        raise ConfigError("min_confidence must be between 0 and 1")
    if cfg.n_clusters < 2:
        raise ConfigError("n_clusters must be at least 2")
    if cfg.max_epochs < MIN_EPOCHS:
        # VAME only saves a model once KL annealing has finished (about epoch 7 with its defaults).
        raise ConfigError(f"max_epochs must be at least {MIN_EPOCHS}")
    if cfg.patience < 1:
        raise ConfigError("patience must be at least 1")
    if cfg.time_window < 2:
        raise ConfigError("time_window must be at least 2")
    if cfg.community_cut_tree < 0:
        raise ConfigError("community_cut_tree must be a non-negative integer")
    if cfg.align_center == cfg.align_direction:
        raise ConfigError("align_center and keypoints.align_direction must be different body parts")
    if {cfg.align_center, cfg.align_direction} & set(cfg.exclude):
        raise ConfigError("exclude cannot contain the alignment body parts")
    if cfg.zdims < 1:
        raise ConfigError("zdims must be at least 1")
    return cfg


def write_yaml(path: str | Path, data: dict) -> None:
    """Write data as the experiment YAML at path, through a temporary file,
    so a failed write never leaves half a file."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False) as fh:
            temporary = Path(fh.name)
            yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)
        os.replace(temporary, path)  # atomic: readers see the old file or the new one, never half
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
