"""Reading what an experiment produced: motifs per frame, usage, communities, map, clips, GIFs.

Nothing here runs VAME or draws anything: it only reads the files the commands wrote,
so any interface (the web app, a notebook) can show them.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from vame_motifs import communities, evaluation
from vame_motifs.config import ExperimentConfig
from vame_motifs.export import labels_to_frames
from vame_motifs.pipeline import VamePipeline
from vame_motifs.pose_io import count_frames

NO_MOTIF = -1
NAMES_FILE = "recording_names.csv"  # in the output folder: the names given to the recordings


def bouts(mask: np.ndarray) -> np.ndarray:
    """(first, last) frame of each run of True values, in order."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=int), [0]]))
    return np.column_stack([np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1])


def short_names(sessions: list[str]) -> dict[str, str]:
    """A short name per recording: the last digits of its video's name, i.e. of the pose file's
    name before DeepLabCut's "DLC..." ('01022023_001_16236-62755DLC_Resnet50_...' -> '62755').
    A recording without digits there, or sharing them with another, keeps its full name."""
    digits = {}
    for session in sessions:
        found = re.findall(r"\d+", session.split("DLC", 1)[0])
        digits[session] = found[-1] if found else None
    counts = Counter(digits.values())
    return {session: d if d and counts[d] == 1 else session for session, d in digits.items()}


class Results:
    """What the analysis produced so far, for one segmentation algorithm. Missing parts are None."""

    def __init__(self, cfg: ExperimentConfig, algorithm: str):
        self.cfg, self.algorithm = cfg, algorithm
        self.pipeline = VamePipeline(cfg)
        self.sessions = [p.stem for p in cfg.pose_files]
        self.pose_files = dict(zip(self.sessions, cfg.pose_files))
        self.videos: dict[str, Path] = dict(zip(self.sessions, cfg.videos))
        self._frame_motifs: dict[str, np.ndarray] = {}
        self.display = self._load_display_names()  # session -> the name shown to the user
        self.tree = self._load_tree()
        self.bag = self._load_bag()
        self.umap = dict(np.load(self.pipeline.umap_path)) if self.pipeline.umap_path.exists() else None
        self.point_motif = self._point_motifs()
        self.evaluation = evaluation.load(self.pipeline.evaluation_path)

    # --- recording names ------------------------------------------------
    @property
    def names_path(self) -> Path:
        return self.cfg.output / NAMES_FILE

    def _load_display_names(self) -> dict[str, str]:
        names = short_names(self.sessions)
        if self.names_path.exists():
            saved = pd.read_csv(self.names_path, dtype=str, keep_default_na=False)
            names.update({s: n for s, n in zip(saved["recording"], saved["name"]) if s in names and n.strip()})
        return names

    def rename(self, session: str, name: str) -> None:
        """Give a recording a new name (empty: back to its short name) and save it.

        Only names that differ from the short name are saved, one row per recording.
        """
        default = short_names(self.sessions)[session]
        name = name.strip() or default
        if name in {n for s, n in self.display.items() if s != session}:
            raise ValueError(f"'{name}' is already the name of another recording")
        self.display[session] = name
        custom = {s: n for s, n in self.display.items() if n != short_names(self.sessions)[s]}
        self.names_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"recording": list(custom), "name": list(custom.values())}).to_csv(self.names_path, index=False)

    # --- motifs ---------------------------------------------------------
    def window_labels(self, session: str) -> np.ndarray | None:
        path = self.pipeline.label_file(session, self.algorithm)
        return np.load(path) if path.exists() else None

    def frame_motifs(self, session: str) -> np.ndarray | None:
        """Motif of every frame of the recording (NO_MOTIF at the window edges), as in the motif CSV."""
        if session not in self._frame_motifs:
            labels = self.window_labels(session)
            if labels is None:
                return None
            n_frames = count_frames(self.pose_files[session])
            per_frame = labels_to_frames(labels, n_frames, self.cfg.time_window)
            self._frame_motifs[session] = per_frame.fillna(NO_MOTIF).to_numpy(dtype=int)
        return self._frame_motifs[session]

    def usage(self, session: str | None = None) -> np.ndarray:
        """Share of the labelled time windows in each motif, in one recording (or over all of them)."""
        counts = np.zeros(self.cfg.n_clusters)
        for one in [session] if session else self.sessions:
            labels = self.window_labels(one)
            if labels is not None:
                counts += np.bincount(labels.astype(int), minlength=self.cfg.n_clusters)[: self.cfg.n_clusters]
        return counts / counts.sum() if counts.sum() else counts

    # --- communities ----------------------------------------------------
    def _load_tree(self):
        path = self.pipeline.community_bag_file(self.algorithm).parent / "tree.graphml"
        return communities.load_tree(path) if path.exists() else None

    def _load_bag(self) -> list[list[int]] | None:
        """The communities VAME saved with the last 'communities' run."""
        path = self.pipeline.community_bag_file(self.algorithm)
        if not path.exists():
            return None
        return [[int(m) for m in motifs] for motifs in np.load(path, allow_pickle=True)]

    def community_of(self) -> dict[int, int]:
        return communities.motif_to_community(self.bag) if self.bag else {}

    def names(self) -> dict[int, str]:
        return communities.read_labels(self.cfg, self.algorithm, self.bag) if self.bag else {}

    # --- map and clips --------------------------------------------------
    def _point_motifs(self) -> np.ndarray | None:
        """Motif of every point of the UMAP (NO_MOTIF if its recording has no labels)."""
        if self.umap is None:
            return None
        motifs = np.full(len(self.umap["window"]), NO_MOTIF)
        for i, session in enumerate(self.umap["sessions"]):
            labels = self.window_labels(str(session))
            points = self.umap["session"] == i
            if labels is not None:
                windows = self.umap["window"][points]
                ok = windows < len(labels)
                motifs[np.flatnonzero(points)[ok]] = labels[windows[ok]]
        return motifs

    def clips(self, session: str) -> dict[int, Path]:
        return self.pipeline.motif_clips(session, self.algorithm)

    def gifs(self) -> list[Path]:
        return sorted((self.cfg.output / "gifs").glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
