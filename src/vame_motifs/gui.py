"""Desktop window for Motivan: ``Motivan-gui`` (or ``python -m vame_motifs.gui``).

The window adds nothing to the analysis. It does three things:

- edits the experiment YAML (the same file the CLI reads), as a form;
- runs the CLI commands on that file and shows their output;
- 'Explore results' (viewer.py): plays the input videos and motif clips, shows
  the motif tree and UMAP, lets the user pick the tree cut, name the
  communities (saved as *_labeled.csv) and make GIFs (vame.gif).

Each command runs as its own process (``python -m vame_motifs <command> -c
<yaml>``), exactly as it would from a terminal: same steps, same run records,
same exit codes. That keeps the window responsive during training and lets a
run be stopped.

Tkinter ships with Python, so no extra package is needed.
"""

from __future__ import annotations

import copy
import datetime as dt
import os
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import yaml

from vame_motifs import __version__
from vame_motifs.cli import COMMANDS
from vame_motifs.config import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_PROJECT_NAME,
    METHODS,
    VIDEO_SUFFIXES,
    ConfigError,
    check_project_name,
    load_config,
)
from vame_motifs.pipeline import VamePipeline

# Commands in pipeline order, as buttons. 'run' gets its own, larger button.
STEP_COMMANDS = ["validate", "init", "prepare", "train", "train-more", "segment", "export", "communities", "videos"]
# Numeric fields of the YAML's 'advanced' section, as shown in the Advanced tab.
ADVANCED_KEYS = ("max_epochs", "patience", "time_window", "zdims", "seed", "community_cut_tree")

# Form defaults: the same values as the README's example experiment and config.py.
DEFAULTS = {
    "pose_files": "",
    "videos": "",
    "fps": "30",
    "align_center": "",
    "align_direction": "",
    "exclude": [],
    "min_confidence": str(DEFAULT_MIN_CONFIDENCE),
    "n_clusters": "15",
    "method": "hmm",
    "output": "outputs",
    "project_name": DEFAULT_PROJECT_NAME,
    "max_epochs": "100",
    "patience": "50",
    "time_window": "30",
    "zdims": "30",
    "seed": "42",
    "community_cut_tree": "3",
}


def read_bodyparts(path: Path) -> list[str]:
    """Body part names from a DLC CSV header (only the header rows are read)."""
    import pandas as pd

    columns = pd.read_csv(path, header=[0, 1, 2], index_col=0, nrows=0).columns
    return list(dict.fromkeys(columns.get_level_values(1)))


def trained_model(yaml_path: Path) -> Path | None:
    """The model file this experiment already has, or None if it has none yet."""
    try:
        cfg = load_config(yaml_path)
    except ConfigError:
        return None  # an invalid experiment cannot train: the command itself reports why
    model_path = VamePipeline(cfg).model_path
    return model_path if model_path.exists() else None


def yaml_to_form(raw: dict) -> dict:
    """Experiment YAML (as loaded) -> form values. Missing keys keep their defaults."""
    def section(name: str) -> dict:
        value = raw.get(name)
        return value if isinstance(value, dict) else {}

    inp, kp, motifs = section("input"), section("keypoints"), section("motifs")
    cleaning, adv = section("cleaning"), section("advanced")
    values = dict(DEFAULTS)
    sources = {
        "pose_files": inp, "videos": inp, "fps": inp,
        "align_center": kp, "align_direction": kp, "exclude": kp,
        "min_confidence": cleaning,
        "n_clusters": motifs, "method": motifs,
        **dict.fromkeys(ADVANCED_KEYS, adv),
    }
    for key, sec in sources.items():
        if sec.get(key) is not None:
            values[key] = sec[key] if key == "exclude" else str(sec[key])
    for key in ("output", "project_name"):
        if raw.get(key) is not None:
            values[key] = str(raw[key])
    values["exclude"] = list(values["exclude"] or [])
    return values


def form_to_yaml(values: dict, original: dict | None = None) -> dict:
    """Form values -> experiment YAML. Numbers are written as numbers when they parse.

    ``original`` is the YAML as it was opened: its keys the form does not show
    are kept, and the form's fields overwrite the rest.
    """
    def number(text: str):
        text = str(text).strip()
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                pass
        return text  # left as typed: load_config reports the wrong type

    inp = {"pose_files": values["pose_files"].strip()}
    if values["videos"].strip():
        inp["videos"] = values["videos"].strip()
    inp["fps"] = number(values["fps"])
    form = {
        "input": inp,
        "keypoints": {
            "align_center": values["align_center"].strip(),
            "align_direction": values["align_direction"].strip(),
            "exclude": list(values["exclude"]),
        },
        "cleaning": {"min_confidence": number(values["min_confidence"])},
        "motifs": {"n_clusters": number(values["n_clusters"]), "method": values["method"]},
        "output": values["output"].strip() or "outputs",
        "project_name": values["project_name"].strip() or DEFAULT_PROJECT_NAME,
        "advanced": {key: number(values[key]) for key in ADVANCED_KEYS},
    }
    merged = copy.deepcopy(original or {})
    if isinstance(merged.get("input"), dict) and "videos" not in inp:
        merged["input"].pop("videos", None)  # emptied in the form
    for key, value in form.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def write_yaml(path: Path, data: dict) -> None:
    """Write ``data`` to ``path`` through a temporary file, so a failed write never leaves half a file."""
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


class App(ttk.Frame):
    def __init__(self, root: tk.Tk, yaml_path: Path | None = None):
        super().__init__(root, padding=8)
        self.root = root
        self.yaml_path: Path | None = None
        self.raw: dict = {}  # the YAML as opened: save keeps the keys the form does not show
        self.process: subprocess.Popen | None = None  # the running command
        self.running = False                           # a command is running
        self.stopping = False
        # ('line', text) | ('info', text) | ('done', exit code)
        self.output_queue: queue.Queue[tuple] = queue.Queue()
        self.vars: dict[str, tk.StringVar] = {}
        self.dirty = False
        self.on_done = None                            # called with the exit code when the command ends
        self.explorer = None                           # the 'Explore results' window, when open

        root.title("Motivan")
        root.minsize(980, 680)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        self._build_file_bar()
        self._build_form()
        self._build_steps()
        self._build_log()
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(2, weight=1)

        self.set_values(DEFAULTS)
        if yaml_path:
            self.load(Path(yaml_path))
        self._update_title()
        self.after(100, self._drain_output)

    # ==================================================================
    # Layout
    # ==================================================================
    def _build_file_bar(self) -> None:
        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(bar, text="Experiment file:").pack(side="left")
        self.path_label = ttk.Label(bar, text="(not saved)", foreground="gray")
        self.path_label.pack(side="left", padx=6, fill="x", expand=True)
        for text, command in [("New", self.new), ("Open…", self.open), ("Save", self.save), ("Save as…", self.save_as),
                              ("Duplicate…", self.duplicate)]:
            ttk.Button(bar, text=text, command=command).pack(side="left", padx=2)

    def _entry(self, parent, row: int, label: str, key: str, browse: str | None = None, hint: str = "", width: int = 40):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        var = self.vars.setdefault(key, tk.StringVar())
        var.trace_add("write", lambda *_: self._mark_dirty())
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, sticky="ew", pady=2)
        col = 2
        if browse in ("file+folder", "folder"):
            if browse == "file+folder":
                ttk.Button(parent, text="File…", width=6, command=lambda: self._browse_file(key)).grid(row=row, column=col, padx=2)
                col += 1
            ttk.Button(parent, text="Folder…", width=7, command=lambda: self._browse_folder(key)).grid(row=row, column=col, padx=2)
            col += 1
        if hint:
            ttk.Label(parent, text=hint, foreground="gray").grid(row=row, column=4, sticky="w", padx=4)
        return var

    def _build_form(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.grid(row=1, column=0, sticky="nsew", padx=(0, 8))

        # --- Data ---------------------------------------------------------
        data = ttk.Frame(notebook, padding=10)
        data.columnconfigure(1, weight=1)
        notebook.add(data, text="Data")
        self._entry(data, 0, "Pose files", "pose_files", browse="file+folder")
        ttk.Label(data, text="A folder of DeepLabCut .csv files, or one .csv file.", foreground="gray").grid(
            row=1, column=1, columnspan=4, sticky="w")
        self._entry(data, 2, "Videos (optional)", "videos", browse="file+folder")
        ttk.Label(data, text="Raw videos, only needed for 'videos'. Leave empty to skip.", foreground="gray").grid(
            row=3, column=1, columnspan=4, sticky="w")
        self._entry(data, 4, "Frame rate (fps)", "fps", width=10)
        self._entry(data, 5, "Output folder", "output", browse="folder")
        ttk.Label(data, text="Paths are relative to the experiment file.", foreground="gray").grid(
            row=6, column=0, columnspan=5, sticky="w", pady=(10, 0))
        ttk.Button(data, text="Watch the videos…", command=lambda: self.open_explorer("recordings")).grid(
            row=7, column=0, sticky="w", pady=(10, 0))

        # --- Body parts ---------------------------------------------------
        kp = ttk.Frame(notebook, padding=10)
        kp.columnconfigure(1, weight=1)
        notebook.add(kp, text="Body parts")
        ttk.Button(kp, text="Read body parts from pose files", command=self.load_bodyparts).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.combos: list[ttk.Combobox] = []
        for row, (label, key) in enumerate([("Align center", "align_center"), ("Align direction", "align_direction")], 1):
            ttk.Label(kp, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=2)
            var = self.vars.setdefault(key, tk.StringVar())
            var.trace_add("write", lambda *_: self._mark_dirty())
            combo = ttk.Combobox(kp, textvariable=var, width=30)
            combo.grid(row=row, column=1, sticky="w", pady=2)
            self.combos.append(combo)
        ttk.Label(kp, text="Center is placed at (0, 0); direction sets the orientation.", foreground="gray").grid(
            row=3, column=1, sticky="w")
        ttk.Label(kp, text="Exclude").grid(row=4, column=0, sticky="nw", padx=(0, 6), pady=(10, 2))
        self.exclude_list = tk.Listbox(kp, selectmode="multiple", height=8, exportselection=False)
        self.exclude_list.grid(row=4, column=1, sticky="nsew", pady=(10, 2))
        self.exclude_list.bind("<<ListboxSelect>>", lambda _: self._mark_dirty())
        ttk.Label(kp, text="Selected body parts are left out of the model.", foreground="gray").grid(
            row=5, column=1, sticky="w")
        kp.rowconfigure(4, weight=1)

        # --- Motifs -------------------------------------------------------
        motifs = ttk.Frame(notebook, padding=10)
        motifs.columnconfigure(1, weight=1)
        notebook.add(motifs, text="Motifs")
        self._entry(motifs, 0, "Number of motifs", "n_clusters", width=10)
        ttk.Label(motifs, text="Method").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        var = self.vars.setdefault("method", tk.StringVar())
        var.trace_add("write", lambda *_: self._mark_dirty())
        ttk.Combobox(motifs, textvariable=var, values=list(METHODS), state="readonly", width=10).grid(
            row=1, column=1, sticky="w", pady=2)
        ttk.Label(motifs, text="hmm is slower but smoother; kmeans is much faster.", foreground="gray").grid(
            row=2, column=1, sticky="w")
        self._entry(motifs, 3, "Min. confidence", "min_confidence", width=10)
        ttk.Label(motifs, text="DLC likelihood below this: point discarded and filled in.", foreground="gray").grid(
            row=4, column=1, sticky="w")
        ttk.Label(motifs, text="New number of motifs or method: just 'Run all' again (the model is reused).",
                  foreground="gray").grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

        # --- Advanced -----------------------------------------------------
        adv = ttk.Frame(notebook, padding=10)
        adv.columnconfigure(1, weight=1)
        notebook.add(adv, text="Advanced")
        self._entry(adv, 0, "Experiment name", "project_name", width=20,
                    hint="its folder in the output folder, e.g. z30")
        for row, (label, key, hint) in enumerate([
            ("Max epochs", "max_epochs", "at least 10"),
            ("Patience", "patience", "stop after this many epochs without improvement"),
            ("Time window", "time_window", "frames per window"),
            ("Latent dims (zdims)", "zdims", "size of the latent space"),
            ("Seed", "seed", "same seed + same data = same motifs"),
            ("Community cut tree", "community_cut_tree", "higher = more, smaller communities"),
        ], 1):
            self._entry(adv, row, label, key, hint=hint, width=10)
        ttk.Label(adv, text="Changing these numbers, the alignment, exclusions or min. confidence needs retraining:\n"
                            "tick 'Force' and use 'Run all'. To keep this model and try other settings,\n"
                            "use 'Duplicate…' (top bar): a new experiment, with its own name and output folder.",
                  foreground="gray").grid(row=7, column=0, columnspan=5, sticky="w", pady=(10, 0))

    def _build_steps(self) -> None:
        box = ttk.LabelFrame(self, text="Run", padding=10)
        box.grid(row=1, column=1, sticky="nsew")
        box.columnconfigure(0, weight=1)

        self.run_button = ttk.Button(box, text="Run all", command=lambda: self.run_command("run"))
        self.run_button.grid(row=0, column=0, columnspan=2, sticky="ew", ipady=4)
        ttk.Label(box, text=COMMANDS["run"][1], foreground="gray", wraplength=330).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 8))

        self.step_buttons = [self.run_button]
        for i, name in enumerate(STEP_COMMANDS):
            action = self.train_more if name == "train-more" else lambda n=name: self.run_command(n)
            button = ttk.Button(box, text=name, width=12, command=action)
            button.grid(row=2 + i, column=0, sticky="w", pady=1)
            ttk.Label(box, text=COMMANDS[name][1], foreground="gray", wraplength=230).grid(
                row=2 + i, column=1, sticky="w", padx=6)
            self.step_buttons.append(button)

        row = 2 + len(STEP_COMMANDS)
        options = ttk.Frame(box)
        options.grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.force = tk.BooleanVar()
        self.verbose = tk.BooleanVar()
        ttk.Checkbutton(options, text="Force (redo steps)", variable=self.force).pack(side="left")
        ttk.Checkbutton(options, text="Verbose", variable=self.verbose).pack(side="left", padx=10)

        actions = ttk.Frame(box)
        actions.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        status = ttk.Button(actions, text="Status", command=lambda: self.run_command("status"))
        status.pack(side="left")
        self.step_buttons.append(status)
        self.stop_button = ttk.Button(actions, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=4)
        ttk.Button(actions, text="Open output folder", command=self.open_output).pack(side="right")
        ttk.Button(box, text="Explore results…", command=lambda: self.open_explorer("map")).grid(
            row=row + 2, column=0, columnspan=2, sticky="ew", pady=(10, 0), ipady=4)
        ttk.Label(box, text="Videos, motif tree and map, community names, GIFs", foreground="gray").grid(
            row=row + 3, column=0, columnspan=2, sticky="w")

    def _build_log(self) -> None:
        frame = ttk.LabelFrame(self, text="Output", padding=4)
        frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(8, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.log = tk.Text(frame, height=14, wrap="word", state="disabled", font=("Menlo", 11) if sys.platform == "darwin" else "TkFixedFont")
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        self.log.tag_configure("error", foreground="#c62828")
        self.log.tag_configure("info", foreground="#1565c0")

        bottom = ttk.Frame(frame)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.status_label = ttk.Label(bottom, text="Ready")
        self.status_label.pack(side="left")
        ttk.Button(bottom, text="Clear", command=self.clear_log).pack(side="right")

    # ==================================================================
    # Form values
    # ==================================================================
    def set_values(self, values: dict) -> None:
        for key, value in values.items():
            if key != "exclude":
                self.vars.setdefault(key, tk.StringVar()).set(value)
        known = [name for name in (values["align_center"], values["align_direction"]) if name]
        self._set_bodyparts(known, selected=values["exclude"])
        self.dirty = False
        self._update_title()

    def get_values(self) -> dict:
        values = {key: var.get() for key, var in self.vars.items()}
        values["exclude"] = [self.exclude_list.get(i) for i in self.exclude_list.curselection()]
        return values

    def _set_bodyparts(self, bodyparts: list[str], selected: list[str]) -> None:
        """Fill the alignment choices and the exclude list. Names from the YAML stay listed even if unknown."""
        names = list(dict.fromkeys(bodyparts + list(selected)))
        for combo in self.combos:
            combo.configure(values=names)
        self.exclude_list.delete(0, "end")
        for i, name in enumerate(names):
            self.exclude_list.insert("end", name)
            if name in selected:
                self.exclude_list.selection_set(i)

    def load_bodyparts(self) -> None:
        pose = self.vars["pose_files"].get().strip()
        if not pose:
            messagebox.showinfo("Body parts", "Set the pose files first (Data tab).")
            return
        path = self._resolve(pose)
        files = [path] if path.is_file() else sorted(path.glob("*.csv")) if path.is_dir() else []
        if not files:
            messagebox.showerror("Body parts", f"No .csv files found at {path}")
            return
        try:
            bodyparts = read_bodyparts(files[0])
        except Exception as e:  # noqa: BLE001  any unreadable file: show it, 'validate' gives the full check
            messagebox.showerror("Body parts", f"Cannot read {files[0].name}: {e}")
            return
        self._set_bodyparts(bodyparts, selected=self.get_values()["exclude"])
        self._write_log(f"Body parts in {files[0].name}: {', '.join(bodyparts)}\n", "info")

    # ==================================================================
    # Experiment file
    # ==================================================================
    def _base_dir(self) -> Path:
        return self.yaml_path.parent if self.yaml_path else Path.cwd()

    def _resolve(self, text: str) -> Path:
        path = Path(text).expanduser()
        return path if path.is_absolute() else self._base_dir() / path

    def _relative(self, path: str) -> str:
        """A chosen path, relative to the experiment file when it is inside its folder."""
        try:
            return str(Path(path).relative_to(self._base_dir()))
        except ValueError:
            return path

    def _browse_file(self, key: str) -> None:
        if key == "videos":
            types = [("Videos", " ".join(f"*{s}" for s in VIDEO_SUFFIXES)), ("All files", "*")]
        else:
            types = [("DeepLabCut CSV", "*.csv"), ("All files", "*")]
        path = filedialog.askopenfilename(initialdir=self._base_dir(), filetypes=types)
        if path:
            self.vars[key].set(self._relative(path))

    def _browse_folder(self, key: str) -> None:
        path = filedialog.askdirectory(initialdir=self._base_dir())
        if path:
            self.vars[key].set(self._relative(path))

    def _mark_dirty(self) -> None:
        if not self.dirty:
            self.dirty = True
            self._update_title()

    def _update_title(self) -> None:
        name = self.yaml_path.name if self.yaml_path else "untitled"
        self.root.title(f"Motivan {__version__} — {name}{' *' if self.dirty else ''}")
        self.path_label.configure(text=str(self.yaml_path) if self.yaml_path else "(not saved)",
                                  foreground="" if self.yaml_path else "gray")

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("Unsaved changes", "Save changes to the experiment file first?")
        if answer is None:
            return False
        return self.save() if answer else True

    def new(self) -> None:
        if self._confirm_discard():
            self.yaml_path = None
            self.raw = {}
            self.set_values(DEFAULTS)

    def open(self) -> None:
        if not self._confirm_discard():
            return
        path = filedialog.askopenfilename(filetypes=[("Experiment YAML", "*.yaml *.yml"), ("All files", "*")])
        if path:
            self.load(Path(path))

    def load(self, path: Path) -> None:
        try:
            with open(path) as fh:
                raw = yaml.safe_load(fh) or {}
            if not isinstance(raw, dict):
                raise TypeError("not a set of 'key: value' sections")
        except (OSError, yaml.YAMLError, TypeError) as e:
            messagebox.showerror("Open", f"Cannot read {path.name}: {e}")
            return
        self.yaml_path = path.resolve()
        self.raw = raw
        self.set_values(yaml_to_form(raw))
        self._write_log(f"Opened {self.yaml_path}\n", "info")

    def save(self) -> bool:
        if self.yaml_path is None:
            return self.save_as()
        data = form_to_yaml(self.get_values(), self.raw)
        write_yaml(self.yaml_path, data)
        self.raw = data
        self.dirty = False
        self._update_title()
        return True

    def save_as(self) -> bool:
        path = self._ask_yaml_path("experiment.yaml")
        if path is None:
            return False
        self._move_to(path)
        return self.save()

    def duplicate(self) -> bool:
        """Save the form as a new experiment, to try other settings without touching this one's model.

        The copy gets its own name and output folder (next to this one's, named
        after it): its model, motifs and run records never mix with this one's.
        """
        current = self.vars["project_name"].get().strip() or DEFAULT_PROJECT_NAME
        name = simpledialog.askstring(
            "Duplicate experiment",
            "Name of the new experiment (e.g. z10, cohort2_no_tail):\n\n"
            "It is saved as a new experiment file, with its own output folder, so this experiment and its "
            "model stay as they are. Unsaved changes in the form go to the new one.",
            initialvalue=f"{current}_copy", parent=self)
        if name is None:  # cancelled
            return False
        name = name.strip()
        try:
            check_project_name(name)
        except ConfigError as e:
            messagebox.showerror("Duplicate experiment", str(e))
            return False
        output = self._resolve(self.vars["output"].get().strip() or "outputs").parent / name
        if output.exists():
            messagebox.showerror("Duplicate experiment", f"{output} already exists: choose another name.")
            return False
        path = self._ask_yaml_path(f"{name}.yaml", initialdir=self._base_dir())
        if path is None:
            return False
        self._move_to(path)
        self.vars["output"].set(os.path.relpath(output, path.parent))
        self.vars["project_name"].set(name)
        if not self.save():
            return False
        self._write_log(f"New experiment '{name}': {path}, results in {output}. Change its settings and "
                        "'Run all'.\n", "info")
        return True

    def _ask_yaml_path(self, initialfile: str, initialdir: Path | None = None) -> Path | None:
        path = filedialog.asksaveasfilename(defaultextension=".yaml", initialfile=initialfile, initialdir=initialdir,
                                            filetypes=[("Experiment YAML", "*.yaml *.yml")])
        return Path(path).resolve() if path else None

    def _move_to(self, path: Path) -> None:
        """Make ``path`` the experiment file, keeping relative paths pointing at the same place from its folder."""
        old_base = self._base_dir()
        self.yaml_path = path
        for key in ("pose_files", "videos", "output"):
            text = self.vars[key].get().strip()
            if text and not Path(text).expanduser().is_absolute():
                self.vars[key].set(os.path.relpath(old_base / text, self.yaml_path.parent))

    # ==================================================================
    # Running commands
    # ==================================================================
    def run_command(self, command: str, extra: list[str] | None = None, on_done=None) -> None:
        """Run a Motivan command on the experiment. ``extra``: more command-line arguments;
        ``on_done(exit code)`` is called when it ends (the Explorer reloads its results)."""
        if self.running:
            return
        if (self.dirty or self.yaml_path is None) and not self.save():
            return
        flags = list(extra or [])
        if self.force.get() and command in ("run", "segment"):
            flags.append("--force")
        if self.verbose.get():
            flags.append("-v")
        # 'train', and 'run' with --force, start training from scratch and replace the model
        retrains = command == "train" or (command == "run" and "--force" in flags)
        if retrains and not self._confirm_retrain():
            return

        args = [sys.executable, "-m", "vame_motifs", command, "-c", str(self.yaml_path), *flags]
        self.stopping = False
        self.on_done = on_done
        self._set_running(command)
        threading.Thread(target=self._run_process, args=(args, self.yaml_path.parent), daemon=True).start()

    def _confirm_retrain(self) -> bool:
        """Ask before replacing a trained model; True if there is none, or the user says yes."""
        model_path = trained_model(self.yaml_path)
        if model_path is None:
            return True
        trained_on = dt.datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        return messagebox.askyesno(
            "Train again?",
            f"This experiment already has a trained model (from {trained_on}):\n{model_path}\n\n"
            "Training again replaces it, and can take hours. Train again?",
            icon="warning", default="no", parent=self)

    def train_more(self) -> None:
        """Ask how many more epochs, then run 'train-more' (the CLI needs --epochs)."""
        if self.running:
            return
        epochs = simpledialog.askinteger(
            "Train more",
            "How many more epochs?\n\nTraining continues from the current model. The previous model, "
            "its plots and its motifs move to vame_project/history/; run 'segment' (or 'Run all') afterwards.",
            initialvalue=50, minvalue=1, parent=self)
        if epochs is not None:  # None: cancelled
            self.run_command("train-more", ["--epochs", str(epochs)])

    def _run_process(self, args: list[str], cwd: Path) -> None:
        """Background thread: run the command, forwarding its output to the window (Tk is not thread-safe)."""
        self.output_queue.put(("info", f"\n$ Motivan {' '.join(args[3:])}\n"))
        env = dict(os.environ, PYTHONUNBUFFERED="1")  # stream output as it is printed
        try:
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd, env=env)
        except OSError as e:
            self.output_queue.put(("line", f"Error: cannot start {args[0]}: {e}\n"))
            self.output_queue.put(("done", 127))
            return
        self.process = process
        for line in process.stdout:
            self.output_queue.put(("line", line))
        process.wait()
        self.process = None
        self.output_queue.put(("done", process.returncode))

    def _drain_output(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item[0] == "done":
                    self._finished(item[1])
                elif item[0] == "info":
                    self._write_log(item[1], "info")
                else:
                    line = item[1]
                    self._write_log(line, "error" if line.startswith(("Error:", "Traceback")) else None)
        except queue.Empty:
            pass
        self.after(100, self._drain_output)

    def _finished(self, code: int) -> None:
        self.running = False
        messages = {0: "Finished", 1: "Failed: see the error above", 2: "Wrong command-line arguments",
                    127: "Program not found: see the error above"}
        text = "Stopped" if self.stopping else messages.get(code, f"Stopped (exit code {code})")
        self._write_log(f"[{text}]\n", "info" if code == 0 else "error")
        self.status_label.configure(text=text)
        for button in self.step_buttons:
            button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        on_done, self.on_done = self.on_done, None
        if on_done is not None:
            on_done(-1 if self.stopping else code)

    def _set_running(self, command: str) -> None:
        self.running = True
        for button in self.step_buttons:
            button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_label.configure(text=f"Running '{command}'…")

    def stop(self) -> None:
        if self.running and messagebox.askyesno("Stop", "Stop the running command?"):
            self._stop_job()

    def _stop_job(self) -> None:
        self.stopping = True
        process = self.process
        if process is not None:
            process.terminate()

    # ==================================================================
    # Explore results (viewer.py)
    # ==================================================================
    def open_explorer(self, tab: str = "map") -> None:
        """The 'Explore results' window, on one of its tabs (viewer.Explorer.TABS)."""
        if self.explorer is None:
            if (self.dirty or self.yaml_path is None) and not self.save():
                return
            try:
                from vame_motifs import viewer  # matplotlib, OpenCV: loaded only when needed
                cfg = load_config(self.yaml_path)
            except ImportError as e:
                messagebox.showerror("Explore results", f"Missing package: {e}. Install VAME (section 1 of the README).")
                return
            except Exception as e:  # noqa: BLE001  ConfigError and friends: what 'validate' would say
                messagebox.showerror("Explore results", f"Cannot read the experiment: {e}")
                return
            self.explorer = viewer.Explorer(self, cfg)
        self.explorer.show(tab)

    def open_output(self) -> None:
        folder = self._resolve(self.vars["output"].get().strip() or "outputs")
        if not folder.exists():
            messagebox.showinfo("Output", f"{folder} does not exist yet.")
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        elif sys.platform == "win32":
            os.startfile(folder)  # Windows only
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    # ==================================================================
    # Log
    # ==================================================================
    def _write_log(self, text: str, tag: str | None = None) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text, (tag,) if tag else ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def on_close(self) -> None:
        if self.explorer is not None:
            self.explorer.close()
        if self.running:
            if not messagebox.askyesno("Quit", "A command is still running. Stop it and quit?"):
                return
            self._stop_job()
        if self._confirm_discard():
            self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    """``Motivan-gui [experiment.yaml]``"""
    argv = sys.argv[1:] if argv is None else argv
    root = tk.Tk()
    App(root, Path(argv[0]) if argv else None)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
