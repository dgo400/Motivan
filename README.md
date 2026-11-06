# MOTIVAN

We present MOTIVAN, an application for extracting behavioural motifs over time from DeepLabCut pose estimations, using [VAME](https://github.com/EthoML/VAME).
The command is called `Motivan` (CLI). A web interface is in progress.

**Input:** a folder of DeepLabCut `.csv` files (one per recording) and a short YAML file.
**Output:** one `.csv` per recording with the motif of every frame, plus a motif-usage summary.

No DeepLabCut installation or DeepLabCut project is needed: just the CSV file, as body part names are read from the CSV header. Raw videos are optional, and only needed for `videos`, the step that cuts short example clips of each motif.

Future steps: multi-animal analysis

```
pose .csv files ──▶ validate ──▶ prepare ──▶ train ──▶ segment ──▶ export ──▶ motif .csv per recording
+ experiment.yaml   (seconds)    (align,     (slow,    (hmm and/   (frame,                  │
                                  clean)      GPU)      or kmeans)  time_s, motif)           ▼
                                                                                      videos (optional)
                                                                            short .mp4 clips, one per motif
```

## 1. Install

Both ways below install `vame-py==0.14.4`, the exact VAME version the code is written
against, into a conda environment called `vame`. VAME picks the device by itself: CUDA GPU
if present, then Apple MPS, then CPU.

### On Windows: `install.bat`

1. Install [Miniconda](https://www.anaconda.com/download/success) if you don't have conda yet
   (Anaconda or Miniforge work too).
2. Get this repository: `git clone <repo-url> Motivan`, or on GitHub *Code → Download ZIP*
   and unzip it.
3. Double-click **`install.bat`** in the `Motivan` folder.

The installer finds conda, creates the `vame` environment with Python 3.12 (only if it
does not exist yet) and installs the package into it. The first install downloads VAME
and PyTorch, so it takes several minutes. If something fails, the console stays open with
the error.

| To… | Do this |
|---|---|
| Update after new code (`git pull` or a new ZIP) | Double-click `install.bat` again: the environment is reused |
| Use the commands in a terminal | Open *Anaconda Prompt*, then `conda activate vame` and `Motivan --help` |
| Develop (editable install + pytest, ruff) | In a terminal in `Motivan`: `install.bat dev` |

On Windows, pip installs the CPU-only build of PyTorch. That is enough for every step
except `train`; training on a PC without a GPU will be slow.

### On macOS, Linux or a server: by hand

Python 3.12 or newer. Use a dedicated environment (not conda's `base`):

```bash
conda create -n vame python=3.12
conda activate vame
git clone <repo-url> Motivan
cd Motivan
python -m pip install .            # or: python -m pip install -e ".[dev]" to develop
Motivan --help
```

## 2. Prepare your data

- **Pose files:** single-animal DeepLabCut `.csv` files (three header rows:
  `scorer`, `bodyparts`, `coords`). Multi-animal files are not supported yet.
  All files must have the same body parts. The file name (without `.csv`) becomes
  the recording's name in every output. Use either the raw or the `_filtered` output
  of each recording, not both: a folder holding `rat01DLC_....csv` and
  `rat01DLC_..._filtered.csv` is rejected, as it would analyse the same recording twice.

- **Videos (optional):** the raw recordings DeepLabCut analysed, only needed for the
  `videos` step. Each video is matched to its pose file the way DeepLabCut names them:
  a video `rat01.mp4` is paired with the pose file `rat01DLC_resnet50_....csv` because
  the video's name is a prefix of the pose file's. `.mp4` and `.avi` are supported.
  Pose files and videos can share one folder: other files are ignored, and when a
  labeled video (`..._labeled.mp4`) sits next to the raw one, the raw one is used.

- **Experiment file:** save the example below as `experiment.yaml` next to your
  data and edit it. Paths in it are relative
  to the YAML file.

```yaml
input:
  pose_files: data/pose/          # folder of DLC .csv files, or one .csv file
  videos: data/videos/            # optional: folder of raw videos, or one file; needed for 'videos'
  fps: 30
keypoints:
  align_center: snout             # placed at (0, 0) by egocentric alignment
  align_direction: tailbase       # sets the orientation
  exclude: []                     # body parts to leave out, e.g. [tail_end]
cleaning:
  min_confidence: 0.9             # DLC likelihood below this: point discarded and filled in
motifs:
  n_clusters: 15                  # number of motifs
  method: hmm                     # hmm | kmeans | both
output: outputs
project_name: vame_project        # optional: the experiment's name, its folder in output (e.g. z30)
advanced:                         # optional; defaults shown
  max_epochs: 100                 # at least 10
  patience: 50                    # stop early after this many epochs without a better test loss
  time_window: 30
  zdims: 30
  seed: 42
```

Don't know the body part names? Run `validate`: it lists them.

## 3. Run

```bash
Motivan validate -c experiment.yaml   # check files + YAML, list body parts, warn on poor tracking
Motivan run      -c experiment.yaml   # prepare -> train -> segment -> export
```

`run` skips steps that are already done, so it is safe to run again. The steps can also
be run one by one:

| Command   | What it does | Time |
|-----------|--------------|------|
| `validate`| Reads every CSV, checks body parts match and the YAML names exist, reports the share of low-confidence frames per body part | seconds |
| `init`    | Creates the VAME project from the CSVs and writes your settings into VAME's config | seconds |
| `prepare` | Low-confidence cleaning, egocentric alignment, outlier removal, smoothing, training set | seconds–minutes |
| `train`   | Trains and evaluates the VAME model | the slow part: use a GPU |
| `evaluate` | Scores how well the model reconstructs and predicts unseen data, and prints what to do about it. `train` and `train-more` run it too; use it for models trained before it existed | seconds |
| `train-more` | Trains the existing model `--epochs N` more epochs, starting from its weights (an underfit model: loss still going down when training stopped). The previous model and everything made from it move to `history/` | like `train`, for N epochs |
| `segment` | Assigns a motif to every time window (`--force` recomputes) | minutes (kmeans is much faster than hmm) |
| `export`  | Writes the motif CSVs and the usage summary | seconds |
| `communities` | Groups motifs into communities by cutting VAME's motif tree at `advanced.community_cut_tree` | seconds |
| `videos`  | Cuts a short `.mp4` per motif per recording, from `input.videos` (needs `input.videos`), in one folder per community | minutes |
| `umap`    | 2-D map (UMAP) of the latent space (`--force` redoes it) | about a minute |
| `gif`     | `vame.gif`: the animal next to its path through the UMAP, as a `.gif` (needs `input.videos`) | under a minute |
| `status`  | Shows which steps are done, and the settings the model in the folder was made with (warns if the experiment file now differs) | instant |
| `run`     | `validate` then all of the above (`--force` redoes every step); `videos` only runs if `input.videos` is set | |

**Changing settings later**

- New `n_clusters` or `method`: just `Motivan run -c experiment.yaml` again. The model is
  reused; only segmentation and export run.
- New `time_window`, `zdims`, `max_epochs`, `patience`, `min_confidence`, alignment or `exclude`:
  the model must be retrained, so use `Motivan run -c experiment.yaml --force`.
  Without `--force`, every step stops with an error naming what changed, instead of
  using a model made with the old settings. To keep the old model, copy the experiment
  file and give the copy a new `project_name` and `output` folder.
- Model still improving when training stopped (underfit): train it further with
  `Motivan train-more -c experiment.yaml --epochs 50`, then `run` again to segment
  with the new model. The settings must not have changed since `prepare`. The previous
  model, its loss curves and plots, and its motifs are kept in
  `vame_project/history/round_<k>/`.

Exit codes: `0` success, `1` a problem with the input or a missing step (printed as one
`Error:` line, no traceback), `2` wrong command-line arguments. This makes the
command safe to use in scripts and Slurm jobs.

### On a Slurm cluster (e.g. Pedraforca)

Everything runs on the server. Submit the slow command as a job:

```bash
sbatch --gres=gpu:1 --wrap "conda run -n vame Motivan run -c /path/to/experiment.yaml"
```

(Ask the admins for the right partition and GPU options.)

## 4. Outputs

```
outputs/
├── motifs/
│   └── hmm/                         (and/or kmeans/)
│       ├── <recording>_motifs.csv   frame, time_s, motif
│       ├── motif_usage.csv          share of time in each motif, one row per recording
│       ├── community_labels.csv     community, label, motifs   (after naming the communities)
│       └── <recording>_motifs_labeled.csv   frame, time_s, motif, community, label
├── gifs/<recording>_<algorithm>_<label>_<start>-<end>.gif   (after 'gif')
├── recording_names.csv              names given to the recordings
├── runs/<date>_<command>/           experiment.yaml copy, versions.txt, run.log
└── vame_project/                    VAME's own project (model, logs, plots in model/evaluate/)
    ├── model_settings.yaml          the settings 'prepare' used (+ more_epochs of each train-more)
    ├── model/evaluate/evaluation.npz  test and training windows with the model's reconstruction and prediction ('evaluate')
    ├── history/round_<k>/           after 'train-more': the previous model, losses, plots, results, motifs
    └── results/
        ├── umap_embedding.npz       the map of 'umap'
        ├── community_cohort/hmm-<n_clusters>/   tree.graphml, tree.png, cohort_community_bag.npy
        └── <recording>/VAME/hmm-<n_clusters>/cluster_videos/
            ├── community_0/<recording>-motif_2.mp4  (only if input.videos was set, after 'videos')
            ├── community_0/<recording>-motif_6.mp4
            └── ...
```

`<recording>_motifs.csv`:

```
frame,time_s,motif
...
14,0.4667,
15,0.5,3
16,0.5333,3
```

**Why the first and last frames have no motif.** VAME labels sliding windows of
`time_window` frames, so a recording of N frames gets N − time_window + 1 labels.
Following VAME's own convention (the one its motif videos use), each label belongs to
the centre of its window: label *i* ↔ frame *i* + `time_window // 2`. With
`time_window: 30`, the first 15 and the last 14 frames are left empty. Every row
still matches the frame with the same number in the original video.

`motif_usage.csv` gives, per recording, the fraction of labelled frames spent in each
motif (each row sums to 1).

Each `run`/`train`/... also leaves a run record in `outputs/runs/`, with a copy of the
YAML and the package versions, so any result can be traced back to the exact settings
that produced it.

## 5. Troubleshooting

| Message | Meaning |
|---|---|
| `Unknown body part 'nose'. Available: ...` | Use a name exactly as listed (case matters). |
| `... has body parts [...], but ... has [...]` | All recordings must be tracked with the same DLC body parts. |
| `<part>: 60% of frames below likelihood 0.9` | A warning: that body part will be mostly filled in. Improve tracking in DLC or add it to `keypoints.exclude`. |
| `Training finished without saving a model` | VAME only saves a model after its warm-up epochs; raise `advanced.max_epochs`. |
| `The training set in ... was made with other settings (zdims: 30 -> 10)` | Settings that shape the model changed since `prepare`. Use `run --force` to redo it with the new ones, or a new `output` folder to keep the old model. |
| `VAME could not load the previous weights, so this was not a continuation` | `train-more` trained from scratch instead. The previous model is in the `history/round_<k>/` folder named in the message; see `logs/train_model.log`. |
| `No hmm labels for '...'. Run 'segment' first.` | `export` was run before `segment`, or with a different `n_clusters`. |
| `No raw videos configured (input.videos). ...` | `videos` was run without setting `input.videos` in the YAML. |
| `input.videos: no video matches pose file '...'` | No video in `input.videos` has a name that is a prefix of that pose file's name (the DeepLabCut naming convention). |
| `VAME is not installed in this environment` | Activate the right environment (`conda activate vame`). On Windows, run `install.bat` again. |
| `Error: conda was not found.` (from `install.bat`) | Install Miniconda, or Anaconda, in its default folder and run `install.bat` again. |

## 6. Code layout

```
src/vame_motifs/
├── cli.py          commands: parse arguments, call functions, print, exit codes
├── config.py       load_config()   experiment YAML -> ExperimentConfig
├── pose_io.py      read_dlc_csv()  DLC CSV -> PoseFile
├── validation.py   validate()      checks before any VAME step (uses config.py + pose_io.py)
├── pipeline.py     VamePipeline    the VAME steps (vame-py 0.14.4)
├── export.py       export_motifs() labels -> motif CSVs + usage summary; run records
├── evaluation.py   scores()        the model's reconstruction/prediction -> scores + suggestions
├── communities.py  cut_tree()      VAME's motif tree -> communities; names -> *_labeled.csv
└── results.py      Results         reads what the commands produced (motifs per frame, usage, map, clips)
```

Only `cli.py` prints. The other modules return data or raise one of four errors
(`ConfigError`, `PoseFileError`, `PipelineError`, `ExportError`), so they can be used
from a notebook or a test without a terminal:

```python
from vame_motifs.config import load_config
from vame_motifs.validation import validate

cfg = load_config("experiment.yaml")
poses, reports = validate(cfg)
```

Tests (no VAME run needed, synthetic DLC files are generated on the fly):

```bash
python -m pip install -e ".[dev]"
pytest
```

## License

VAME is GPL-3.0; this project is distributed under GPL-3.0-or-later.
