"""Motivan web API: receives HTTP requests and calls vame_motifs.

How a job goes, from the client's side:

    1. POST   /jobs               upload the pose CSVs (and videos)  -> job_id, body parts
    2. POST   /jobs/{id}/run      send the settings                  -> the job is queued
    3. GET    /jobs/{id}          ask how it is going (again and again: "polling")
    4. GET    /jobs/{id}/download when it is finished: the motif CSVs, as a .zip

Start the server with:  uv run uvicorn motivan_web.main:app --reload
and try every endpoint at http://127.0.0.1:8000/docs
"""

import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from motivan_web import database, worker
from vame_motifs import __version__
from vame_motifs.config import (
    DEFAULT_PROJECT_NAME,
    METHODS,
    VIDEO_SUFFIXES,
    ConfigError,
    ExperimentConfig,
    load_config,
    write_yaml,
)
from vame_motifs.pose_io import PoseFileError, read_bodyparts
from vame_motifs.validation import validate


# "lifespan": code that runs once when the server starts (before "yield")
# and once when it stops (after "yield").
@asynccontextmanager
async def lifespan(app: FastAPI):
    database.create_table()
    worker.start()
    yield


app = FastAPI(title="Motivan", version=__version__, lifespan=lifespan)


# ======================================================================
# The settings the user chooses in the form
# ======================================================================
class ExperimentSettings(BaseModel):
    """The analysis settings: the experiment YAML without the file paths (the server sets those).

    FastAPI reads them from the JSON body of the request and checks the types: a text
    where a number goes is answered with a 422 error before our code even runs.
    Whether the VALUES make sense (e.g. n_clusters >= 2) is checked by load_config,
    as for the CLI: the rules are written in one place only.
    The defaults are taken from ExperimentConfig, for the same reason.
    """
    # Reject unknown fields
    model_config = ConfigDict(extra="forbid")

    fps: float
    align_center: str
    align_direction: str
    exclude: list[str] = []
    min_confidence: float = ExperimentConfig.min_confidence
    n_clusters: int = 15
    method: str = ExperimentConfig.method
    max_epochs: int = ExperimentConfig.max_epochs
    patience: int = ExperimentConfig.patience
    time_window: int = ExperimentConfig.time_window
    zdims: int = ExperimentConfig.zdims
    seed: int = ExperimentConfig.seed
    community_cut_tree: int = ExperimentConfig.community_cut_tree


def settings_to_yaml(settings: ExperimentSettings, has_videos: bool) -> dict:
    """The experiment YAML for a job, as the CLI would read it. Paths are relative to the job's folder."""
    data = {
        "input": {"pose_files": "inputs/pose", "fps": settings.fps},
        "keypoints": {
            "align_center": settings.align_center,
            "align_direction": settings.align_direction,
            "exclude": settings.exclude,
        },
        "cleaning": {"min_confidence": settings.min_confidence},
        "motifs": {"n_clusters": settings.n_clusters, "method": settings.method},
        "output": "outputs",
        "project_name": DEFAULT_PROJECT_NAME,
        "advanced": {
            "max_epochs": settings.max_epochs,
            "patience": settings.patience,
            "time_window": settings.time_window,
            "zdims": settings.zdims,
            "seed": settings.seed,
            "community_cut_tree": settings.community_cut_tree,
        },
    }
    if has_videos:
        data["input"]["videos"] = "inputs/videos"
    return data


# ======================================================================
# Helpers
# ======================================================================
def find_job(job_id: str) -> dict:
    """The job with this id, or a 404 error if there is none.

    The id comes from the URL, so anyone can type anything there. It is only
    used to look for an exact match in the database: an id such as "../../" is
    simply not found, so it can never reach a folder outside jobs/.
    """
    job = database.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job with id '{job_id}'")
    return job


def file_names(folder: Path) -> list[str]:
    """The names of the files in a folder, sorted (an empty list if the folder does not exist)."""
    if not folder.exists():
        return []
    return sorted(path.name for path in folder.iterdir() if path.is_file())


# ======================================================================
# Endpoints
# ======================================================================
@app.get("/health")
def health() -> dict:
    """Check that the server is alive."""
    return {"status": "ok", "version": __version__}


@app.get("/methods")
def methods() -> dict:
    """The segmentation methods the user can choose from."""
    return {"methods": list(METHODS)}


@app.get("/defaults")
def defaults() -> dict:
    """The default value of every optional setting, to pre-fill the form.

    Taken from ExperimentSettings itself, so the form and the API never disagree.
    """
    values = {}
    for name, field in ExperimentSettings.model_fields.items():
        if not field.is_required():
            values[name] = field.default
    return values


@app.post("/jobs", status_code=201)  # 201 Created: this request creates a new job
def create_job(files: list[UploadFile]) -> dict:
    """Create a job from uploaded files: DeepLabCut CSVs and, optionally, the raw videos.

    Other files are ignored (so a whole folder can be uploaded). Returns the job id
    and the body parts of the first CSV, to fill the form's body part lists.
    """
    # 1. Sort the uploads by type. Path(...).name keeps only the file name, so a
    #    malicious name such as "../../important.txt" cannot write outside the job's folder.
    pose_uploads = []
    video_uploads = []
    ignored = []
    for upload in files:
        name = Path(upload.filename).name
        suffix = Path(name).suffix.lower()
        if suffix == ".csv":
            pose_uploads.append((name, upload))
        elif suffix in VIDEO_SUFFIXES:
            video_uploads.append((name, upload))
        else:
            ignored.append(name)

    # 2. Check before writing anything: a rejected request leaves nothing behind.
    if not pose_uploads:
        # 400 Bad Request: the client sent something wrong.
        raise HTTPException(status_code=400, detail="No .csv file among the uploaded files")

    # 3. Create the job's folders. uuid4().hex: a random 32-character id, impossible to guess.
    job_id = uuid.uuid4().hex
    job_dir = database.JOBS_DIR / job_id
    pose_dir = job_dir / "inputs" / "pose"
    video_dir = job_dir / "inputs" / "videos"
    pose_dir.mkdir(parents=True)
    if video_uploads:
        video_dir.mkdir(parents=True)

    # 4. Save the files. copyfileobj copies in chunks: a big video never fills the memory.
    try:
        for name, upload in pose_uploads:
            with open(pose_dir / name, "wb") as out:
                shutil.copyfileobj(upload.file, out)
        for name, upload in video_uploads:
            with open(video_dir / name, "wb") as out:
                shutil.copyfileobj(upload.file, out)
    except OSError as error:
        # Usually on Windows: a path longer than 260 characters (long DeepLabCut file names).
        shutil.rmtree(job_dir, ignore_errors=True)  # do not keep a half-saved job
        raise HTTPException(
            status_code=400,
            detail=f"Could not save '{name}' ({error.strerror}). If the file name is very long, "
            "rename the files to shorter names, or enable long paths in Windows (LongPathsEnabled).",
        )

    # 5. The body parts, read with the same function the CLI uses.
    first_csv = pose_dir / pose_uploads[0][0]
    try:
        bodyparts = read_bodyparts(first_csv)
    except ValueError as error:  # pandas raises ValueError (or a subclass) for unreadable CSVs
        shutil.rmtree(job_dir)  # do not keep a broken job
        raise HTTPException(status_code=400, detail=f"{first_csv.name}: not a readable DeepLabCut CSV ({error})")

    database.add_job(job_id)
    return {
        "job_id": job_id,
        "pose_files": file_names(pose_dir),
        "videos": file_names(video_dir),
        "ignored": ignored,
        "bodyparts": bodyparts,
    }


@app.get("/jobs")
def list_jobs() -> list[dict]:
    """All the jobs, the newest first."""
    return database.list_jobs()


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """A job's state ("status", and "step" while it runs), its input files and body parts.

    The frontend calls this every few seconds while the job runs ("polling").
    """
    job = find_job(job_id)
    job_dir = database.JOBS_DIR / job_id
    job["pose_files"] = file_names(job_dir / "inputs" / "pose")
    job["videos"] = file_names(job_dir / "inputs" / "videos")
    # The body parts, so the form can be filled again when a job is reopened.
    job["bodyparts"] = read_bodyparts(job_dir / "inputs" / "pose" / job["pose_files"][0])
    return job


@app.post("/jobs/{job_id}/run", status_code=202)  # 202 Accepted: received, it will be done later
def run_job(job_id: str, settings: ExperimentSettings) -> dict:
    """Save the job's settings, check them, and put the job in the worker's queue.

    The checks that take seconds (YAML values, body parts, the CSVs themselves) run
    here, so a mistake is answered at once instead of failing in the queue later.
    """
    job = find_job(job_id)
    if job["status"] != database.UPLOADED:
        # 409 Conflict: the request is fine, but not for a job in this state.
        raise HTTPException(status_code=409, detail=f"The job is already '{job['status']}': create a new job")

    job_dir = database.JOBS_DIR / job_id
    has_videos = (job_dir / "inputs" / "videos").exists()
    yaml_path = job_dir / "experiment.yaml"
    write_yaml(yaml_path, settings_to_yaml(settings, has_videos))

    try:
        cfg = load_config(yaml_path)
        _, reports = validate(cfg)  # validate also returns the files it read: not needed here
    except (ConfigError, PoseFileError) as error:
        # The settings stay saved, but the job stays "uploaded": fix them and send them again.
        raise HTTPException(status_code=400, detail=str(error))

    database.set_status(job_id, database.QUEUED)
    worker.job_queue.put(job_id)

    # The warnings (e.g. a poorly tracked body part) do not stop the job, but the user should see them.
    warnings = {report.name: report.warnings for report in reports if report.warnings}
    return {"job_id": job_id, "status": database.QUEUED, "warnings": warnings}


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
def get_log(job_id: str) -> str:
    """What the analysis has written so far (ours and VAME's messages), as plain text."""
    find_job(job_id)
    log_path = database.JOBS_DIR / job_id / "job.log"
    if not log_path.exists():
        return ""  # the job has not started yet
    return log_path.read_text(encoding="utf-8", errors="replace")


@app.get("/jobs/{job_id}/outputs")
def list_outputs(job_id: str) -> list[str]:
    """Every file the job produced, as paths inside its outputs folder (to download them one by one)."""
    find_job(job_id)
    outputs = database.JOBS_DIR / job_id / "outputs"
    if not outputs.exists():
        return []
    # as_posix(): always "/" between folders, as in URLs, also on Windows.
    return sorted(path.relative_to(outputs).as_posix() for path in outputs.rglob("*") if path.is_file())


@app.get("/jobs/{job_id}/outputs/{file_path:path}")  # ":path": file_path may contain "/"
def download_output(job_id: str, file_path: str) -> FileResponse:
    """Download one output file, e.g. motifs/hmm/motif_usage.csv."""
    find_job(job_id)
    outputs = (database.JOBS_DIR / job_id / "outputs").resolve()
    path = (outputs / file_path).resolve()
    # file_path comes from the URL: "../../jobs.db" would point outside the outputs folder.
    # resolve() turns the path into the real, full one; it must still be inside outputs.
    if outputs not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail=f"No output file '{file_path}'")
    return FileResponse(path)


@app.get("/jobs/{job_id}/download")
def download_motifs(job_id: str) -> FileResponse:
    """The main result: the motif CSVs and the usage summary of every recording, as one .zip."""
    job = find_job(job_id)
    if job["status"] != database.FINISHED:
        raise HTTPException(status_code=409, detail=f"The job is '{job['status']}', not finished yet")
    job_dir = database.JOBS_DIR / job_id
    # make_archive(name without .zip, format, folder to compress) -> the .zip's path
    zip_path = shutil.make_archive(str(job_dir / "motifs"), "zip", job_dir / "outputs" / "motifs")
    return FileResponse(zip_path, filename=f"motifs_{job_id[:8]}.zip")


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Delete a job and all its files (inputs and outputs)."""
    job = find_job(job_id)
    if job["status"] in (database.QUEUED, database.RUNNING):
        raise HTTPException(status_code=409, detail="The job is queued or running: it cannot be deleted now")
    shutil.rmtree(database.JOBS_DIR / job_id, ignore_errors=True)
    database.delete_job(job_id)
    return {"deleted": job_id}


# ======================================================================
# The web page (frontend/): index.html, style.css, app.js
# ======================================================================
# The same server gives the page and answers the API, so the page can call the
# API with short URLs such as "/jobs" (same address: the browser allows it).
# This must stay at the END: the API routes above are checked first, and only
# a URL that matches none of them is looked for as a file in frontend/.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"  # main.py -> motivan_web -> src -> project
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
