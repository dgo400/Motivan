"""The worker: runs the analysis of the queued jobs, one at a time, in a background thread.

Why in the background: training takes hours, but an HTTP request must be answered in
seconds. So "POST /jobs/{id}/run" only puts the job in a queue and answers at once;
the worker thread takes the jobs from the queue and runs them while the API keeps
answering other requests (such as "how is my job going?").

Why one at a time: there is a single GPU. Two trainings at once would share it,
and its memory, badly.

The worker only uses the job's folder and the database, never the API itself.
That is on purpose: later it can become a separate program on another machine
(the GPU server).
"""

import logging
import queue
import threading

from motivan_web import database
from vame_motifs.config import load_config
from vame_motifs.export import export_motifs, write_run_record
from vame_motifs.pipeline import VamePipeline
from vame_motifs.validation import validate

logger = logging.getLogger("motivan_web")

# The jobs waiting to run, in arrival order. queue.Queue is safe to use from
# several threads: the API puts job ids in, the worker takes them out.
job_queue: queue.Queue[str] = queue.Queue()


def start() -> None:
    """Start the worker thread. Called once, when the server starts."""
    # Jobs that were waiting or running when the server stopped:
    for job in database.list_jobs():
        if job["status"] == database.QUEUED:
            job_queue.put(job["id"])  # still waiting: back in the queue
        elif job["status"] == database.RUNNING:
            database.set_failed(job["id"], "The server stopped while this job was running. Create a new job.")

    # daemon=True: the thread stops when the server stops, instead of keeping it alive.
    thread = threading.Thread(target=work_forever, daemon=True)
    thread.start()


def work_forever() -> None:
    while True:
        job_id = job_queue.get()  # waits here until there is a job in the queue
        run_job(job_id)


def run_job(job_id: str) -> None:
    """Run the whole analysis of one job: the same steps as 'Motivan run', plus communities."""
    job_dir = database.JOBS_DIR / job_id

    # Everything the analysis logs (ours and VAME's) also goes to jobs/<id>/job.log,
    # so the user can follow the job with "GET /jobs/{id}/log".
    log_file = logging.FileHandler(job_dir / "job.log", encoding="utf-8")
    log_file.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_file)
    root_logger.setLevel(logging.INFO)

    # Called before each step: write it in the log and in the database.
    def progress(step: str) -> None:
        logger.info("Step: %s", step)
        database.set_step(job_id, step)

    try:
        cfg = load_config(job_dir / "experiment.yaml")
        write_run_record(cfg, "run")  # experiment.yaml copy + package versions, as the CLI does

        progress("validate")
        validate(cfg)

        pipeline = VamePipeline(cfg)
        pipeline.run(progress=progress)  # init, prepare, train, segment (and videos)

        progress("export")
        export_motifs(cfg)

        progress("communities")
        if not pipeline.has_community():  # 'videos' may have made them already
            pipeline.community()

        database.set_status(job_id, database.FINISHED)
        logger.info("Job finished")

    except Exception as error:
        # Any error: the job fails, but the worker goes on with the next job.
        logger.exception("Job failed")  # writes the full error (traceback) to job.log
        database.set_failed(job_id, str(error))

    finally:
        # Always stop writing to this job's log, even after an error.
        root_logger.removeHandler(log_file)
        log_file.close()
