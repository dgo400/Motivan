"""The list of jobs and their state, kept in a small SQLite database (jobs/jobs.db).

SQLite is a whole database in a single file, and Python includes it (module sqlite3):
there is nothing to install or start. Each function below opens its own connection
and closes it at the end, so the API and the worker thread can use it at the same time.

The files of each job (inputs, outputs, log) live in its folder, jobs/<job_id>/.
The database only says which jobs exist and how far each one got.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

# Relative path: created in the folder where the server is started.
JOBS_DIR = Path("jobs")
DATABASE_FILE = JOBS_DIR / "jobs.db"

# The states of a job, in order. A job ends either "finished" or "failed".
UPLOADED = "uploaded"  # files received, waiting for the settings
QUEUED = "queued"  # settings received, waiting for the worker
RUNNING = "running"  # the worker is on it ("step" says which step)
FINISHED = "finished"
FAILED = "failed"  # "error" says why


def run_sql(sql: str, parameters: tuple = ()) -> list[dict]:
    """Run one SQL statement and return the rows it gives, as dictionaries."""
    JOBS_DIR.mkdir(exist_ok=True)
    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row  # rows can be read by column name: row["status"]
    try:
        # The "?" in the SQL are filled with the parameters by sqlite3 itself.
        # Never build SQL by pasting values into the string: that allows "SQL injection".
        rows = connection.execute(sql, parameters).fetchall()
        connection.commit()  # save the changes to the file
    finally:
        connection.close()  # always, even after an error
    return [dict(row) for row in rows]


def create_table() -> None:
    """Create the jobs table, if it does not exist yet. Called when the server starts."""
    run_sql(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id      TEXT PRIMARY KEY,
            status  TEXT NOT NULL,
            step    TEXT,
            error   TEXT,
            created TEXT NOT NULL,
            updated TEXT NOT NULL
        )
        """
    )


def now() -> str:
    """The current local date and time as text, with the time zone: e.g. '2026-11-06T17:30:00+01:00'."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def add_job(job_id: str) -> None:
    run_sql(
        "INSERT INTO jobs (id, status, created, updated) VALUES (?, ?, ?, ?)",
        (job_id, UPLOADED, now(), now()),
    )


def get_job(job_id: str) -> dict | None:
    """The job with this id, or None if there is none."""
    rows = run_sql("SELECT * FROM jobs WHERE id = ?", (job_id,))
    if not rows:
        return None
    return rows[0]


def list_jobs() -> list[dict]:
    """All the jobs, the newest first."""
    return run_sql("SELECT * FROM jobs ORDER BY created DESC")


def set_status(job_id: str, status: str) -> None:
    run_sql("UPDATE jobs SET status = ?, updated = ? WHERE id = ?", (status, now(), job_id))


def set_step(job_id: str, step: str) -> None:
    """The worker started a new step: the job is running, on that step."""
    run_sql("UPDATE jobs SET status = ?, step = ?, updated = ? WHERE id = ?", (RUNNING, step, now(), job_id))


def set_failed(job_id: str, error: str) -> None:
    """The job stopped with an error. "step" keeps the step where it happened."""
    run_sql("UPDATE jobs SET status = ?, error = ?, updated = ? WHERE id = ?", (FAILED, error, now(), job_id))


def delete_job(job_id: str) -> None:
    run_sql("DELETE FROM jobs WHERE id = ?", (job_id,))
