// The page's logic: it talks to the API with fetch() and changes the page with what it gets.
//
// Three ideas run through this file:
// - document.getElementById("x") finds the element with id="x" in index.html.
// - addEventListener("submit", f) means "when this form is sent, call f".
// - fetch(url) sends an HTTP request. It takes time, so it is used with "await"
//   inside "async" functions: the function pauses there until the answer arrives,
//   while the page stays usable.

// ======================================================================
// State: the job shown in "3. Progress", and the timer that asks for its news
// ======================================================================
let currentJobId = null;
let pollTimer = null;
const POLL_EVERY_MS = 3000; // ask the server how the job goes every 3 seconds

// The elements we use more than once
const uploadForm = document.getElementById("upload-form");
const settingsForm = document.getElementById("settings-form");

// ======================================================================
// Small helpers
// ======================================================================

// Show a text in a message paragraph, red if it is an error.
function showMessage(elementId, text, isError) {
  const element = document.getElementById(elementId);
  element.textContent = text;
  element.className = isError ? "message error" : "message ok";
}

// The API's error answers look like {"detail": "text"} (our HTTPException),
// or {"detail": [{"loc": [...], "msg": "..."}, ...]} (a 422 from FastAPI's own checks).
// This turns either into readable text.
function errorText(data) {
  if (typeof data.detail === "string") {
    return data.detail;
  }
  if (Array.isArray(data.detail)) {
    return data.detail.map((problem) => `${problem.loc.at(-1)}: ${problem.msg}`).join("\n");
  }
  return "Unexpected error";
}

// GET a URL and return its JSON. Throws an Error with the API's message if it fails.
async function getJSON(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(errorText(data));
  }
  return data;
}

// Fill a <select> with one <option> per value.
function fillSelect(select, values) {
  select.innerHTML = ""; // remove the old options
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  }
}

// ======================================================================
// When the page opens
// ======================================================================
async function start() {
  // The method list and the default values come from the API: nothing is repeated here.
  const methods = await getJSON("/methods");
  fillSelect(settingsForm.elements["method"], methods.methods);

  const defaults = await getJSON("/defaults");
  for (const [name, value] of Object.entries(defaults)) {
    const input = settingsForm.elements[name]; // the input with name="..." in the form
    if (input) {
      input.value = value;
    }
  }

  await loadJobs();
}

// ======================================================================
// 1. Upload the files -> a new job
// ======================================================================
uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault(); // without this, the browser would send the form itself and reload the page

  const includeVideos = document.getElementById("include-videos").checked;
  const chosen = [
    ...document.getElementById("folder-input").files,
    ...document.getElementById("files-input").files,
  ];

  // Keep only what the analysis uses. A folder may hold big files we do not want to upload.
  const wanted = chosen.filter((file) => {
    const name = file.name.toLowerCase();
    if (name.endsWith(".csv")) {
      return true;
    }
    return includeVideos && (name.endsWith(".mp4") || name.endsWith(".avi"));
  });

  if (wanted.length === 0) {
    showMessage("upload-message", "Choose a folder or files with at least one .csv file.", true);
    return;
  }

  // FormData builds a "multipart/form-data" body: the format browsers use to send files.
  // Each file goes under the name "files", the parameter's name in create_job (main.py).
  const formData = new FormData();
  for (const file of wanted) {
    formData.append("files", file);
  }

  const button = uploadForm.querySelector("button");
  button.disabled = true;
  showMessage("upload-message", `Uploading ${wanted.length} file(s)...`, false);

  try {
    const response = await fetch("/jobs", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) {
      showMessage("upload-message", errorText(data), true);
      return;
    }
    let text = `Uploaded ${data.pose_files.length} pose file(s)`;
    if (data.videos.length > 0) {
      text += ` and ${data.videos.length} video(s)`;
    }
    showMessage("upload-message", text + ". Now choose the settings below.", false);
    await openJob(data.job_id);
    await loadJobs();
  } catch (error) {
    // fetch itself fails only when the server cannot be reached at all
    showMessage("upload-message", `Could not reach the server: ${error.message}`, true);
  } finally {
    button.disabled = false; // always, also after an error
  }
});

// ======================================================================
// 2. The settings
// ======================================================================

// Put the job's body parts in the two lists and in the "Exclude" checkboxes.
function fillBodyparts(bodyparts) {
  fillSelect(settingsForm.elements["align_center"], bodyparts);
  fillSelect(settingsForm.elements["align_direction"], bodyparts);
  // A sensible start: the first body part as center, the last as direction
  settingsForm.elements["align_direction"].value = bodyparts[bodyparts.length - 1];

  const excludeList = document.getElementById("exclude-list");
  excludeList.innerHTML = "";
  for (const bodypart of bodyparts) {
    const label = document.createElement("label");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = bodypart;
    label.appendChild(checkbox);
    label.append(" " + bodypart);
    excludeList.appendChild(label);
  }
}

// Read the form into the object the API expects (ExperimentSettings in main.py).
// Inputs always give text, so the numbers are converted with Number().
function readSettings() {
  const fields = settingsForm.elements;
  const excluded = document.querySelectorAll("#exclude-list input:checked");
  return {
    fps: Number(fields["fps"].value),
    align_center: fields["align_center"].value,
    align_direction: fields["align_direction"].value,
    exclude: Array.from(excluded).map((checkbox) => checkbox.value),
    min_confidence: Number(fields["min_confidence"].value),
    n_clusters: Number(fields["n_clusters"].value),
    method: fields["method"].value,
    max_epochs: Number(fields["max_epochs"].value),
    patience: Number(fields["patience"].value),
    time_window: Number(fields["time_window"].value),
    zdims: Number(fields["zdims"].value),
    seed: Number(fields["seed"].value),
    community_cut_tree: Number(fields["community_cut_tree"].value),
  };
}

settingsForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = settingsForm.querySelector("button");
  button.disabled = true;
  showMessage("settings-message", "Checking the settings and the files...", false);

  try {
    // This time the body is JSON: an object turned into text with JSON.stringify,
    // and a header that tells the server so.
    const response = await fetch(`/jobs/${currentJobId}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readSettings()),
    });
    const data = await response.json();
    if (!response.ok) {
      showMessage("settings-message", errorText(data), true);
      return;
    }

    let text = "The analysis is queued. You can close this page: the job goes on in the server.";
    const warnings = Object.entries(data.warnings); // [[file name, [warning, ...]], ...]
    if (warnings.length > 0) {
      text += "\n\nWarnings (the analysis runs anyway):";
      for (const [fileName, fileWarnings] of warnings) {
        text += `\n${fileName}: ${fileWarnings.join("; ")}`;
      }
    }
    // Shown in "3. Progress": the settings section is hidden now that the job is queued
    showMessage("job-message", text, false);
    document.getElementById("settings-section").hidden = true;
    startPolling();
    await loadJobs();
  } catch (error) {
    showMessage("settings-message", `Could not reach the server: ${error.message}`, true);
  } finally {
    button.disabled = false;
  }
});

// ======================================================================
// 3. Progress: ask the server how the job goes, every few seconds ("polling")
// ======================================================================

// Show a job: fill the form if it still needs settings, and follow its progress.
async function openJob(jobId) {
  currentJobId = jobId;
  const job = await getJSON(`/jobs/${jobId}`);
  showMessage("job-message", "", false); // the message of the job shown before, if any

  const needsSettings = job.status === "uploaded";
  document.getElementById("settings-section").hidden = !needsSettings;
  if (needsSettings) {
    fillBodyparts(job.bodyparts);
    showMessage("settings-message", "", false);
  }

  document.getElementById("job-section").hidden = false;
  startPolling();
}

function startPolling() {
  stopPolling(); // never two timers at once
  refreshJob(); // once now, then every POLL_EVERY_MS
  pollTimer = setInterval(refreshJob, POLL_EVERY_MS);
}

function stopPolling() {
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

async function refreshJob() {
  let job;
  try {
    job = await getJSON(`/jobs/${currentJobId}`);
  } catch (error) {
    stopPolling(); // the job was deleted, or the server stopped
    return;
  }

  document.getElementById("job-id").textContent = job.id.slice(0, 8);
  const status = document.getElementById("job-status");
  status.textContent = job.status;
  status.className = `status-${job.status}`;
  document.getElementById("job-step").textContent = job.status === "running" ? `(step: ${job.step})` : "";
  document.getElementById("job-error").textContent = job.error || "";

  // The log, as plain text. Scroll to the end, where the newest lines are.
  const logResponse = await fetch(`/jobs/${currentJobId}/log`);
  const log = document.getElementById("job-log");
  log.textContent = await logResponse.text();
  log.scrollTop = log.scrollHeight;

  const finished = job.status === "finished";
  document.getElementById("job-results").hidden = !finished;
  if (finished) {
    await showOutputs();
  }

  // Only queued and running jobs can still change: stop asking about the others.
  if (job.status !== "queued" && job.status !== "running") {
    stopPolling();
    await loadJobs(); // the table shows the new status too
  }
}

// The download link and the list of every output file, each one a link.
async function showOutputs() {
  document.getElementById("download-link").href = `/jobs/${currentJobId}/download`;
  const files = await getJSON(`/jobs/${currentJobId}/outputs`);
  const list = document.getElementById("outputs-list");
  list.innerHTML = "";
  for (const file of files) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = `/jobs/${currentJobId}/outputs/${file}`;
    link.textContent = file;
    item.appendChild(link);
    list.appendChild(item);
  }
}

// ======================================================================
// All jobs
// ======================================================================
async function loadJobs() {
  const jobs = await getJSON("/jobs");
  const table = document.getElementById("jobs-table");
  table.innerHTML = "";

  if (jobs.length === 0) {
    table.innerHTML = "<tr><td colspan='5'>No jobs yet.</td></tr>";
    return;
  }

  for (const job of jobs) {
    const row = document.createElement("tr");
    // textContent (not innerHTML) for the values: they are shown as text, never run as HTML
    for (const value of [job.id.slice(0, 8), job.status, job.step || "", job.created.slice(0, 16).replace("T", " ")]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.appendChild(cell);
    }
    row.children[1].className = `status-${job.status}`;

    const buttons = document.createElement("td");
    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.className = "small";
    openButton.textContent = "Open";
    openButton.addEventListener("click", () => openJob(job.id));
    buttons.appendChild(openButton);

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "small danger";
    deleteButton.textContent = "Delete";
    deleteButton.addEventListener("click", () => deleteJob(job.id));
    buttons.appendChild(deleteButton);

    row.appendChild(buttons);
    table.appendChild(row);
  }
}

async function deleteJob(jobId) {
  // confirm(): the browser's own "OK / Cancel" box
  if (!confirm(`Delete job ${jobId.slice(0, 8)} and all its files?`)) {
    return;
  }
  const response = await fetch(`/jobs/${jobId}`, { method: "DELETE" });
  if (!response.ok) {
    const data = await response.json();
    alert(errorText(data));
    return;
  }
  if (jobId === currentJobId) {
    // The job on screen is gone: hide its sections
    stopPolling();
    currentJobId = null;
    document.getElementById("job-section").hidden = true;
    document.getElementById("settings-section").hidden = true;
  }
  await loadJobs();
}

document.getElementById("refresh-jobs").addEventListener("click", loadJobs);

start();
