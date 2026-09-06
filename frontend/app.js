const API_BASE = "";

async function checkHealth() {
  const badge = document.getElementById("health-badge");
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (data.ollama_available) {
      badge.textContent = `Ollama ready (${data.datasets_loaded} datasets, ${data.documents_indexed} documents)`;
      badge.className = "badge badge-ok";
    } else {
      badge.textContent = `Ollama unavailable: ${data.ollama_message}`;
      badge.className = "badge badge-bad";
    }
  } catch (err) {
    badge.textContent = "Backend unreachable";
    badge.className = "badge badge-bad";
  }
}

function renderDatasets(datasets) {
  const list = document.getElementById("dataset-list");
  list.innerHTML = "";
  if (!datasets.length) {
    list.innerHTML = '<li class="empty">None uploaded yet</li>';
    return;
  }
  for (const ds of datasets) {
    const li = document.createElement("li");
    li.textContent = `${ds.name} — ${ds.row_count} rows, ${ds.columns.length} columns (${ds.file_type})`;
    list.appendChild(li);
  }
}

function renderDocuments(documents) {
  const list = document.getElementById("document-list");
  list.innerHTML = "";
  if (!documents.length) {
    list.innerHTML = '<li class="empty">None uploaded yet</li>';
    return;
  }
  for (const doc of documents) {
    const li = document.createElement("li");
    li.textContent = `${doc.filename} — ${doc.chunk_count} chunks (${doc.doc_type})`;
    list.appendChild(li);
  }
}

async function refreshDatasets() {
  const res = await fetch(`${API_BASE}/datasets`);
  const data = await res.json();
  renderDatasets(data.datasets);
}

async function refreshDocuments() {
  const res = await fetch(`${API_BASE}/documents`);
  const data = await res.json();
  renderDocuments(data.documents);
}

function setStatus(elId, message, kind) {
  const el = document.getElementById(elId);
  el.textContent = message;
  el.className = `status-msg ${kind || ""}`;
}

async function uploadFile(url, fileInputId, statusId, onDone) {
  const input = document.getElementById(fileInputId);
  const file = input.files[0];
  if (!file) {
    setStatus(statusId, "Please choose a file first.", "error");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);
  setStatus(statusId, "Uploading...", "");
  try {
    const res = await fetch(`${API_BASE}${url}`, { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) {
      setStatus(statusId, data.detail || "Upload failed.", "error");
      return;
    }
    setStatus(statusId, data.message, "success");
    input.value = "";
    onDone && onDone();
  } catch (err) {
    setStatus(statusId, `Upload failed: ${err}`, "error");
  }
}

function renderTable(columns, rows) {
  if (!rows.length) return "<p>No rows returned.</p>";
  const head = columns.map((c) => `<th>${c}</th>`).join("");
  const body = rows
    .map((row) => `<tr>${columns.map((c) => `<td>${row[c] ?? ""}</td>`).join("")}</tr>`)
    .join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

async function askQuestion(question) {
  const resultsSection = document.getElementById("results");
  const askBtn = document.getElementById("ask-btn");
  askBtn.disabled = true;
  askBtn.textContent = "Thinking...";
  try {
    const res = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || "Query failed.");
      return;
    }
    resultsSection.hidden = false;

    document.getElementById("final-answer").textContent = data.answer;
    const badge = document.getElementById("validation-badge");
    badge.innerHTML = `<span class="${data.validation.passed ? "valid-pass" : "valid-fail"}">${
      data.validation.passed ? "Validated" : "Needs review"
    }</span>`;

    const insightsList = document.getElementById("insights-list");
    insightsList.innerHTML = "";
    if (data.insights.length) {
      for (const insight of data.insights) {
        const li = document.createElement("li");
        li.textContent = insight;
        insightsList.appendChild(li);
      }
    } else {
      insightsList.innerHTML = '<li class="empty">No computed insights for this question.</li>';
    }

    const chartCard = document.getElementById("chart-card");
    if (data.visualization && data.visualization.generated) {
      chartCard.hidden = false;
      document.getElementById("chart-img").src = data.visualization.chart_url + `?t=${Date.now()}`;
    } else {
      chartCard.hidden = true;
    }

    const sqlCard = document.getElementById("sql-card");
    if (data.sql) {
      sqlCard.hidden = false;
      document.getElementById("sql-query").textContent = data.sql.query || "(no query generated)";
      document.getElementById("sql-table").innerHTML = data.sql.error
        ? `<p class="status-msg error">${data.sql.error}</p>`
        : renderTable(data.sql.columns, data.sql.rows);
    } else {
      sqlCard.hidden = true;
    }

    const docsCard = document.getElementById("docs-card");
    const docsList = document.getElementById("docs-list");
    if (data.retrieved_documents.length) {
      docsCard.hidden = false;
      docsList.innerHTML = data.retrieved_documents
        .map(
          (hit) =>
            `<div class="doc-hit"><div class="source">${hit.source} (similarity ${hit.similarity})</div><div>${hit.text}</div></div>`
        )
        .join("");
    } else {
      docsCard.hidden = true;
    }

    const errorsCard = document.getElementById("errors-card");
    const errorsList = document.getElementById("errors-list");
    const allIssues = [...(data.errors || []), ...(data.validation.issues || [])];
    if (allIssues.length) {
      errorsCard.hidden = false;
      errorsList.innerHTML = allIssues.map((e) => `<li>${e}</li>`).join("");
    } else {
      errorsCard.hidden = true;
    }
  } catch (err) {
    alert(`Request failed: ${err}`);
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
  }
}

document.getElementById("upload-dataset-btn").addEventListener("click", () =>
  uploadFile("/upload/dataset", "dataset-file", "dataset-upload-status", () => {
    refreshDatasets();
    checkHealth();
  })
);

document.getElementById("upload-document-btn").addEventListener("click", () =>
  uploadFile("/upload/document", "document-file", "document-upload-status", () => {
    refreshDocuments();
    checkHealth();
  })
);

document.getElementById("ask-btn").addEventListener("click", () => {
  const question = document.getElementById("question-input").value.trim();
  if (question) askQuestion(question);
});

document.getElementById("question-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("ask-btn").click();
});

document.querySelectorAll(".example-chip").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.getElementById("question-input").value = btn.dataset.q;
    askQuestion(btn.dataset.q);
  });
});

checkHealth();
refreshDatasets();
refreshDocuments();
