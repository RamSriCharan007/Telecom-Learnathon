document.addEventListener("DOMContentLoaded", () => {

  /* ---------------------------------------------------------------- */
  /* Upload stage: drag-and-drop file picker                          */
  /* ---------------------------------------------------------------- */
  const dropZone = document.getElementById("uploadDrop");
  const fileInput = document.getElementById("datasetInput");
  const filenameLabel = document.getElementById("uploadFilename");

  if (dropZone && fileInput && filenameLabel) {
    const showFilename = () => {
      filenameLabel.textContent = fileInput.files.length ? fileInput.files[0].name : "";
    };

    fileInput.addEventListener("change", showFilename);

    ["dragenter", "dragover"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.add("is-dragover");
      });
    });

    ["dragleave", "drop"].forEach((evt) => {
      dropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        dropZone.classList.remove("is-dragover");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const dropped = e.dataTransfer.files;
      if (dropped.length) {
        fileInput.files = dropped;
        showFilename();
      }
    });
  }

  /* ---------------------------------------------------------------- */
  /* Analyse Features stage: two dropdowns + live AJAX analysis       */
  /* ---------------------------------------------------------------- */
  const featureForm = document.getElementById("featureForm");
  if (featureForm) {
  const featureSelect = document.getElementById("featureSelect");
  const methodSelect = document.getElementById("methodSelect");
  const runBtn = document.getElementById("runAnalysisBtn");
  const numericHint = document.getElementById("numericHint");
  const resultPanel = document.getElementById("resultPanel");
  const emptyPanel = document.getElementById("emptyPanel");
  const resultTitle = document.getElementById("resultTitle");
  const resultStatus = document.getElementById("resultStatus");
  const resultBody = document.getElementById("resultBody");

  function selectedFeatureIsNumeric() {
    const opt = featureSelect.options[featureSelect.selectedIndex];
    return opt ? opt.dataset.numeric === "true" : true;
  }

  function selectedMethodIsNumericOnly() {
    const opt = methodSelect.options[methodSelect.selectedIndex];
    return opt ? opt.dataset.numericOnly === "true" : false;
  }

  function validateCombination() {
    const bothChosen = featureSelect.value && methodSelect.value;
    const mismatch = bothChosen && selectedMethodIsNumericOnly() && !selectedFeatureIsNumeric();
    numericHint.hidden = !mismatch;
    runBtn.disabled = !bothChosen || mismatch;
    return !mismatch && bothChosen;
  }

  featureSelect.addEventListener("change", validateCombination);
  methodSelect.addEventListener("change", validateCombination);
  validateCombination();

  function renderTable(rows) {
    const table = document.createElement("table");
    table.className = "result-table";
    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      const tdLabel = document.createElement("td");
      tdLabel.textContent = row.label;
      const tdValue = document.createElement("td");
      tdValue.textContent = row.value;
      tr.appendChild(tdLabel);
      tr.appendChild(tdValue);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
  }

  featureForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!validateCombination()) return;

    const feature = featureSelect.value;
    const method = methodSelect.value;

    runBtn.disabled = true;
    runBtn.textContent = "Running…";
    emptyPanel.hidden = true;
    resultPanel.hidden = false;
    resultTitle.textContent = `${feature} — ${methodSelect.options[methodSelect.selectedIndex].text}`;
    resultStatus.textContent = "Working…";
    resultBody.innerHTML = "";

    try {
      const res = await fetch("/api/analyse-feature", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feature, method }),
      });
      const data = await res.json();

      resultBody.innerHTML = "";

      if (!data.ok) {
        resultStatus.textContent = "Error";
        const p = document.createElement("p");
        p.className = "result-error";
        p.textContent = data.error || "Something went wrong running that analysis.";
        resultBody.appendChild(p);
      } else {
        resultStatus.textContent = "Done";
        resultTitle.textContent = data.title || resultTitle.textContent;
        if (data.kind === "table") {
          resultBody.appendChild(renderTable(data.rows));
        } else if (data.kind === "image") {
          const img = document.createElement("img");
          img.className = "result-image";
          img.src = data.image;
          img.alt = data.title || "Analysis chart";
          resultBody.appendChild(img);
        }
      }
    } catch (err) {
      resultStatus.textContent = "Error";
      const p = document.createElement("p");
      p.className = "result-error";
      p.textContent = "Could not reach the server. Try again.";
      resultBody.appendChild(p);
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "Run analysis";
    }
  });
  } // end if (featureForm)

  /* ---------------------------------------------------------------- */
  /* Missing Values stage: mean/median imputation                     */
  /* ---------------------------------------------------------------- */
  const imputeForm = document.getElementById("imputeForm");
  if (imputeForm) {
    const imputeFeatureSelect = document.getElementById("imputeFeatureSelect");
    const imputeMethodSelect = document.getElementById("imputeMethodSelect");
    const imputeBtn = document.getElementById("imputeBtn");
    const imputeError = document.getElementById("imputeError");

    imputeForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const feature = imputeFeatureSelect.value;
      const method = imputeMethodSelect.value;
      imputeError.hidden = true;

      if (!feature || !method) {
        imputeError.textContent = "Choose a feature and a method first.";
        imputeError.hidden = false;
        return;
      }

      imputeBtn.disabled = true;
      imputeBtn.textContent = "Applying…";

      try {
        const res = await fetch("/api/impute-feature", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ feature, method }),
        });
        const data = await res.json();

        if (!data.ok) {
          imputeError.textContent = data.error || "Could not apply imputation.";
          imputeError.hidden = false;
          imputeBtn.disabled = false;
          imputeBtn.textContent = "Apply imputation";
          return;
        }

        // Reload with a cache-busting param so the browser can't serve a
        // stale cached copy of this page — otherwise the overview stats,
        // breakdown table, and chart can appear unchanged even though the
        // dataset on the server was actually updated.
        const url = new URL(window.location.href);
        url.searchParams.set("_", Date.now());
        window.location.href = url.toString();
      } catch (err) {
        imputeError.textContent = "Could not reach the server. Try again.";
        imputeError.hidden = false;
        imputeBtn.disabled = false;
        imputeBtn.textContent = "Apply imputation";
      }
    });
  }

  /* ---------------------------------------------------------------- */
  /* Data Visualization stage: numeric-vs-target, categorical-vs-target */
  /* ---------------------------------------------------------------- */
  function renderLabelValueTable(rows) {
    const table = document.createElement("table");
    table.className = "result-table";
    const tbody = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      const tdLabel = document.createElement("td");
      tdLabel.textContent = row.label;
      const tdValue = document.createElement("td");
      tdValue.textContent = row.value;
      tr.appendChild(tdLabel);
      tr.appendChild(tdValue);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    return table;
  }

  function wireVisualizeForm(formId, selectId, btnId, panelId, titleId, statusId, bodyId, chartType) {
    const form = document.getElementById(formId);
    if (!form) return;

    const select = document.getElementById(selectId);
    const btn = document.getElementById(btnId);
    const panel = document.getElementById(panelId);
    const title = document.getElementById(titleId);
    const status = document.getElementById(statusId);
    const body = document.getElementById(bodyId);

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const feature = select.value;
      if (!feature) return;

      btn.disabled = true;
      btn.textContent = "Building…";
      panel.hidden = false;
      title.textContent = feature;
      status.textContent = "Working…";
      body.innerHTML = "";

      try {
        const res = await fetch("/api/visualize-feature", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ feature, chart_type: chartType }),
        });
        const data = await res.json();

        body.innerHTML = "";
        if (!data.ok) {
          status.textContent = "Error";
          const p = document.createElement("p");
          p.className = "result-error";
          p.textContent = data.error || "Could not build that chart.";
          body.appendChild(p);
        } else {
          status.textContent = "Done";
          title.textContent = data.title || feature;
          if (data.image) {
            const img = document.createElement("img");
            img.className = "result-image";
            img.src = data.image;
            img.alt = data.title || "Chart";
            body.appendChild(img);
          }
          if (data.rows && data.rows.length) {
            const caption = document.createElement("h4");
            caption.className = "result-subcaption";
            caption.textContent = "Breakdown";
            body.appendChild(caption);
            body.appendChild(renderLabelValueTable(data.rows));
          }
        }
      } catch (err) {
        status.textContent = "Error";
        const p = document.createElement("p");
        p.className = "result-error";
        p.textContent = "Could not reach the server. Try again.";
        body.appendChild(p);
      } finally {
        btn.disabled = false;
        btn.textContent = "Visualize";
      }
    });
  }

  wireVisualizeForm(
    "vizNumericForm", "vizNumericSelect", "vizNumericBtn",
    "vizNumericPanel", "vizNumericTitle", "vizNumericStatus", "vizNumericBody",
    "numeric"
  );
  wireVisualizeForm(
    "vizCategoricalForm", "vizCategoricalSelect", "vizCategoricalBtn",
    "vizCategoricalPanel", "vizCategoricalTitle", "vizCategoricalStatus", "vizCategoricalBody",
    "categorical"
  );
});
