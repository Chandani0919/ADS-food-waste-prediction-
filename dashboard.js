"use strict";

const numericFields = [
  ["guests", "Number of guests"],
  ["quantity_prepared_kg", "Quantity prepared (kg)"],
  ["purchase_history_kg", "Purchase history (kg)"],
  ["price_per_kg", "Price per kg"]
];
const categoricalFields = [
  ["food_type", "Food type"],
  ["event_type", "Event type"],
  ["storage_condition", "Storage condition"],
  ["season", "Season"],
  ["preparation_method", "Preparation method"],
  ["location", "Location"]
];

function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
}

function showNotice(message, isError = false) {
  const notice = document.getElementById("predictionNotice");
  if (notice) {
    notice.textContent = message;
    notice.classList.toggle("error", isError);
    notice.hidden = !message;
  }
  const uploadStatus = document.getElementById("uploadStatus");
  if (uploadStatus) {
    uploadStatus.textContent = message;
    uploadStatus.classList.toggle("error-text", isError);
  }
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The request failed.");
  return result;
}

function renderField(parent, key, labelText, type, value, options = []) {
  const wrapper = node("div", undefined, "field");
  const label = node("label", labelText);
  label.htmlFor = `field-${key}`;
  const input = document.createElement(type === "select" ? "select" : "input");
  input.id = `field-${key}`;
  input.name = key;

  if (type === "select") {
    for (const optionText of options) {
      const option = node("option", optionText);
      option.value = optionText;
      input.append(option);
    }
  } else {
    input.type = "number";
    input.min = "0";
    input.step = key === "guests" ? "1" : "0.1";
    input.required = true;
    input.value = Number(value || 0).toFixed(1);
  }

  wrapper.append(label, input);
  parent.append(wrapper);
}

function renderPredictionInputs(summary) {
  const parent = document.getElementById("featureFields");
  if (!parent) return;
  parent.replaceChildren();
  for (const [key, label] of numericFields) {
    renderField(parent, key, label, "number", summary.numeric_defaults[key]);
  }
  for (const [key, label] of categoricalFields) {
    const values = summary.feature_options[key] || [];
    renderField(parent, key, label, "select", "", [...new Set(values)]);
  }
}

function renderKpis(summary) {
  const parent = document.getElementById("kpiGrid");
  if (!parent) return;
  const items = [
    ["Records analyzed", summary.rows.toLocaleString(), "Labeled observations"],
    ["Total food waste", `${summary.total_waste_kg.toFixed(1)} kg`, "Across this dataset"],
    ["Average waste", `${summary.average_waste_kg.toFixed(1)} kg`, "Per restaurant service"],
    ["Waste-to-prepared", `${summary.waste_ratio_percent.toFixed(1)}%`, "Share of prepared food"]
  ];
  parent.replaceChildren();
  for (const [label, value, note] of items) {
    const card = node("article", undefined, "kpi");
    card.append(
      node("div", label, "kpi-label"),
      node("div", value, "kpi-value"),
      node("div", note, "kpi-note")
    );
    parent.append(card);
  }
}

function renderBars(parentId, entries, labelKey, valueKey, formatter, warm = false) {
  const parent = document.getElementById(parentId);
  if (!parent) return;
  parent.replaceChildren();
  if (!entries.length) {
    parent.append(node("p", "Feature importance is available for tree models.", "muted"));
    return;
  }

  const maximum = Math.max(...entries.map(item => Number(item[valueKey])), 0.000001);
  for (const item of entries) {
    const row = node("div", undefined, "bar-row");
    const caption = node("div", undefined, "bar-caption");
    caption.append(node("span", item[labelKey]), node("strong", formatter(Number(item[valueKey]))));
    const track = node("div", undefined, "bar-track");
    const fill = node("div", undefined, warm ? "bar-fill warm" : "bar-fill");
    fill.style.width = `${Math.max(2, Number(item[valueKey]) / maximum * 100)}%`;
    track.append(fill);
    row.append(caption, track);
    parent.append(row);
  }
}

function renderModels(summary) {
  const parent = document.getElementById("modelCards");
  if (!parent) return;
  parent.replaceChildren();
  const best = [...summary.metrics].sort((a, b) => a.mae - b.mae)[0];

  for (const metric of summary.metrics) {
    const card = node("article", undefined, "metric-card");
    card.append(
      node("div", metric.model, "metric-name"),
      node("div", `MAE ${metric.mae.toFixed(2)} kg`, "metric-mae"),
      node("div", `RMSE ${metric.rmse.toFixed(2)} kg · R² ${metric.r2.toFixed(3)}`, "metric-secondary")
    );
    if (metric.model === best.model) {
      card.append(node("span", "Best holdout MAE", "best-badge"));
    }
    parent.append(card);
  }

  renderBars(
    "errorChart",
    summary.metrics.map(item => ({ model: item.model, error: item.mae })),
    "model",
    "error",
    value => `${value.toFixed(2)} kg`,
    true
  );
}

function renderImportance(summary) {
  renderBars(
    "importanceChart",
    summary.feature_importance,
    "feature",
    "importance",
    value => `${(value * 100).toFixed(1)}%`
  );
}

function renderFoodChart(summary) {
  renderBars(
    "foodChart",
    summary.waste_by_food,
    "food_type",
    "average_waste_kg",
    value => `${value.toFixed(1)} kg`
  );
}

function renderSummary(summary) {
  const badge = document.getElementById("sourceBadge");
  if (badge) badge.textContent = summary.source;

  const modelSelect = document.getElementById("modelSelect");
  const page = document.body.dataset.page;
  if (modelSelect) {
    if (page === "model-evaluation") {
      if (summary.selected_model !== "Linear Regression") {
        modelSelect.value = summary.selected_model;
      } else if (!modelSelect.dataset.initialized) {
        modelSelect.value = "Decision Tree";
      }
      modelSelect.dataset.initialized = "true";
    } else if (page === "prediction") {
      modelSelect.value = summary.default_model;
    }
  }

  renderKpis(summary);
  renderModels(summary);
  renderPredictionInputs(summary);
  renderImportance(summary);
  renderFoodChart(summary);
}

async function loadSummary(modelName) {
  const query = modelName ? `?model=${encodeURIComponent(modelName)}` : "";
  const summary = await requestJson(`/api/summary${query}`);
  renderSummary(summary);
}

function connectModelSelection() {
  const modelSelect = document.getElementById("modelSelect");
  if (!modelSelect || document.body.dataset.page !== "model-evaluation") return;
  modelSelect.addEventListener("change", async event => {
    try {
      const summary = await requestJson(
        `/api/summary?model=${encodeURIComponent(event.target.value)}`
      );
      renderImportance(summary);
    } catch (error) {
      showNotice(error.message, true);
    }
  });
}

function connectPredictionForm() {
  const form = document.getElementById("predictionForm");
  if (!form) return;

  form.addEventListener("submit", async event => {
    event.preventDefault();
    showNotice("");
    const button = document.getElementById("predictButton");
    button.disabled = true;
    button.textContent = "Predicting…";
    try {
      const formData = new FormData(form);
      const inputs = {};
      for (const [key] of [...numericFields, ...categoricalFields]) {
        inputs[key] = formData.get(key);
      }
      const result = await requestJson("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: formData.get("model"), inputs })
      });

      const prepared = Number(inputs.quantity_prepared_kg);
      document.getElementById("predictionValue").textContent =
        `${result.prediction_kg.toFixed(1)} kg`;
      document.getElementById("predictionDetail").textContent =
        `${prepared ? (result.prediction_kg / prepared * 100).toFixed(1) : "0.0"}% of prepared food · ${result.model}`;
      document.getElementById("predictionResult").hidden = false;
      if (result.prediction_kg > prepared) {
        showNotice("The estimate is above the amount prepared. Check your inputs and interpret this result cautiously.", true);
      }
    } catch (error) {
      showNotice(error.message, true);
    } finally {
      button.disabled = false;
      button.textContent = "Predict expected waste";
    }
  });
}

function connectCsvUpload() {
  const upload = document.getElementById("csvUpload");
  if (!upload) return;

  upload.addEventListener("change", async event => {
    const file = event.target.files[0];
    if (!file) return;

    const label = document.querySelector('label[for="csvUpload"]');
    const priorText = label.textContent;
    label.textContent = "Training models…";
    label.setAttribute("aria-disabled", "true");
    try {
      await requestJson("/api/upload", {
        method: "POST",
        headers: { "Content-Type": "text/csv; charset=utf-8" },
        body: await file.text()
      });
      const status = document.getElementById("uploadStatus");
      if (status) status.textContent = `${file.name} uploaded and models evaluated.`;
      await loadSummary();
    } catch (error) {
      showNotice(error.message, true);
    } finally {
      label.textContent = priorText;
      label.removeAttribute("aria-disabled");
      event.target.value = "";
    }
  });
}

async function initializeDashboard() {
  connectModelSelection();
  connectPredictionForm();
  connectCsvUpload();
  try {
    const page = document.body.dataset.page;
    const defaultModel = page === "model-evaluation" ? "Decision Tree" : undefined;
    await loadSummary(defaultModel);
  } catch (error) {
    const badge = document.getElementById("sourceBadge");
    if (badge) badge.textContent = "Dashboard could not load";
    showNotice(error.message, true);
  }
}

document.addEventListener("DOMContentLoaded", initializeDashboard);
