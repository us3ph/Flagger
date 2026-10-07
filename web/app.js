"use strict";

const $ = (id) => document.getElementById(id);
const examples = {
  software: "Python is a programming language. Python was created by Guido van Rossum. The latest Python version is 3.13. Python 3.0 was released in 2008.",
  travel: "A visa is an authorization to enter a country. The visa fee is €50. Visitors need a visa to enter the country. The visa fee was €50 in 2010.",
  mixed: "The Earth orbits the Sun. Python was created by Guido van Rossum. The current CEO is John Smith. The current price is €50. A triangle has three sides.",
};
let currentResult = null;
let analyzedAnswer = "";
let requestSequence = 0;
let busy = false;
let toastTimer;

function countCharacters() {
  $("character-count").textContent = Array.from($("answer").value).length.toLocaleString();
}

function resetResults() {
  requestSequence += 1;
  currentResult = null;
  $("empty-state").hidden = false;
  $("result-content").hidden = true;
  $("result-count").textContent = "Awaiting analysis";
  $("input-error").hidden = true;
  $("analyze-button").disabled = false;
  $("analyze-button").querySelector("span").textContent = "Analyze answer";
  document.querySelector(".results-panel").setAttribute("aria-busy", "false");
  busy = false;
}

function selectTab(name) {
  for (const tab of ["highlight", "sentence"]) {
    const active = name === tab;
    $(tab + "-tab").classList.toggle("active", active);
    $(tab + "-tab").setAttribute("aria-selected", String(active));
    $(tab + "-tab").tabIndex = active ? 0 : -1;
    $(tab + "-view").hidden = !active;
  }
}

function renderResults(result, answer) {
  currentResult = result;
  analyzedAnswer = answer;
  $("empty-state").hidden = true;
  $("result-content").hidden = false;
  $("result-count").textContent = `${result.summary.total} sentences`;
  $("stable-count").textContent = result.summary.stable;
  $("volatile-count").textContent = result.summary.volatile;
  $("inference-time").textContent = `${result.elapsed_ms} ms · local inference`;
  $("highlight-view").replaceChildren();
  $("sentence-view").replaceChildren();
  // Python uses Unicode code points; Array.from keeps emoji offsets aligned.
  const characters = Array.from(answer);
  let cursor = 0;
  for (const claim of result.claims) {
    $("highlight-view").append(document.createTextNode(characters.slice(cursor, claim.start).join("")));
    const highlight = document.createElement("mark");
    highlight.className = `claim-highlight ${claim.label ? "volatile" : "stable"}`;
    highlight.dataset.label = claim.label;
    highlight.title = `${claim.label} = ${claim.label_name}; volatility score ${claim.volatility_probability.toFixed(2)}`;
    highlight.textContent = characters.slice(claim.start, claim.end).join("");
    $("highlight-view").append(highlight);
    cursor = claim.end;

    const row = document.createElement("div");
    row.className = `sentence-row ${claim.label ? "volatile" : "stable"}`;
    const label = document.createElement("span");
    label.className = "sentence-label";
    label.textContent = claim.label;
    label.setAttribute("aria-label", claim.label_name);
    const content = document.createElement("div");
    const sentence = document.createElement("p");
    sentence.textContent = claim.text;
    const meta = document.createElement("div");
    meta.className = "sentence-meta";
    meta.textContent = `${claim.label_name} · volatility score ${claim.volatility_probability.toFixed(2)}`;
    content.append(sentence, meta);
    row.append(label, content);
    $("sentence-view").append(row);
  }
  $("highlight-view").append(document.createTextNode(characters.slice(cursor).join("")));
  $("announcement").textContent = `${result.summary.total} sentences analyzed. ${result.summary.stable} stable and ${result.summary.volatile} time-sensitive.`;
}

async function analyzeAnswer() {
  if (busy) return;
  const answer = $("answer").value;
  if (!answer.trim()) {
    $("input-error").textContent = "Paste an answer containing at least one sentence.";
    $("input-error").hidden = false;
    $("answer").focus();
    return;
  }
  const sequence = ++requestSequence;
  busy = true;
  $("input-error").hidden = true;
  $("analyze-button").disabled = true;
  $("analyze-button").querySelector("span").textContent = "Analyzing…";
  document.querySelector(".results-panel").setAttribute("aria-busy", "true");
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer, threshold: Number($("threshold").value) }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : "Please use an answer under 25,000 characters and at most 150 sentences.");
    }
    if (sequence !== requestSequence) return;
    renderResults(data, answer);
  } catch (error) {
    if (sequence !== requestSequence) return;
    $("input-error").textContent = error instanceof TypeError ? "The local classifier is unavailable. Check that the server is running." : error.message;
    $("input-error").hidden = false;
  } finally {
    if (sequence === requestSequence) {
      busy = false;
      $("analyze-button").disabled = false;
      $("analyze-button").querySelector("span").textContent = "Analyze answer";
      document.querySelector(".results-panel").setAttribute("aria-busy", "false");
    }
  }
}

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 2500);
}

$("answer").addEventListener("input", () => { countCharacters(); resetResults(); });
$("answer").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    analyzeAnswer();
  }
});
$("analyze-button").addEventListener("click", analyzeAnswer);
$("clear-button").addEventListener("click", () => {
  $("answer").value = "";
  countCharacters();
  resetResults();
  $("answer").focus();
});
document.querySelectorAll("[data-example]").forEach((button) => button.addEventListener("click", () => {
  $("answer").value = examples[button.dataset.example];
  countCharacters();
  resetResults();
  $("answer").focus();
}));
for (const name of ["highlight", "sentence"]) {
  $(name + "-tab").addEventListener("click", () => selectTab(name));
  $(name + "-tab").addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const other = name === "highlight" ? "sentence" : "highlight";
      selectTab(other);
      $(other + "-tab").focus();
    }
  });
}
$("threshold").addEventListener("input", () => {
  $("threshold-value").textContent = Number($("threshold").value).toFixed(2);
});
$("copy-button").addEventListener("click", async () => {
  if (!currentResult) return;
  const text = currentResult.claims.map((claim) => `${claim.label ? "🔴" : "🟢"} ${claim.label} = ${claim.label_name}: ${claim.text}`).join("\n");
  try { await navigator.clipboard.writeText(text); toast("Sentence labels copied"); }
  catch { toast("Clipboard unavailable. Use Export JSON to save the results."); }
});
$("export-button").addEventListener("click", () => {
  if (!currentResult) return;
  const blob = new Blob([JSON.stringify({ answer: analyzedAnswer, ...currentResult }, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "staleness-analysis.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("Analysis exported");
});

async function loadModel() {
  try {
    const response = await fetch("/api/model");
    if (!response.ok) throw new Error("Model unavailable");
    const metadata = await response.json();
    $("connection-status").textContent = "Local classifier ready";
    $("model-description").textContent = `${metadata.algorithm}. ${metadata.training_rows.toLocaleString()} training examples, ${metadata.feature_count.toLocaleString()} text features, trained in ${metadata.train_seconds.toFixed(2)} seconds on CPU.`;
  } catch {
    $("connection-status").textContent = "Classifier unavailable";
    document.querySelector(".status-dot").style.background = "#c7726b";
  }
}
countCharacters();
selectTab("highlight");
loadModel();
