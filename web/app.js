"use strict";

const $ = (id) => document.getElementById(id);
let currentResult = null;
let analyzedAnswer = "";
let requestSequence = 0;
let busy = false;
let toastTimer;

function countCharacters() {
  $("character-count").textContent = Array.from($("answer").value).length.toLocaleString();
  const words = $("answer").value.trim().split(/\s+/).filter(Boolean).length;
  $("word-count").textContent = words.toLocaleString();
  $("word-label").textContent = words === 1 ? "word" : "words";
}

function resetResults() {
  requestSequence += 1;
  currentResult = null;
  $("empty-state").hidden = false;
  $("result-content").hidden = true;
  document.querySelector(".results-panel").hidden = true;
  $("result-count").textContent = "Awaiting analysis";
  $("empty-title").textContent = "A clearer view starts here.";
  $("empty-copy").textContent = "Analyze your answer to find the claims that may need current verification.";
  $("announcement").textContent = "";
  $("input-error").hidden = true;
  $("analyze-button").disabled = false;
  $("analyze-label").textContent = "Analyze";
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
  // Use the displayed predictions as the single source for counts and exports.
  const claims = result.claims.map((claim) => {
    if (![0, 1, "0", "1"].includes(claim.label)) {
      throw new Error("The classifier returned an unsupported label. Please analyze again.");
    }
    const label = Number(claim.label);
    return { ...claim, label, label_name: label === 1 ? "TIME-SENSITIVE" : "STABLE" };
  });
  const volatile = claims.filter((claim) => claim.label === 1).length;
  result = { ...result, claims, summary: { total: claims.length, stable: claims.length - volatile, volatile } };
  currentResult = result;
  analyzedAnswer = answer;
  document.querySelector(".results-panel").hidden = false;
  $("empty-state").hidden = true;
  $("result-content").hidden = false;
  $("result-count").textContent = `${result.summary.total} sentences`;
  $("stable-count").textContent = result.summary.stable;
  $("volatile-count").textContent = result.summary.volatile;
  $("stable-label").textContent = result.summary.stable === 1 ? "stable claim" : "stable claims";
  $("volatile-label").textContent = result.summary.volatile === 1 ? "time-sensitive claim" : "time-sensitive claims";
  // Counts identify claims to check; they never produce an answer-level verdict.
  const verificationMessage = result.summary.volatile === 0
    ? "No time-sensitive claims flagged. Stable claims can still be incorrect."
    : `${result.summary.volatile} ${result.summary.volatile === 1 ? "claim" : "claims"} may need current verification`;
  $("verification-message").textContent = verificationMessage;
  $("verification-notice").dataset.count = result.summary.volatile;
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
  $("announcement").textContent = `${result.summary.total} sentences analyzed. ${result.summary.stable} stable and ${result.summary.volatile} time-sensitive. ${verificationMessage}. Counts do not assess truth, accuracy, reliability, or trustworthiness.`;
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
  $("analyze-label").textContent = "Analyzing…";
  $("result-count").textContent = "Analyzing…";
  $("empty-title").textContent = "Looking for what may change…";
  $("empty-copy").textContent = "The local classifier is checking each sentence for temporal stability.";
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
    // Refresh the model description after a possible automatic model reload.
    void loadModel();
  } catch (error) {
    if (sequence !== requestSequence) return;
    $("input-error").textContent = error instanceof TypeError ? "The local classifier is unavailable. Check that the server is running." : error.message;
    $("input-error").hidden = false;
    $("result-count").textContent = currentResult ? `${currentResult.summary.total} sentences` : "Analysis unavailable";
    $("empty-title").textContent = "Try your analysis again.";
    $("empty-copy").textContent = "Check the message below your answer, then try again.";
  } finally {
    if (sequence === requestSequence) {
      busy = false;
      $("analyze-button").disabled = false;
      $("analyze-label").textContent = "Analyze";
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
$("analysis-form").addEventListener("submit", (event) => { event.preventDefault(); analyzeAnswer(); });
$("clear-button").addEventListener("click", () => {
  $("answer").value = "";
  countCharacters();
  resetResults();
  $("answer").focus();
});
for (const name of ["highlight", "sentence"]) {
  $(name + "-tab").addEventListener("click", () => selectTab(name));
  $(name + "-tab").addEventListener("keydown", (event) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const other = event.key === "Home" ? "highlight" : event.key === "End" ? "sentence" : name === "highlight" ? "sentence" : "highlight";
      selectTab(other);
      $(other + "-tab").focus();
    }
  });
}
$("threshold").addEventListener("input", () => {
  $("threshold-value").textContent = Number($("threshold").value).toFixed(2);
  const hadAnalysis = currentResult || busy;
  resetResults();
  if (hadAnalysis) {
    $("empty-title").textContent = "Threshold updated.";
    $("empty-copy").textContent = "Analyze your answer again to update the stability labels.";
    $("announcement").textContent = "Threshold updated. Analyze again to update the labels.";
  }
});
$("copy-button").addEventListener("click", async () => {
  if (!currentResult) return;
  const text = [
    "Stability summary",
    `${currentResult.summary.stable} ${$("stable-label").textContent}`,
    `${currentResult.summary.volatile} ${$("volatile-label").textContent}`,
    $("verification-message").textContent,
    "Counts do not assess truth, accuracy, reliability, or trustworthiness.",
    "Labels apply per sentence; a sentence may contain several claims.",
    "",
    ...currentResult.claims.map((claim) => `[${claim.label}] ${claim.label_name}: ${claim.text}`),
  ].join("\n");
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
    $("connection-status").textContent = metadata.semantic_encoder ? "Semantic classifier ready" : "Local classifier ready";
    $("model-status").dataset.state = "ready";
    $("model-description").textContent = metadata.semantic_encoder
      ? `${metadata.algorithm}. A pretrained sentence encoder supplies contextual features; its weights stay fixed. The temporal classifier learns from ${metadata.training_rows.toLocaleString()} training examples. Local CPU run: ${metadata.encoding_seconds.toFixed(2)} seconds preparing sentence features and ${metadata.head_train_seconds.toFixed(2)} seconds fitting the selected classifier. Inference runs locally.`
      : `${metadata.algorithm}. ${metadata.training_rows.toLocaleString()} training examples, ${metadata.feature_count.toLocaleString()} text features, trained in ${metadata.train_seconds.toFixed(2)} seconds on CPU.`;
  } catch {
    $("connection-status").textContent = "Classifier unavailable";
    $("model-status").dataset.state = "unavailable";
  }
}
countCharacters();
if (/Mac|iPhone|iPad/.test(navigator.platform)) $("shortcut-modifier").textContent = "⌘";
selectTab("highlight");
loadModel();
