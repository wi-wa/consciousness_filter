const DATA_URLS = [
  "../data/fineweb_edu_100k_rated.jsonl",
];

const LEGACY_FILTER = "philosophy_of_mind";
const LEGACY_MODEL = "(legacy)";

const state = {
  documents: [],
  visible: [],
  selectedIndex: 0,
  filterName: "",
  ratingFilter: "all",
  searchQuery: "",
};

const els = {
  statusText: document.getElementById("statusText"),
  filterSelect: document.getElementById("filterSelect"),
  ratingSelect: document.getElementById("ratingSelect"),
  searchInput: document.getElementById("searchInput"),
  prevButton: document.getElementById("prevButton"),
  nextButton: document.getElementById("nextButton"),
  positionText: document.getElementById("positionText"),
  chartSummary: document.getElementById("chartSummary"),
  histogram: document.getElementById("histogram"),
  chartTooltip: document.getElementById("chartTooltip"),
  documentList: document.getElementById("documentList"),
  documentTitle: document.getElementById("documentTitle"),
  documentSubhead: document.getElementById("documentSubhead"),
  meanBadge: document.getElementById("meanBadge"),
  judgementList: document.getElementById("judgementList"),
  documentText: document.getElementById("documentText"),
};

function isValidRating(value) {
  return Number.isInteger(value) && value >= 0 && value <= 10;
}

// Returns {filterName: {model: {rating, explanation, quote}}} for a JSONL row,
// accepting both the current "ratings" schema and the legacy "pom_rating" one.
function extractRatings(row) {
  const result = {};

  if (row?.ratings && typeof row.ratings === "object") {
    for (const [filterName, entries] of Object.entries(row.ratings)) {
      if (!Array.isArray(entries)) continue;
      for (const entry of entries) {
        if (typeof entry?.model === "string" && isValidRating(entry?.rating)) {
          (result[filterName] ??= {})[entry.model] = {
            rating: entry.rating,
            explanation: typeof entry.explanation === "string" ? entry.explanation : "",
            quote: typeof entry.quote === "string" ? entry.quote : "",
          };
        }
      }
    }
  }

  const legacy = row?.pom_rating?.rating;
  if (isValidRating(legacy) && !(LEGACY_FILTER in result)) {
    result[LEGACY_FILTER] = {
      [LEGACY_MODEL]: { rating: legacy, explanation: "", quote: "" },
    };
  }

  return result;
}

function parseJsonl(text) {
  const rows = [];
  const lines = text.split(/\r?\n/);

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index].trim();
    if (!line) continue;

    let row;
    try {
      row = JSON.parse(line);
    } catch {
      continue; // tolerate a partial trailing line from an interrupted run
    }
    const ratings = extractRatings(row);
    if (
      Object.keys(ratings).length > 0 &&
      typeof row.text === "string" &&
      row.text.length > 0
    ) {
      rows.push({
        originalIndex: index + 1,
        ratings,
        text: row.text,
      });
    }
  }

  return rows;
}

function getEntries(doc) {
  return doc.ratings[state.filterName] ?? null;
}

function getMean(doc) {
  const entries = getEntries(doc);
  if (!entries) return null;
  const ratings = Object.values(entries).map((entry) => entry.rating);
  if (ratings.length === 0) return null;
  return ratings.reduce((sum, rating) => sum + rating, 0) / ratings.length;
}

function getBin(doc) {
  const mean = getMean(doc);
  return mean === null ? null : Math.round(mean);
}

function formatMean(mean) {
  if (mean === null) return "-";
  return Number.isInteger(mean) ? String(mean) : mean.toFixed(1);
}

function collectFilterNames() {
  const names = new Set();
  for (const doc of state.documents) {
    for (const name of Object.keys(doc.ratings)) {
      names.add(name);
    }
  }
  return [...names].sort();
}

function populateFilterControls() {
  const filterNames = collectFilterNames();
  els.filterSelect.replaceChildren(
    ...filterNames.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      return option;
    }),
  );
  state.filterName = filterNames.includes(state.filterName)
    ? state.filterName
    : (filterNames[0] ?? "");
  els.filterSelect.value = state.filterName;
}

function summarize(text) {
  const normalized = text.replace(/\s+/g, " ").trim();
  return normalized.length > 200 ? `${normalized.slice(0, 200)}...` : normalized;
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(value);
}

function applyFilters() {
  const query = state.searchQuery.trim().toLowerCase();
  const ratingFilter = state.ratingFilter;

  let docs = state.documents.filter((doc) => {
    const mean = getMean(doc);
    if (mean === null) {
      return false;
    }
    if (ratingFilter !== "all" && Math.round(mean) !== Number(ratingFilter)) {
      return false;
    }
    if (query && !doc.text.toLowerCase().includes(query)) {
      return false;
    }
    return true;
  });

  // Fixed sort: highest mean rating first.
  docs = docs.toSorted(
    (a, b) => getMean(b) - getMean(a) || a.originalIndex - b.originalIndex,
  );

  state.visible = docs;
  state.selectedIndex = Math.min(state.selectedIndex, Math.max(docs.length - 1, 0));
}

/* ---------- Histogram ---------- */

function computeHistogram() {
  const counts = Array.from({ length: 11 }, () => 0);
  for (const doc of state.documents) {
    const bin = getBin(doc);
    if (bin !== null) counts[bin] += 1;
  }
  return counts;
}

function showBinTooltip(binButton, count, bin) {
  const tooltip = els.chartTooltip;
  tooltip.replaceChildren();
  const strong = document.createElement("strong");
  strong.textContent = `${formatNumber(count)} document${count === 1 ? "" : "s"}`;
  tooltip.append(strong, document.createTextNode(` · mean ≈ ${bin}`));
  tooltip.hidden = false;

  const cardRect = els.histogram.parentElement.getBoundingClientRect();
  const binRect = binButton.getBoundingClientRect();
  const left = binRect.left - cardRect.left + binRect.width / 2;
  tooltip.style.left = `${Math.max(8, Math.min(left, cardRect.width - 8))}px`;
  tooltip.style.top = `${binRect.top - cardRect.top - 6}px`;
  tooltip.style.transform = "translate(-50%, -100%)";
}

function hideBinTooltip() {
  els.chartTooltip.hidden = true;
}

function renderHistogram() {
  const counts = computeHistogram();
  const maxCount = Math.max(...counts, 1);
  const total = counts.reduce((sum, count) => sum + count, 0);
  const hasSelection = state.ratingFilter !== "all";

  els.chartSummary.textContent =
    `${formatNumber(total)} documents · mean across models · ${state.filterName}`;

  els.histogram.replaceChildren(
    ...counts.map((count, bin) => {
      const isSelected = state.ratingFilter === String(bin);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "bin";
      if (isSelected) button.classList.add("is-selected");
      if (hasSelection && !isSelected) button.classList.add("is-dimmed");
      button.setAttribute("aria-pressed", String(isSelected));
      button.setAttribute(
        "aria-label",
        `Mean rating ${bin}: ${formatNumber(count)} documents`,
      );
      button.addEventListener("click", () => {
        state.ratingFilter = isSelected ? "all" : String(bin);
        els.ratingSelect.value = state.ratingFilter;
        state.selectedIndex = 0;
        render();
      });
      button.addEventListener("pointerenter", () => showBinTooltip(button, count, bin));
      button.addEventListener("pointerleave", hideBinTooltip);
      button.addEventListener("focus", () => showBinTooltip(button, count, bin));
      button.addEventListener("blur", hideBinTooltip);

      const track = document.createElement("div");
      track.className = "bin-track";
      const heightPct = count > 0 ? Math.max(3, (count / maxCount) * 88) : 0;
      const fill = document.createElement("div");
      fill.className = "bin-fill";
      fill.style.height = `${heightPct}%`;
      track.append(fill);

      if (count > 0) {
        const countEl = document.createElement("div");
        countEl.className = "bin-count";
        countEl.textContent = formatNumber(count);
        countEl.style.bottom = `calc(${heightPct}% + 4px)`; // ride the bar cap
        track.append(countEl);
      }

      const label = document.createElement("div");
      label.className = "bin-x";
      label.textContent = String(bin);

      button.append(track, label);
      return button;
    }),
  );
}

/* ---------- Document list ---------- */

function renderList() {
  if (state.visible.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No rated documents match the current view.";
    els.documentList.replaceChildren(empty);
    return;
  }

  const activeDoc = state.visible[state.selectedIndex];
  const nearActive = state.visible
    .map((doc, index) => ({ doc, index }))
    .filter(({ index }) => {
      if (state.visible.length <= 150) return true;
      return Math.abs(index - state.selectedIndex) <= 75;
    });

  els.documentList.replaceChildren(
    ...nearActive.map(({ doc, index }) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "document-list-item";
      if (doc === activeDoc) {
        button.classList.add("is-active");
      }
      button.addEventListener("click", () => {
        state.selectedIndex = index;
        render();
      });

      const indexEl = document.createElement("div");
      indexEl.className = "list-index";
      indexEl.textContent = `#${doc.originalIndex}`;

      const snippet = document.createElement("div");
      snippet.className = "list-snippet";
      snippet.textContent = summarize(doc.text);

      const mean = document.createElement("div");
      mean.className = "list-mean";
      mean.textContent = formatMean(getMean(doc));

      button.append(indexEl, snippet, mean);
      return button;
    }),
  );

  const active = els.documentList.querySelector(".is-active");
  active?.scrollIntoView({ block: "nearest" });
}

/* ---------- Document detail ---------- */

function renderJudgements(doc) {
  const entries = getEntries(doc);
  if (!entries || Object.keys(entries).length === 0) {
    els.judgementList.hidden = true;
    els.judgementList.replaceChildren();
    return;
  }

  els.judgementList.hidden = false;
  els.judgementList.replaceChildren(
    ...Object.entries(entries).map(([model, entry]) => {
      const card = document.createElement("div");
      card.className = "judgement";

      const header = document.createElement("div");
      header.className = "judgement-header";

      const name = document.createElement("span");
      name.className = "judgement-model";
      name.textContent = model;

      const meter = document.createElement("div");
      meter.className = "judgement-meter";
      const fill = document.createElement("div");
      fill.className = "judgement-meter-fill";
      fill.style.width = `${(entry.rating / 10) * 100}%`;
      meter.append(fill);

      const rating = document.createElement("span");
      rating.className = "judgement-rating";
      rating.textContent = String(entry.rating);

      header.append(name, meter, rating);
      card.append(header);

      if (entry.explanation) {
        const explanation = document.createElement("div");
        explanation.className = "judgement-explanation";
        explanation.textContent = entry.explanation;
        card.append(explanation);
      }

      if (entry.quote) {
        const quote = document.createElement("blockquote");
        quote.className = "judgement-quote";
        quote.textContent = entry.quote;
        card.append(quote);
      }

      return card;
    }),
  );
}

function renderDocument() {
  const total = state.visible.length;
  const doc = state.visible[state.selectedIndex];

  els.prevButton.disabled = total === 0 || state.selectedIndex === 0;
  els.nextButton.disabled = total === 0 || state.selectedIndex >= total - 1;
  els.positionText.value = total === 0 ? "0 / 0" : `${state.selectedIndex + 1} / ${total}`;

  if (!doc) {
    els.documentTitle.textContent = "No document selected";
    els.documentSubhead.textContent = "";
    els.meanBadge.textContent = "–";
    els.documentText.textContent = "";
    els.judgementList.hidden = true;
    els.judgementList.replaceChildren();
    return;
  }

  const entries = getEntries(doc) ?? {};
  const modelCount = Object.keys(entries).length;
  els.documentTitle.textContent = `Document #${doc.originalIndex}`;
  els.documentSubhead.textContent =
    `${formatNumber(doc.text.length)} characters · ` +
    `${modelCount} model${modelCount === 1 ? "" : "s"} · ${state.filterName}`;
  els.meanBadge.textContent = formatMean(getMean(doc));
  els.documentText.textContent = doc.text;
  renderJudgements(doc);
}

function renderStatus() {
  const ratedCount = state.documents.filter((doc) => getMean(doc) !== null).length;
  els.statusText.textContent =
    `${formatNumber(state.visible.length)} visible of ${formatNumber(ratedCount)} ` +
    `documents rated on ${state.filterName || "—"}`;
}

function render() {
  applyFilters();
  renderStatus();
  renderHistogram();
  renderList();
  renderDocument();
}

function moveSelection(delta) {
  if (state.visible.length === 0) return;
  state.selectedIndex = Math.max(
    0,
    Math.min(state.visible.length - 1, state.selectedIndex + delta),
  );
  render();
}

function bindEvents() {
  els.filterSelect.addEventListener("change", () => {
    state.filterName = els.filterSelect.value;
    state.selectedIndex = 0;
    render();
  });

  els.ratingSelect.addEventListener("change", () => {
    state.ratingFilter = els.ratingSelect.value;
    state.selectedIndex = 0;
    render();
  });

  els.searchInput.addEventListener("input", () => {
    state.searchQuery = els.searchInput.value;
    state.selectedIndex = 0;
    render();
  });

  els.prevButton.addEventListener("click", () => moveSelection(-1));
  els.nextButton.addEventListener("click", () => moveSelection(1));

  window.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      moveSelection(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      moveSelection(1);
    }
  });
}

async function loadDocuments() {
  const errors = [];
  for (const url of DATA_URLS) {
    let response;
    try {
      response = await fetch(`${url}?t=${Date.now()}`);
    } catch (error) {
      errors.push(`${url}: ${error.message}`);
      continue;
    }
    if (!response.ok) {
      errors.push(`${url}: HTTP ${response.status}`);
      continue;
    }

    const text = await response.text();
    const documents = parseJsonl(text);
    if (documents.length === 0) {
      errors.push(`${url}: no rated rows`);
      continue;
    }

    state.documents = documents;
    state.visible = documents;
    return;
  }

  throw new Error(`No rated JSONL could be loaded.\n${errors.join("\n")}`);
}

async function main() {
  bindEvents();
  try {
    await loadDocuments();
    populateFilterControls();
    render();
  } catch (error) {
    els.statusText.textContent = "Could not load rated JSONL";
    els.documentList.innerHTML = "";
    els.documentText.textContent =
      `${error.message}\n\n` +
      "Serve the repository root (e.g. python -m http.server) and open /viewer/ " +
      "so the browser can fetch the rated JSONL from /data/.";
  }
}

main();
