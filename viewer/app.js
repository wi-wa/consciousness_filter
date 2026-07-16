const DATA_URLS = [
  "../data/fineweb_edu_100k_rated.jsonl",
];

const LEGACY_FILTER = "philosophy_of_mind";
const LEGACY_MODEL = "(legacy)";

const CONFIG_URL = "../config.json";
const ANNOTATIONS_URL = "../data/hand_annotated_samples.jsonl";

// Hand-annotation JSONL keys -> rated-output filter names.
const HUMAN_LABEL_FILTERS = [
  { key: "pom-rating", filter: "philosophy_of_mind", short: "pom" },
  { key: "reification-rating", filter: "reified_experience", short: "reif" },
  { key: "experience-rating", filter: "experience_descriptions", short: "exp" },
];

const state = {
  documents: [],
  visible: [],
  selectedIndex: 0,
  filterName: "",
  ratingFilter: "all",
  labelsFilter: HUMAN_LABEL_FILTERS[0].filter, // labels page: which filter to show
  labelsSort: "hand", // labels page: hand-label diff or inter-model variance
  labelsThreshold: 5, // labels page: classify mean >= threshold as 1 (0.1 steps)
  labelsItems: [], // labels page: current filter's items with fresh means/diffs
  disabledModels: new Set(), // models unchecked in the "Model agreement" box
  promptPaths: {}, // filter name -> prompt file path (from config.json)
  promptCache: {}, // filter name -> fetched prompt text
  annotations: null, // lazy-loaded hand-annotated samples joined to documents
};

// Resolves once the rated JSONL has loaded (or failed); the labels page waits
// on this so hand-annotated samples can join against the documents.
let documentsReady = Promise.resolve();

const els = {
  homeView: document.getElementById("homeView"),
  labelsView: document.getElementById("labelsView"),
  dataView: document.getElementById("dataView"),
  labelsStatusText: document.getElementById("labelsStatusText"),
  labelsFilterSelect: document.getElementById("labelsFilterSelect"),
  labelsSortSelect: document.getElementById("labelsSortSelect"),
  labelsSortNote: document.getElementById("labelsSortNote"),
  labelsBody: document.getElementById("labelsBody"),
  correlationCaption: document.getElementById("correlationCaption"),
  correlationMatrix: document.getElementById("correlationMatrix"),
  modelStatsList: document.getElementById("modelStatsList"),
  modelStatsCaption: document.getElementById("modelStatsCaption"),
  thresholdRange: document.getElementById("thresholdRange"),
  thresholdMinus: document.getElementById("thresholdMinus"),
  thresholdPlus: document.getElementById("thresholdPlus"),
  thresholdValue: document.getElementById("thresholdValue"),
  accuracyList: document.getElementById("accuracyList"),
  overallValue: document.getElementById("overallValue"),
  overallN: document.getElementById("overallN"),
  statusText: document.getElementById("statusText"),
  filterSelect: document.getElementById("filterSelect"),
  promptButton: document.getElementById("promptButton"),
  promptModal: document.getElementById("promptModal"),
  promptModalTitle: document.getElementById("promptModalTitle"),
  promptModalText: document.getElementById("promptModalText"),
  promptModalClose: document.getElementById("promptModalClose"),
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

// Ratings from models unchecked in the "Model agreement" box are excluded
// from every mean (and everything derived from one).
function getMeanForFilter(doc, filterName) {
  const entries = doc.ratings[filterName];
  if (!entries) return null;
  const ratings = Object.entries(entries)
    .filter(([model]) => !state.disabledModels.has(model))
    .map(([, entry]) => entry.rating);
  if (ratings.length === 0) return null;
  return ratings.reduce((sum, rating) => sum + rating, 0) / ratings.length;
}

function getMean(doc) {
  return getMeanForFilter(doc, state.filterName);
}

function getBin(doc) {
  const mean = getMean(doc);
  return mean === null ? null : Math.round(mean);
}

const DISAGREEMENT_THRESHOLD = 5;

function hasHighDisagreement(doc) {
  const entries = getEntries(doc);
  if (!entries) return false;
  const ratings = Object.entries(entries)
    .filter(([model]) => !state.disabledModels.has(model))
    .map(([, entry]) => entry.rating);
  if (ratings.length < 2) return false;
  return Math.max(...ratings) - Math.min(...ratings) >= DISAGREEMENT_THRESHOLD;
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
  const ratingFilter = state.ratingFilter;

  let docs = state.documents.filter((doc) => {
    const mean = getMean(doc);
    if (mean === null) {
      return false;
    }
    if (ratingFilter !== "all" && Math.round(mean) !== Number(ratingFilter)) {
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

  const unchecked = state.disabledModels.size;
  els.chartSummary.textContent =
    `${formatNumber(total)} documents · mean across models` +
    (unchecked > 0 ? ` (${unchecked} unchecked)` : "") +
    ` · ${state.filterName}`;

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

      const meanCell = document.createElement("div");
      meanCell.className = "list-mean-cell";

      if (hasHighDisagreement(doc)) {
        const flag = document.createElement("span");
        flag.className = "list-disagreement";
        flag.textContent = "!";
        flag.title =
          `Models disagree by ${DISAGREEMENT_THRESHOLD}+ points on this document`;
        meanCell.append(flag);
      }

      const mean = document.createElement("div");
      mean.className = "list-mean";
      mean.textContent = formatMean(getMean(doc));
      meanCell.append(mean);

      button.append(indexEl, snippet, meanCell);
      return button;
    }),
  );

  const active = els.documentList.querySelector(".is-active");
  active?.scrollIntoView({ block: "nearest" });
}

/* ---------- Document detail ---------- */

function buildJudgementCard(model, entry) {
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

  if (state.disabledModels.has(model)) {
    card.classList.add("is-disabled");
  }

  return card;
}

function renderJudgements(doc) {
  const entries = getEntries(doc);
  if (!entries || Object.keys(entries).length === 0) {
    els.judgementList.hidden = true;
    els.judgementList.replaceChildren();
    return;
  }

  els.judgementList.hidden = false;
  els.judgementList.replaceChildren(
    ...Object.entries(entries).map(([model, entry]) => buildJudgementCard(model, entry)),
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
  const models = Object.keys(entries);
  const enabledCount = models.filter((m) => !state.disabledModels.has(m)).length;
  const modelsLabel = enabledCount === models.length
    ? `${models.length} model${models.length === 1 ? "" : "s"}`
    : `${enabledCount} of ${models.length} models checked`;
  els.documentTitle.textContent = `Document #${doc.originalIndex}`;
  els.documentSubhead.textContent =
    `${formatNumber(doc.text.length)} characters · ${modelsLabel} · ${state.filterName}`;
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

/* ---------- Classifier prompt modal ---------- */

async function loadPromptPaths() {
  try {
    const response = await fetch(`${CONFIG_URL}?t=${Date.now()}`);
    if (!response.ok) return;
    const config = await response.json();
    for (const filter of config.filters ?? []) {
      if (typeof filter?.name === "string" && typeof filter?.prompt_path === "string") {
        state.promptPaths[filter.name] = filter.prompt_path;
      }
    }
  } catch {
    // Non-fatal: the button will report that no prompt is available.
  }
}

async function fetchPromptText(filterName) {
  if (filterName in state.promptCache) {
    return state.promptCache[filterName];
  }

  const path = state.promptPaths[filterName];
  let text = `No prompt file is known for the filter "${filterName}".`;
  if (path) {
    try {
      const response = await fetch(`../${path}?t=${Date.now()}`);
      text = response.ok
        ? await response.text()
        : `Could not load ${path} (HTTP ${response.status}).`;
    } catch (error) {
      text = `Could not load ${path}: ${error.message}`;
    }
  }
  state.promptCache[filterName] = text;
  return text;
}

async function openPromptModal() {
  els.promptModalTitle.textContent = `Classifier prompt — ${state.filterName}`;
  els.promptModalText.textContent = "Loading prompt…";
  els.promptModal.hidden = false;
  els.promptModalText.textContent = await fetchPromptText(state.filterName);
}

function closePromptModal() {
  els.promptModal.hidden = true;
}

/* ---------- Hand-annotation comparison page ---------- */

// Hand labels are binary (0 = negative, 1 = positive); map them to the 0/10
// poles of the model scale. Values above 1 are taken as already on the 0-10
// scale, in case labeling ever switches to it.
function humanTarget(label) {
  return label <= 1 ? label * 10 : label;
}

async function loadAnnotations() {
  if (state.annotations) return state.annotations;

  // Without the rated documents the join below matches nothing, so wait for
  // them; if they failed to load, still show the hand labels on their own.
  try {
    await documentsReady;
  } catch {}

  const response = await fetch(`${ANNOTATIONS_URL}?t=${Date.now()}`);
  if (!response.ok) {
    throw new Error(`Could not load ${ANNOTATIONS_URL} (HTTP ${response.status}).`);
  }
  const text = await response.text();

  const byText = new Map(state.documents.map((doc) => [doc.text, doc]));
  const items = [];
  let badLines = 0;
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    let row;
    try {
      row = JSON.parse(line);
    } catch {
      badLines += 1;
      continue;
    }
    if (typeof row?.text !== "string" || row.text.length === 0) continue;

    let doc = byText.get(row.text) ?? null;
    if (!doc) {
      // Tolerate hand-edited tails: fall back to a long-prefix match.
      const prefix = row.text.slice(0, 200);
      doc = state.documents.find((d) => d.text.startsWith(prefix)) ?? null;
    }

    // Only the raw hand labels are stored here; means and diffs are computed
    // at render time (filterAnnotationItems) so they track the model checkboxes.
    const labels = [];
    for (const { key, filter, short } of HUMAN_LABEL_FILTERS) {
      const human = row[key];
      if (!Number.isInteger(human) || human < 0) continue; // -1 = not yet labeled
      labels.push({ filter, short, human });
    }
    items.push({ annotationIndex: items.length, text: row.text, doc, labels });
  }

  state.annotations = { items, badLines };
  return state.annotations;
}

function buildAnnotationItem(item) {
  const details = document.createElement("details");
  details.className = "annotation-item";

  const summary = document.createElement("summary");
  summary.className = "annotation-summary";

  const score = document.createElement("span");
  score.className = "ann-score";
  score.textContent =
    item.score === null
      ? "n/a"
      : item.sortMode === "models"
        ? item.score.toFixed(2)
        : item.score.toFixed(1);
  if (item.sortMode === "hand" && item.score !== null && item.score >= 5) {
    score.classList.add("is-high");
  }
  score.title =
    item.sortMode === "models"
      ? `Model rating variance across ${item.modelCount} checked model${item.modelCount === 1 ? "" : "s"}`
      : "|mean model rating − hand label| for the selected filter";

  const chips = document.createElement("span");
  chips.className = "ann-chips";
  for (const c of item.categories) {
    const chip = document.createElement("span");
    chip.className = "ann-chip";
    chip.textContent = `hand ${c.human} → model ${c.mean === null ? "?" : formatMean(c.mean)}`;
    chip.title = `${c.filter}: hand label ${c.human}, model mean ${c.mean === null ? "unknown" : formatMean(c.mean)}`;
    chips.append(chip);
  }

  const snippet = document.createElement("span");
  snippet.className = "ann-snippet";
  snippet.textContent = summarize(item.text);

  summary.append(score, chips, snippet);
  details.append(summary);

  const body = document.createElement("div");
  body.className = "ann-body";

  if (item.categories.length === 0) {
    const note = document.createElement("div");
    note.className = "ann-note";
    note.textContent = `Not yet hand-labeled for ${state.labelsFilter}.`;
    body.append(note);
  }
  if (!item.doc) {
    const note = document.createElement("div");
    note.className = "ann-note";
    note.textContent =
      "No matching document found in the rated JSONL, so model reviews are unavailable.";
    body.append(note);
  }

  for (const c of item.categories) {
    const section = document.createElement("div");
    section.className = "ann-section";

    const head = document.createElement("div");
    head.className = "ann-section-head";
    head.textContent =
      `${c.filter} · hand label ${c.human}` +
      (c.mean === null
        ? ""
        : ` · model mean ${formatMean(c.mean)} · Δ ${c.diff.toFixed(1)}`);
    section.append(head);

    const entries = item.doc?.ratings[c.filter];
    if (entries) {
      for (const [model, entry] of Object.entries(entries)) {
        section.append(buildJudgementCard(model, entry));
      }
    }
    body.append(section);
  }

  const pre = document.createElement("pre");
  pre.className = "ann-doc-text";
  pre.textContent = item.text;
  body.append(pre);

  details.append(body);
  return details;
}

function populateLabelsFilterControls() {
  els.labelsFilterSelect.replaceChildren(
    ...HUMAN_LABEL_FILTERS.map(({ filter }) => {
      const option = document.createElement("option");
      option.value = filter;
      option.textContent = filter;
      return option;
    }),
  );
  els.labelsFilterSelect.value = state.labelsFilter;
}

// Population variance across the checked model ratings for one document.
// With one rating the variance is 0; with no ratings it is unavailable.
function getModelVarianceForFilter(doc, filterName) {
  const entries = doc?.ratings[filterName];
  if (!entries) return { value: null, n: 0 };
  const ratings = Object.entries(entries)
    .filter(([model]) => !state.disabledModels.has(model))
    .map(([, entry]) => entry.rating);
  if (ratings.length === 0) return { value: null, n: 0 };
  const mean = ratings.reduce((sum, rating) => sum + rating, 0) / ratings.length;
  const value =
    ratings.reduce((sum, rating) => sum + (rating - mean) ** 2, 0) /
    ratings.length;
  return { value, n: ratings.length };
}

function compareNullableScoresDescending(a, b) {
  if (a.score === null && b.score === null) {
    return a.annotationIndex - b.annotationIndex;
  }
  if (a.score === null) return 1;
  if (b.score === null) return -1;
  return b.score - a.score || a.annotationIndex - b.annotationIndex;
}

// Restrict each sample to the selected filter, recompute metrics from the
// checked models, and sort by either hand-label error or inter-model variance.
function filterAnnotationItems(items, filterName) {
  return items
    .map((item) => {
      const label = item.labels.find((l) => l.filter === filterName) ?? null;
      const mean = item.doc ? getMeanForFilter(item.doc, filterName) : null;
      const { value: modelVariance, n: modelCount } = getModelVarianceForFilter(
        item.doc,
        filterName,
      );
      const categories = label
        ? [{
            ...label,
            mean,
            diff: mean === null ? null : Math.abs(mean - humanTarget(label.human)),
          }]
        : [];
      const handDiff = categories[0]?.diff ?? null;
      return {
        ...item,
        categories,
        handDiff,
        modelVariance,
        modelCount,
        sortMode: state.labelsSort,
        score: state.labelsSort === "models" ? modelVariance : handDiff,
      };
    })
    .toSorted(compareNullableScoresDescending);
}

/* ---------- Pairwise model correlations (labels page sidebar) ---------- */

function collectCheckedModels(items, filterName) {
  const models = new Set();
  for (const item of items) {
    for (const model of Object.keys(item.doc?.ratings[filterName] ?? {})) {
      if (!state.disabledModels.has(model)) models.add(model);
    }
  }
  return [...models].sort((a, b) => a.localeCompare(b));
}

// Pearson correlation using pairwise-complete observations. Correlation is
// undefined with fewer than two pairs or when either series has zero variance.
function computeModelCorrelation(items, filterName, modelA, modelB) {
  const pairs = [];
  for (const item of items) {
    const entries = item.doc?.ratings[filterName];
    const ratingA = entries?.[modelA]?.rating;
    const ratingB = entries?.[modelB]?.rating;
    if (Number.isFinite(ratingA) && Number.isFinite(ratingB)) {
      pairs.push([ratingA, ratingB]);
    }
  }
  const n = pairs.length;
  if (n < 2) return { value: null, n };

  const meanA = pairs.reduce((sum, pair) => sum + pair[0], 0) / n;
  const meanB = pairs.reduce((sum, pair) => sum + pair[1], 0) / n;
  let covariance = 0;
  let varianceA = 0;
  let varianceB = 0;
  for (const [ratingA, ratingB] of pairs) {
    const deltaA = ratingA - meanA;
    const deltaB = ratingB - meanB;
    covariance += deltaA * deltaB;
    varianceA += deltaA ** 2;
    varianceB += deltaB ** 2;
  }
  const denominator = Math.sqrt(varianceA * varianceB);
  return { value: denominator === 0 ? null : covariance / denominator, n };
}

function renderCorrelationMatrix(items) {
  const models = collectCheckedModels(items, state.labelsFilter);
  els.correlationCaption.textContent =
    `${models.length} checked model${models.length === 1 ? "" : "s"}`;

  if (models.length === 0) {
    const empty = document.createElement("div");
    empty.className = "ann-note";
    empty.textContent = "Check at least one model to show correlations.";
    els.correlationMatrix.replaceChildren(empty);
    return;
  }

  const table = document.createElement("table");
  table.className = "correlation-table";
  table.setAttribute("aria-label", "Pairwise Pearson correlations between checked models");

  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  const corner = document.createElement("th");
  corner.scope = "col";
  corner.textContent = "";
  headRow.append(corner);
  models.forEach((model, index) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = String(index + 1);
    th.title = model;
    th.setAttribute("aria-label", `${index + 1}: ${model}`);
    headRow.append(th);
  });
  head.append(headRow);

  const body = document.createElement("tbody");
  models.forEach((modelA, rowIndex) => {
    const row = document.createElement("tr");
    const rowHead = document.createElement("th");
    rowHead.scope = "row";
    rowHead.textContent = String(rowIndex + 1);
    rowHead.title = modelA;
    rowHead.setAttribute("aria-label", `${rowIndex + 1}: ${modelA}`);
    row.append(rowHead);

    models.forEach((modelB) => {
      const { value: rawValue, n } = computeModelCorrelation(
        items,
        state.labelsFilter,
        modelA,
        modelB,
      );
      const value =
        rawValue === null ? null : Math.max(-1, Math.min(1, rawValue));
      const cell = document.createElement("td");
      cell.textContent = value === null ? "–" : value.toFixed(2);
      cell.title =
        `${modelA} × ${modelB}: ` +
        (value === null ? `undefined (n=${n})` : `r=${value.toFixed(3)} (n=${n})`);
      cell.setAttribute("aria-label", cell.title);
      if (value !== null) {
        cell.classList.add(value < 0 ? "is-negative" : "is-positive");
        cell.style.setProperty(
          "--correlation-alpha",
          `${8 + 32 * Math.abs(value)}%`,
        );
      }
      row.append(cell);
    });
    body.append(row);
  });
  table.append(head, body);

  const legend = document.createElement("ol");
  legend.className = "correlation-legend";
  models.forEach((model) => {
    const item = document.createElement("li");
    item.textContent = model;
    legend.append(item);
  });

  const tableWrap = document.createElement("div");
  tableWrap.className = "correlation-table-wrap";
  tableWrap.tabIndex = 0;
  tableWrap.setAttribute("aria-label", "Scrollable model-correlation matrix");
  tableWrap.append(table);
  els.correlationMatrix.replaceChildren(tableWrap, legend);
}

/* ---------- Per-model agreement stats (labels page sidebar) ---------- */

// For one filter, score each model over the hand-annotated documents:
//   maeHuman  - mean absolute error vs the hand labels mapped to 0/10,
//               over samples that are labeled for this filter
//   maeOthers - mean absolute error vs the mean of the other CHECKED models'
//               ratings on the same document, over all matched samples
function computeModelStats(items, filterName) {
  const perModel = new Map(); // model -> {human: number[], others: number[]}

  for (const item of items) {
    const entries = item.doc?.ratings[filterName];
    if (!entries) continue;
    const models = Object.keys(entries);
    const labeled = item.categories.find((c) => c.filter === filterName);
    const target = labeled ? humanTarget(labeled.human) : null;

    for (const model of models) {
      const stats = perModel.get(model) ?? { human: [], others: [] };
      const rating = entries[model].rating;
      if (target !== null) {
        stats.human.push(Math.abs(rating - target));
      }
      const others = models.filter((m) => m !== model && !state.disabledModels.has(m));
      if (others.length > 0) {
        const otherMean =
          others.reduce((sum, m) => sum + entries[m].rating, 0) / others.length;
        stats.others.push(Math.abs(rating - otherMean));
      }
      perModel.set(model, stats);
    }
  }

  const mae = (values) =>
    values.length === 0
      ? null
      : values.reduce((sum, value) => sum + value, 0) / values.length;

  return [...perModel.entries()]
    .map(([model, stats]) => ({
      model,
      maeHuman: mae(stats.human),
      nHuman: stats.human.length,
      maeOthers: mae(stats.others),
      nOthers: stats.others.length,
    }))
    .sort(
      (a, b) =>
        (a.maeHuman ?? Infinity) - (b.maeHuman ?? Infinity) ||
        a.model.localeCompare(b.model),
    );
}

function buildModelStatRow(label, maeValue, n) {
  const row = document.createElement("div");
  row.className = "model-stat-row";

  const rowLabel = document.createElement("span");
  rowLabel.className = "model-stat-label";
  rowLabel.textContent = label;

  // MAE of 0-10 ratings is bounded by 10; show it as a fraction of that.
  const meter = document.createElement("div");
  meter.className = "model-stat-meter";
  const fill = document.createElement("div");
  fill.className = "model-stat-meter-fill";
  fill.style.width = maeValue === null ? "0%" : `${Math.min(100, maeValue * 10)}%`;
  meter.append(fill);

  const value = document.createElement("span");
  value.className = "model-stat-value";
  value.textContent = maeValue === null ? "–" : maeValue.toFixed(1);

  const count = document.createElement("span");
  count.className = "model-stat-n";
  count.textContent = `n=${n}`;

  row.append(rowLabel, meter, value, count);
  return row;
}

function renderModelStats(items) {
  els.modelStatsCaption.textContent = state.labelsFilter;
  const stats = computeModelStats(items, state.labelsFilter);

  if (stats.length === 0) {
    const empty = document.createElement("div");
    empty.className = "ann-note";
    empty.textContent =
      "No model ratings for this filter yet. Run scripts/rerate_hand_annotated.py " +
      "to rate the hand-annotated samples, then reload.";
    els.modelStatsList.replaceChildren(empty);
    return;
  }

  els.modelStatsList.replaceChildren(
    ...stats.map((s) => {
      const card = document.createElement("div");
      card.className = "model-stat";
      const enabled = !state.disabledModels.has(s.model);
      if (!enabled) card.classList.add("is-disabled");

      const name = document.createElement("label");
      name.className = "model-stat-name";

      const toggle = document.createElement("input");
      toggle.type = "checkbox";
      toggle.className = "model-stat-toggle";
      toggle.checked = enabled;
      toggle.title = "Include this model's ratings in the computed numbers";
      toggle.addEventListener("change", () => setModelEnabled(s.model, toggle.checked));

      const nameText = document.createElement("span");
      nameText.textContent = s.model;
      name.append(toggle, nameText);

      card.append(
        name,
        buildModelStatRow("vs you", s.maeHuman, s.nHuman),
        buildModelStatRow("vs models", s.maeOthers, s.nOthers),
      );
      return card;
    }),
  );
}

// Unchecking a model drops its ratings from every computed number (means,
// diffs, MAE baselines, accuracies) on both pages; its own rows stay visible,
// dimmed, so it can be re-checked.
function setModelEnabled(model, enabled) {
  if (enabled) state.disabledModels.delete(model);
  else state.disabledModels.add(model);
  render();
  renderLabelsPage();
}

/* ---------- Classification threshold & accuracy (labels page sidebar) ---------- */

// Classify every matched sample by the models' mean rating for one filter
// (mean >= threshold -> 1, else 0) and report the filtering tradeoff:
//   posCorrect/posTotal      - positive accuracy: fraction of hand-label-1
//                              samples classified 1 (the complement is the
//                              true positives leaking through the filter)
//   classNegative/classTotal - negative %: fraction of ALL samples classified
//                              0, i.e. the share of data the filter keeps
function computeAccuracy(items, filterName, threshold) {
  const tally = { posTotal: 0, posCorrect: 0, classTotal: 0, classNegative: 0 };

  for (const item of items) {
    if (!item.doc) continue;
    const mean = getMeanForFilter(item.doc, filterName);
    if (mean === null) continue;

    const predicted = mean >= threshold ? 1 : 0;
    tally.classTotal += 1;
    if (predicted === 0) tally.classNegative += 1;

    const labeled = item.categories.find((c) => c.filter === filterName);
    if (labeled && labeled.human >= 1) {
      tally.posTotal += 1;
      if (predicted === 1) tally.posCorrect += 1;
    }
  }

  return tally;
}

// Threshold-free accuracy over the checked models: a sample's mean rating
// becomes p = mean/10, scored as p when the hand label is 1 and 1 - p when it
// is 0, then averaged over all samples labeled for the filter. Equivalent to
// 1 - MAE/10 on the 0-1 scale.
function computeOverallAccuracy(items, filterName) {
  let sum = 0;
  let n = 0;
  for (const item of items) {
    const labeled = item.categories.find((c) => c.filter === filterName);
    if (!labeled || labeled.mean === null) continue;
    const p = labeled.mean / 10;
    sum += labeled.human >= 1 ? p : 1 - p;
    n += 1;
  }
  return { value: n > 0 ? sum / n : null, n };
}

function renderOverallAccuracy() {
  const { value, n } = computeOverallAccuracy(state.labelsItems, state.labelsFilter);
  els.overallValue.textContent = value === null ? "–" : `${(value * 100).toFixed(1)}%`;
  els.overallN.textContent = n > 0 ? `n=${n}` : "";
}

function buildAccuracyRow(label, correct, total) {
  const row = document.createElement("div");
  row.className = "accuracy-row";

  const name = document.createElement("span");
  name.className = "accuracy-label";
  name.textContent = label;

  const pct = document.createElement("span");
  pct.className = "accuracy-pct";
  pct.textContent = total === 0 ? "–" : `${((correct / total) * 100).toFixed(1)}%`;

  const count = document.createElement("span");
  count.className = "accuracy-n";
  count.textContent = `${correct}/${total}`;

  row.append(name, pct, count);
  return row;
}

function renderAccuracy() {
  const threshold = state.labelsThreshold;
  els.thresholdValue.textContent = `x = ${threshold.toFixed(1)}`;
  els.thresholdRange.value = String(threshold);

  const tally = computeAccuracy(state.labelsItems, state.labelsFilter, threshold);
  els.accuracyList.replaceChildren(
    buildAccuracyRow("positive accuracy", tally.posCorrect, tally.posTotal),
    buildAccuracyRow("negative %", tally.classNegative, tally.classTotal),
  );
}

function renderLabelsSortNote() {
  els.labelsSortNote.innerHTML = "";
  if (state.labelsSort === "models") {
    els.labelsSortNote.append(
      "Sorted by model disagreement: population variance ",
      "avg((model score − mean(model scores))²) across checked models for the ",
      "selected filter. Samples without checked-model ratings sort last. Click a row to expand it.",
    );
    return;
  }
  els.labelsSortNote.append(
    "Sorted by hand-label disagreement: |mean model rating − hand label| for the ",
    "selected filter, with binary hand labels 0/1 mapped to 0/10. Unlabeled or ",
    "unmatched samples sort last. Click a row to expand it.",
  );
}

function setThreshold(value) {
  const clamped = Math.min(10, Math.max(0, value));
  state.labelsThreshold = Math.round(clamped * 10) / 10; // keep clean 0.1 steps
  renderAccuracy();
}

let labelsRenderToken = 0;

async function renderLabelsPage() {
  const token = ++labelsRenderToken;
  if (!state.annotations) {
    els.labelsStatusText.textContent = "Loading hand-annotated samples…";
    els.labelsBody.textContent = "Loading hand-annotated samples…";
    els.correlationMatrix.textContent = "Loading…";
    els.modelStatsList.textContent = "Loading…";
  }
  try {
    const { items, badLines } = await loadAnnotations();
    if (token !== labelsRenderToken) return; // superseded by a newer render
    const shown = filterAnnotationItems(items, state.labelsFilter);
    state.labelsItems = shown;
    renderCorrelationMatrix(shown);
    renderModelStats(shown);
    renderOverallAccuracy();
    renderAccuracy();
    renderLabelsSortNote();
    const sortLabel =
      state.labelsSort === "models" ? "model variance" : "hand-label error";
    els.labelsStatusText.textContent =
      `${shown.length} samples · ${state.labelsFilter} · sorted by ${sortLabel}`;
    els.labelsBody.replaceChildren(...shown.map(buildAnnotationItem));
    if (badLines > 0) {
      const note = document.createElement("div");
      note.className = "ann-note";
      note.textContent = `${badLines} line${badLines === 1 ? "" : "s"} in ${ANNOTATIONS_URL} could not be parsed and ${badLines === 1 ? "was" : "were"} skipped.`;
      els.labelsBody.prepend(note);
    }
  } catch (error) {
    if (token !== labelsRenderToken) return;
    els.labelsStatusText.textContent = "Could not load hand labels";
    els.correlationCaption.textContent = "";
    els.correlationMatrix.replaceChildren();
    els.modelStatsList.replaceChildren();
    els.accuracyList.replaceChildren();
    els.labelsBody.textContent = `${error.message}\n\nMake sure ${ANNOTATIONS_URL} exists and the repository root is being served.`;
  }
}

/* ---------- Views ---------- */

function currentView() {
  if (location.hash.startsWith("#/labels")) return "labels";
  if (location.hash.startsWith("#/data")) return "data";
  return "home";
}

function renderRoute() {
  const view = currentView();
  els.homeView.hidden = view !== "home";
  els.labelsView.hidden = view !== "labels";
  els.dataView.hidden = view !== "data";
  if (view === "labels") renderLabelsPage();
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

  els.prevButton.addEventListener("click", () => moveSelection(-1));
  els.nextButton.addEventListener("click", () => moveSelection(1));

  els.promptButton.addEventListener("click", openPromptModal);
  els.promptModalClose.addEventListener("click", closePromptModal);
  els.promptModal.addEventListener("click", (event) => {
    if (event.target === els.promptModal) closePromptModal();
  });

  els.labelsFilterSelect.addEventListener("change", () => {
    state.labelsFilter = els.labelsFilterSelect.value;
    renderLabelsPage();
  });

  els.labelsSortSelect.addEventListener("change", () => {
    state.labelsSort = els.labelsSortSelect.value;
    renderLabelsPage();
  });

  els.thresholdRange.addEventListener("input", () => {
    setThreshold(Number(els.thresholdRange.value));
  });
  els.thresholdMinus.addEventListener("click", () => {
    setThreshold(state.labelsThreshold - 0.1);
  });
  els.thresholdPlus.addEventListener("click", () => {
    setThreshold(state.labelsThreshold + 0.1);
  });

  window.addEventListener("hashchange", renderRoute);

  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !els.promptModal.hidden) {
      closePromptModal();
      return;
    }
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
      return;
    }
    if (currentView() !== "data") {
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
  populateLabelsFilterControls();
  loadPromptPaths();
  documentsReady = loadDocuments();
  renderRoute();
  try {
    await documentsReady;
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
