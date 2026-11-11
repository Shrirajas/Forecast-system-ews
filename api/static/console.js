"use strict";
// EWS console: plain DOM, no build step, no external libraries.
// All server text goes through textContent, never innerHTML.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const state = { tables: {}, columns: [], source: null, uploadId: null, jobTimer: null };

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) e.setAttribute(k, v);
  }
  for (const c of children.flat()) e.append(c instanceof Node ? c : document.createTextNode(c ?? ""));
  return e;
}

function toast(msg, ms = 4000) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let body = text;
  try { body = JSON.parse(text); } catch (_) { /* plain text */ }
  if (!res.ok) {
    const d = body && body.detail !== undefined ? body.detail : body;
    const msg = typeof d === "string" ? d : (d && d.problems ? d.problems.join("\n") : JSON.stringify(d));
    throw new Error(msg || `HTTP ${res.status}`);
  }
  return body;
}
const post = (path, data) => api(path, {
  method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(data ?? {}),
});

const fmt = {
  num: (x, d = 0) => (x === null || x === undefined ? "–" : Number(x).toLocaleString(undefined, { maximumFractionDigits: d })),
  pct: (x) => (x === null || x === undefined ? "–" : (100 * x).toFixed(1) + "%"),
  str: (x) => (x === null || x === undefined || x === "" ? "–" : String(x)),
};

function kv(target, pairs) {
  const box = $(target);
  box.replaceChildren(...pairs.flatMap(([k, v]) => [el("div", { class: "k" }, k), el("div", { class: "v" }, fmt.str(v))]));
}

// Plain-English meaning of column headers; shown on hover (dotted underline).
const HELP = {
  priority: "Ranking score: expected loss weighted by how soon the problem is due. Highest first.",
  PD: "Probability of default: chance an EMI bounces within the horizon.",
  EAD: "Exposure at default: amount at risk if it does default.",
  "expected loss": "PD × EAD × loss rate: the loss to provision for.",
  "breach day": "Days until the balance is likely too low to cover an EMI.",
  "zero day": "Days until the balance is likely to hit (near) zero.",
  type: "CREDIT_EWS = loan repayment at risk. DEPOSIT_ATTRITION = balance/spend projected to fall sharply (customer may leave).",
  tier: "GREEN = fine; SMA-0 / SMA-1 / SMA-2 = special-mention account, rising severity.",
  RM: "Relationship manager / territory the account belongs to.",
};

function table(target, headers, rows) {
  const t = $(target);
  t.replaceChildren(
    el("thead", {}, el("tr", {}, headers.map((h) => el("th", HELP[h] ? { title: HELP[h], class: "has-help" } : {}, h)))),
    el("tbody", {}, rows.length ? rows : [el("tr", {}, el("td", { colspan: headers.length, class: "muted" }, "Nothing yet."))]),
  );
}

function tierBadge(t) { return el("span", { class: `tier ${t}` }, t); }

// ------------------------------------------------------------------------------ tabs
function showTab(name) {
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${name}`));
  if (name === "overview") loadOverview();
  if (name === "database") loadConnection();
  if (name === "jobs") loadJobs();
  if (name === "alerts") loadAlerts();
  if (name === "forecast") loadAccountList();
  if (name === "training") loadTraining();
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ------------------------------------------------------------------------------ overview
async function loadOverview() {
  try {
    const o = await api("/api/v1/console/overview");
    const c = o.connection;
    $("#conn-badge").textContent = c.backend === "duckdb" ? `DuckDB · ${c.duckdb_path}` : `PostgreSQL · ${c.postgres_dsn}`;
    kv("#ov-db", [["backend", c.backend], ["target", c.duckdb_path || c.postgres_dsn], ["device", o.device]]);
    kv("#ov-data", o.data ? [["accounts", fmt.num(o.data.accounts)], ["from", o.data.first_day], ["to", o.data.last_day]]
      : [["status", "no daily balances yet"]]);
    const m = o.model;
    kv("#ov-model", [["backend", m.backend], ["trained", m.exists ? (m.trained_at || "yes") : "no checkpoint"],
      ["version", m.model_version], ...(m.error ? [["error", m.error]] : [])]);
    const b = o.latest_book_run;
    kv("#ov-book", b ? [["scored", String(b.scored_at).replace("T", " ").slice(0, 19)],
      ["up to date", b.up_to_date ? "yes" : "no: data changed since"], ["alerts", fmt.num(b.alerts)],
      ["expected loss", fmt.num(b.total_expected_loss)],
      ["tiers", Object.entries(b.tiers).map(([k, v]) => `${k} ${v}`).join(" · ")]]
      : [["status", "not scored yet"]]);
    $("#ov-tables").replaceChildren(...Object.entries(o.tables)
      .filter(([k]) => !["agg_watermarks", "run_scope"].includes(k))
      .map(([k, v]) => el("span", { class: "chip" }, `${k} `, el("b", {}, fmt.num(v)))));
    nextStep(o);
  } catch (e) { toast("Overview: " + e.message); }
}

function nextStep(o) {
  const box = $("#next-step");
  let msg = null, action = null;
  if (!o.tables.daily_balances) { msg = "No data yet. Import daily balances (and ideally accounts and loan schedules)."; action = ["Import data", () => showTab("import")]; }
  else if (!o.model.exists && o.model.backend === "stformer") { msg = "Data is loaded but no model is trained yet."; action = ["Train model", () => startJob("train")]; }
  else if (!o.latest_book_run || !o.latest_book_run.up_to_date) { msg = "The book has not been scored on the current data."; action = ["Refresh & score", () => startJob("refresh")]; }
  box.classList.toggle("hidden", !msg);
  if (msg) box.replaceChildren(el("span", {}, msg), el("button", { class: "primary", onclick: action[1] }, action[0]));
}

async function startJob(kind) {
  try {
    const body = kind === "train" ? { epochs: Number($("#train-epochs").value || 6) } : {};
    const r = await post(`/api/v1/jobs/${kind}`, body);
    toast(`Started: ${r.job.description}`);
    watchJobs();
  } catch (e) { toast(e.message); }
}
$$("[data-job]").forEach((b) => b.addEventListener("click", () => startJob(b.dataset.job)));

// ------------------------------------------------------------------------------ database
function backendChoice() { return $("input[name=backend]:checked").value; }
$$("input[name=backend]").forEach((r) => r.addEventListener("change", () => {
  $("#db-duckdb").classList.toggle("hidden", backendChoice() !== "duckdb");
  $("#db-postgres").classList.toggle("hidden", backendChoice() !== "postgres");
}));

async function loadConnection() {
  try {
    const c = await api("/api/v1/console/connection");
    kv("#db-current", [["backend", c.backend], ["target", c.duckdb_path || c.postgres_dsn], ["tensor cache", c.cache_root]]);
  } catch (e) { toast(e.message); }
}

function connSpec() {
  const b = backendChoice();
  return b === "duckdb" ? { backend: b, duckdb_path: $("#duckdb-path").value.trim() }
    : { backend: b, postgres_dsn: $("#pg-dsn").value.trim() };
}

async function withButton(btn, fn) {
  btn.disabled = true;
  try { await fn(); } finally { btn.disabled = false; }
}

$("#db-test").addEventListener("click", (ev) => withButton(ev.target, async () => {
  const out = $("#db-result");
  try {
    const r = await post("/api/v1/console/connection/test", connSpec());
    out.className = "result-ok";
    out.textContent = "Connected. Rows already there: " + Object.entries(r.tables).map(([k, v]) => `${k} ${v}`).join(", ");
  } catch (e) { out.className = "result-bad"; out.textContent = e.message; }
}));

$("#db-use").addEventListener("click", (ev) => withButton(ev.target, async () => {
  const out = $("#db-result");
  try {
    const r = await post("/api/v1/console/connection", connSpec());
    out.className = "result-ok";
    out.textContent = "Switched." + (r.job ? " Refreshing and scoring in the background (see Jobs)." : " The database is empty: import data next.");
    loadConnection(); loadOverview(); if (r.job) watchJobs();
  } catch (e) { out.className = "result-bad"; out.textContent = e.message; }
}));

// ------------------------------------------------------------------------------ import
const TABLE_HELP = {
  accounts: "Account master: one row per account (loan amount, EMI, credit limit, LGD, RM).",
  daily_balances: "End-of-day balance per account per day. The one table the forecaster needs.",
  transactions: "Credits and debits. Used for salary, payment returns and outward-payment volatility.",
  credit_profiles: "Bureau pulls (CIBIL score) per account and date.",
  loan_schedules: "Upcoming EMI due dates and amounts: what the early-warning trigger checks against.",
  holidays: "Bank holidays (Sundays and 2nd/4th Saturdays are built in).",
};

async function initImport() {
  state.tables = await api("/api/v1/import/tables");
  const sel = $("#imp-table");
  sel.replaceChildren(...Object.keys(state.tables).map((t) => el("option", { value: t }, t)));
  sel.value = "daily_balances";
  $("#imp-templates").replaceChildren(...Object.keys(state.tables).flatMap((t, i) => [
    i ? " · " : "", el("a", { href: `/api/v1/import/template/${t}` }, t)]));
  sel.addEventListener("change", onTableChange);
  onTableChange();
}

async function onTableChange() {
  const t = $("#imp-table").value;
  $("#imp-table-help").textContent = TABLE_HELP[t] || "";
  $("#signed").closest("label").classList.toggle("hidden", t !== "transactions");
  if (state.columns.length) {
    const r = await post("/api/v1/import/suggest", { table: t, columns: state.columns });
    renderMapping(r.mapping);
  }
}

$$("input[name=src]").forEach((r) => r.addEventListener("change", () => {
  const v = $("input[name=src]:checked").value;
  $("#src-file").classList.toggle("hidden", v !== "file");
  $("#src-sql").classList.toggle("hidden", v !== "sql");
}));

const drop = $("#drop");
["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (ev) => { if (ev.dataTransfer.files[0]) uploadFile(ev.dataTransfer.files[0]); });
$("#file-input").addEventListener("change", (ev) => { if (ev.target.files[0]) uploadFile(ev.target.files[0]); });

async function uploadFile(file) {
  $("#file-name").textContent = `Uploading ${file.name} (${fmt.num(file.size / 1e6, 1)} MB)…`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await api(`/api/v1/import/upload?table=${encodeURIComponent($("#imp-table").value)}`, { method: "POST", body: fd });
    $("#file-name").textContent = `${r.filename} · ${r.columns.length} columns`;
    state.source = { kind: "upload" }; state.uploadId = r.upload_id;
    showPreview(r.columns, r.sample, r.suggested_mapping);
  } catch (e) { $("#file-name").textContent = ""; toast(e.message, 7000); }
}

$("#sql-preview").addEventListener("click", (ev) => withButton(ev.target, async () => {
  try {
    const src = { url: $("#sql-url").value.trim(), query: $("#sql-query").value.trim() };
    const r = await post(`/api/v1/import/sql/preview?table=${encodeURIComponent($("#imp-table").value)}`, src);
    state.source = { kind: "sql", ...src }; state.uploadId = null;
    showPreview(r.columns, r.sample, r.suggested_mapping);
  } catch (e) { toast(e.message, 7000); }
}));

function showPreview(columns, sample, mapping) {
  state.columns = columns;
  table("#imp-preview", columns, sample.map((r) => el("tr", {}, columns.map((c) => el("td", {}, fmt.str(r[c]))))));
  $("#imp-preview-card").classList.remove("hidden");
  renderMapping(mapping);
  $("#imp-map-card").classList.remove("hidden");
  $("#imp-result").textContent = "";
}

function renderMapping(mapping) {
  const t = $("#imp-table").value;
  const rows = state.tables[t].map((col) => {
    const sel = el("select", { "data-target": col.name },
      el("option", { value: "" }, "— not in the source —"),
      state.columns.map((c) => el("option", { value: c }, c)));
    sel.value = mapping[col.name] || "";
    const hint = col.copy_from ? `copies ${col.copy_from}` : (col.default !== null && col.default !== undefined ? String(col.default) : "");
    const constant = el("input", { "data-const": col.name, placeholder: hint || (col.required ? "required" : "optional") });
    return el("tr", {},
      el("td", {}, col.name, col.required ? el("span", { class: "req" }, "required") : ""),
      el("td", { class: "muted" }, col.type),
      el("td", {}, sel),
      el("td", {}, constant));
  });
  table("#imp-map", ["EWS column", "type", "source column", "constant if not mapped"], rows);
}

$("#imp-go").addEventListener("click", (ev) => withButton(ev.target, async () => {
  const out = $("#imp-result");
  const mapping = {}, constants = {};
  $$("#imp-map select").forEach((s) => { mapping[s.dataset.target] = s.value || null; });
  $$("#imp-map input[data-const]").forEach((i) => { if (i.value.trim() !== "") constants[i.dataset.const] = i.value.trim(); });
  const body = {
    table: $("#imp-table").value, mapping, constants,
    date_format: $("#date-fmt").value || null,
    signed_amounts: $("#signed").checked, decimal_comma: $("#dec-comma").checked,
    refresh_after: $("#refresh-after").checked,
  };
  if (state.source && state.source.kind === "sql") body.sql = { url: state.source.url, query: state.source.query };
  else body.upload_id = state.uploadId;
  try {
    const r = await post("/api/v1/import/commit", body);
    out.className = "result-ok";
    out.textContent = `Import started (${r.job.id}). Progress is on the Jobs tab.`;
    watchJobs(r.job.id, (job) => {
      if (job.status === "done") {
        out.className = "result-ok";
        out.textContent = `Imported ${fmt.num(job.result.rows)} rows into ${job.result.table}` +
          (job.result.first_date ? ` (${job.result.first_date} → ${job.result.last_date})` : "") +
          (job.result.refresh ? ". Features refreshed and the book rescored." : ".");
      } else if (job.status === "failed") { out.className = "result-bad"; out.textContent = job.error; }
    });
  } catch (e) { out.className = "result-bad"; out.textContent = e.message; }
}));

// ------------------------------------------------------------------------------ jobs
async function loadJobs() {
  const jobs = await api("/api/v1/jobs");
  table("#jobs-table", ["job", "what", "status", "submitted", "finished", "result / error"], jobs.map((j) => el("tr", {},
    el("td", { class: "muted" }, j.id), el("td", {}, j.description),
    el("td", {}, el("span", { class: `status ${j.status}` }, j.status)),
    el("td", {}, (j.submitted_at || "").replace("T", " ")), el("td", {}, (j.finished_at || "").replace("T", " ")),
    el("td", { class: "muted" }, j.error || summarise(j.result)))));
  const active = jobs.filter((j) => j.status === "queued" || j.status === "running").length;
  $("#jobs-badge").textContent = active; $("#jobs-badge").classList.toggle("hidden", !active);
  return jobs;
}

function summarise(r) {
  if (!r) return "";
  if (r.rows !== undefined) return `${r.rows} rows`;
  if (r.training) return `val pinball ${r.training.best_val_pinball}, ${r.training.seconds}s`;
  if (r.book && r.book.alerts !== undefined) return `${r.book.alerts} alerts`;
  if (r.alerts !== undefined) return `${r.alerts} alerts`;
  return JSON.stringify(r).slice(0, 80);
}

function watchJobs(jobId, onFinish) {
  clearInterval(state.jobTimer);
  state.jobTimer = setInterval(async () => {
    try {
      const jobs = await loadJobs();
      if (jobId) {
        const j = jobs.find((x) => x.id === jobId);
        if (j && (j.status === "done" || j.status === "failed")) { onFinish && onFinish(j); jobId = null; }
      }
      if (!jobs.some((j) => j.status === "queued" || j.status === "running")) {
        clearInterval(state.jobTimer); loadOverview();
      }
    } catch (_) { /* keep polling */ }
  }, 1500);
}

// ------------------------------------------------------------------------------ alerts
async function loadAlerts() {
  const q = new URLSearchParams({ limit: "200" });
  if ($("#al-type").value) q.set("alert_type", $("#al-type").value);
  if ($("#al-tier").value) q.set("tier", $("#al-tier").value);
  try {
    const r = await api("/api/v1/alerts/active?" + q);
    fillAccountList(r.alerts);
    table("#alerts-table", ["#", "account", "RM", "type", "tier", "priority", "PD", "EAD", "expected loss",
      "breach day", "zero day", "next EMI", "triggers"],
    r.alerts.map((a) => el("tr", {},
      el("td", { class: "num" }, a.rank), el("td", {}, el("a", { class: "acct", onclick: () => openForecast(a.account_id) }, a.account_id)),
      el("td", {}, fmt.str(a.rm_id)), el("td", {}, a.alert_type), el("td", {}, tierBadge(a.risk_tier)),
      el("td", { class: "num" }, fmt.num(a.priority_score)), el("td", { class: "num" }, fmt.pct(a.probability_of_default)),
      el("td", { class: "num" }, fmt.num(a.exposure_at_default)), el("td", { class: "num" }, fmt.num(a.expected_loss)),
      el("td", { class: "num" }, fmt.str(a.days_to_breach)), el("td", { class: "num" }, fmt.str(a.days_to_zero)),
      el("td", {}, a.next_emi_date ? `${a.next_emi_date} · ${fmt.num(a.next_emi_amount)}` : "–"),
      el("td", { class: "muted" }, (a.triggers || []).join(", ")))));
  } catch (e) { toast(e.message); }
}
$("#al-reload").addEventListener("click", loadAlerts);
$("#al-type").addEventListener("change", loadAlerts);
$("#al-tier").addEventListener("change", loadAlerts);

// ------------------------------------------------------------------------------ forecast
function openForecast(id) { $("#fc-account").value = id; showTab("forecast"); runForecast(); }

// Suggest alerted accounts in the account box so nobody has to guess an id.
function fillAccountList(alerts) {
  const seen = new Set();
  $("#fc-accounts").replaceChildren(...alerts.filter((a) => !seen.has(a.account_id) && seen.add(a.account_id))
    .map((a) => el("option", { value: a.account_id }, `${a.risk_tier} · ${a.alert_type}`)));
}
async function loadAccountList() {
  if ($("#fc-accounts").children.length) return;
  try { fillAccountList((await api("/api/v1/alerts/active?limit=200")).alerts); } catch (_) { /* optional */ }
}
$("#fc-go").addEventListener("click", runForecast);
$("#fc-account").addEventListener("keydown", (e) => { if (e.key === "Enter") runForecast(); });

async function runForecast() {
  const id = $("#fc-account").value.trim();
  if (!id) return;
  try {
    const r = await post("/api/v1/predict/ews", {
      account_ids: [id], horizon: Number($("#fc-h").value), include_paths: true,
      return_quantiles: [0.1, 0.5, 0.9], persist_alerts: false,
    });
    if (!r.accounts.length) { toast(`Unknown account ${id}`); return; }
    const a = r.accounts[0];
    const cards = [
      ["Tier", tierBadge(a.risk_tier), "Risk bucket: GREEN = fine; SMA-0/1/2 = special-mention account, rising severity."],
      ["Balance today", fmt.num(a.current_balance), "Latest end-of-day balance."],
      ["Next EMI", a.next_emi ? `${fmt.num(a.next_emi.amount)} on ${a.next_emi.date}` : "–", "Next loan instalment due on this account."],
      ["Days to breach", fmt.str(a.days_to_breach), "Days until the balance is likely too low to cover an EMI."],
      ["PD", fmt.pct(a.probability_of_default), "Probability of default: chance an EMI bounces within the horizon."],
      ["EAD", fmt.num(a.exposure_at_default), "Exposure at default: amount at risk if it does default."],
      ["Expected loss", fmt.num(a.expected_loss), "PD × EAD × loss rate: the loss to provision for."],
      ["Trend", `${a.behavior.trend} (${fmt.num(a.behavior.projected_change_pct, 1)}%)`, "Projected change in balance over the horizon."],
    ];
    $("#fc-summary").replaceChildren(...cards.map(([k, v, tip]) => el("div", { class: "card", title: tip },
      el("div", { class: "muted" }, k, el("span", { class: "q", "aria-label": tip }, "?")), el("div", { class: "big" }, v))));
    $("#fc-triggers").replaceChildren(...(a.triggers.length ? a.triggers : ["no triggers"]).map((t) => el("span", { class: "chip" }, t)));
    $("#fc-empty").classList.add("hidden");
    drawChart(a.forecast, /CARD$/.test(a.product_type || "") ? "30-day spend" : "balance");
  } catch (e) { toast(e.message); }
}

function drawChart(f, what = "balance") {
  const W = 960, H = 320, L = 88, R = 16, T = 16, B = 34;
  const n = f.dates.length, lo = f.balance.q10, md = f.balance.q50, hi = f.balance.q90;
  const emis = f.emi_schedule.map((e) => ({ i: f.dates.indexOf(e.date), amt: e.amount })).filter((e) => e.i >= 0);
  const ymin = Math.min(0, ...lo), ymax = Math.max(...hi, ...emis.map((e) => e.amt), 1);
  const x = (i) => L + (i / Math.max(n - 1, 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - T - B);
  const path = (arr) => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const band = path(hi) + lo.map((v, i) => `L${x(n - 1 - i).toFixed(1)},${y(lo[n - 1 - i]).toFixed(1)}`).join("") + "Z";
  const NS = "http://www.w3.org/2000/svg";
  const s = (tag, attrs, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    return e;
  };
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "balance forecast" });
  for (let k = 0; k <= 4; k++) {
    const v = ymin + (k / 4) * (ymax - ymin);
    svg.append(s("line", { class: "axis", x1: L, x2: W - R, y1: y(v), y2: y(v) }),
      s("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, Math.round(v).toLocaleString()));
  }
  if (ymin < 0) svg.append(s("line", { class: "zero", x1: L, x2: W - R, y1: y(0), y2: y(0) }));
  svg.append(s("path", { class: "band", d: band }), s("path", { class: "med", d: path(md) }), s("path", { class: "low", d: path(lo) }));
  for (const e of emis) {
    svg.append(s("line", { class: "emi-line", x1: x(e.i), x2: x(e.i), y1: T, y2: H - B }),
      s("circle", { class: "emi", cx: x(e.i), cy: y(e.amt), r: 4 }, undefined));
  }
  [[0, "start"], [Math.floor((n - 1) / 4), "middle"], [Math.floor((n - 1) / 2), "middle"],
   [Math.floor(3 * (n - 1) / 4), "middle"], [n - 1, "end"]].forEach(([i, anchor]) =>
    svg.append(s("text", { x: x(i), y: H - 10, "text-anchor": anchor }, f.dates[i])));
  svg.append(s("text", { x: 12, y: T + (H - T - B) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${T + (H - T - B) / 2})` }, what[0].toUpperCase() + what.slice(1)));

  // hover: guide line + tooltip for the nearest day
  const guide = s("line", { class: "guide hidden", y1: T, y2: H - B });
  svg.append(guide);
  const tip = el("div", { class: "tip hidden" });
  const emiOn = Object.fromEntries(emis.map((e) => [e.i, e.amt]));
  svg.addEventListener("mousemove", (ev) => {
    const vx = (ev.offsetX / svg.clientWidth) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((vx - L) / (W - L - R)) * (n - 1))));
    guide.setAttribute("x1", x(i)); guide.setAttribute("x2", x(i)); guide.classList.remove("hidden");
    tip.replaceChildren(el("b", {}, f.dates[i]),
      el("div", {}, `Expected: ${fmt.num(md[i])}`),
      el("div", {}, `Likely range: ${fmt.num(lo[i])} – ${fmt.num(hi[i])}`),
      ...(i in emiOn ? [el("div", { class: "tip-emi" }, `EMI due: ${fmt.num(emiOn[i])}`)] : []),
      ...(lo[i] < (emiOn[i] ?? 0) ? [el("div", { class: "tip-bad" }, "Risk: low case can't cover the EMI")] : []));
    tip.style.left = `${Math.min(ev.offsetX + 14, svg.clientWidth - 220)}px`;
    tip.style.top = `${ev.offsetY + 10}px`;
    tip.classList.remove("hidden");
  });
  svg.addEventListener("mouseleave", () => { guide.classList.add("hidden"); tip.classList.add("hidden"); });

  const key = (cls, label, desc) => el("div", { class: "key" }, el("span", { class: `sw ${cls}` }), el("b", {}, label), el("span", { class: "muted" }, desc));
  const legend = el("div", { class: "legend" },
    key("sw-med", `Expected ${what}`, "most likely path (median)"),
    key("sw-band", "Likely range", `80% chance the ${what} stays inside`),
    key("sw-low", "Bad case", "only 10% chance of being lower"),
    ...(emis.length ? [key("sw-emi", "EMI due", "dot height = instalment amount")] : []),
    ...(ymin < 0 ? [key("sw-zero", "Zero", what === "balance" ? "below = overdrawn" : "below = more refunds than spend")] : []));
  const last = n - 1;
  const read = el("p", { class: "hint" }, `By ${f.dates[last]} the ${what} is expected to be about ${fmt.num(md[last])}, ` +
    `likely between ${fmt.num(lo[last])} and ${fmt.num(hi[last])}. ` +
    (emis.length ? "Where the red dashed line drops below an orange dot, the account may not cover that EMI. "
      : what === "balance" ? "" : "A falling line means the customer is using the card less. ") +
    "Hover the chart for daily values.");
  $("#fc-chart").replaceChildren(legend, svg, tip, read);
}

// ------------------------------------------------------------------------------ training
const NS_SVG = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, text) {
  const e = document.createElementNS(NS_SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== undefined) e.textContent = text;
  return e;
}

$("#tr-go").addEventListener("click", (ev) => withButton(ev.target, async () => {
  const steps = $("#tr-steps").value.trim();
  const r = await post("/api/v1/jobs/train", { epochs: Number($("#tr-epochs").value || 6), ...(steps ? { steps_per_epoch: Number(steps) } : {}) });
  toast(`Started: ${r.job.description}`);
  watchJobs();
  setTimeout(loadTraining, 800);
}).catch((e) => toast(e.message)));

async function loadTraining() {
  clearTimeout(loadTraining._t);
  let p;
  try { p = await api("/api/v1/console/training"); } catch (e) { toast(e.message); return; }
  if (p.status === "none") {
    $("#tr-status").replaceChildren(el("p", { class: "hint" }, `No training run recorded yet. The model will be saved to ${p.checkpoint}`));
    $("#tr-fill").style.width = "0%"; $("#tr-chart").replaceChildren(); $("#tr-table").replaceChildren();
    return;
  }
  const h = p.history || [], spe = p.steps_per_epoch || 1;
  const doneSteps = h.length * spe + (p.status === "running" && p.epoch > h.length ? p.step : 0);
  const pct = p.status === "done" ? 100 : Math.min(100, (100 * doneSteps) / (p.epochs * spe));
  $("#tr-fill").style.width = `${pct.toFixed(1)}%`;
  $("#tr-fill").className = `fill ${p.status}`;
  const best = h.filter((r) => r.saved).pop();
  $("#tr-status").replaceChildren(el("div", { class: "kv" }, ...[
    ["status", p.status === "running" ? `running · epoch ${p.epoch}/${p.epochs} · step ${p.step}/${spe}` : p.status + (p.error ? `: ${p.error}` : "")],
    ["progress", `${pct.toFixed(0)}%`], ["device", p.device],
    ["data", `${fmt.num(p.train_windows)} training windows · ${fmt.num(p.val_windows)} check windows · batch ${p.batch_size}`],
    ["model size", `${fmt.num(p.params)} weights`],
    ["best so far", best ? `epoch ${best.epoch}: error ${best.pinball.toFixed(4)}, ${(100 * best.coverage_10_90).toFixed(1)}% inside band` : "–"],
    ["saved to", p.checkpoint],
  ].flatMap(([k, v]) => [el("div", { class: "k" }, k), el("div", { class: "v" }, v)])));
  drawTrainChart(h, p.epochs);
  table("#tr-table", ["epoch", "train error", "check error", "inside band", "typical miss", "saved", "elapsed"],
    h.map((r) => el("tr", {}, el("td", { class: "num" }, r.epoch), el("td", { class: "num" }, r.train_pinball.toFixed(4)),
      el("td", { class: "num" }, r.pinball.toFixed(4)), el("td", { class: "num" }, fmt.pct(r.coverage_10_90)),
      el("td", { class: "num" }, fmt.pct(r.median_mape)), el("td", {}, r.saved ? "✓" : ""), el("td", { class: "num" }, `${r.seconds}s`))));
  if (p.status === "running" && $("#tab-training").classList.contains("active")) loadTraining._t = setTimeout(loadTraining, 1500);
}

function drawTrainChart(h, epochs) {
  if (!h.length) { $("#tr-chart").replaceChildren(el("p", { class: "hint" }, "Waiting for the first epoch to finish…")); return; }
  const W = 960, H = 260, L = 64, R = 16, T = 14, B = 34;
  const vals = h.flatMap((r) => [r.train_pinball, r.pinball]);
  const lo = Math.min(...vals) * 0.95, hi = Math.max(...vals) * 1.05;
  const n = Math.max(epochs, 2);
  const x = (e) => L + ((e - 1) / (n - 1)) * (W - L - R);
  const y = (v) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B);
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "training error per epoch" });
  for (let k = 0; k <= 4; k++) {
    const v = lo + (k / 4) * (hi - lo);
    svg.append(svgEl("line", { class: "axis", x1: L, x2: W - R, y1: y(v), y2: y(v) }),
      svgEl("text", { x: L - 8, y: y(v) + 4, "text-anchor": "end" }, v.toFixed(3)));
  }
  for (let e = 1; e <= epochs; e++) svg.append(svgEl("text", { x: x(e), y: H - 10, "text-anchor": "middle" }, `ep ${e}`));
  const line = (key, cls) => svgEl("path", { class: cls, d: h.map((r, i) => `${i ? "L" : "M"}${x(r.epoch)},${y(r[key])}`).join("") });
  svg.append(line("train_pinball", "tr-train"), line("pinball", "med"));
  for (const r of h) {
    svg.append(svgEl("circle", { class: "tr-dot", cx: x(r.epoch), cy: y(r.pinball), r: r.saved ? 5 : 3 }));
    if (r.saved) svg.append(svgEl("text", { x: x(r.epoch), y: y(r.pinball) - 9, "text-anchor": "middle", class: "tr-saved" }, "✓"));
  }
  const key = (cls, label, desc) => el("div", { class: "key" }, el("span", { class: `sw ${cls}` }), el("b", {}, label), el("span", { class: "muted" }, desc));
  $("#tr-chart").replaceChildren(el("div", { class: "legend" },
    key("sw-med", "Check error", "on accounts it never trained on: this is what matters"),
    key("sw-train", "Training error", "on the data it learns from"),
    el("div", { class: "key" }, el("b", {}, "✓"), el("span", { class: "muted" }, "weights saved (best so far)"))), svg);
}

// ------------------------------------------------------------------------------ boot
initImport().catch((e) => toast(e.message));
loadOverview();
api("/api/v1/jobs").then((j) => { if (j.some((x) => x.status === "queued" || x.status === "running")) watchJobs(); }).catch(() => {});
