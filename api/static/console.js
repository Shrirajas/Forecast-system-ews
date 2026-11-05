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

function table(target, headers, rows) {
  const t = $(target);
  t.replaceChildren(
    el("thead", {}, el("tr", {}, headers.map((h) => el("th", {}, h)))),
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
      ["Tier", tierBadge(a.risk_tier)], ["Balance today", fmt.num(a.current_balance)],
      ["Next EMI", a.next_emi ? `${fmt.num(a.next_emi.amount)} on ${a.next_emi.date}` : "–"],
      ["Days to breach", fmt.str(a.days_to_breach)], ["PD", fmt.pct(a.probability_of_default)],
      ["EAD", fmt.num(a.exposure_at_default)], ["Expected loss", fmt.num(a.expected_loss)],
      ["Trend", `${a.behavior.trend} (${fmt.num(a.behavior.projected_change_pct, 1)}%)`],
    ];
    $("#fc-summary").replaceChildren(...cards.map(([k, v]) => el("div", { class: "card" }, el("div", { class: "muted" }, k), el("div", {}, v))));
    $("#fc-triggers").replaceChildren(...(a.triggers.length ? a.triggers : ["no triggers"]).map((t) => el("span", { class: "chip" }, t)));
    drawChart(a.forecast);
  } catch (e) { toast(e.message); }
}

function drawChart(f) {
  const W = 960, H = 320, L = 70, R = 16, T = 16, B = 34;
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
  [[0, "start"], [Math.floor((n - 1) / 2), "middle"], [n - 1, "end"]].forEach(([i, anchor]) =>
    svg.append(s("text", { x: x(i), y: H - 10, "text-anchor": anchor }, f.dates[i])));
  const legend = el("p", { class: "hint" }, "Shaded: q10–q90 band · solid: median (q50) · dashed red: worst-case q10 track · orange: EMI due (amount).");
  $("#fc-chart").replaceChildren(svg, legend);
}

// ------------------------------------------------------------------------------ boot
initImport().catch((e) => toast(e.message));
loadOverview();
api("/api/v1/jobs").then((j) => { if (j.some((x) => x.status === "queued" || x.status === "running")) watchJobs(); }).catch(() => {});
