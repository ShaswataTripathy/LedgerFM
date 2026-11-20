// LedgerFM in-browser demo: CSV -> LedgerFM tokens (tokenizer.js) -> ONNX Runtime Web -> risk scores / embeddings.
"use strict";
const ORT_VER = "1.20.1";
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
ort.env.wasm.numThreads = 1;

const MAX_ROWS = 20000, L = 128;
const REQUIRED = ["txn_id", "ts", "account", "counterparty", "amount"];
const OPT = ["currency", "currency_recv", "category", "channel", "location", "instrument", "src_bank", "dst_bank", "errors"];
const $ = (id) => document.getElementById(id);
const sessions = {};
let last = null;

async function session(name) {
  if (!sessions[name]) {
    status(`Loading the ${name === "scorer" ? "risk model" : "embedding model"} (15 MB, cached after the first time)…`);
    sessions[name] = await ort.InferenceSession.create(`models/${name}.onnx`, { executionProviders: ["wasm"] });
  }
  return sessions[name];
}
function status(msg, cls = "") { const s = $("status"); s.textContent = msg; s.className = "status " + cls; }
function progress(f) { $("bar").style.width = (100 * f).toFixed(1) + "%"; }
const tick = () => new Promise((r) => setTimeout(r, 0));

function parseCsv(text) {
  const p = Papa.parse(text, { header: true, skipEmptyLines: true, dynamicTyping: false });
  if (p.errors.length && !p.data.length) throw new Error("Could not read the CSV: " + p.errors[0].message);
  const cols = p.meta.fields.map((c) => c.trim());
  const miss = REQUIRED.filter((c) => !cols.includes(c));
  if (miss.length) throw new Error(`Missing required column(s): ${miss.join(", ")}. Required: ${REQUIRED.join(", ")}.`);
  if (p.data.length > MAX_ROWS) throw new Error(`${p.data.length.toLocaleString()} rows: this in-browser demo takes up to ${MAX_ROWS.toLocaleString()}. Use "pip install ledgerfm" for more.`);
  const hasLabel = cols.includes("label");
  const nul = (v) => (v === undefined || v === null || String(v).trim() === "" ? null : String(v).trim());
  const rows = p.data.map((r, i) => {
    const o = { _i: i, txn_id: Number.isFinite(+r.txn_id) && nul(r.txn_id) !== null ? +r.txn_id : i,
      ts: LedgerTok.parseTs(r.ts), account: nul(r.account), counterparty: nul(r.counterparty),
      amount: nul(r.amount) === null ? null : +r.amount, amount_recv: nul(r.amount_recv) === null ? null : +r.amount_recv,
      label: hasLabel ? +r.label : null, _raw: r };
    for (const c of OPT) o[c] = nul(r[c]);
    if (o.currency === null) o.currency = "USD";
    return o;
  });
  const bad = rows.findIndex((r) => Number.isNaN(r.ts) || r.account === null || r.counterparty === null || r.amount === null || Number.isNaN(r.amount));
  if (bad >= 0) throw new Error(`Row ${bad + 2}: ts, account, counterparty and amount must be filled (ts like 2024-01-31 13:45:00).`);
  return { rows, hasLabel };
}

async function runModel(sess, tok, out, dim, label) {
  const W = LedgerTok.windows(tok.offsets, L);
  for (let w = 0; w < W.length; w++) {
    const [s, n, from] = W[w];
    const ints = new BigInt64Array(n * tok.NI), fl = tok.floats.subarray(s * tok.NF, (s + n) * tok.NF);
    for (let j = 0; j < n * tok.NI; j++) ints[j] = BigInt(tok.ints[s * tok.NI + j]);
    const res = await sess.run({ ints: new ort.Tensor("int64", ints, [1, n, tok.NI]), floats: new ort.Tensor("float32", Float32Array.from(fl), [1, n, tok.NF]) });
    const data = Object.values(res)[0].data;
    for (let j = from; j < n; j++) {
      if (dim === 1) out[s + j] = data[j]; else out.set(data.subarray(j * dim, (j + 1) * dim), (s + j) * dim);
    }
    if (w % 2 === 0) { progress((w + 1) / W.length); status(`${label}: window ${w + 1} of ${W.length}`); await tick(); }
  }
}

function prAuc(y, s) {
  const o = s.map((v, i) => i).sort((a, b) => s[b] - s[a]);
  const P = y.reduce((a, b) => a + b, 0); if (!P) return NaN;
  let tp = 0, ap = 0, prevRec = 0;
  for (let k = 0; k < o.length; k++) {
    tp += y[o[k]];
    if (k === o.length - 1 || s[o[k + 1]] !== s[o[k]]) { const rec = tp / P; ap += (rec - prevRec) * (tp / (k + 1)); prevRec = rec; }
  }
  return ap;
}

async function run(text) {
  $("results").hidden = true; progress(0);
  const t0 = performance.now();
  try {
    const { rows, hasLabel } = parseCsv(text);
    status(`Tokenising ${rows.length.toLocaleString()} transactions…`); await tick();
    const tok = LedgerTok.tokenizeRows(rows);
    const N = rows.length, score = new Float32Array(N);
    await runModel(await session("scorer"), tok, score, 1, "Scoring");
    let emb = null;
    if ($("wantEmb").checked) { emb = new Float32Array(N * 256); progress(0); await runModel(await session("encoder"), tok, emb, 256, "Embedding"); }
    progress(1);
    const sorted = tok.order.map((ri, k) => ({ r: rows[ri], s: score[k], k }));
    last = { sorted, emb, hasLabel };
    const accounts = tok.offsets.length - 1, secs = ((performance.now() - t0) / 1000).toFixed(1);
    let msg = `Scored <b>${N.toLocaleString()}</b> transactions from <b>${accounts.toLocaleString()}</b> accounts in ${secs}s, on your device.`;
    if (hasLabel) {
      const y = sorted.map((x) => (x.r.label === 1 ? 1 : 0)), s = sorted.map((x) => x.s), k = y.reduce((a, b) => a + b, 0);
      if (k > 0 && k < N) {
        const hit = [...sorted].sort((a, b) => b.s - a.s).slice(0, k).filter((x) => x.r.label === 1).length;
        msg += `<br>Your file has a <code>label</code> column: PR-AUC <b>${prAuc(y, s).toFixed(3)}</b>. The ${k} highest scores contain <b>${hit} of the ${k}</b> labelled frauds.`;
      }
    }
    $("summary").innerHTML = msg;
    renderTable([...sorted].sort((a, b) => b.s - a.s).slice(0, 50), hasLabel);
    $("dlEmb").hidden = !emb;
    $("results").hidden = false; status("Done.", "ok");
  } catch (e) { console.error(e); status(e.message || String(e), "err"); }
}

function renderTable(top, hasLabel) {
  const esc = (v) => String(v ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const head = ["risk", "txn_id", "ts", "account", "counterparty", "amount", "category"].concat(hasLabel ? ["label"] : []);
  let h = "<thead><tr>" + head.map((c) => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
  for (const { r, s } of top) {
    const raw = r._raw;
    h += `<tr><td class="num"><span class="pill" style="--p:${s}">${s.toFixed(3)}</span></td><td>${esc(raw.txn_id)}</td><td>${new Date(r.ts).toISOString().slice(0, 19).replace("T", " ")}</td>` +
      `<td>${esc(raw.account)}</td><td>${esc(raw.counterparty)}</td><td class="num">${esc(raw.amount)}</td><td>${esc(raw.category)}</td>` +
      (hasLabel ? `<td>${r.label === 1 ? "fraud" : ""}</td>` : "") + "</tr>";
  }
  $("table").innerHTML = h + "</tbody>";
}

function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/csv" })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
$("dlScores").onclick = () => {
  const lines = ["txn_id,risk_score"]; for (const x of [...last.sorted].sort((a, b) => a.r._i - b.r._i)) lines.push(`${x.r._raw.txn_id},${x.s}`);
  download("ledgerfm_scores.csv", lines.join("\n"));
};
$("dlEmb").onclick = () => {
  const lines = ["txn_id," + Array.from({ length: 256 }, (_, j) => "emb" + j).join(",")];
  for (const x of last.sorted) lines.push(x.r._raw.txn_id + "," + Array.from(last.emb.subarray(x.k * 256, (x.k + 1) * 256), (v) => v.toPrecision(5)).join(","));
  download("ledgerfm_embeddings.csv", lines.join("\n"));
};
$("file").onchange = async (e) => { const f = e.target.files[0]; if (f) run(await f.text()); };
$("example").onclick = async () => { status("Loading example…"); run(await (await fetch("example_sparkov_cards.csv")).text()); };
