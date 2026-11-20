// LedgerFM tokenizer, a JavaScript port of ledgerfm/tokens.py (context_columns + tokenize).
// Works in the browser (globalThis.LedgerTok) and in Node (module.exports) for parity tests.
(function (root) {
  "use strict";

  // ---------- MD5 (RFC 1321) on UTF-8 bytes ----------
  function md5Bytes(bytes) {
    const K = new Uint32Array(64), S = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];
    for (let i = 0; i < 64; i++) K[i] = Math.floor(Math.abs(Math.sin(i + 1)) * 4294967296) >>> 0;
    const n = bytes.length, nb = ((n + 8) >> 6) + 1, words = new Uint32Array(nb * 16);
    for (let i = 0; i < n; i++) words[i >> 2] |= bytes[i] << ((i % 4) * 8);
    words[n >> 2] |= 0x80 << ((n % 4) * 8);
    words[nb * 16 - 2] = (n * 8) >>> 0; words[nb * 16 - 1] = Math.floor(n / 536870912) >>> 0;
    let a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;
    for (let blk = 0; blk < nb; blk++) {
      let A = a0, B = b0, C = c0, D = d0;
      for (let i = 0; i < 64; i++) {
        let F, g;
        if (i < 16) { F = (B & C) | (~B & D); g = i; }
        else if (i < 32) { F = (D & B) | (~D & C); g = (5 * i + 1) % 16; }
        else if (i < 48) { F = B ^ C ^ D; g = (3 * i + 5) % 16; }
        else { F = C ^ (B | ~D); g = (7 * i) % 16; }
        F = (F + A + K[i] + words[blk * 16 + g]) >>> 0;
        A = D; D = C; C = B;
        const s = S[(i >> 4) * 4 + (i % 4)];
        B = (B + ((F << s) | (F >>> (32 - s)))) >>> 0;
      }
      a0 = (a0 + A) >>> 0; b0 = (b0 + B) >>> 0; c0 = (c0 + C) >>> 0; d0 = (d0 + D) >>> 0;
    }
    return [a0, b0, c0, d0]; // little-endian 32-bit words of the digest
  }
  const enc = new TextEncoder();
  const hashCache = new Map();
  // python: int.from_bytes(md5(salt+v).digest()[:8], "little") % (buckets-1) + 1 ; 0 = null
  function stableHash(v, buckets, salt) {
    if (v === null || v === undefined) return 0;
    const key = salt + "\u0000" + buckets + "\u0000" + v;
    let h = hashCache.get(key);
    if (h === undefined) {
      const w = md5Bytes(enc.encode(salt + String(v)));
      const x = (BigInt(w[1]) << 32n) | BigInt(w[0]);
      h = Number(x % BigInt(buckets - 1)) + 1;
      hashCache.set(key, h);
    }
    return h;
  }

  // ---------- fields (must match tokens.py) ----------
  const INT_FIELDS = [["amt_b", 64], ["gap_b", 32], ["hour", 25], ["wday", 8], ["category", 4096], ["channel", 512],
    ["location", 2048], ["currency", 256], ["currency_recv", 256], ["errors", 256], ["cp", 32768], ["flags", 64],
    ["r_n_b", 32], ["r_gap_b", 32]];
  const NI = INT_FIELDS.length, NF = 12;
  const ISO = { "US Dollar": "USD", "Euro": "EUR", "Swiss Franc": "CHF", "Yuan": "CNY", "Shekel": "ILS", "Rupee": "INR",
    "UK Pound": "GBP", "Yen": "JPY", "Ruble": "RUB", "Bitcoin": "BTC", "Canadian Dollar": "CAD",
    "Australian Dollar": "AUD", "Mexican Peso": "MXN", "Saudi Riyal": "SAR", "Brazil Real": "BRL" };
  const USD_RATE = { USD: 1.0, EUR: 1.17, CHF: 1.09, CNY: 0.149, ILS: 0.296, INR: 0.0136, GBP: 1.29, JPY: 0.0095,
    RUB: 0.0129, BTC: 11900.0, CAD: 0.758, AUD: 0.708, MXN: 0.0473, SAR: 0.267, BRL: 0.177 };
  const normCcy = (c) => (c === null || c === undefined) ? null : (ISO[c] !== undefined ? ISO[c] : String(c)).toUpperCase();

  const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
  function logb(x, scale, n) { // 0 = missing
    if (x === null || Number.isNaN(x)) return 0;
    const v = Math.floor(Math.log1p(Math.max(x, 0)) * scale) + 1;
    return Math.min(Math.max(v, 1), n - 1);
  }

  // rolling count over (t - 24h, t] within one group; rows in time order; equal timestamps all count
  function rolling24(idx, ts, ones) {
    const out = new Float64Array(idx.length); let lo = 0, sum = 0, hi = 0;
    const W = 24 * 3600 * 1000;
    for (let i = 0; i < idx.length; i++) {
      const t = ts[idx[i]];
      while (hi < idx.length && ts[idx[hi]] <= t) { sum += ones[idx[hi]]; hi++; }
      while (ts[idx[lo]] <= t - W) { sum -= ones[idx[lo]]; lo++; }
      out[i] = sum;
    }
    return out;
  }

  // rows: [{txn_id, ts (ms since epoch, naive time read as UTC), account, counterparty, amount, currency, amount_recv,
  //         currency_recv, category, channel, location, instrument, src_bank, dst_bank, errors}]
  // returns {order (row indices sorted by account, ts, txn_id), ints Int32Array[N*NI], floats Float32Array[N*NF], offsets}
  function tokenizeRows(rows) {
    const N = rows.length, ts = new Float64Array(N), amt = new Float64Array(N), la = new Float64Array(N), ones = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const r = rows[i]; ts[i] = r.ts;
      const c = normCcy(r.currency), rate = c !== null && USD_RATE[c] !== undefined ? USD_RATE[c] : 1.0;
      amt[i] = (r.amount === null || r.amount === undefined) ? NaN : r.amount * rate;
      la[i] = Math.log1p(Math.abs(amt[i])); ones[i] = Number.isNaN(amt[i]) ? 0 : 1;
    }
    const byKey = (k1, k2) => {
      const o = Array.from({ length: N }, (_, i) => i);
      o.sort((a, b) => cmp(rows[a][k1], rows[b][k1]) || (k2 ? cmp(rows[a][k2], rows[b][k2]) : 0) || (ts[a] - ts[b]) || cmp(rows[a].txn_id, rows[b].txn_id));
      return o;
    };
    function groupStats(order, key, pre) {
      const nPrev = new Float64Array(N), gap = new Float64Array(N), mean = new Float64Array(N), cnt = new Float64Array(N);
      let s = 0;
      while (s < N) {
        let e = s; const g = rows[order[s]][key];
        while (e < N && rows[order[e]][key] === g) e++;
        const idx = order.slice(s, e), roll = rolling24(idx, ts, ones);
        let cum = 0;
        for (let j = 0; j < idx.length; j++) {
          const i = idx[j];
          nPrev[i] = j; gap[i] = j === 0 ? NaN : Math.trunc((ts[i] - ts[idx[j - 1]]) / 1000);
          cum += la[i]; mean[i] = j === 0 ? NaN : (cum - la[i]) / j; cnt[i] = roll[j];
        }
        s = e;
      }
      return { nPrev, gap, mean, cnt };
    }
    const R = groupStats(byKey("counterparty"), "counterparty");
    const order = byKey("account");
    const S = groupStats(order, "account");
    const newPair = new Uint8Array(N);
    { const o = byKey("account", "counterparty");
      for (let j = 0; j < N; j++) {
        const i = o[j], p = j ? o[j - 1] : -1;
        newPair[i] = (p < 0 || rows[p].account !== rows[i].account || rows[p].counterparty !== rows[i].counterparty) ? 1 : 0;
      } }

    const ints = new Int32Array(N * NI), floats = new Float32Array(N * NF), offsets = [0];
    for (let k = 0; k < N; k++) {
      const i = order[k], r = rows[i];
      if (k > 0 && rows[order[k - 1]].account !== r.account) offsets.push(k);
      const d = new Date(r.ts), hour = d.getUTCHours(), wday = ((d.getUTCDay() + 6) % 7) + 1;
      const a = amt[i], lamt = Math.log1p(Math.abs(a));
      const ccy = normCcy(r.currency), ccyr = normCcy(r.currency_recv);
      const flags = (a < 0 ? 1 : 0)
        | ((r.account === r.counterparty ? 1 : 0) << 1)
        | ((ccyr !== null && ccy !== null && ccy !== ccyr ? 1 : 0) << 2)
        | ((r.src_bank != null && r.dst_bank != null && r.src_bank !== r.dst_bank ? 1 : 0) << 3)
        | (((Math.abs(a) % 1) === 0 ? 1 : 0) << 4)
        | (newPair[i] << 5);
      const o = k * NI;
      ints[o] = Number.isNaN(lamt) ? 1 : Math.min(Math.max(Math.floor(lamt * 4) + 1, 1), 63);
      ints[o + 1] = logb(S.gap[i], 2.0, 32);
      ints[o + 2] = hour + 1; ints[o + 3] = wday;
      ints[o + 4] = stableHash(r.category, 4096, "category");
      ints[o + 5] = stableHash(r.channel, 512, "channel");
      ints[o + 6] = stableHash(r.location, 2048, "location");
      ints[o + 7] = stableHash(ccy, 256, "currency");
      ints[o + 8] = stableHash(ccyr, 256, "currency");
      ints[o + 9] = stableHash(r.errors, 256, "errors");
      ints[o + 10] = stableHash(r.counterparty, 32768, "cp");
      ints[o + 11] = flags;
      ints[o + 12] = logb(R.nPrev[i], 2.0, 32);
      ints[o + 13] = logb(R.gap[i], 2.0, 32);
      const gap = S.gap[i], rm = R.mean[i], sm = S.mean[i];
      const fv = [lamt / 10, Math.log1p(Number.isNaN(gap) ? 0 : Math.max(gap, 0)) / 15,
        Math.log1p(R.nPrev[i]) / 10, Math.log1p(R.cnt[i]) / 7,
        (Number.isNaN(rm) ? 0 : rm) / 10, Number.isNaN(rm) ? 0 : 1,
        (Number.isNaN(lamt - sm) ? 0 : lamt - sm) / 5, Math.log1p(S.cnt[i]) / 7,
        Math.sin(2 * Math.PI * hour / 24), Math.cos(2 * Math.PI * hour / 24),
        Math.sin(2 * Math.PI * wday / 7), Math.cos(2 * Math.PI * wday / 7)];
      for (let f = 0; f < NF; f++) floats[k * NF + f] = Number.isFinite(fv[f]) ? fv[f] : 0;
    }
    offsets.push(N);
    return { order, ints, floats, offsets, NI, NF };
  }

  // windows over each account (same as TokenStore.eval_chunks): length <= L, stride L/2, score from L/2 after the first
  function windows(offsets, L) {
    const H = L >> 1, out = [];
    for (let g = 0; g + 1 < offsets.length; g++) {
      const a = offsets[g], n = offsets[g + 1] - a; let s = 0;
      for (;;) { const e = Math.min(n, s + L); out.push([a + s, e - s, s === 0 ? 0 : H]); if (e === n) break; s += H; }
    }
    return out;
  }

  // naive timestamps ("2024-01-31 13:45[:00[.123]]", "2024/01/31 13:45", ISO with T) are read as UTC wall-clock time,
  // like polars does for naive datetimes; returns ms or NaN
  function parseTs(s) {
    if (s === null || s === undefined) return NaN;
    const m = String(s).trim().match(/^(\d{4})[-\/](\d{1,2})[-\/](\d{1,2})(?:[ T](\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?)?/);
    if (!m) return NaN;
    const ms = m[7] ? Math.floor(Number((m[7] + "000000").slice(0, 6)) / 1000) : 0;
    return Date.UTC(+m[1], +m[2] - 1, +m[3], +(m[4] || 0), +(m[5] || 0), +(m[6] || 0), ms);
  }

  const api = { tokenizeRows, windows, stableHash, md5Bytes, parseTs, NI, NF };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.LedgerTok = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
