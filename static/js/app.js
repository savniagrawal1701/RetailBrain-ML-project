/* RetailBrain front-end. No framework, no chart library: forms, fetch, and hand-drawn SVG. */
(function () {
  'use strict';

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (v) => String(v).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const int = (n) => Math.round(n).toLocaleString('en-US');
  const pct = (x, d = 0) => (x * 100).toFixed(d) + '%';
  const nice = (s) => String(s).replace(/_/g, ' ');
  const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
  const reduced = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  let META = null;

  async function api(path, body) {
    const opts = body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined;
    let res;
    try { res = await fetch(path, opts); } catch (e) { throw new Error('Could not reach the server. Check your connection and try again.'); }
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(j.error || 'Something went wrong. Try again.');
    return j;
  }
  function showErr(id, msg) { const el = $(id); if (!el) return; el.textContent = msg || ''; el.hidden = !msg; }
  async function busy(btn, fn) {
    const label = btn.textContent; btn.disabled = true; btn.textContent = 'Working…';
    try { await fn(); } finally { btn.disabled = false; btn.textContent = label; }
  }

  /* ---------------------------------------------------------------- theme */
  const root = document.documentElement;
  const themeBtn = $('#themeBtn');
  const syncTheme = () => themeBtn.setAttribute('aria-pressed', root.dataset.theme === 'dark');
  themeBtn.addEventListener('click', () => {
    const t = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = t;
    try { localStorage.setItem('rb-theme', t); } catch (e) { /* private mode */ }
    syncTheme();
  });
  syncTheme();

  /* ---------------------------------------------------------------- tiny SVG chart kit */
  const niceMax = (m) => { if (m <= 0) return 1; const p = Math.pow(10, Math.floor(Math.log10(m))) / 2; return Math.ceil(m / p) * p; };

  function lineChart(el, o) {
    const W = 640, H = 290, L = 52, R = 14, T = 14, B = 30;
    const all = o.series.flatMap((s) => s.y.filter((v) => v != null));
    const lo = Math.min(...all, 0), hi = Math.max(...all, 1e-9);
    const step = niceMax((hi - lo) / 4), ymax = Math.ceil(hi / step) * step, ymin = Math.floor(lo / step) * step;
    const X = (i) => L + ((W - L - R) * i) / (o.n - 1);
    const Y = (v) => T + (H - T - B) * (1 - (v - ymin) / (ymax - ymin));
    let g = '';
    for (let k = 0; k <= Math.round((ymax - ymin) / step); k++) {
      const v = ymin + step * k, y = Y(v);
      g += `<line class="gridline" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text class="axis-txt" x="${L - 8}" y="${y + 4}" text-anchor="end">${o.yFmt ? o.yFmt(v) : int(v)}</text>`;
    }
    if (o.band) g += `<rect class="band" x="${X(o.band.from)}" y="${T}" width="${X(o.band.to) - X(o.band.from)}" height="${H - T - B}"/><text class="band-txt" x="${X(o.band.from) + 6}" y="${T + 14}">${esc(o.band.label)}</text>`;
    o.xTicks.forEach((t) => { g += `<text class="axis-txt" x="${X(t.i)}" y="${H - 9}" text-anchor="middle">${esc(t.label)}</text>`; });
    o.series.forEach((s) => {
      let d = '', pen = false;
      s.y.forEach((v, i) => { if (v == null) { pen = false; return; } d += (pen ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1); pen = true; });
      g += `<path class="ln" d="${d}" style="stroke:var(${s.color})" stroke-width="${s.w || 2}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''}/>`;
    });
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" aria-hidden="true">${g}</svg>`;
  }
  const legend = (el, items) => {
    el.innerHTML = items.map((s) => `<span><svg viewBox="0 0 26 8" aria-hidden="true"><line x1="0" x2="26" y1="4" y2="4" style="stroke:var(${s.color})" stroke-width="${Math.min(s.w || 2, 3.2)}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} stroke-linecap="round"/></svg>${esc(s.label)}</span>`).join('');
  };

  /* ---------------------------------------------------------------- 1-2. order oracle */
  const PRESETS = [
    { from: 'SP', to: 'AM', cat: 'toys', weight: 2, price: 90, items: 1, month: 12 },
    { from: 'SP', to: 'SP', cat: 'health_beauty', weight: 0.5, price: 35, items: 1, month: 5 },
    { from: 'PR', to: 'CE', cat: 'furniture_decor', weight: 18, price: 420, items: 1, month: 7 },
  ];
  const STAMP = { low: 'On time', medium: 'Watch it', high: 'Likely late' };
  let raf = 0;

  function placeParcel(t) {
    const path = $('#routePath'), p = path.getPointAtLength(path.getTotalLength() * t);
    $('#parcel').setAttribute('transform', `translate(${p.x} ${p.y})`);
  }
  function flyParcel() {
    cancelAnimationFrame(raf);
    if (reduced) return placeParcel(1);
    const t0 = performance.now(), D = 1500;
    const step = (now) => { const k = Math.min(1, (now - t0) / D); placeParcel(1 - Math.pow(1 - k, 3)); if (k < 1) raf = requestAnimationFrame(step); };
    placeParcel(0); raf = requestAnimationFrame(step);
  }
  const whyList = (el, items, cls, emptyText) => {
    el.className = 'why-list ' + cls;
    el.innerHTML = items.length ? items.map((i) => `<li>${esc(i.feature)}</li>`).join('') : `<li class="empty">${emptyText}</li>`;
  };

  function readOrder() {
    const v = (id) => $(id).value;
    const body = { seller_state: v('#f-from'), customer_state: v('#f-to'), category: v('#f-cat'), weight_kg: v('#f-weight'),
      price: v('#f-price'), n_items: v('#f-items'), month: v('#f-month') };
    if (v('#f-promised') !== '') body.promised_days = v('#f-promised');
    return body;
  }
  async function runOrder() {
    showErr('#orderErr');
    const body = readOrder();
    try {
      const j = await api('/api/order', body);
      $('#rFrom').textContent = body.seller_state; $('#rTo').textContent = body.customer_state;
      $('#rDist').textContent = int(j.distance_km) + ' km apart';
      $('#vDays').textContent = j.predicted_days + (j.predicted_days === 1 ? ' day' : ' days');
      $('#vRange').textContent = `most likely ${j.range[0]} to ${j.range[1]} days`;
      $('#vPromised').textContent = j.promised_days + ' days';
      const gap = j.predicted_days - j.promised_days;
      $('#vGap').textContent = gap > 0 ? `${gap} day${gap > 1 ? 's' : ''} over the promise` : gap === 0 ? 'right on the promise' : `${-gap} day${gap < -1 ? 's' : ''} to spare`;
      $('#vRisk').textContent = pct(j.late_probability);
      const x = j.late_probability / j.base_rate;
      $('#vBase').textContent = `${x.toFixed(1)}× the usual ${pct(j.base_rate)}`;
      const st = $('#stamp'); st.dataset.level = j.verdict; $('#stampTxt').textContent = STAMP[j.verdict];
      st.classList.remove('pop'); void st.offsetWidth; st.classList.add('pop');
      whyList($('#whyUp'), j.pushes_up, 'up', 'Nothing stands out');
      whyList($('#whyDown'), j.pushes_down, 'down', 'Nothing stands out');
      flyParcel();
    } catch (e) { showErr('#orderErr', e.message); }
  }
  function applyPreset(i) {
    const p = PRESETS[i];
    $('#f-from').value = p.from; $('#f-to').value = p.to; $('#f-cat').value = p.cat; $('#f-weight').value = p.weight;
    $('#f-price').value = p.price; $('#f-items').value = p.items; $('#f-month').value = p.month; $('#f-promised').value = '';
    runOrder();
  }

  /* ---------------------------------------------------------------- 5. reviews */
  async function runReview() {
    showErr('#reviewErr');
    try {
      const j = await api('/api/review', { text: $('#reviewText').value });
      $('#sentPin').style.left = (j.positive * 100).toFixed(1) + '%';
      const verdict = { positive: 'Happy customer', negative: 'Unhappy customer', unsure: 'Not enough to go on' }[j.label];
      $('#sentVerdict').textContent = verdict;
      $('#sentSub').textContent = j.label === 'unsure'
        ? 'None of these words carry a clear signal. Try a full sentence about delivery or quality.'
        : `${pct(j.label === 'positive' ? j.positive : 1 - j.positive)} confident · about ${j.stars.toFixed(1)} stars`;
      $('#sentWords').innerHTML = j.words.map((w) => `<span class="word ${w.weight > 0 ? 'pos' : 'neg'}" title="${w.weight > 0 ? 'pulls towards delighted' : 'pulls towards unhappy'}">${esc(w.word)}</span>`).join('');
    } catch (e) { showErr('#reviewErr', e.message); }
  }

  /* ---------------------------------------------------------------- 4,6,7,8. customers */
  const recCard = (p, extra) => `<div class="rec"><b>${esc(nice(p.category))}</b>${p.price != null ? `<span>R$ ${p.price.toFixed(0)} · ★ ${p.rating.toFixed(1)}</span>` : ''}<span class="tag">${esc(p.product_id)}${extra ? ' · ' + extra : ''}</span></div>`;

  function renderCustomer(j) {
    const si = Math.max(0, META.segments.indexOf(j.segment));
    const coupon = j.discount === 0 ? 'no coupon' : `a ${j.discount}% coupon`;
    $('#custCard').innerHTML = `
      <div class="cust-top"><span class="cust-id">${esc(j.id)}</span><span class="badge" data-seg="${si}">${esc(j.segment)}</span><span class="muted">${esc(j.state)}</span></div>
      <div class="kv"><div><b>R$ ${int(j.spend)}</b><span>Total spend</span></div><div><b>${j.orders}</b><span>${j.orders === 1 ? 'Order' : 'Orders'}</span></div><div><b>${int(j.recency)} d</b><span>Since last order</span></div></div>
      <div>
        <div class="prob-line"><span>Chance of a second purchase</span><span>${pct(j.repeat_probability)}</span></div>
        <div class="bar"><i style="width:${Math.max(2, j.repeat_probability * 100)}%"></i></div>
        <p class="muted" style="font-size:.84rem;margin-top:6px">Judged from their first order. The average customer: ${pct(j.base_rate)}</p>
        <ul class="reasons">${j.reasons.map((r) => `<li><span class="${r.effect > 0 ? 'up' : 'down'}">${r.effect > 0 ? '▲' : '▼'}</span>${esc(nice(r.feature))} ${r.effect > 0 ? 'raises' : 'lowers'} the odds</li>`).join('')}</ul>
      </div>
      <div class="callout"><b>Best offer: ${coupon}.</b> ${esc(j.action)}</div>`;
    $('#recStrategy').textContent = `Based on ${j.history_len} browsed or bought item${j.history_len === 1 ? '' : 's'}: ${j.strategy}.`;
    $('#recs').innerHTML = j.recommendations.map((p) => recCard(p)).join('');
    $('#custId').value = j.id;
  }
  async function lookup(path) {
    showErr('#custErr');
    try { renderCustomer(await api(path)); } catch (e) { showErr('#custErr', e.message); }
  }
  async function loadSimilar() {
    const pid = $('#simProd').value; if (!pid) return;
    try {
      const j = await api('/api/similar/' + encodeURIComponent(pid));
      $('#simList').innerHTML = j.neighbors.map((p) => recCard(p, Math.round(p.match * 100) + '% match')).join('');
    } catch (e) { $('#simList').innerHTML = `<p class="form-err">${esc(e.message)}</p>`; }
  }
  async function loadProducts() {
    const list = await api('/api/products?category=' + encodeURIComponent($('#simCat').value));
    $('#simProd').innerHTML = list.map((p) => `<option value="${esc(p.product_id)}">${esc(p.product_id)} · R$ ${p.price.toFixed(0)} · ★ ${p.rating.toFixed(1)}</option>`).join('');
    loadSimilar();
  }

  /* ---------------------------------------------------------------- 9. pricing agent */
  const STRAT = {
    thompson: { label: 'Learning agent', color: '--c-thompson', w: 3.4 },
    oracle: { label: 'Perfect knowledge', color: '--c-oracle', dash: '7 5', w: 2 },
    epsilon_greedy: { label: 'Epsilon-greedy', color: '--c-eg', w: 2 },
    flat_10: { label: 'Everyone 10% off', color: '--c-flat10', w: 2 },
    flat_20: { label: 'Everyone 20% off', color: '--c-flat20', w: 2 },
    no_discount: { label: 'No coupons', color: '--c-none', w: 2 },
    random: { label: 'Random coupons', color: '--c-random', dash: '2 5', w: 2 },
  };
  function renderSim(j) {
    const keys = Object.keys(STRAT);
    const k = (v) => { const a = Math.abs(v), t = a >= 1000 ? (a / 1000).toFixed(a % 1000 ? 1 : 0) + 'k' : String(Math.round(a)); return (v < 0 ? '-' : '') + t; };
    const base = j.curves.no_discount;                     // plot extra profit versus doing nothing
    lineChart($('#simChart'), {
      n: j.x.length, series: keys.map((s) => ({ y: j.curves[s].map((v, i) => v - base[i]), color: STRAT[s].color, w: STRAT[s].w, dash: STRAT[s].dash })),
      xTicks: [0, 0.25, 0.5, 0.75, 1].map((f) => { const i = Math.round(f * (j.x.length - 1)); return { i, label: k(j.x[i]) }; }),
      yFmt: (v) => (v < 0 ? '-R$' : 'R$') + k(Math.abs(v)),
    });
    legend($('#simLegend'), keys.map((s) => STRAT[s]));
    const kpi = (v, text) => `<div class="kpi"><b class="${v < 0 ? 'neg' : ''}">${v > 0 ? '+' : ''}${v.toFixed(0)}%</b><span>${text}</span></div>`;
    $('#simKpis').innerHTML = kpi(j.uplift_vs_none_pct, 'profit versus giving no coupons') + kpi(j.uplift_vs_flat_pct, 'versus 10% off for everyone') + kpi(j.uplift_vs_flat20_pct, 'versus 20% off for everyone') +
      `<div class="kpi"><b>${j.regret_pct.toFixed(0)}%</b><span>short of an agent that knew every answer in advance</span></div>`;
    const rows = Object.entries(j.learned).map(([s, v]) => `<tr><td>${esc(s)}</td><td class="n">${pct(v.share)}</td><td class="n ${v.learned_discount === v.true_best ? 'best' : ''}">${v.learned_discount}%</td><td class="n">${v.true_best}%</td></tr>`).join('');
    $('#simLearned').innerHTML = `<thead><tr><th>Customer type</th><th class="n">Share</th><th class="n">Agent chose</th><th class="n">Best possible</th></tr></thead><tbody>${rows}</tbody>`;
  }
  async function runSim() {
    showErr('#simErr');
    try { renderSim(await api('/api/pricing', { rounds: +$('#rounds').value, seed: Math.floor(Math.random() * 100000) })); }
    catch (e) { showErr('#simErr', e.message); }
  }

  /* ---------------------------------------------------------------- 10. forecast */
  async function runForecast() {
    try {
      const j = await api('/api/forecast/' + encodeURIComponent($('#fcCat').value));
      const nw = j.weeks.length, n = nw + j.future_weeks.length, labels = j.weeks.concat(j.future_weeks);
      const pad = (arr, start) => Array.from({ length: n }, (_, i) => (i >= start && i - start < arr.length ? arr[i - start] : null));
      const s = j.series, ts = j.test_start;
      const mk = (name, test, fut, st) => ({ y: pad(test.concat(fut), ts), color: st.color, w: st.w, dash: st.dash });
      const defs = [
        { y: pad(s.history, 0), color: '--c-actual', w: 2.4, label: 'What actually sold' },
        { ...mk('x', s.test['XGBoost'], s.future['XGBoost'], { color: '--c-thompson', w: 2.6 }), label: 'XGBoost forecast' },
        { ...mk('m', s.test['Neural Net (MLP)'], [], { color: '--c-flat10', w: 2, dash: '7 5' }), label: 'Neural network (held-out weeks only)' },
        { y: pad(s.test['Naive (last week)'], ts), color: '--c-random', w: 1.8, dash: '2 5', label: '"Same as last week"' },
      ];
      const ticks = []; for (let i = 0; i < n; i += 13) ticks.push({ i, label: labels[i].slice(0, 7) });
      lineChart($('#fcChart'), { n, series: defs, xTicks: ticks, band: { from: ts, to: n - 1, label: 'held-out + forecast' }, yFmt: (v) => int(v) });
      legend($('#fcLegend'), defs);
      $('#fcMetrics').innerHTML = j.table.map((r) => `<span class="fc-pill">${esc(r.model)}: off by <b>${pct(r.wape)}</b></span>`).join('');
    } catch (e) { $('#fcChart').innerHTML = `<p class="form-err">${esc(e.message)}</p>`; }
  }

  /* ---------------------------------------------------------------- model lab */
  function tbl(rows, cols) {
    const best = {};
    cols.forEach((c) => { if (c.dir) { const v = rows.map((r) => r[c.k]).filter((x) => x != null); best[c.k] = c.dir === 'max' ? Math.max(...v) : Math.min(...v); } });
    const head = cols.map((c) => `<th class="${c.num ? 'n' : ''}">${esc(c.label)}</th>`).join('');
    const body = rows.map((r) => `<tr>${cols.map((c) => { const v = r[c.k]; const txt = v == null ? '' : c.fmt ? c.fmt(v) : esc(v); return `<td class="${(c.num ? 'n ' : '') + (c.dir && v === best[c.k] ? 'best' : '')}">${txt}</td>`; }).join('')}</tr>`).join('');
    return `<div class="scroll"><table class="table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  }
  const bars = (items, key, label, signed) => {
    const mx = Math.max(...items.map((i) => Math.abs(i[key])));
    return `<div class="imp">${items.map((i) => `<div class="imp-row"><span><em style="font-style:normal">${esc(i.feature)}</em><b>${signed && i[key] > 0 ? '+' : ''}${i[key]}</b></span><div class="bar thin"><i style="width:${(Math.abs(i[key]) / mx) * 100}%;${signed && i[key] < 0 ? 'background:var(--warm)' : ''}"></i></div></div>`).join('')}</div>`;
  };
  const P = (v) => pct(v, 1);

  function scatter(points) {
    const W = 520, H = 320, xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const X = (v) => 16 + ((W - 32) * (v - x0)) / (x1 - x0), Y = (v) => H - 16 - ((H - 32) * (v - y0)) / (y1 - y0);
    const dots = points.map((p) => `<circle cx="${X(p.x).toFixed(1)}" cy="${Y(p.y).toFixed(1)}" r="3.2" style="fill:var(--seg${Math.max(0, META.segments.indexOf(p.s))})" opacity=".72"/>`).join('');
    return `<div class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Customers plotted by recency, frequency and spend, coloured by segment"><rect x="0" y="0" width="${W}" height="${H}" fill="none" class="gridline" rx="10"/>${dots}</svg></div>`;
  }

  const LAB = {
    delivery: ['Delivery time', (A) => {
      const m = A.m1_delivery_time;
      return `<p class="note" style="margin:0 0 14px">Predicting days from purchase to doorstep. The "mean baseline" always guesses the average, so every real model should beat it.</p>
        <div class="lab-grid"><div>${tbl(m.table, [{ k: 'model', label: 'Model' }, { k: 'mae', label: 'Average miss (days)', num: 1, dir: 'min' }, { k: 'rmse', label: 'RMSE', num: 1, dir: 'min' }, { k: 'r2', label: 'R²', num: 1, dir: 'max' }, { k: 'note', label: 'Notes' }])}</div>
        <div><h3 class="lab-sub">Biggest drivers (${esc(m.best)} weights, in days)</h3>${bars(m.top_coefs, 'coef', '', true)}</div></div>`;
    }],
    late: ['Late orders', (A) => {
      const m = A['m2_3_late'];
      return `<p class="note" style="margin:0 0 14px">Only ${P(m.base_rate)} of orders arrive late, so accuracy would be misleading. These scores ask a practical question: flag the riskiest 10% of orders, how many late ones do you catch?</p>
        ${tbl(m.table, [{ k: 'model', label: 'Model' }, { k: 'roc_auc', label: 'ROC-AUC', num: 1, dir: 'max' }, { k: 'pr_auc', label: 'PR-AUC', num: 1, dir: 'max' }, { k: 'recall_top10', label: 'Late orders caught', num: 1, dir: 'max', fmt: P }, { k: 'precision_top10', label: 'Flags that were right', num: 1, dir: 'max', fmt: P }])}
        <h3 class="lab-sub" style="margin-top:22px">Decision tree: the rules behind late orders</h3>
        ${m.tree_rules.map((r) => `<div class="rule"><b>${P(r.late_rate)} late</b> across ${int(r.orders)} orders when: ${r.conditions.map(esc).join(', and ')}</div>`).join('')}
        <details><summary>See the whole tree</summary><pre>${esc(m.tree_text)}</pre></details>`;
    }],
    sentiment: ['Sentiment', (A) => {
      const m = A.m5_sentiment;
      return `<p class="note" style="margin:0 0 14px">${int(m.n_reviews)} written reviews, labelled happy (4 to 5 stars) or unhappy (1 to 2 stars). Review text here is generated from phrase templates with noise, so the real-world score would be lower.</p>
        ${tbl(m.table, [{ k: 'model', label: 'Model' }, { k: 'accuracy', label: 'Accuracy', num: 1, dir: 'max', fmt: P }, { k: 'f1', label: 'F1', num: 1, dir: 'max' }, { k: 'roc_auc', label: 'ROC-AUC', num: 1, dir: 'max' }])}
        <div class="lab-grid" style="margin-top:20px"><div><h3 class="lab-sub">Words Naive Bayes links to happy customers</h3><div class="words">${m.top_positive.map((w) => `<span class="word pos">${esc(w)}</span>`).join('')}</div></div>
        <div><h3 class="lab-sub">…and to unhappy ones</h3><div class="words">${m.top_negative.map((w) => `<span class="word neg">${esc(w)}</span>`).join('')}</div></div></div>`;
    }],
    repeat: ['Repeat buyers', (A) => {
      const m = A.m4_repeat;
      return `<p class="note" style="margin:0 0 14px">Which first-time customers will buy again? ${P(m.base_rate)} do, so the table asks: contact the 20% most likely, how many returners do you reach, and how much better is that than picking at random (lift)?</p>
        <div class="lab-grid"><div>${tbl(m.table, [{ k: 'model', label: 'Model' }, { k: 'roc_auc', label: 'ROC-AUC', num: 1, dir: 'max' }, { k: 'pr_auc', label: 'PR-AUC', num: 1, dir: 'max' }, { k: 'recall_top20', label: 'Returners reached', num: 1, dir: 'max', fmt: P }, { k: 'lift_top20', label: 'Lift', num: 1, dir: 'max', fmt: (v) => v.toFixed(2) + '×' }])}</div>
        <div><h3 class="lab-sub">What the tuned XGBoost relies on (SHAP)</h3>${bars(m.importance, 'shap', '')}</div></div>`;
    }],
    reco: ['Recommender', (A) => {
      const m = A.m8_recommender, k = A.m7_knn, w = m.weights;
      return `<p class="note" style="margin:0 0 14px">Hide one item each customer touched, then check whether it appears in the top 10 suggestions (${int(m.eval_users)} test customers; ${m.cold_users} of them are "cold": we know only one other item about them).</p>
        ${tbl(m.table, [{ k: 'model', label: 'Method' }, { k: 'hit10', label: 'Hit rate @10', num: 1, dir: 'max', fmt: P }, { k: 'cold_hit10', label: 'Hit rate, cold customers', num: 1, dir: 'max', fmt: P }])}
        <p class="note">The hybrid blends three signals. Blend weights (collaborative / content / popularity) were tuned on a separate group of customers: warm ${w.warm.join(' / ')}, cold ${w.cold.join(' / ')}. KNN look-alikes stay in the same category ${P(k.category_purity_at5)} of the time across ${int(k.n_products)} products.</p>`;
    }],
    forecast: ['Forecast', (A) => `<p class="note" style="margin:0 0 14px">Next-week demand for every category, scored on the last 12 weeks, which the models never saw. "Off by" is total absolute error divided by total units sold. On a dataset this small the neural network struggles, and XGBoost is the safer pick.</p>
        ${tbl(A.m10_forecast.table, [{ k: 'model', label: 'Model' }, { k: 'mae', label: 'Average miss (units/week)', num: 1, dir: 'min' }, { k: 'wape', label: 'Off by', num: 1, dir: 'min', fmt: P }])}`],
    segments: ['Segments', (A) => {
      const m = A.m6_segments, sil = Object.entries(m.silhouette).map(([k, v]) => `k=${k}: ${v}`).join(' · ');
      const rows = m.profiles.map((p) => ({ ...p, dot: `<span class="dot-seg" style="background:var(--seg${META.segments.indexOf(p.segment)})"></span>${esc(p.segment)}` }));
      return `<div class="lab-grid"><div>${tbl(rows, [{ k: 'dot', label: 'Segment', fmt: (v) => v }, { k: 'customers', label: 'Customers', num: 1, fmt: int }, { k: 'orders', label: 'Orders each', num: 1 }, { k: 'spend', label: 'Spend (R$)', num: 1, fmt: int }, { k: 'recency_days', label: 'Days since last order', num: 1, fmt: int }, { k: 'share', label: 'Share of customers', num: 1, fmt: P }])}
        <p class="note">K-Means on log recency, frequency and spend, with 4 clusters chosen for business meaning. Silhouette scores for other k: ${sil}.</p></div>
        <div>${scatter(m.scatter)}<p class="chart-cap">1,500 customers flattened to two dimensions with PCA (${pct(m.pca_var[0] + m.pca_var[1])} of the variation kept).</p></div></div>`;
    }],
  };

  function buildLab(A) {
    const tabs = $('#labTabs'), body = $('#labBody'), keys = Object.keys(LAB);
    tabs.innerHTML = keys.map((k, i) => `<button type="button" role="tab" class="tab" id="tab-${k}" data-k="${k}" aria-selected="${i === 0}" aria-controls="labBody">${LAB[k][0]}</button>`).join('');
    const show = (k) => { $$('.tab', tabs).forEach((t) => t.setAttribute('aria-selected', t.dataset.k === k)); body.setAttribute('aria-labelledby', 'tab-' + k); body.innerHTML = LAB[k][1](A); };
    tabs.addEventListener('click', (e) => { const b = e.target.closest('.tab'); if (b) show(b.dataset.k); });
    show(keys[0]);
  }

  /* ---------------------------------------------------------------- boot */
  async function boot() {
    try {
      META = await api('/api/meta');
    } catch (e) { showErr('#orderErr', e.message); return; }
    const opt = (arr, fmt) => arr.map((v) => `<option value="${esc(v)}">${esc(fmt ? fmt(v) : v)}</option>`).join('');
    $('#f-from').innerHTML = opt(META.states); $('#f-to').innerHTML = opt(META.states);
    $('#f-cat').innerHTML = opt(META.categories, nice);
    $('#f-month').innerHTML = MONTHS.map((m, i) => `<option value="${i + 1}">${m}</option>`).join('');
    $('#simCat').innerHTML = opt(META.categories, nice); $('#fcCat').innerHTML = opt(META.categories, nice);
    const d = META.dataset;
    $('#footStats').textContent = `${int(d.orders)} simulated orders · ${int(d.customers)} customers · ${int(d.products)} products · ${d.start.slice(0, 4)} to ${d.end.slice(0, 4)}`;

    $('#orderForm').addEventListener('submit', (e) => { e.preventDefault(); busy($('#btnOrder'), runOrder); });
    $$('[data-preset]').forEach((b) => b.addEventListener('click', () => applyPreset(+b.dataset.preset)));
    $$('[data-review]').forEach((b) => b.addEventListener('click', () => { $('#reviewText').value = b.dataset.review; runReview(); }));
    $('#btnReview').addEventListener('click', () => busy($('#btnReview'), runReview));
    $('#custForm').addEventListener('submit', (e) => { e.preventDefault(); lookup('/api/customer/' + encodeURIComponent($('#custId').value.trim() || 'C00042')); });
    $('#btnRandom').addEventListener('click', () => lookup('/api/customers/random'));
    $('#simCat').addEventListener('change', loadProducts); $('#simProd').addEventListener('change', loadSimilar);
    $('#rounds').addEventListener('input', (e) => { $('#roundsOut').textContent = int(+e.target.value); });
    $('#btnSim').addEventListener('click', () => busy($('#btnSim'), runSim));
    $('#fcCat').addEventListener('change', runForecast);

    applyPreset(0);
    $('#reviewText').value = 'the product never arrived on time and the box was damaged'; runReview();
    lookup('/api/customers/random'); loadProducts(); runSim(); runForecast();
    try { buildLab(await api('/api/lab')); } catch (e) { $('#labBody').innerHTML = `<p class="form-err">${esc(e.message)}</p>`; }
  }
  boot();
})();
