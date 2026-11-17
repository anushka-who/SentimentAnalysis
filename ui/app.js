const format = new Intl.NumberFormat('en-US');
const fixed = (value, digits = 4) => Number(value).toFixed(digits);
const el = (id) => document.getElementById(id);
const endpoints = {
  audit: '/outputs/metrics/dataset_audit.json',
  models: '/outputs/metrics/model_comparison.json',
  terms: '/outputs/metrics/top_terms.json',
  trend: '/outputs/metrics/daily_trend_summary.json',
  streams: [1, 2, 4].map((n) => `/outputs/streaming/local_${n}/stream_metrics.json`),
};

async function getJson(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json();
}

function fillOverview(audit, models) {
  const best = [...models.metrics].sort((a, b) => b.weighted_f1 - a.weighted_f1)[0];
  el('kpi-raw').textContent = format.format(audit.raw_rows);
  el('kpi-retained').textContent = format.format(audit.retained_rows);
  el('kpi-retained-note').textContent = `${format.format(audit.rows_removed_or_invalid)} removed or invalid`;
  el('kpi-holdout').textContent = format.format(models.holdout_rows);
  el('kpi-best-f1').textContent = fixed(best.weighted_f1, 4);
  el('kpi-best-method').textContent = `${best.method} · weighted F1`;
  el('quality-ratio').textContent = `${(audit.completeness_ratio * 100).toFixed(2)}%`;
  el('quality-fill').style.width = `${audit.completeness_ratio * 100}%`;
  el('split-train').textContent = format.format(audit.train_rows);
  el('split-holdout').textContent = format.format(audit.holdout_rows);
  const training = audit.train_label_counts_target;
  const holding = audit.holdout_label_counts_target;
  el('class-balance').innerHTML = `
    <div class="class-pill"><span>Train · negative</span><strong>${format.format(training['0'])}</strong></div>
    <div class="class-pill"><span>Train · positive</span><strong>${format.format(training['4'])}</strong></div>
    <div class="class-pill"><span>Holdout · negative</span><strong>${format.format(holding['0'])}</strong></div>
    <div class="class-pill"><span>Holdout · positive</span><strong>${format.format(holding['4'])}</strong></div>`;
}

function fillModels(data) {
  const names = {
    'Logistic Regression': 'logistic-regression',
    'Naive Bayes': 'naive-bayes',
    VADER: 'vader',
  };
  el('model-rows').innerHTML = data.metrics.map((item) => `
    <tr><td>${item.method}</td><td>${fixed(item.accuracy, 4)}</td><td><strong>${fixed(item.weighted_f1, 4)}</strong></td><td>${item.fit_seconds == null ? 'No fit step' : `${fixed(item.fit_seconds, 2)} s`}</td></tr>`).join('');
  el('metric-bars').innerHTML = data.metrics.map((item) => `
    <div class="bar-row"><span class="bar-name">${names[item.method] || item.method}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, item.weighted_f1 * 100))}%"></div></div><strong class="bar-value">${fixed(item.weighted_f1, 4)}</strong></div>`).join('');
}

function fillStreams(runs) {
  const base = runs.find((r) => r.cores === 1)?.stream_elapsed_seconds;
  el('stream-rows').innerHTML = runs.map((run) => {
    const speedup = base ? base / run.stream_elapsed_seconds : null;
    const isPass = run.window_reconciliation_passed;
    return `<tr><td>local[${run.cores}]</td><td>${format.format(run.records_scored)}</td><td>${fixed(run.stream_elapsed_seconds, 2)} s</td><td>${fixed(run.throughput_records_per_second, 0)} /s</td><td>${format.format(run.batch_duration_median_ms)} ms</td><td>${format.format(run.batch_duration_p95_ms)} ms</td><td>${fixed(speedup, 3)}×</td><td>${isPass ? 'Pass' : 'Fail'}</td></tr>`;
  }).join('');
  const allPass = runs.length === 3 && runs.every((r) => r.window_reconciliation_passed);
  el('reconciliation-status').textContent = allPass ? 'All counts reconcile' : 'Review reconciliation';
  const fastest = [...runs].sort((a, b) => a.stream_elapsed_seconds - b.stream_elapsed_seconds)[0];
  el('stream-kpis').innerHTML = `
    <article class="stream-kpi"><span>Fastest measured setting</span><strong>local[${fastest.cores}]</strong><small>${fixed(fastest.stream_elapsed_seconds, 2)} seconds · ${fixed(fastest.throughput_records_per_second, 0)} records/s</small></article>
    <article class="stream-kpi"><span>Rows scored per run</span><strong>${format.format(fastest.records_scored)}</strong><small>Same shared holdout across all core settings</small></article>
    <article class="stream-kpi"><span>Replay workload</span><strong>${runFileCount(runs)} files</strong><small>Four files per trigger · finite simulated replay</small></article>`;
}

function runFileCount(runs) {
  return runs[0]?.source_files ?? '—';
}

function fillTrend(data) {
  el('trend-window-count').textContent = format.format(data.one_day_utc_windows);
  const neg = data.peak_daily_counts.predicted_negative;
  const pos = data.peak_daily_counts.predicted_positive;
  el('peak-negative').textContent = `${format.format(neg.count)} · ${neg.day_utc.slice(0, 10)}`;
  el('peak-positive').textContent = `${format.format(pos.count)} · ${pos.day_utc.slice(0, 10)}`;
}

function fillTerms(data) {
  const groups = [
    { label: 0, title: 'Predicted negative', className: 'negative' },
    { label: 1, title: 'Predicted positive', className: 'positive' },
  ];
  el('term-lists').innerHTML = groups.map((group) => {
    const items = data.terms.filter((t) => Number(t.prediction) === group.label).sort((a, b) => a.rank - b.rank);
    return `<article class="term-list ${group.className}"><h3>${group.title}</h3><ol>${items.slice(0, 8).map((item) => `<li>${item.word}<span class="term-count">${format.format(item.count)}</span></li>`).join('')}</ol></article>`;
  }).join('');
}

function fillDetails(audit) {
  const details = [
    ['Source', 'Sentiment140 · headerless latin-1 CSV'],
    ['Source size', `${format.format(audit.source_bytes)} bytes`],
    ['Split seed', String(audit.split_seed)],
    ['Replay files', String(audit.stream_files_requested)],
    ['Event time', 'PDT −07:00 / PST −08:00, normalized to UTC'],
    ['Spark', '3.5.8 · local mode in Apache Docker image'],
    ['Runtime', 'Java 17.0.17 · Python 3.10.12 in container'],
    ['Host', 'Core i7-1355U · 12 logical processors · 15.69 GiB RAM'],
    ['Saved identifiers', 'No user names or tweet IDs in derived files'],
  ];
  el('run-details').innerHTML = details.map(([term, value]) => `<dt>${term}</dt><dd>${value}</dd>`).join('');
}

async function main() {
  try {
    const [audit, models, terms, trend, ...runs] = await Promise.all([
      getJson(endpoints.audit), getJson(endpoints.models), getJson(endpoints.terms), getJson(endpoints.trend),
      ...endpoints.streams.map(getJson),
    ]);
    fillOverview(audit, models);
    fillModels(models);
    fillStreams(runs);
    fillTrend(trend);
    fillTerms(terms);
    fillDetails(audit);
    el('data-status').textContent = 'Measured run data loaded';
  } catch (error) {
    console.error(error);
    el('data-status').textContent = 'Could not load run data';
    document.querySelector('main').insertAdjacentHTML('afterbegin', `<div class="load-error" role="alert"><strong>Run data could not be loaded.</strong> Start the local dashboard server from the project root and reload this page.</div>`);
  }
}

main();
