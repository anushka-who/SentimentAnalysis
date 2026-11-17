from __future__ import annotations

import json
from pathlib import Path

from common import METRICS, STREAMING, ROOT


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fmt(value, digits=4):
    return "Not measured" if value is None else f"{value:.{digits}f}"


def main() -> None:
    audit = read_json(METRICS / "dataset_audit.json")
    comparison = read_json(METRICS / "model_comparison.json")
    top_terms = read_json(METRICS / "top_terms.json")
    trend = read_json(METRICS / "daily_trend_summary.json")
    streams = {n: read_json(STREAMING / f"local_{n}" / "stream_metrics.json") for n in (1, 2, 4)}
    lines = [
        "# Real-Time Social Media Sentiment Analytics on Tweet Streams",
        "",
        "**Big Data Analytics (CSE557) | Ramaiah Institute of Technology | Submission deadline: 19 November 2026**  ",
        "**Team:** Anish Krishna (1MS24CS029), Anushka Sharad (1MS24CS035)  ",
        "**Faculty guide:** Not supplied",
        "",
        "## Executive summary",
    ]
    if audit:
        lines += [
            f"The project processed {audit['raw_rows']:,} source records, retained {audit['retained_rows']:,} after cleaning and deduplication ({audit['completeness_ratio']:.4%}), and used a seeded 80/20 split. The 20% holdout was evaluated with Logistic Regression, Naive Bayes, and VADER. Historical JSON files were replayed through Structured Streaming as a **simulated stream**; no live Twitter/X feed was used.",
        ]
    else:
        lines += ["Dataset preparation has not completed; data-derived results are not available."]
    if comparison:
        best = max(comparison["metrics"], key=lambda m: m["weighted_f1"])
        lines.append(f"On the shared holdout, {best['method']} had the highest measured weighted F1 ({best['weighted_f1']:.4f}).")
    else:
        lines.append("Model evaluation has not completed; no classifier metric is reported.")
    lines += ["", "## Scenario and questions", ""]
    lines += [
        "Social-media teams need to summarize sentiment in large volumes of short, noisy posts. This study asks how predicted positive and negative counts vary by event-time day, which cleaned words are frequent by predicted class, and how Logistic Regression, Naive Bayes, and VADER compare on one holdout set. The historical Sentiment140 file replay represents simulated streaming, not live ingestion.",
        "",
        "## Research reference and relation to this project",
        "",
        "Ismail, A., Sazali, F. H., Agos Jawaddi, S. N., & Mutalib, S. (2025). *Stream ETL framework for twitter-based sentiment analysis: Leveraging big data technologies*. Expert Systems with Applications, 261, 125523. DOI: [10.1016/j.eswa.2024.125523](https://doi.org/10.1016/j.eswa.2024.125523). The publisher abstract describes a framework combining Kafka, Spark, Cassandra, HBase, Hive, and HDFS for stream ETL, sentiment analysis, bias detection/correction, and geospatial visualization. It reports use cases and technology comparisons but the detailed datasets and numeric tables were not available in the abstract reviewed for this submission. The DOI contains 2024; the journal issue is dated 2025.",
        "",
        "This implementation narrows the system to local Spark Structured Streaming and MLlib, evaluates three methods on the same fixed holdout, and measures local-core replay performance. It does not reproduce the paper’s multi-component deployment, bias correction, or mapping work. The comparison is architectural and methodological; absolute performance numbers are not comparable.",
        "",
        "## Dataset and preparation",
        "",
        "Sentiment140 is a headerless, latin-1 CSV with fields target, id, date, flag, user, and text. Targets 0 and 4 were mapped to negative 0 and positive 1. Rows without target, ID, text, valid event time, or non-empty cleaned text were excluded. Duplicate tweet IDs were removed before splitting. PDT/PST wall-clock times were normalized to UTC using offsets of -07:00 and -08:00. The username and ID fields were not persisted; VADER text removes handles and URLs, while model text also removes common escaped HTML entities before lowercasing and stripping to alphabetic tokens.",
        "",
    ]
    if audit:
        lines += [
            "| Audit item | Measured value |",
            "|---|---:|",
            f"| Source CSV size | {audit['source_bytes']:,} bytes |",
            f"| Raw rows | {audit['raw_rows']:,} |",
            f"| Retained after cleaning and deduplication | {audit['retained_rows']:,} |",
            f"| Removed or invalid rows | {audit['rows_removed_or_invalid']:,} |",
            f"| Completeness ratio | {audit['completeness_ratio']:.4%} |",
            f"| Training rows | {audit['train_rows']:,} |",
            f"| Holdout rows | {audit['holdout_rows']:,} |",
            f"| Seed | {audit['split_seed']} |",
            f"| JSON replay files | {audit['stream_files_requested']} |",
            "",
            "The split used a seeded random value threshold rather than source-file order, avoiding the original label ordering bias. The recorded split counts are the actual counts from this run.",
            "",
        ]
    lines += ["## Models and evaluation", ""]
    if comparison:
        lines += [
            "All methods scored the same holdout rows. Logistic Regression and multinomial Naive Bayes used word tokenization, English stop-word removal, 262,144-feature HashingTF, and IDF fitted on training data only. Logistic Regression used 20 iterations and regularization 0.01; Naive Bayes used multinomial mode and smoothing 1.0. VADER compound scores at or above 0.05 were positive; neutral and negative scores were mapped to negative.",
            "",
            "| Method | Holdout rows | Accuracy | Weighted F1 | Fit time (s) |",
            "|---|---:|---:|---:|---:|",
        ]
        for metric in comparison["metrics"]:
            lines.append(f"| {metric['method']} | {metric['holdout_rows']:,} | {metric['accuracy']:.4f} | {metric['weighted_f1']:.4f} | {fmt(metric.get('fit_seconds'), 2)} |")
        vader = next((m for m in comparison["metrics"] if m["method"] == "VADER"), None)
        vader_time = fmt(vader.get("scoring_seconds"), 2) if vader else "Not measured"
        lines += ["", f"Metrics are computed over the persisted holdout with no resampling between methods. VADER has no model fitting step; its measured scoring time was {vader_time} seconds.", ""]
    else:
        lines += ["Evaluation did not complete. Accuracy and weighted F1 are not measured.", ""]
    lines += ["## Simulated streaming and performance", ""]
    lines += [
        "The trained Logistic Regression pipeline scored 64 historical JSON files through Spark Structured Streaming with `availableNow` and four files per trigger. This is a finite **simulated file replay**, not a live or paced feed. Stream time excludes model training and starts after the streaming query is started. Throughput is scored records divided by measured replay wait time. Batch durations are Spark trigger-execution times.",
        "",
        "The host has a 13th Gen Intel Core i7-1355U (12 logical processors) and 15.69 GiB physical RAM. Docker Desktop reported about 7.60 GiB available to its Linux runtime. Spark 3.5.8 ran in the official Apache image with Java 17.0.17 and Python 3.10.12.",
        "",
        "| Spark mode | Records scored | Replay time (s) | Throughput (records/s) | Median trigger (ms) | P95 trigger (ms) | Speedup | Window reconciliation |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    baseline = streams[1]["stream_elapsed_seconds"] if streams[1] else None
    for cores in (1, 2, 4):
        result = streams[cores]
        if not result:
            lines.append(f"| local[{cores}] | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |")
            continue
        speedup = baseline / result["stream_elapsed_seconds"] if baseline else None
        lines.append(
            f"| local[{cores}] | {result['records_scored']:,} | {result['stream_elapsed_seconds']:.2f} | {result['throughput_records_per_second']:.2f} | {fmt(result.get('batch_duration_median_ms'), 0)} | {fmt(result.get('batch_duration_p95_ms'), 0)} | {fmt(speedup, 3)} | {'Pass' if result['window_reconciliation_passed'] else 'Fail'} |"
        )
    lines += ["", "Local-mode core scaling measures one laptop and does not establish cluster scalability. Only completed configurations are interpreted.", ""]
    if top_terms:
        lines += ["## Measured findings and figures", ""]
        if comparison:
            best = max(comparison["metrics"], key=lambda m: m["weighted_f1"])
            lines.append(f"{best['method']} had the highest measured weighted F1 ({best['weighted_f1']:.4f}) on this holdout. This result describes Sentiment140's noisy emoticon-derived labels and the listed preprocessing; it does not establish performance on current social-media language.")
        for label, key in (("predicted negative", 0.0), ("predicted positive", 1.0)):
            words = sorted((t for t in top_terms["terms"] if float(t["prediction"]) == key), key=lambda t: t.get("rank", 999))[:5]
            if words:
                listed = ", ".join(f"{t['word']} ({t['count']:,})" for t in words)
                lines.append(f"The most frequent content tokens in {label} Logistic Regression predictions were {listed}.")
        if trend:
            peaks = trend["peak_daily_counts"]
            neg = peaks.get("predicted_negative")
            pos = peaks.get("predicted_positive")
            lines.append(f"The daily trend contains {trend['one_day_utc_windows']:,} one-day UTC windows. Peak predicted-negative count: {neg['count']:,} on {neg['day_utc']}; peak predicted-positive count: {pos['count']:,} on {pos['day_utc']}.")
        completed = [streams[n] for n in (1, 2, 4) if streams[n]]
        if completed:
            fastest = min(completed, key=lambda r: r["stream_elapsed_seconds"])
            if fastest["cores"] == 1 and len(completed) > 1:
                lines.append("The local[1] run was fastest. Additional local cores did not improve throughput for this file replay on this machine.")
            else:
                lines.append(f"The fastest measured replay used local[{fastest['cores']}] at {fastest['throughput_records_per_second']:.2f} records/second.")
        lines += ["", "![Weighted F1 comparison](../outputs/figures/holdout_weighted_f1.png)", "", "![Top words by predicted class](../outputs/figures/top_terms_by_prediction.png)", ""]
        if (ROOT / "outputs" / "figures" / "daily_sentiment_trend.png").exists():
            lines += ["![Daily predicted sentiment counts](../outputs/figures/daily_sentiment_trend.png)", ""]
        if (ROOT / "outputs" / "figures" / "stream_throughput.png").exists():
            lines += ["![Measured replay throughput](../outputs/figures/stream_throughput.png)", ""]
        if (ROOT / "outputs" / "figures" / "stream_speedup.png").exists():
            lines += ["![Measured replay speedup](../outputs/figures/stream_speedup.png)", ""]
    else:
        lines += ["Top-term and chart generation is not complete. No chart-derived interpretation is reported.", ""]
    lines += [
        "## Limitations and incomplete work",
        "",
        "- Sentiment140 is historical (2009) and uses distant-supervision labels derived from emoticons; labels are noisy and language may not represent current users.",
        "- The replay uses historical files and local Spark mode. It does not measure live-feed arrival latency or distributed cluster scaling.",
        "- The selected paper is summarized from its publisher abstract. Dataset names/sizes and numerical benchmark tables from the full article are not asserted here.",
        "- Any performance core setting not listed as measured above was not completed and must not be inferred from another run.",
        "",
        "## Reproduction and evidence",
        "",
        "See `README.md` for environment setup and commands. Machine-readable run evidence is in `outputs/metrics/` and `outputs/streaming/local_N/stream_metrics.json`. The raw CSV is excluded from version control. Usernames, IDs, and raw prediction text are excluded from persisted predictions; the report shows only model metrics, aggregate counts, and token frequencies.",
        "",
        "## References",
        "",
        "1. Ismail, A. et al. (2025). *Stream ETL framework for twitter-based sentiment analysis: Leveraging big data technologies*. Expert Systems with Applications, 261, 125523. https://doi.org/10.1016/j.eswa.2024.125523",
        "2. Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision*. Sentiment140. http://help.sentiment140.com/for-students",
        "3. Apache Spark 3.5.8 documentation. https://spark.apache.org/docs/3.5.8/",
        "4. Apache Spark Structured Streaming programming guide. https://spark.apache.org/docs/3.5.8/structured-streaming-programming-guide.html",
        "",
    ]
    out = ROOT / "report" / "submission_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
