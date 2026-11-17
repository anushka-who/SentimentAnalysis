from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGURES, METRICS, PROCESSED, STREAMING, make_spark, write_json


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((METRICS / "model_comparison.json").read_text(encoding="utf-8"))
    metrics = comparison["metrics"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([m["method"] for m in metrics], [m["weighted_f1"] for m in metrics], color=["#2563eb", "#0f766e", "#d97706"])
    ax.set_ylim(0, 1); ax.set_ylabel("Weighted F1"); ax.set_title("Same holdout set model comparison")
    ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(FIGURES / "holdout_weighted_f1.png", dpi=180); plt.close(fig)

    terms = json.loads((METRICS / "top_terms.json").read_text(encoding="utf-8"))["terms"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, ax in ((0.0, axes[0]), (1.0, axes[1])):
        group = sorted((t for t in terms if float(t["prediction"]) == label), key=lambda r: r["count"], reverse=True)[:15]
        ax.barh([r["word"] for r in reversed(group)], [r["count"] for r in reversed(group)], color="#b91c1c" if label == 0.0 else "#15803d")
        ax.set_title("Predicted negative" if label == 0.0 else "Predicted positive")
        ax.set_xlabel("Token occurrences")
    fig.suptitle("Top content words in Logistic Regression holdout predictions")
    fig.tight_layout(); fig.savefig(FIGURES / "top_terms_by_prediction.png", dpi=180); plt.close(fig)

    runs = []
    for cores in (1, 2, 4):
        path = STREAMING / f"local_{cores}" / "stream_metrics.json"
        if path.exists():
            runs.append(json.loads(path.read_text(encoding="utf-8")))
    if runs:
        benchmark_files_per_trigger = next((r.get("max_files_per_trigger") for r in runs if r["cores"] == 1), None)
        comparable_runs = [r for r in runs if r.get("max_files_per_trigger") == benchmark_files_per_trigger]
        baseline = next((r["stream_elapsed_seconds"] for r in comparable_runs if r["cores"] == 1), None)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([r["cores"] for r in comparable_runs], [r["throughput_records_per_second"] for r in comparable_runs], marker="o", color="#7c3aed")
        ax.set_xticks([r["cores"] for r in comparable_runs]); ax.set_xlabel("Spark local cores"); ax.set_ylabel("Records per second")
        ax.set_title(f"Measured replay throughput ({benchmark_files_per_trigger} files per trigger)"); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(FIGURES / "stream_throughput.png", dpi=180); plt.close(fig)
        if baseline:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot([r["cores"] for r in comparable_runs], [baseline / r["stream_elapsed_seconds"] for r in comparable_runs], marker="o", color="#0891b2")
            ax.axhline(1.0, color="#666", linewidth=1); ax.set_xticks([r["cores"] for r in comparable_runs])
            ax.set_xlabel("Spark local cores"); ax.set_ylabel("Speedup (T1 / Tn)")
            ax.set_title("Measured local-mode speedup"); ax.grid(alpha=.25); fig.tight_layout()
            fig.savefig(FIGURES / "stream_speedup.png", dpi=180); plt.close(fig)

        spark = make_spark("Sentiment140ChartAggregation", "local[1]")
        win_path = STREAMING / "local_1" / "daily_window_counts"
        if win_path.exists():
            rows = spark.read.parquet(str(win_path)).orderBy("day", "prediction").collect()
            by_label = {0.0: [], 1.0: []}
            by_actual = {0.0: [], 1.0: []}
            days = sorted({r["day"] for r in rows})
            for label in (0.0, 1.0):
                counts = {}
                for r in rows:
                    if float(r["prediction"]) == label:
                        counts[r["day"]] = counts.get(r["day"], 0) + r["count"]
                by_label[label] = [counts.get(day, 0) for day in days]
                actual_counts = {}
                for r in rows:
                    if float(r["label"]) == label:
                        actual_counts[r["day"]] = actual_counts.get(r["day"], 0) + r["count"]
                by_actual[label] = [actual_counts.get(day, 0) for day in days]
            peaks = {}
            for label in (0.0, 1.0):
                class_counts = {}
                for r in rows:
                    if float(r["prediction"]) == label:
                        class_counts[r["day"]] = class_counts.get(r["day"], 0) + r["count"]
                peak = max(class_counts.items(), key=lambda kv: kv[1]) if class_counts else None
                peaks["predicted_positive" if label == 1.0 else "predicted_negative"] = (
                    {"day_utc": str(peak[0]), "count": int(peak[1])} if peak else None
                )
            write_json(METRICS / "daily_trend_summary.json", {
                "one_day_utc_windows": len(days),
                "window_counts_total": sum(int(r["count"]) for r in rows),
                "peak_daily_counts": peaks,
            })
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(days, by_label[0.0], label="Predicted negative", color="#b91c1c")
            ax.plot(days, by_label[1.0], label="Predicted positive", color="#15803d")
            ax.plot(days, by_actual[0.0], label="Actual negative", color="#b91c1c", linestyle="--", alpha=.65)
            ax.plot(days, by_actual[1.0], label="Actual positive", color="#15803d", linestyle="--", alpha=.65)
            ax.set_xlabel("Event-time day (UTC)"); ax.set_ylabel("Tweet count")
            ax.set_title("Actual and predicted daily sentiment from simulated replay"); ax.legend(ncol=2); ax.grid(alpha=.2)
            fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIGURES / "daily_sentiment_trend.png", dpi=180); plt.close(fig)
        spark.stop()
    print(f"Charts saved to {FIGURES}")


if __name__ == "__main__":
    main()
