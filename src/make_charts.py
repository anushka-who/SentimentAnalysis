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
        baseline = next((r["stream_elapsed_seconds"] for r in runs if r["cores"] == 1), None)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot([r["cores"] for r in runs], [r["throughput_records_per_second"] for r in runs], marker="o", color="#7c3aed")
        ax.set_xticks([r["cores"] for r in runs]); ax.set_xlabel("Spark local cores"); ax.set_ylabel("Records per second")
        ax.set_title("Measured simulated replay throughput"); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(FIGURES / "stream_throughput.png", dpi=180); plt.close(fig)
        if baseline:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot([r["cores"] for r in runs], [baseline / r["stream_elapsed_seconds"] for r in runs], marker="o", color="#0891b2")
            ax.axhline(1.0, color="#666", linewidth=1); ax.set_xticks([r["cores"] for r in runs])
            ax.set_xlabel("Spark local cores"); ax.set_ylabel("Speedup (T1 / Tn)")
            ax.set_title("Measured local-mode speedup"); ax.grid(alpha=.25); fig.tight_layout()
            fig.savefig(FIGURES / "stream_speedup.png", dpi=180); plt.close(fig)

        spark = make_spark("Sentiment140ChartAggregation", "local[1]")
        win_path = STREAMING / "local_1" / "daily_window_counts"
        if win_path.exists():
            rows = spark.read.parquet(str(win_path)).orderBy("day", "prediction").collect()
            by_label = {0.0: [], 1.0: []}
            days = sorted({r["day"] for r in rows})
            for label in (0.0, 1.0):
                counts = {r["day"]: r["count"] for r in rows if float(r["prediction"]) == label}
                by_label[label] = [counts.get(day, 0) for day in days]
            peaks = {}
            for label in (0.0, 1.0):
                class_rows = [r for r in rows if float(r["prediction"]) == label]
                peak = max(class_rows, key=lambda r: r["count"]) if class_rows else None
                peaks["predicted_positive" if label == 1.0 else "predicted_negative"] = (
                    {"day_utc": str(peak["day"]), "count": int(peak["count"])} if peak else None
                )
            write_json(METRICS / "daily_trend_summary.json", {
                "one_day_utc_windows": len(days),
                "window_counts_total": sum(int(r["count"]) for r in rows),
                "peak_daily_counts": peaks,
            })
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(days, by_label[0.0], label="Predicted negative", color="#b91c1c")
            ax.plot(days, by_label[1.0], label="Predicted positive", color="#15803d")
            ax.set_xlabel("Event-time day (UTC)"); ax.set_ylabel("Predicted tweets")
            ax.set_title("Daily sentiment counts from simulated file replay"); ax.legend(); ax.grid(alpha=.2)
            fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIGURES / "daily_sentiment_trend.png", dpi=180); plt.close(fig)
        spark.stop()
    print(f"Charts saved to {FIGURES}")


if __name__ == "__main__":
    main()
