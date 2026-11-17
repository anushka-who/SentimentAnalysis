from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.window import Window

from common import METRICS, PREDICTIONS, PROCESSED, STREAMING, make_spark, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay historical JSON files as a simulated stream.")
    parser.add_argument("--cores", choices=["1", "2", "4"], default="1")
    parser.add_argument("--max-files-per-trigger", type=int, default=4)
    args = parser.parse_args()
    core_spec = f"local[{args.cores}]"
    spark = make_spark("Sentiment140SimulatedFileReplay", core_spec)
    from pyspark.ml import PipelineModel

    model = PipelineModel.load(str(PREDICTIONS / "logistic_regression_model"))
    source = PROCESSED / "simulated_stream"
    run_root = STREAMING / f"local_{args.cores}"
    output = run_root / "predictions"
    checkpoint = run_root / "checkpoint"
    window_updates = run_root / "window_updates"
    import shutil
    for p in (output, checkpoint, window_updates, run_root / "window_checkpoint"):
        if p.exists():
            shutil.rmtree(p)
    schema = spark.read.parquet(str(PROCESSED / "holdout")).schema
    stream = (spark.readStream.schema(schema)
              .option("maxFilesPerTrigger", args.max_files_per_trigger)
              .json(str(source)))
    scored = model.transform(stream).select("label", "prediction", "event_time", "clean_text")
    scored = scored.withColumn("row_key", F.xxhash64("label", "event_time", "clean_text"))
    # File arrival order is unrelated to historical event time, so the watermark
    # covers the full historical calendar-year range and tolerates file order.
    windowed = (scored.withWatermark("event_time", "365 days")
                .groupBy(F.window("event_time", "1 day").alias("window"), "prediction", "label")
                .count())
    def persist_window_updates(batch_df, batch_id):
        batch_df.withColumn("batch_id", F.lit(batch_id)).write.mode("append").parquet(str(window_updates))

    window_query = (windowed.writeStream.outputMode("update").foreachBatch(persist_window_updates)
                    .option("checkpointLocation", str(run_root / "window_checkpoint"))
                    .trigger(availableNow=True).start())
    query = (scored.writeStream.format("parquet").outputMode("append")
             .option("path", str(output)).option("checkpointLocation", str(checkpoint))
             .trigger(availableNow=True).start())
    start = time.perf_counter()
    query.awaitTermination()
    window_query.awaitTermination()
    elapsed = time.perf_counter() - start
    progress = [p if isinstance(p, dict) else p.json for p in query.recentProgress]
    parsed = [json.loads(p) if isinstance(p, str) else p for p in progress]
    preds = spark.read.parquet(str(output)).cache()
    rows = preds.count()
    valid_event_rows = preds.filter(F.col("event_time").isNotNull()).count()
    latest_update = Window.partitionBy("window", "prediction", "label").orderBy(F.col("batch_id").desc())
    windows = (spark.read.parquet(str(window_updates))
               .withColumn("_latest", F.row_number().over(latest_update)).filter(F.col("_latest") == 1)
               .select(F.col("window.start").alias("day"), "prediction", "label", "count")
               .orderBy("day", "prediction"))
    windows.write.mode("overwrite").parquet(str(run_root / "daily_window_counts"))
    window_total = windows.agg(F.sum("count")).first()[0] or 0
    batch = spark.read.parquet(str(PREDICTIONS / "logistic_regression"))
    batch_counts = (batch.groupBy("row_key", "label", "prediction").count()
                    .withColumnRenamed("count", "batch_count"))
    stream_counts = (preds.groupBy("row_key", "label", "prediction").count()
                     .withColumnRenamed("count", "stream_count"))
    mismatches = (batch_counts.join(stream_counts, ["row_key", "label", "prediction"], "full")
                  .filter(F.coalesce("batch_count", F.lit(0)) != F.coalesce("stream_count", F.lit(0))).count())
    durations = [int(p.get("durationMs", {}).get("triggerExecution", 0)) for p in parsed]
    batch = {
        "cores": int(args.cores),
        "mode": core_spec,
        "replay_type": "simulated streaming: finite replay of historical JSON files",
        "source_files": len(list(source.glob("*.json"))),
        "max_files_per_trigger": args.max_files_per_trigger,
        "records_scored": rows,
        "valid_event_time_rows": valid_event_rows,
        "window_count_sum": int(window_total),
        "window_reconciliation_passed": int(window_total) == valid_event_rows,
        "batch_stream_predictions_match": mismatches == 0,
        "batch_stream_prediction_mismatch_groups": mismatches,
        "watermark": "365 days (covers the source's historical event-time range in this finite replay)",
        "window_aggregation_location": "Structured Streaming watermark aggregation, update-mode Parquet sink",
        "stream_elapsed_seconds": elapsed,
        "throughput_records_per_second": rows / elapsed if elapsed else None,
        "micro_batches": len(parsed),
        "batch_trigger_duration_ms": durations,
        "batch_duration_median_ms": sorted(durations)[len(durations)//2] if durations else None,
        "batch_duration_p95_ms": sorted(durations)[min(len(durations)-1, int(len(durations)*0.95))] if durations else None,
        "progress": parsed,
    }
    write_json(run_root / "stream_metrics.json", batch)
    print({k: v for k, v in batch.items() if k != "progress"})
    preds.unpersist(); spark.stop()


if __name__ == "__main__":
    main()
