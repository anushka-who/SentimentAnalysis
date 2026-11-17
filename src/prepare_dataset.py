from __future__ import annotations

import argparse
import os
import stat
import time
import shutil

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType

from common import METRICS, PROCESSED, RAW, clean_columns, make_spark, write_json

SEED = 42
STREAM_FILES = 64


def remove_generated(path):
    if not path.exists():
        return

    def retry_with_write_access(function, name, exc_info):
        os.chmod(name, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        function(name)

    shutil.rmtree(path, onerror=retry_with_write_access)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Sentiment140 for offline training and simulated file replay.")
    parser.add_argument("--cores", default="local[2]")
    args = parser.parse_args()
    if not RAW.exists():
        raise FileNotFoundError(f"Place Sentiment140 at {RAW}")
    spark = make_spark("Sentiment140Preparation", args.cores)
    schema = StructType(
        [
            StructField("target", IntegerType(), True),
            StructField("ids", LongType(), True),
            StructField("date", StringType(), True),
            StructField("flag", StringType(), True),
            StructField("user", StringType(), True),
            StructField("text", StringType(), True),
        ]
    )
    t0 = time.perf_counter()
    raw = spark.read.schema(schema).option("header", "false").option("encoding", "ISO-8859-1").csv(str(RAW))
    raw_count = raw.count()
    clean = clean_columns(raw, deduplicate=True).cache()
    retained_count = clean.count()
    # Stable text hash keeps identical cleaned tweets together across the split.
    split_fraction = F.pmod(F.xxhash64("clean_text", F.lit(SEED)), F.lit(1_000_000)) / F.lit(1_000_000.0)
    assigned = clean.withColumn("_split", split_fraction).cache()
    assigned.count()
    train = assigned.filter(F.col("_split") < 0.8).drop("_split")
    holdout = assigned.filter(F.col("_split") >= 0.8).drop("_split")

    train_path = PROCESSED / "train"
    holdout_path = PROCESSED / "holdout"
    replay_path = PROCESSED / "simulated_stream"
    for path in (train_path, holdout_path, replay_path):
        remove_generated(path)
    train.write.mode("overwrite").parquet(str(train_path))
    holdout.write.mode("overwrite").parquet(str(holdout_path))
    (holdout.select("target", "label", "date", "event_time", "safe_text", "clean_text")
     .repartition(STREAM_FILES).write.mode("overwrite").json(str(replay_path)))

    def label_counts(df):
        return {str(int(row["target"])): int(row["count"]) for row in df.groupBy("target").count().collect()}

    train_count, holdout_count = train.count(), holdout.count()
    ts_bad = raw_count - clean.count()
    duplicate_text_count = (clean.groupBy("clean_text").count().filter(F.col("count") > 1)
                            .agg(F.sum(F.col("count") - 1).alias("n")).first()[0] or 0)
    summary = {
        "source_file": RAW.name,
        "source_bytes": RAW.stat().st_size,
        "raw_rows": raw_count,
        "retained_rows": retained_count,
        "rows_removed_or_invalid": ts_bad,
        "duplicate_clean_text_excess_rows": int(duplicate_text_count),
        "completeness_ratio": retained_count / raw_count if raw_count else None,
        "train_rows": train_count,
        "holdout_rows": holdout_count,
        "train_label_counts_target": label_counts(train),
        "holdout_label_counts_target": label_counts(holdout),
        "split_seed": SEED,
        "train_fraction_threshold": 0.8,
        "stream_files_requested": STREAM_FILES,
        "simulated_stream_directory": str(replay_path.relative_to(PROCESSED.parent.parent)),
        "timestamp_parse_rule": "PDT=-0700, PST=-0800; Spark session UTC",
        "preparation_seconds": time.perf_counter() - t0,
        "usernames_or_ids_saved": False,
    }
    write_json(METRICS / "dataset_audit.json", summary)
    print(summary)
    clean.unpersist()
    assigned.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
