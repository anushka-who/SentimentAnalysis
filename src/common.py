from __future__ import annotations

import json
import os
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "training.1600000.processed.noemoticon.csv"
PROCESSED = ROOT / "data" / "processed"
METRICS = ROOT / "outputs" / "metrics"
PREDICTIONS = ROOT / "outputs" / "predictions"
STREAMING = ROOT / "outputs" / "streaming"
FIGURES = ROOT / "outputs" / "figures"


def make_spark(app_name: str, cores: str = "local[2]") -> SparkSession:
    spark = (
        SparkSession.builder.master(cores)
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.legacy.timeParserPolicy", "CORRECTED")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def clean_columns(df, deduplicate: bool = False):
    from pyspark.sql import functions as F

    df = df.dropna(subset=["target", "ids", "text"])
    df = df.withColumn("label", (F.col("target") == 4).cast("double"))
    # The source timestamps use PDT/PST abbreviations. Normalize these to
    # explicit UTC offsets before Spark parses them, then store UTC instants.
    date_local = F.regexp_replace(F.col("date"), r" P[DS]T ", " ")
    date_local = F.regexp_replace(date_local, r"^[A-Za-z]{3} ", "")
    parsed_local = F.to_timestamp(date_local, "MMM dd HH:mm:ss yyyy")
    # Pacific wall-clock time is UTC-7 during PDT and UTC-8 during PST.
    df = df.withColumn("_local_time", parsed_local).withColumn(
        "event_time",
        F.when(F.col("date").contains("PDT"), F.col("_local_time") + F.expr("INTERVAL 7 HOURS"))
        .when(F.col("date").contains("PST"), F.col("_local_time") + F.expr("INTERVAL 8 HOURS")),
    ).drop("_local_time")
    df = df.withColumn(
        "safe_text",
        F.regexp_replace(
            F.regexp_replace(F.col("text"), r"https?://\S+|www\.\S+", " "),
            r"@\w+",
            " ",
        ),
    )
    model_input = F.regexp_replace(
        F.lower(F.col("safe_text")),
        r"&(?:amp|quot|apos|lt|gt);|&#(?:x[0-9a-f]+|[0-9]+);",
        " ",
    )
    df = df.withColumn(
        "clean_text",
        F.trim(F.regexp_replace(model_input, r"@\w+|[^a-z\s]", " ")),
    ).filter(F.length("clean_text") > 0)
    df = df.filter(F.col("event_time").isNotNull())
    if deduplicate:
        df = df.dropDuplicates(["ids"])
    # Exclude usernames and raw metadata from all derived artifacts.
    # IDs are used for deduplication only; neither IDs nor usernames/raw text
    # are written to processed artifacts. safe_text removes handles and URLs.
    return df.select("target", "label", "date", "event_time", "safe_text", "clean_text")

