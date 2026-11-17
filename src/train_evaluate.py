from __future__ import annotations

import argparse
import time

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import HashingTF, IDF, RegexTokenizer, StopWordsRemover
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

from common import METRICS, PREDICTIONS, PROCESSED, make_spark, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", default="local[2]")
    args = parser.parse_args()
    spark = make_spark("Sentiment140HoldoutEvaluation", args.cores)
    train = spark.read.parquet(str(PROCESSED / "train")).cache()
    holdout = spark.read.parquet(str(PROCESSED / "holdout")).cache()
    train_count, holdout_count = train.count(), holdout.count()
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    results = []
    lr_scored = None
    for name, classifier in (
        ("Logistic Regression", LogisticRegression(labelCol="label", featuresCol="features", maxIter=20, regParam=0.01)),
        ("Naive Bayes", NaiveBayes(labelCol="label", featuresCol="features", modelType="multinomial", smoothing=1.0)),
    ):
        stages = [
            RegexTokenizer(inputCol="clean_text", outputCol="tokens", pattern="\\s+", minTokenLength=2),
            StopWordsRemover(inputCol="tokens", outputCol="words", stopWords=[w for w in StopWordsRemover.loadDefaultStopWords("english") if w not in {"no", "not", "nor", "never"}]),
            HashingTF(inputCol="words", outputCol="tf", numFeatures=1 << 18),
            IDF(inputCol="tf", outputCol="features"),
            classifier,
        ]
        pipe = Pipeline(stages=stages)
        started = time.perf_counter()
        model = pipe.fit(train)
        fit_seconds = time.perf_counter() - started
        scored = model.transform(holdout)
        pred = scored.select("label", "prediction", "event_time", "clean_text").withColumn("row_key", F.xxhash64("label", "event_time", "clean_text"))
        confusion = pred.groupBy("label", "prediction").count().collect()
        by_pair = {(int(r.label), int(r.prediction)): int(r["count"]) for r in confusion}
        class_metrics = {}
        for label in (0, 1):
            tp = by_pair.get((label, label), 0)
            fp = sum(by_pair.get((actual, label), 0) for actual in (0, 1) if actual != label)
            fn = sum(by_pair.get((label, predicted), 0) for predicted in (0, 1) if predicted != label)
            class_metrics[str(label)] = {"precision": tp / (tp + fp) if tp + fp else 0.0,
                                         "recall": tp / (tp + fn) if tp + fn else 0.0}
        row = {
            "method": name,
            "holdout_rows": holdout_count,
            "accuracy": evaluator_acc.evaluate(pred),
            "weighted_f1": evaluator_f1.evaluate(pred),
            "class_metrics": class_metrics,
            "confusion_matrix": {f"actual_{a}_predicted_{p}": by_pair.get((a, p), 0) for a in (0, 1) for p in (0, 1)},
            "fit_seconds": fit_seconds,
            "hashing_features": 1 << 18,
            "max_iterations": 20 if name == "Logistic Regression" else None,
            "smoothing": 1.0 if name == "Naive Bayes" else None,
        }
        results.append(row)
        prediction_dir = "logistic_regression" if name == "Logistic Regression" else "naive_bayes"
        pred.write.mode("overwrite").parquet(str(PREDICTIONS / prediction_dir))
        if name == "Logistic Regression":
            model.write().overwrite().save(str(PREDICTIONS / "logistic_regression_model"))
            lr_scored = pred.select("prediction", "clean_text")
    # VADER uses the same persisted holdout. A fixed compound >= 0.05 rule is
    # positive; neutral and negative scores map to negative for this binary task.
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()

    @F.udf(DoubleType())
    def vader_prediction(text):
        return 1.0 if analyzer.polarity_scores(text or "")["compound"] >= 0.05 else 0.0

    started = time.perf_counter()
    vader_pred = holdout.withColumn("prediction", vader_prediction("safe_text")).select("label", "prediction", "event_time")
    vader_row = {
        "method": "VADER",
        "holdout_rows": holdout_count,
        "accuracy": evaluator_acc.evaluate(vader_pred),
        "weighted_f1": evaluator_f1.evaluate(vader_pred),
        "fit_seconds": None,
        "scoring_seconds": time.perf_counter() - started,
        "neutral_rule": "compound >= 0.05 => positive; otherwise negative",
    }
    results.append(vader_row)
    vader_pred.write.mode("overwrite").parquet(str(PREDICTIONS / "vader"))
    write_json(METRICS / "model_comparison.json", {
        "holdout_rows": holdout_count,
        "train_rows": train_count,
        "metrics": results,
        "shared_holdout": True,
    })
    # Top words use LR-predicted classes on the same holdout.
    stop = set(StopWordsRemover.loadDefaultStopWords("english"))
    stop_bc = spark.sparkContext.broadcast(stop)
    from pyspark.sql.types import BooleanType

    @F.udf(BooleanType())
    def is_content_word(word):
        return bool(word) and word not in stop_bc.value and len(word) > 1

    top = (lr_scored.select("prediction", F.explode(F.split("clean_text", r"\s+")).alias("word"))
           .filter(is_content_word("word"))
           .groupBy("prediction", "word").count())
    rank = Window.partitionBy("prediction").orderBy(F.col("count").desc(), F.col("word"))
    top_json = [row.asDict() for row in top.withColumn("rank", F.row_number().over(rank))
                .filter(F.col("rank") <= 15).orderBy("prediction", "rank").collect()]
    write_json(METRICS / "top_terms.json", {"source": "Logistic Regression holdout predictions", "terms": top_json})
    print(results)
    train.unpersist(); holdout.unpersist(); spark.stop()


if __name__ == "__main__":
    main()
