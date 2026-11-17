from __future__ import annotations

from pyspark.ml import PipelineModel
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from pyspark.sql.window import Window

from common import METRICS, PREDICTIONS, PROCESSED, make_spark, write_json


def main() -> None:
    spark = make_spark("Sentiment140TopTerms", "local[2]")
    holdout = spark.read.parquet(str(PROCESSED / "holdout"))
    model = PipelineModel.load(str(PREDICTIONS / "logistic_regression_model"))
    scored = model.transform(holdout).select("prediction", "clean_text")
    stop = spark.sparkContext.broadcast(set(StopWordsRemover.loadDefaultStopWords("english")))

    @F.udf(BooleanType())
    def is_content_word(word):
        return bool(word) and len(word) > 1 and word not in stop.value

    counts = (scored.select("prediction", F.explode(F.split("clean_text", r"\s+")).alias("word"))
              .filter(is_content_word("word"))
              .groupBy("prediction", "word").count())
    rank = Window.partitionBy("prediction").orderBy(F.col("count").desc(), F.col("word"))
    top = counts.withColumn("rank", F.row_number().over(rank)).filter(F.col("rank") <= 15)
    terms = [row.asDict() for row in top.orderBy("prediction", "rank").collect()]
    write_json(METRICS / "top_terms.json", {"source": "Logistic Regression holdout predictions", "terms": terms})
    print(f"Saved {len(terms)} top terms")
    spark.stop()


if __name__ == "__main__":
    main()
