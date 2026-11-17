# Real-Time Social Media Sentiment Analytics on Tweet Streams

**Big Data Analytics (CSE557) | Ramaiah Institute of Technology | Submission deadline: 19 November 2026**  
**Team:** Anish Krishna (1MS24CS029), Anushka Sharad (1MS24CS035)  
**Faculty guide:** Not supplied

## Executive summary
The project processed 1,600,000 source records, retained 1,594,559 after cleaning and deduplication (99.6599%), and used a seeded 80/20 split. The 20% holdout was evaluated with Logistic Regression, Naive Bayes, and VADER. Historical JSON files were replayed through Structured Streaming as a **simulated stream**; no live Twitter/X feed was used.
On the shared holdout, Logistic Regression had the highest measured weighted F1 (0.7692).

## Scenario and questions

Social-media teams need to summarize sentiment in large volumes of short, noisy posts. This study asks how predicted positive and negative counts vary by event-time day, which cleaned words are frequent by predicted class, and how Logistic Regression, Naive Bayes, and VADER compare on one holdout set. The historical Sentiment140 file replay represents simulated streaming, not live ingestion.

## Research reference and relation to this project

Ismail, A., Sazali, F. H., Agos Jawaddi, S. N., & Mutalib, S. (2025). *Stream ETL framework for twitter-based sentiment analysis: Leveraging big data technologies*. Expert Systems with Applications, 261, 125523. DOI: [10.1016/j.eswa.2024.125523](https://doi.org/10.1016/j.eswa.2024.125523). The publisher record confirms volume 261, article 125523, and 2025 publication. Its accessible abstract/section text describes a Kafka-Spark-Cassandra/HBase-Hive-HDFS architecture, a Twitter-XLM-RoBERTa-base three-class classifier, and three use cases (COVID English API data, Country English public data, and COVID Spanish public data). The paper evaluates classification and bias detection/correction, and varies dataset size, trigger interval, and storage technology. Numeric tables and detailed dataset sizes could not be verified from the accessible full text, so no values are reproduced here.

This implementation narrows the system to local Spark Structured Streaming and MLlib, evaluates three methods on the same fixed holdout, and measures local-core replay performance. It does not reproduce the paper’s multi-component deployment, bias correction, or mapping work. The comparison is architectural and methodological; absolute performance numbers are not comparable.

## Dataset and preparation

Sentiment140 is a headerless, latin-1 CSV with fields target, id, date, flag, user, and text. Targets 0 and 4 were mapped to negative 0 and positive 1. Rows without target, ID, text, valid event time, or non-empty cleaned text were excluded. Duplicate tweet IDs were removed before splitting. PDT/PST wall-clock times were normalized to UTC using offsets of -07:00 and -08:00. The username and ID fields were not persisted; VADER text removes handles and URLs, while model text also removes common escaped HTML entities before lowercasing and stripping to alphabetic tokens.

| Audit item | Measured value |
|---|---:|
| Source CSV size | 238,803,811 bytes |
| Raw rows | 1,600,000 |
| Retained after cleaning and deduplication | 1,594,559 |
| Removed or invalid rows | 5,441 |
| Completeness ratio | 99.6599% |
| Training rows | 1,276,032 |
| Holdout rows | 318,527 |
| Seed | 42 |
| JSON replay files | 64 |

The split used a seeded random value threshold rather than source-file order, avoiding the original label ordering bias. The recorded split counts are the actual counts from this run.

## Models and evaluation

All methods scored the same holdout rows. Logistic Regression and multinomial Naive Bayes used word tokenization, English stop-word removal except no/not/nor/never, 262,144-feature HashingTF, and IDF fitted on training data only. Logistic Regression used 20 iterations and regularization 0.01; Naive Bayes used multinomial mode and smoothing 1.0. VADER compound scores at or above 0.05 were positive; neutral scores were mapped to negative by design, which can disadvantage the binary VADER comparison.

| Method | Holdout rows | Accuracy | Weighted F1 | Fit time (s) |
|---|---:|---:|---:|---:|
| Logistic Regression | 318,527 | 0.7692 | 0.7692 | 86.65 |
| Naive Bayes | 318,527 | 0.7495 | 0.7495 | 68.16 |
| VADER | 318,527 | 0.6547 | 0.6543 | Not measured |

Per-class precision and recall (class 0 = negative, class 1 = positive):

| Method | Class | Precision | Recall |
|---|---:|---:|---:|
| Logistic Regression | 0 | 0.7798 | 0.7543 |
| Logistic Regression | 1 | 0.7592 | 0.7844 |
| Naive Bayes | 0 | 0.7447 | 0.7639 |
| Naive Bayes | 1 | 0.7546 | 0.7349 |

Metrics are computed over the persisted holdout with no resampling between methods. VADER has no model fitting step; its measured scoring time was 54.25 seconds.

## Simulated streaming and performance

The trained Logistic Regression pipeline scored 64 historical JSON files through Spark Structured Streaming with `availableNow` and 16 files per trigger. A watermarked daily window aggregation runs in the streaming query and writes update records to a Parquet sink. The watermark covers the historical time span; final window totals are checked against valid event-time prediction rows. This is a finite **simulated file replay**, not a live or paced feed. Trigger durations are processing times, not live-feed latency.

The host has a 13th Gen Intel Core i7-1355U (12 logical processors) and 15.69 GiB physical RAM. Docker Desktop reported 12 CPUs and about 7.60 GiB available to its Linux runtime. Spark 3.5.8 ran in the official Apache image with Java 17.0.17 and Python 3.10.12.

| Spark mode | Records scored | Replay time (s) | Throughput (records/s) | Median trigger (ms) | P95 trigger (ms) | Speedup | Window reconciliation |
|---|---:|---:|---:|---:|---:|---:|---|
| local[1] | 318,527 | 253.95 | 1254.30 | 24451 | 30210 | 1.000 | Pass |
| local[2] | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |
| local[4] | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |

Only one successful run is available for this 16-file trigger configuration. The previous local[2]/local[4] runs used a different trigger size and are not compared here. Three-run means and spread, and same-protocol local[2]/local[4] runs, remain not measured; no core-scaling conclusion is drawn.

## Measured findings and figures

Logistic Regression had the highest measured weighted F1 (0.7692) on this holdout. This result describes Sentiment140's noisy emoticon-derived labels and the listed preprocessing; it does not establish performance on current social-media language.
The most frequent content tokens in predicted negative Logistic Regression predictions were work (9,927), go (9,450), get (9,104), like (8,173), day (8,149).
The most frequent content tokens in predicted positive Logistic Regression predictions were good (13,393), love (10,326), day (9,489), lol (7,609), like (7,515).
The daily trend contains 41 one-day UTC windows. Peak predicted-negative count: 9,442 on 2009-06-07 00:00:00; peak predicted-positive count: 12,735 on 2009-06-07 00:00:00.
The fastest measured replay used local[1] at 1254.30 records/second.

![Weighted F1 comparison](../outputs/figures/holdout_weighted_f1.png)

![Top words by predicted class](../outputs/figures/top_terms_by_prediction.png)

![Daily predicted sentiment counts](../outputs/figures/daily_sentiment_trend.png)

![Measured replay throughput](../outputs/figures/stream_throughput.png)

![Measured replay speedup](../outputs/figures/stream_speedup.png)

## Limitations and incomplete work

- Sentiment140 is historical (2009) and uses distant-supervision labels derived from emoticons; labels are noisy and language may not represent current users.
- The replay uses historical files and local Spark mode. It does not measure live-feed arrival latency or distributed cluster scaling.
- The selected paper is summarized from its publisher abstract. Dataset names/sizes and numerical benchmark tables from the full article are not asserted here.
- Any performance core setting not listed as measured above was not completed and must not be inferred from another run.

## Reproduction and evidence

See `README.md` for environment setup and commands. Machine-readable run evidence is in `outputs/metrics/` and `outputs/streaming/local_N/stream_metrics.json`. The raw CSV is excluded from version control. Usernames, IDs, and raw prediction text are excluded from persisted predictions; the report shows only model metrics, aggregate counts, and token frequencies.

## References

1. Ismail, A. et al. (2025). *Stream ETL framework for twitter-based sentiment analysis: Leveraging big data technologies*. Expert Systems with Applications, 261, 125523. https://doi.org/10.1016/j.eswa.2024.125523
2. Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision*. Sentiment140. http://help.sentiment140.com/for-students
3. Apache Spark 3.5.8 documentation. https://spark.apache.org/docs/3.5.8/
4. Apache Spark Structured Streaming programming guide. https://spark.apache.org/docs/3.5.8/structured-streaming-programming-guide.html
