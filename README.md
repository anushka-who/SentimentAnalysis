# Sentiment140 Spark Project

This project evaluates Logistic Regression, Naive Bayes, and VADER on the same deterministic 20% holdout from Sentiment140. It replays those historical holdout records as JSON files through Spark Structured Streaming. The replay is **simulated streaming**, not a live Twitter/X feed.

## Environment

- Windows host: Python 3.12.10 and Java 8u401 are installed; a project-local venv is available for Python utilities.
- Spark execution: the Windows Hadoop helper is blocked by this machine's application-control policy. The reproducible local fallback is Docker Desktop with Apache's official `apache/spark:3.5.8-java17-python3` image; it runs Spark 3.5.8 on the same laptop with Java 17 and Python 3.10.12.
- PySpark 3.5.8, VADER 3.3.2, Matplotlib 3.10.7, and NumPy 1.26.4 are pinned for the Windows venv. The Docker image adds VADER, NumPy, and Matplotlib to the official Spark image.

The raw CSV is expected at `data/raw/training.1600000.processed.noemoticon.csv`. It is ignored by Git because it is a large source dataset. Tweet IDs are used only for deduplication; derived files omit IDs and usernames, and the VADER text removes handles and URLs. Model text also removes common HTML entities before tokenization. Saved predictions contain only labels, predictions, and event timestamps.

## Setup

PowerShell from the repository directory:

```powershell
python -m venv .venv
$env:TEMP = (Join-Path $PWD '.tmp')
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Build the Spark container once (Docker Desktop must be running):

```powershell
docker build -t bda-sentiment-spark:3.5.8 .
```

## Run

```powershell
$workspace = (Get-Location).Path
docker run --rm -v "${workspace}:/workspace" -w /workspace bda-sentiment-spark:3.5.8 /opt/spark/bin/spark-submit --driver-memory 4g --master 'local[2]' /workspace/src/prepare_dataset.py --cores 'local[2]'
docker run --rm -v "${workspace}:/workspace" -w /workspace bda-sentiment-spark:3.5.8 /opt/spark/bin/spark-submit --driver-memory 4g --master 'local[2]' /workspace/src/train_evaluate.py --cores 'local[2]'
docker run --rm -v "${workspace}:/workspace" -w /workspace bda-sentiment-spark:3.5.8 /opt/spark/bin/spark-submit --driver-memory 4g --master 'local[1]' /workspace/src/sentiment_stream.py --cores 1
docker run --rm -v "${workspace}:/workspace" -w /workspace bda-sentiment-spark:3.5.8 /opt/spark/bin/spark-submit --driver-memory 4g --master 'local[2]' /workspace/src/sentiment_stream.py --cores 2
# Optional if the laptop can handle it:
docker run --rm -v "${workspace}:/workspace" -w /workspace bda-sentiment-spark:3.5.8 /opt/spark/bin/spark-submit --driver-memory 4g --master 'local[4]' /workspace/src/sentiment_stream.py --cores 4
docker run --rm -v "${workspace}:/workspace" -w /workspace bda-sentiment-spark:3.5.8 /opt/spark/bin/spark-submit --driver-memory 2g --master 'local[1]' /workspace/src/make_charts.py
.\.venv\Scripts\python.exe src\build_report.py
```

Run each streaming configuration separately. The script clears that configuration's previous output and checkpoint before measuring. Results include stream-only elapsed time, rows per second, micro-batch trigger durations, and daily event-time aggregates.

## Outputs

- `outputs/metrics/dataset_audit.json`: source size, row counts, cleaning retention, class counts and split seed.
- `outputs/metrics/model_comparison.json`: accuracy and weighted F1 for all methods on the shared holdout.
- `outputs/metrics/top_terms.json`: top token counts by Logistic Regression predicted class.
- `outputs/metrics/daily_trend_summary.json`: measured peak-day counts for each predicted class.
- `outputs/streaming/local_N/stream_metrics.json`: measured per-configuration replay evidence.
- `outputs/figures/`: charts derived from saved run outputs.
- `report/submission_report.md`: submission-ready summary, updated after successful runs.
- `ui/`: professional, read-only dashboard for the measured findings.

No values are filled in until a corresponding run succeeds. Missing benchmarks are explicitly marked not measured.

## Open the local dashboard

After the pipeline has produced its outputs, start the allow-listed local server:

```powershell
.\.venv\Scripts\python.exe src\serve_dashboard.py
```

Open `http://127.0.0.1:8765/`. The server only exposes `ui/`, `outputs/metrics/`, `outputs/figures/`, and the submission report; it does not serve the raw CSV, processed tweet files, or prediction data.
