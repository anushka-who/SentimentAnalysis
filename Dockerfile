FROM apache/spark:3.5.8-java17-python3
USER root
RUN python3 -m pip install --no-cache-dir vaderSentiment==3.3.2 numpy==1.26.4 matplotlib==3.10.7
WORKDIR /workspace
