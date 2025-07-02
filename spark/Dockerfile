FROM apache/spark:4.0.0-scala2.13-java17-python3-r-ubuntu

ENV JMX_EXPORTER_VERSION=0.20.0

USER root

RUN apt-get update && apt-get install -y curl && \
    curl -LO https://repo1.maven.org/maven2/io/prometheus/jmx/jmx_prometheus_javaagent/${JMX_EXPORTER_VERSION}/jmx_prometheus_javaagent-${JMX_EXPORTER_VERSION}.jar && \
    mv jmx_prometheus_javaagent-${JMX_EXPORTER_VERSION}.jar /opt/spark/ && \
    mkdir -p /opt/spark/history

COPY jmx_exporter_config.yml /opt/spark/
COPY start-history-server.sh /opt/spark/
RUN chmod +x /opt/spark/start-history-server.sh

ENV SPARK_HISTORY_OPTS="-Dspark.history.fs.logDirectory=/opt/spark/history"
ENV SPARK_DAEMON_JAVA_OPTS="-javaagent:/opt/spark/jmx_prometheus_javaagent-${JMX_EXPORTER_VERSION}.jar=7071:/opt/spark/jmx_exporter_config.yml"
