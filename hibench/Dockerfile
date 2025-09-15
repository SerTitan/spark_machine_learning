FROM openjdk:8-jdk-bullseye AS build

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-lib2to3 bc maven git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Клонируем HiBench
RUN git clone https://github.com/Intel-bigdata/HiBench.git hibench
WORKDIR /opt/hibench

# Сборка под Hadoop 3.2 / Spark 3.0
RUN mvn -Psparkbench -Phadoopbench -Dspark=3.0 -Dscala=2.12 -Dhadoop=3.2 \
        clean package -DskipTests

FROM bitnami/spark:3.3

USER root
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8

# Утилиты
RUN apt-get update && apt-get install -y \
    git maven wget curl tar dos2unix ca-certificates python3 python3-lib2to3 bc \
 && rm -rf /var/lib/apt/lists/*

# Симлинк на Spark
ENV SPARK_HOME=/opt/spark \
    HIBENCH_HOME=/opt/hibench
RUN test -d /opt/spark || ln -s /opt/bitnami/spark /opt/spark

WORKDIR /opt

# --- Hadoop 3.3.6 (client only) ---
ARG HADOOP_URL=https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
RUN set -eux; \
    wget -qO /tmp/hadoop.tgz "$HADOOP_URL"; \
    tar -xzf /tmp/hadoop.tgz -C /opt; \
    ln -s /opt/hadoop-3.3.6 /opt/hadoop; \
    rm -f /tmp/hadoop.tgz
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

# --- JDK 8 ---
ARG JDK8_URL=https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u382-b05/OpenJDK8U-jdk_x64_linux_hotspot_8u382b05.tar.gz
RUN set -eux; \
    wget -qO /tmp/jdk8.tgz "$JDK8_URL"; \
    mkdir -p /opt/java; \
    tar -xzf /tmp/jdk8.tgz -C /opt/java; \
    rm -f /tmp/jdk8.tgz; \
    ln -s /opt/java/*jdk8* /opt/java8

COPY --from=build /opt/hibench /opt/hibench

WORKDIR /opt/hibench

# --- Прогоняем 2to3 + фиксим скрипты Python ---
RUN 2to3 -w bin/functions/*.py || true && \
    sed -i '1s|/usr/bin/env python2|/usr/bin/env python3|' bin/functions/*.py || true && \
    chmod +x bin/functions/*.py

# --- Копируем фиксированные версии файлов ---
COPY fix/execute_with_log.py /opt/hibench/bin/functions/execute_with_log.py
COPY fix/terminalsize.py     /opt/hibench/bin/functions/terminalsize.py
COPY fix/load_config.py      /opt/hibench/bin/functions/load_config.py

# --- Конфиги HiBench ---
COPY conf/hibench.conf conf/hibench.conf
COPY conf/spark.conf   conf/spark.conf

# --- Утилиты: JavaWordCount и парсер CSV ---
COPY scripts/wordcount_runs_java.sh /usr/local/bin/wordcount_runs_java.sh
COPY scripts/parse_report_to_csv.py /usr/local/bin/parse_report_to_csv.py
RUN chmod +x /usr/local/bin/wordcount_runs_java.sh /usr/local/bin/parse_report_to_csv.py

CMD ["/bin/bash"]
