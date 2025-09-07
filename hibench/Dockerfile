FROM bitnami/spark:3.3

USER root
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8

# Инструменты
RUN apt-get update && apt-get install -y \
    git maven wget curl tar dos2unix ca-certificates python3 \
 && rm -rf /var/lib/apt/lists/*

# Удобный симлинк на Spark (в bitnami: /opt/bitnami/spark)
ENV SPARK_HOME=/opt/spark \
    HIBENCH_HOME=/opt/hibench
RUN test -d /opt/spark || ln -s /opt/bitnami/spark /opt/spark

WORKDIR /opt

# --- Hadoop 3.3.6 (клиент) ---
ARG HADOOP_URL=https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
RUN set -eux; \
    wget -qO /tmp/hadoop.tgz "$HADOOP_URL"; \
    tar -xzf /tmp/hadoop.tgz -C /opt; \
    ln -s /opt/hadoop-3.3.6 /opt/hadoop; \
    rm -f /tmp/hadoop.tgz
ENV HADOOP_HOME=/opt/hadoop
ENV HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV PATH=$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$PATH

# --- JDK 8 (только для сборки HiBench; в рантайме будем использовать Java 17 из bitnami) ---
ARG JDK8_URL=https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u382-b05/OpenJDK8U-jdk_x64_linux_hotspot_8u382b05.tar.gz
RUN set -eux; \
    wget -qO /tmp/jdk8.tgz "$JDK8_URL"; \
    mkdir -p /opt/java; \
    tar -xzf /tmp/jdk8.tgz -C /opt/java; \
    rm -f /tmp/jdk8.tgz; \
    ln -s /opt/java/*jdk8* /opt/java8
# ВАЖНО: не выставляем JAVA_HOME глобально, чтобы рантайм Spark использовал /opt/bitnami/java (Java 17)

# --- Клонируем HiBench ---
RUN set -eux; \
  git clone --depth 1 --branch master https://github.com/Intel-bigdata/HiBench.git hibench || \
  git clone --depth 1 --branch main   https://github.com/Intel-bigdata/HiBench.git hibench; \
  cd /opt/hibench; \
  if [ ! -f bin/build_all.sh ] && [ ! -f bin/build-all.sh ]; then \
    git fetch --tags --depth 1; \
    for T in v7.1.1 HiBench-7.1 v7.1 7.1 HiBench-7.0 v7.0 7.0; do \
      if git rev-parse -q --verify "refs/tags/$T" >/dev/null; then \
        git checkout -q "tags/$T"; \
        break; \
      fi; \
    done; \
  fi

# Настройки Maven (опционально: зеркала/таймауты)
COPY maven-settings.xml /opt/hibench/maven-settings.xml

# --- Перевод legacy python-скриптов HiBench на Python3 (shebang + print) ---
RUN set -eux; \
  sed -i '1s|/usr/bin/env python2|/usr/bin/env python3|; 1s|/usr/bin/env python|/usr/bin/env python3|' \
    /opt/hibench/bin/functions/execute_with_log.py \
    /opt/hibench/bin/functions/terminalsize.py || true; \
  python3 - <<'PY'
from pathlib import Path
import re
for p in ['/opt/hibench/bin/functions/terminalsize.py','/opt/hibench/bin/functions/execute_with_log.py']:
    fp = Path(p)
    if not fp.exists(): 
        continue
    s = fp.read_text(encoding='utf-8', errors='ignore')
    s = re.sub(r'(?m)^[ \t]*print (.+)$', lambda m: 'print(' + m.group(1).rstrip() + ')', s)
    fp.write_text(s, encoding='utf-8')
print('patched python3 prints')
PY

# --- Сборка common + sparkbench с использованием Java8 только в build-слое ---
RUN --mount=type=cache,target=/root/.m2,sharing=locked \
    set -eux; \
    cd /opt/hibench; \
    export JAVA_HOME=/opt/java8; export PATH="$JAVA_HOME/bin:$PATH"; \
    java -version; mvn -version; \
    mvn -s /opt/hibench/maven-settings.xml -B -U -DskipTests -T 1C \
        -pl common,sparkbench -am dependency:go-offline; \
    mvn -s /opt/hibench/maven-settings.xml -B    -DskipTests -T 1C \
        -pl common,sparkbench -am clean package

# --- Конфиги HiBench ---
WORKDIR /opt/hibench
COPY conf/hibench.conf conf/hibench.conf
COPY conf/spark.conf   conf/spark.conf

# --- Наши утилиты: JavaWordCount-бенч и парсер CSV (сразу в образ) ---
COPY scripts/wordcount_runs_java.sh     /usr/local/bin/wordcount_runs_java.sh
COPY scripts/parse_report_to_csv.py     /usr/local/bin/parse_report_to_csv.py
RUN chmod +x /usr/local/bin/wordcount_runs_java.sh /usr/local/bin/parse_report_to_csv.py

# По умолчанию — интерактивный шелл
CMD ["/bin/bash"]
