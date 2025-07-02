FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    openjdk-8-jdk wget curl git maven scala python3 \
    ssh rsync nano net-tools netcat gnupg lsb-release && \
    apt-get clean

# Переменные окружения
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV HADOOP_HOME=/opt/hadoop
ENV SPARK_HOME=/opt/spark
ENV PATH=$HADOOP_HOME/bin:$SPARK_HOME/bin:$PATH

# Установка Spark 3.3.2 с зеркала Huawei
RUN mkdir -p /opt && \
    curl -fSL -o /tmp/spark.tgz https://mirrors.huaweicloud.com/apache/spark/spark-3.3.2/spark-3.3.2-bin-hadoop3.tgz && \
    tar -xzf /tmp/spark.tgz -C /opt && \
    mv /opt/spark-3.3.2-bin-hadoop3 /opt/spark && \
    rm /tmp/spark.tgz

# Установка Hadoop 3.3.6 с зеркала Яндекса
RUN curl -fSL -o /tmp/hadoop.tgz https://mirror.yandex.ru/mirrors/apache/hadoop/core/hadoop-3.3.6/hadoop-3.3.6.tar.gz && \
    tar -xzf /tmp/hadoop.tgz -C /opt && \
    mv /opt/hadoop-3.3.6 /opt/hadoop && \
    rm /tmp/hadoop.tgz

# Клонирование и сборка HiBench с правильными параметрами
RUN git clone https://github.com/Intel-bigdata/HiBench.git /opt/hibench && \
    cd /opt/hibench && \
    mvn -Psparkbench \
        -Dmodules=micro,ml,sql \
        -Dspark.version=3.3.2 \
        -Dscala=2.12 \
        -Dhadoop.mr2.version=3.3.6 \
        clean package

# Копирование настроек (если файл есть в контексте сборки)
COPY hibench.conf /opt/hibench/conf/

WORKDIR /opt/hibench
