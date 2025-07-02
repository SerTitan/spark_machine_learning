from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DemoJob") \
    .master("spark://spark-master:7077") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "/opt/spark/history") \
    .getOrCreate()

df = spark.range(1000000)
print("Count:", df.count())

spark.stop()
