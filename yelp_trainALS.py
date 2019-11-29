from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession, Row, functions, Column
from pyspark.sql.types import *
import sys

spark = SparkSession.builder.appName('YelpTorontoRS').getOrCreate()
spark.sparkContext.setCheckpointDir('/spark-date/checkpoint')
schema = StructType([
    StructField('user_id', LongType(), False),
    StructField('item_id', LongType(), False),
    StructField('rate', FloatType(), False)
])

def main():

    data=spark.read.json("/ALS_all", schema=schema)    
    als = ALS(rank=20, regParam=0.4, maxIter=30, userCol='user_id',
    itemCol='item_id', ratingCol='rate', coldStartStrategy="drop")   
    model = als.fit(data)
    model.save("/model_ALS_all")

    data=spark.read.json("/ALS_rest", schema=schema)
    als = ALS(rank=20, regParam=0.4, maxIter=30, userCol='user_id',
    itemCol='item_id', ratingCol='rate', coldStartStrategy="drop")   
    model = als.fit(data)
    model.save("/model_ALS_rest")
     
if __name__ == "__main__":
    main()