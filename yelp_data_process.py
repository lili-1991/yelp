from pyspark.sql import SparkSession
from pyspark.sql.types import *
import sys

spark = SparkSession.builder.appName('data_process').getOrCreate()

def main():
    #step 1 convert string business id to int id
    schema_bid = StructType([
        StructField('business_id', StringType(), False)
    ])

    yelp_business = spark.read.json("/business.json", schema_bid)
    yelp_business.createOrReplaceTempView('yelp_business')
    business_id = spark.sql(""" SELECT DISTINCT business_id FROM yelp_business""")
    business_id = business_id.rdd.map(lambda x: x['business_id']).zipWithIndex().toDF(['business_id','bid'])
    # business_id.write.save("/Bid", format='csv', mode='overwrite')
    business_id.show()
    bid_total = business_id.count()
    print(bid_total) 
    spark.stop()

if __name__ == "__main__":
    main()
