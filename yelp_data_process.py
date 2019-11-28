from pyspark.sql import SparkSession
from pyspark.sql.types import *
import sys

spark = SparkSession.builder.appName('data_process').getOrCreate()

def main():
	#step 1 convert string busines id to int id
	schema_bid = StructType([
        StructField('business_id', StringType(), False)       
    ])
    yelp_business = spark.read.json("/business.json", schema_bid)
    yelp_business.createOrReplaceTempView('yelp_business')
    businiess_id = spark.sql(""" SELECT DISTINCT business_id FROM yelp_business""")
    businiess_id = businiess_id.rdd.map(lambda x: x['business_id']).zipWithIndex().toDF(['business_id','bid'])
    # businiess_id.write.save("/Bid", format='csv', mode='overwrite')
    businiess_id.show()

if __name__ == "__main__":
    main()
