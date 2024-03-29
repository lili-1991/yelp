from pyspark.sql import SparkSession
from pyspark.sql.types import *
import sys
import os
from pyspark.sql import Row
from pyspark.sql.functions import udf,explode, lit, desc, count

spark = SparkSession.builder.appName('data_process').getOrCreate()

new_cats = [   
   'Restaurants',
   'Food',
   'Nightlife',
   'Shopping',
   'Beauty & Spas',
   'Event Planning & Services',
   'Arts & Entertainment',
   'Active Life',
   'Hotels & Travel',
   'Health & Medical',
   'Local Services',
   'Pets',
   'Automotive',
   'Home Services',
   'Professional Services',
   'Public Services & Government',
   'Education',
   'Financial Services',
   'Local Flavor'
   ]

def add_new_categories(categories):
    cate = str(categories)
    cate_list = cate.split(',')
    for nc in new_cats:
        if nc in cate_list:
            return nc
    return 'Others'

def read_file():

    schema1= StructType([
        StructField('user_id', StringType(), False),
        StructField('name', StringType(), False),
        StructField('review_count', IntegerType(), False),
        StructField('yelping_since', StringType(), False),
        StructField('friends', StringType(), False),
        StructField('useful', IntegerType(), False),
        StructField('funny', IntegerType(), False),
        StructField('cool', IntegerType(), False),
        StructField('fans', IntegerType(), False),
        StructField('elite', StringType(), False),
        StructField('average_stars', FloatType(), False),
        StructField('compliment_hot', IntegerType(), False),
        StructField('compliment_more', IntegerType(), False),
        StructField('compliment_profile', IntegerType(), False),
        StructField('compliment_cute', IntegerType(), False),
        StructField('compliment_list', IntegerType(), False),
        StructField('compliment_note', IntegerType(), False),
        StructField('compliment_plain', IntegerType(), False),
        StructField('compliment_cool', IntegerType(), False),
        StructField('compliment_funny', IntegerType(), False),
        StructField('compliment_writer', IntegerType(), False),
        StructField('compliment_photos', IntegerType(), False)
    ])

    schema2= StructType([
        StructField('business_id', StringType(), False),
        StructField('name', StringType(), False),
        StructField('neighborhood', StringType(), False),
        StructField('address', StringType(), False),
        StructField('city', StringType(), False),
        StructField('state', StringType(), False),
        StructField('postal_code', StringType(), False),
        StructField('latitude', FloatType(), False),
        StructField('longitude', FloatType(), False),
        StructField('stars', FloatType(), False),
        StructField('review_count', IntegerType(), False),
        StructField('is_open', IntegerType(), False),
        StructField('attributes', StringType(), False),
        StructField('categories', StringType(), False),
        StructField('hours', StringType(), False)
        
    ])

    schema3=StructType([
        StructField('review_id', StringType(), False),
        StructField('user_id', StringType(), False),
        StructField('business_id', StringType(), False),
        StructField('stars', FloatType(), False),
        StructField('date', DateType(), False),
        StructField('text', StringType(), False),
        StructField('useful', IntegerType(), False),
        StructField('funny', IntegerType(), False),
        StructField('cool', IntegerType(), False)     
    ])

    yelp_user = spark.read.json("/user.json", schema1)
    yelp_business = spark.read.json("/business.json", schema2)
    yelp_review = spark.read.json("/review.json", schema3)

    return yelp_user,yelp_business,yelp_review

def main():


    yelp_user,yelp_business,yelp_review = read_file()

    yelp_business.createOrReplaceTempView('yelp_business')
    bid_convert = spark.sql(""" SELECT DISTINCT business_id FROM yelp_business""")
    bid_convert = bid_convert.rdd.map(lambda x: x['business_id']).zipWithIndex().toDF(['business_id','bid'])
    bid_convert.createOrReplaceTempView('bid_convert')

    # bid_convert.show()
    bid_total = bid_convert.count()

    yelp_user.createOrReplaceTempView('yelp_user')
    uid_convert = spark.sql(""" SELECT DISTINCT user_id FROM yelp_user""")
    uid_convert = uid_convert.rdd.map(lambda x: x['user_id']).zipWithIndex().toDF(['user_id','uid'])
    uid_convert.createOrReplaceTempView('uid_convert')
    uid_convert = spark.sql(""" SELECT user_id, uid + {} as uid FROM uid_convert""".format(bid_total))
    uid_convert.createOrReplaceTempView('uid_convert')

    addNewCatetory = udf(add_new_categories, StringType())
    yelp_business = yelp_business.withColumn('category', addNewCatetory('categories')) 
    yelp_business.createOrReplaceTempView('yelp_business')

    yelp_review.createOrReplaceTempView('yelp_review')

    toronto_business = spark.sql("""
            SELECT bc.bid as business_id, b.business_id as business_sid, b.name as business_name,
            b.latitude as business_latitude, b.longitude as business_longitude, b.category as business_category,
            b.is_open as business_is_open,
            r.stars as user_rate_stars, r.date as review_date, uc.uid as user_id, u.user_id as user_sid, u.name as user_name
            FROM yelp_business b JOIN yelp_review r
            ON b.business_id = r.business_id
            JOIN yelp_user u
            ON r.user_id = u.user_id
            JOIN bid_convert bc
            ON  bc.business_id = b.business_id
            JOIN uid_convert uc
            ON uc.user_id = u.user_id
            WHERE lower(b.city) like '%toronto%' """)
    toronto_business.createOrReplaceTempView('toronto_business')
    
    user_review_count = spark.sql("""
            SELECT u.user_id as user_sid, u.review_count as review_count  
            FROM yelp_user u  """)

    user_review_count = toronto_business.join(user_review_count, 'user_sid')

    user_review_count_node = user_review_count.select('user_sid', 'review_count').distinct() \
                                .sort(desc('review_count')) \
                                .limit(100) \
                                .select('user_sid')

    toronto_user_node = toronto_business.join(user_review_count_node, 'user_sid', 'inner') \
                                .select('user_sid','business_sid').distinct() \
                                .groupby('user_sid').agg(count('business_sid')) \
                                .sort(desc('count(business_sid)'))

    toronto_user_node.createOrReplaceTempView('toronto_user_node')
    toronto_user_node.write.save('/toronto_TOP_user_node', format='json', mode='overwrite')

    
    ALS_data_All = spark.sql("""
            SELECT tb.user_id as user_id, tb.business_id as item_id, tb.user_rate_stars as rate  
            FROM toronto_business tb  """)
    # ALS_data_All.show()
    ALS_data_All.write.save("/ALS_all", format='json', mode='overwrite')

    ALS_data_Rest = spark.sql("""
            SELECT tb.user_id as user_id, tb.business_id as item_id, tb.user_rate_stars as rate  
            FROM toronto_business tb  
            WHERE tb.business_category='Restaurants' """)

    # ALS_data_Rest.show()
    ALS_data_Rest.write.save("/ALS_rest", format='json', mode='overwrite')
   

    spark.stop()

if __name__ == "__main__":
    main()
