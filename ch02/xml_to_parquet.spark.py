#!/usr/bin/env python3

#
# Convert the Stack Overflow data from XML format to Parquet format for performance reasons.
# Run me with: PYSPARK_PYTHON=python3 PYSPARK_DRIVER_PYTHON=ipython3 pyspark/spark-submit --packages com.databricks:spark-xml_2.11:0.7.0
#

import json

from pyspark.sql import SparkSession


spark = SparkSession.builder.appName('Weakly Supervised Learning - Convert XML to Parquet').getOrCreate()
sc = spark.sparkContext

PATH_SET = 's3' # 'local'

# Load the many paths from a JSON file
PATHS = json.load(
    open('paths.json')
)

# Spark-XML DataFrame method
posts_df = spark.read.format('xml').options(rowTag='row').options(rootTag='posts')\
                .load(PATHS['posts_xml'][PATH_SET])
posts_df.write.mode('overwrite')\
        .parquet(PATHS['posts'][PATH_SET])

users_df = spark.read.format('xml').options(rowTag='row').options(rootTag='users')\
                .load(PATHS['users_xml'][PATH_SET])
users_df.write.mode('overwrite')\
        .parquet(PATHS['users'][PATH_SET])

# tags_df = spark.read.format('xml').options(rowTag='row').options(rootTag='tags')\
#                .load('s3://stackoverflow-events/06-24-2019/Tags.xml.lzo')
# tags_df.write.mode('overwrite')\
#        .parquet('s3://stackoverflow-events/06-24-2019/Tags.df.parquet')

# badges_df = spark.read.format('xml').options(rowTag='row').options(rootTag='badges')\
#                  .load('s3://stackoverflow-events/06-24-2019/Badges.xml.lzo')
# badges_df.write.mode('overwrite')\
#          .parquet('s3://stackoverflow-events/06-24-2019/Badges.df.parquet')

# comments_df = spark.read.format('xml').options(rowTag='row').options(rootTag='comments')\
#                    .load('s3://stackoverflow-events/06-24-2019/Comments.xml.lzo')
# comments_df.write.mode('overwrite')\
#            .parquet('s3://stackoverflow-events/06-24-2019/Comments.df.parquet')

# history_df = spark.read.format('xml').options(rowTag='row')\
#                   .options(rootTag='posthistory')\
#                   .load('s3://stackoverflow-events/06-24-2019/PostHistory.xml.lzo')
# history_df.write.mode('overwrite')\
#           .parquet('s3://stackoverflow-events/06-24-2019/PostHistory.df.parquet')
# votes_df = spark.read.format('xml').options(rowTag='row').options(rootTag='votes')\
#                 .load('s3://stackoverflow-events/06-24-2019/Votes.xml.lzo')
# votes_df.write.mode('overwrite')\
#         .parquet('s3://stackoverflow-events/06-24-2019/Votes.df.parquet')
