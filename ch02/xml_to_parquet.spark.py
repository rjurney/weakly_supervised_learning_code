#!/usr/bin/env python3

#
# Convert the Stack Overflow data from XML format to Parquet format for performance reasons.
# Run me with: PYSPARK_PYTHON=python3 PYSPARK_DRIVER_PYTHON=ipython3 pyspark/spark-submit --packages com.databricks:spark-xml_2.11:0.8.0
#

import json

from pyspark.sql import SparkSession


# Initialize PySpark
spark = SparkSession.builder.appName('Weakly Supervised Learning - Convert XML to Parquet').getOrCreate()
sc = spark.sparkContext

# Load the many paths from a JSON file
PATHS = json.load(
    open('paths.json')
)
PATH_SET = 's3' # 'local'

# Use Spark-XML to split the XML file into records
posts_df = spark.read.format('xml')\
                .options(rowTag='row')\
                .options(rootTag='posts')\
                .load(PATHS['posts_xml'][PATH_SET])

# Write the DataFrame out to Parquet format
posts_df.write\
        .mode('overwrite')\
        .parquet(PATHS['posts'][PATH_SET])

# Use Spark-XML to split the XML file into records
users_df = spark.read.format('xml')\
                .options(rowTag='row')\
                .options(rootTag='users')\
                .load(PATHS['users_xml'][PATH_SET])

# Write the DataFrame out to Parquet format
users_df.write\
        .mode('overwrite')\
        .parquet(PATHS['users'][PATH_SET])

tags_df = spark.read.format('xml')\
               .options(rowTag='row')\
               .options(rootTag='tags')\
               .load(PATHS['tags_xml'][PATH_SET])

tags_df.write\
       .mode('overwrite')\
       .parquet(PATHS['tags'][PATH_SET])

badges_df = spark.read.format('xml')\
                      .options(rowTag='row')\
                      .options(rootTag='badges')\
                      .load(PATHS['badges_xml'][PATH_SET])

badges_df.write\
         .mode('overwrite')\
         .parquet(PATHS['badges'][PATH_SET])

comments_df = spark.read.format('xml')\
                        .options(rowTag='row')\
                        .options(rootTag='comments')\
                        .load(PATHS['comments_xml'][PATH_SET])

comments_df.write\
           .mode('overwrite')\
           .parquet(PATHS['comments'][PATH_SET])

post_links_df = spark.read.format('xml')\
                     .options(rowTag='row')\
                     .options(rootTag='postlinks')\
                     .load(PATHS['postlinks_xml'][PATH_SET])

post_links_df.write\
             .mode('overwrite')\
             .parquet(PATHS['postlinks'][PATH_SET])
