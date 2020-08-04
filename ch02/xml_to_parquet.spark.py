#!/usr/bin/env python3

#
# Convert the Stack Overflow data from XML format to Parquet format for performance reasons.
# Run me with: PYSPARK_DRIVER_PYTHON=ipython3 PYSPARK_PYTHON=python3 pyspark --packages com.databricks:spark-xml_2.11:0.9.0
#

import json

from pyspark.sql import SparkSession
import pyspark.sql.functions as F


# Initialize PySpark
spark = SparkSession.builder.appName('Weakly Supervised Learning - Convert XML to Parquet').getOrCreate()
sc = spark.sparkContext


def remove_prefix(df):
    """Remove the _ prefix that Spark-XML adds to all attributes"""
    field_names = [x.name for x in df.schema]
    new_field_names = [x[1:] for x in field_names]
    s = []

    # Substitute the old name for the new one
    for old, new in zip(field_names, new_field_names):
        s.append(
            F.col(old).alias(new)
        )
    return df.select(s)


# Use Spark-XML to split the XML file into records
posts_df = spark.read.format('xml')\
                .options(rowTag='row')\
                .options(rootTag='posts')\
                .load('s3://stackoverflow-events/2020-06-01/Posts.xml')

# Remove the _ prefix from field names
posts_df = remove_prefix(posts_df)

# Write the DataFrame out to Parquet format
posts_df.write\
        .mode('overwrite')\
        .parquet('s3://stackoverflow-events/2020-06-01/Posts.parquet')


# Use Spark-XML to split the XML file into records
users_df = spark.read.format('xml')\
                .options(rowTag='row')\
                .options(rootTag='users')\
                .load('s3://stackoverflow-events/2020-06-01/Users.xml')

# Remove the _ prefix from field names
users_df = remove_prefix(users_df)

# Write the DataFrame out to Parquet format
users_df.write\
        .mode('overwrite')\
        .parquet('s3://stackoverflow-events/2020-06-01/Users.parquet')


# # Use Spark-XML to split the XML file into records
# tags_df = spark.read.format('xml')\
#                .options(rowTag='row')\
#                .options(rootTag='tags')\
#                .load('s3://stackoverflow-events/2020-06-01/Tags.xml')

# # Remove the _ prefix from field names
# tags_df = remove_prefix(tags_df)

# # Write the DataFrame out to Parquet format
# tags_df.write\
#        .mode('overwrite')\
#        .parquet('s3://stackoverflow-events/2020-06-01/Tags.parquet')


# # Use Spark-XML to split the XML file into records
# badges_df = spark.read.format('xml')\
#                       .options(rowTag='row')\
#                       .options(rootTag='badges')\
#                       .load('s3://stackoverflow-events/2020-06-01/Badges.xml')

# # Remove the _ prefix from field names
# badges_df = remove_prefix(badges_df)

# # Write the DataFrame out to Parquet format
# badges_df.write\
#          .mode('overwrite')\
#          .parquet('s3://stackoverflow-events/2020-06-01/Badges.parquet')


# # Use Spark-XML to split the XML file into records
# comments_df = spark.read.format('xml')\
#                         .options(rowTag='row')\
#                         .options(rootTag='comments')\
#                         .load('s3://stackoverflow-events/2020-06-01/Comments.xml')

# # Remove the _ prefix from field names
# comments_df = remove_prefix(comments_df)

# # Write the DataFrame out to Parquet format
# comments_df.write\
#            .mode('overwrite')\
#            .parquet('s3://stackoverflow-events/2020-06-01/Comments.parquet')


# # Use Spark-XML to split the XML file into records
# post_links_df = spark.read.format('xml')\
#                      .options(rowTag='row')\
#                      .options(rootTag='postlinks')\
#                      .load('s3://stackoverflow-events/2020-06-01/PostLinks.xml')

# # Remove the _ prefix from field names
# post_links_df = remove_prefix(post_links_df)

# # Write the DataFrame out to Parquet format
# post_links_df.write\
#              .mode('overwrite')\
#              .parquet('s3://stackoverflow-events/2020-06-01/PostLinks.parquet')
