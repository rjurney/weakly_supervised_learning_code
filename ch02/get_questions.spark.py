#!/usr/bin/env python

#
# This script extracts the text and code of Stack Overflow questions related to Python.
#
# Run me with: PYSPARK_DRIVER_PYTHON=ipython3 PYSPARK_PYTHON=python3 pyspark
#

import re

from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
from pyspark.sql.functions import udf
import pyspark.sql.types as T


DEBUG = True


#
# Initialize Spark with dynamic allocation enabled to (hopefully) use less RAM
#
spark = SparkSession.builder\
    .appName('Weakly Supervised Learning - Extract Questions')\
    .getOrCreate()
sc = spark.sparkContext


#
# Get answered questions and not their answers
#
posts = spark.read.parquet('s3://stackoverflow-events/2020-06-01/Posts.parquet')
posts.show(3)

if DEBUG:
    print('Total posts count:       {:,}'.format(
        posts.count()
    ))

# Questions are posts without a parent ID
questions = posts.filter(posts.ParentId.isNull())

if DEBUG:
    print(
        f'Total questions count:   {questions.count():,}'
    )

# Quality questions have at least one answer and at least one vote
quality_questions = questions.filter(posts.AnswerCount > 0)\
                             .filter(posts.Score > 1)

if DEBUG:
    print(f'Quality questions count: {quality_questions.count():,}')

# Combine title with body
tb_questions = quality_questions.withColumn(
    'Title_Body',
    F.concat(
        F.col("Title"),
        F.lit(" "),
        F.col("Body")
    ),
)

# Split the tags and replace the Tags column
@udf(T.ArrayType(T.StringType()))
def split_tags(tag_string):
    return re.sub('[<>]', ' ', tag_string).split()

# Just look at Python questions
python_questions = tb_questions.filter(tb_questions.Tags.contains('python'))

if DEBUG:
    print(f'Python questions count: {python_questions.count()}')

# Make tags a list all pretty like
tag_questions = python_questions.withColumn(
    'Tags',
    split_tags(
        F.col('Tags')
    )
)

# Show 5 records' Title and Tag fields, full field length
tag_questions.select('Title', 'Tags').show()

# Write all questions to a Parquet file
tag_questions\
    .write.mode('overwrite')\
    .parquet('s3://stackoverflow-events/2020-06-01/PythonQuestions.parquet')
