#!/usr/bin/env python

#
# This script extracts the text and code of Stack Overflow questions (not answers) in separate fields along with one-hot 
# encoded labels (folksonomy tags, 1-5 each question) for records having at least so many occurrences. To run it locally
# set PATH_SET to 'local'. For AWS using PATH_SET of 's3'.
#
# Run me with: PYSPARK_PYTHON=python3 PYSPARK_DRIVER_PYTHON=ipython3 pyspark
#

import gc
import json
import random
import re

import boto3
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
import pyspark.sql.types as T

from lib.utils import (
    create_labeled_schema, create_label_row_columns, extract_text, extract_text_plain, 
    extract_code_plain, get_indexes, one_hot_encode,
)


# Set the minimum count a tag must occur to be included in our dataset
TAG_LIMIT = 50

# Set the maximum number of records to sample for each tags
SAMPLE_LIMIT = 500

# Print debug info as we compute, takes extra time
DEBUG = False

# Print a report on record/label duplication at the end
REPORT = True

# Define a set of paths for each step for local and S3
PATH_SET = 'local' # 's3'

PATHS = {
    's3_bucket': 'stackoverflow-events',
    'posts': {
        'local': 'data/stackoverflow/Posts.df.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Posts.df.parquet',
    },
    'questions': {
        'local': 'data/stackoverflow/Questions.Answered.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Answered.parquet',
    },
    'users_parquet': {
        'local': 'data/stackoverflow/Users.df.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Users.df.parquet',
    },
    'questions_users': {
        'local': 'data/stackoverflow/QuestionsUsers.df.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/QuestionsUsers.df.parquet',
    },
    'tag_counts': {
        'local': 'data/stackoverflow/Questions.TagCounts.All.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.TagCounts.All.parquet',
    },
    'questions_tags': {
        'local': 'data/stackoverflow/Questions.Tags.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Tags.{}.parquet',
    },
    'per_tag': {
        'local': 'data/stackoverflow/Questions.PerTag.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.PerTag.{}.parquet',
    },
    'sample_ratios': {
        'local': 'data/stackoverflow/Tag.SampleRatios.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Tag.SampleRatios.{}.parquet',
    },
    'sample': {
        'local': 'data/stackoverflow/Questions.Stratified.All.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Stratified.All.{}.parquet',
    },
    'tag_index': {
        'local': 'data/stackoverflow/all_tag_index.{}.json',
        's3': '08-05-2019/tag_index.{}.json',
    },
    'index_tag': {
        'local': 'data/stackoverflow/all_index_tag.{}.json',
        's3': '08-05-2019/index_tag.{}.json',
    },
}

#
# Initialize Spark with dynamic allocation enabled to (hopefully) use less RAM
#
spark = SparkSession.builder\
    .appName('Weakly Supervised Learning - Extract Questions') \
    .config('spark.dynamicAllocation.enabled', True) \
    .config('spark.shuffle.service.enabled', True) \
    .getOrCreate()
sc = spark.sparkContext


#
# Get answered questions and not their answers
#
posts = spark.read.parquet(PATHS['posts'][PATH_SET])
if DEBUG is True:
    print('Total posts count: {:,}'.format(
        posts.count()
    ))
questions = posts.filter(posts._ParentId.isNull())\
                 .filter(posts._AnswerCount > 0)\
                 .filter(posts._Score > 1)
if DEBUG is True:
    print('Total questions count: {:,}'.format(questions.count()))

# Combine title with body
questions = questions.select(
    F.col('_Id').alias('_PostId'),
    '_AcceptedAnswerId',
    F.concat(
        F.col("_Title"),
        F.lit(" "),
        F.col("_Body")
    ).alias('_Body'),
    '_Tags',
    '_AnswerCount',
    '_CommentCount',
    '_FavoriteCount',
    '_OwnerUserId',
    '_OwnerDisplayName',
    '_Score',
    '_ViewCount',
)
questions.show()

# Write all questions to a Parquet file, then trim fields
questions\
    .write.mode('overwrite')\
    .parquet(PATHS['questions'][PATH_SET])
questions_df = spark.read.parquet(PATHS['questions'][PATH_SET])

#
# Join User records from ch02/xml_to_parquet.py
#
users_df = spark.read.parquet(PATHS['users_parquet'][PATH_SET])
users_df = users_df.withColumn(
    '_UserId',
    F.col('_Id')
).drop('_Id')

questions_users_df = questions_df.join(
    users_df,
    on=questions_df._OwnerUserId == users_df._UserId,
    how='left_outer'
)
questions_users_df = questions_users_df.selectExpr(
    '_PostId',
    '_AcceptedAnswerId',
    '_Body',
    '_Tags',
    '_AnswerCount',
    '_CommentCount',
    '_FavoriteCount',
    '_OwnerUserId',
    '_OwnerDisplayName',
    '_Score',
    '_ViewCount',
    '_AboutMe AS _UserAboutMe',
    '_AccountId',
    '_UserId',
    '_DisplayName AS _UserDisplayName',
    '_DownVotes AS _UserDownVotes',
    '_Location AS _UserLocation',
    '_ProfileImageUrl',
    '_Reputation AS _UserReputation',
    '_UpVotes AS _UserUpVotes',
    '_Views AS _UserViews',
    '_WebsiteUrl AS _UserWebsiteUrl',
)
questions_users_df.write.mode('overwrite').parquet(PATHS['questions_users'][PATH_SET])
questions_users_df = spark.read.parquet(PATHS['questions_users'][PATH_SET])
if DEBUG is True:
    questions_users_df.show()

# Count the number of each tag
all_tags = questions_users_df.rdd.flatMap(lambda x: re.sub('[<>]', ' ', x['_Tags']).split())

# Count the instances of each tag
tag_counts_df = all_tags\
    .groupBy(lambda x: x)\
    .map(lambda x: Row(tag=x[0], total=len(x[1])))\
    .toDF()\
    .select('tag', 'total').orderBy(['total'], ascending=False)
tag_counts_df.write.mode('overwrite').parquet(PATHS['tag_counts'][PATH_SET])
tag_counts_df = spark.read.parquet(PATHS['tag_counts'][PATH_SET])

if DEBUG is True:
    tag_counts_df.show(100)

# Create a local dict of tag counts
local_tag_counts = tag_counts_df.rdd.collect()
tag_counts = {x.tag: x.total for x in local_tag_counts}

# Use tags with at least 50 instances
TAG_LIMIT = 50

# Count the good tags
remaining_tags_df = tag_counts_df.filter(tag_counts_df.total > TAG_LIMIT)
tag_total = remaining_tags_df.count()
print(f'\n\nNumber of tags with > {TAG_LIMIT:,} instances: {tag_total:,}')
valid_tags = remaining_tags_df.rdd.map(lambda x: x['tag']).collect()

# Create forward and backward indexes for good/bad tags
tag_index, index_tag, enumerated_labels = get_indexes(remaining_tags_df)

# Turn text of body and tags into lists of words
def tag_list_record(x, valid_tags):
    d = x.asDict()
    
    body = extract_text_plain(d['_Body']) 
    code = extract_code_plain(x['_Body'])
    tags = re.sub('[<>]', ' ', x['_Tags']).split()
    valid_tags = [y for y in tags if y in valid_tags]


    d['_Body'] = body
    d['_Code'] = code
    d['_Tags'] = valid_tags
    d['_Label'] = 0
    
    return Row(**d)

questions_lists = questions_users_df.rdd.map(lambda x: tag_list_record(x, valid_tags))

filtered_lists = questions_lists\
    .filter(lambda x: bool(set(x._Tags) & set(valid_tags)))

# Create a DataFrame to persist this progress
tag_list_schema = T.StructType([
    T.StructField('_PostId', T.IntegerType(), True),
    T.StructField('_AcceptedAnswerId', T.IntegerType(), True),
    T.StructField('_Body', T.StringType(), True),
    T.StructField('_Code', T.StringType(), True),
    T.StructField(
        "_Tags", 
        T.ArrayType(
            T.StringType()
        )
    ),
    T.StructField('_Label', T.IntegerType(), True),
    T.StructField('_AnswerCount', T.IntegerType(), True),
    T.StructField('_CommentCount', T.IntegerType(), True),
    T.StructField('_FavoriteCount', T.IntegerType(), True),
    T.StructField('_OwnerUserId', T.IntegerType(), True),
    T.StructField('_OwnerDisplayName', T.StringType(), True),
    T.StructField('_Score', T.IntegerType(), True),
    T.StructField('_ViewCount', T.IntegerType(), True),
    T.StructField('_UserAboutMe', T.StringType(), True),
    T.StructField('_AccountId',T.IntegerType(), True),
    T.StructField('_UserId', T.IntegerType(), True),
    T.StructField('_UserDisplayName', T.StringType(),True),
    T.StructField('_UserDownVotes', T.IntegerType(), True),
    T.StructField('_UserLocation', T.StringType(), True),
    T.StructField('_ProfileImageUrl', T.StringType(), True),
    T.StructField('_UserReputation', T.IntegerType() ,True),
    T.StructField('_UserUpVotes', T.IntegerType(), True),
    T.StructField('_UserViews', T.IntegerType(), True),
    T.StructField('_UserWebsiteUrl', T.StringType(), True),
])

questions_tags_df = spark.createDataFrame(
    filtered_lists,
    tag_list_schema
)

questions_tags_df.write.mode('overwrite').parquet(PATHS['questions_tags'][PATH_SET].format(TAG_LIMIT))
questions_tags_df = spark.read.parquet(PATHS['questions_tags'][PATH_SET].format(TAG_LIMIT))
questions_tags_df.show()

# # Emit one record per tag
# def emit_tag_records(x, tag_index):
#     d = x.asDict()

#     for tag in d['_Tags']:

#         n = d.copy()
#         n['_LabelIndex'] = tag_index[tag]
#         n['_LabelString'] = tag
#         n['_LabelValue'] = 1
#         del n['_Tags']

#         yield(Row(**n))

# per_tag_questions = questions_tags_df.rdd.flatMap(lambda x: emit_tag_records(x, tag_index))

# # Create a DataFrame out of the one-hot encoded RDD
# per_tag_schema = T.StructType([
#     T.StructField('_PostId', T.IntegerType(), True),
#     T.StructField('_AcceptedAnswerId', T.IntegerType(), True),
#     T.StructField('_Body', T.StringType(), True),
#     T.StructField('_Code', T.StringType(), True),
#     T.StructField('_LabelIndex', T.IntegerType(), True),
#     T.StructField('_LabelString', T.StringType(), True),
#     T.StructField('_LabelValue', T.IntegerType(), True),
#     T.StructField('_AnswerCount', T.IntegerType(), True),
#     T.StructField('_CommentCount', T.IntegerType(), True),
#     T.StructField('_FavoriteCount', T.IntegerType(), True),
#     T.StructField('_OwnerUserId', T.IntegerType(), True),
#     T.StructField('_OwnerDisplayName', T.StringType(), True),
#     T.StructField('_Score', T.IntegerType(), True),
#     T.StructField('_ViewCount', T.IntegerType(), True),
#     T.StructField('_UserAboutMe', T.StringType(), True),
#     T.StructField('_AccountId',T.IntegerType(), True),
#     T.StructField('_UserId', T.IntegerType(), True),
#     T.StructField('_UserDisplayName', T.StringType(),True),
#     T.StructField('_UserDownVotes', T.IntegerType(), True),
#     T.StructField('_UserLocation', T.StringType(), True),
#     T.StructField('_ProfileImageUrl', T.StringType(), True),
#     T.StructField('_UserReputation', T.IntegerType() ,True),
#     T.StructField('_UserUpVotes', T.IntegerType(), True),
#     T.StructField('_UserViews', T.IntegerType(), True),
#     T.StructField('_UserWebsiteUrl', T.StringType(), True),
# ])

# per_tag_df = spark.createDataFrame(
#     per_tag_questions,
#     per_tag_schema
# )

# # Save as Parquet format, partitioned by the label index
# per_tag_df.write.mode('overwrite').parquet(
#     PATHS['per_tag'][PATH_SET].format(TAG_LIMIT),
#     partitionBy=['_LabelIndex']
# )

# per_tag_df = spark.read.parquet(
#     PATHS['per_tag'][PATH_SET].format(TAG_LIMIT)
# )
# per_tag_df.registerTempTable('per_tag')

# # #
# # # 1) Use GROUP BY to get sample ratios
# # #
# # from datetime import datetime

# # # Get the counts for tags all at once
# # total_records_df = spark.sql('SELECT COUNT(*) AS total FROM per_tag')
# # total_records = total_records_df.first().total

# # query = f"""
# #     SELECT 
# #         _LabelIndex,
# #         COUNT(*) AS total,
# #         COUNT(*)/{total_records} AS sample_ratio
# #     FROM per_tag 
# #     GROUP BY _LabelIndex
# # """
# # sample_ratios_df = spark.sql(query)
# # sample_ratios_df.write.mode('overwrite').parquet(
# #     PATHS['sample_ratios'][PATH_SET].format(TAG_LIMIT)
# # )
# # sample_ratios = sample_ratios_df.rdd.map(lambda x: x.asDict()).collect()
# # sample_ratios_d = {x['_LabelIndex'] : x for x in sample_ratios}

# start = datetime.now()

# def sample_group(x, sample_ratios_d):
#     sample_n = 50
#     rs = random.Random()
#     yield rs.sample(list(x), sample_n)


# groupable = stratified_down_sample = per_tag_df.rdd \
#     .map(lambda x: (x._LabelIndex, x))

# grouped = groupable.groupByKey()

# .flatMap(
#         lambda x: sample_group(x[1] if len(x) > 0 else [], sample_ratios_d)
#     ) \
#     .flatMapValues(lambda x: x[1])

# stratified_df = spark.createDataFrame(
#     stratified_down_sample,
#     per_tag_schema
# )
# stratified_df.write.mode('overwrite').parquet(
#     PATHS['sample'][PATH_SET].format(TAG_LIMIT),
#     partitionBy=['_LabelIndex']
# )

# end = datetime.now()
# speed = end - start
# print(speed)

# # diff = speed_3 - speed_2
# # print(diff)


# # # Write out a stratify_limit sized stratified sample for each tag
# # for i in range(0, 10):#tag_total):
# #     print(f'\n\nProcessing tag {i:,} of {tag_total:,} total tags\n\n')
    
# #     one_label_df = one_hot_df.rdd.map(
# #         lambda x: Row(
# #             _Body=x._Body, 
# #             _Code=x._Code,
# #             _Label=x._Tags[i]
# #         )
# #     ).toDF()

# #     one_label_df = one_hot_df.select(
# #         '_Body',
# #         '_Code',
# #         one_hot_df['_Tags'].getItem(i).alias('_Label')
# #     )

# #     # Select records with a positive value for this tag
# #     positive_examples = one_label_df.filter(one_label_df._Label == 1)
# #     negative_examples = one_label_df.filter(one_label_df._Label == 0)
    
# #     # Sample the positive examples to equal the stratify limit
# #     positive_count = positive_examples.count()
# #     ratio = min(1.0, SAMPLE_LIMIT / positive_count)
# #     sample_ratio = max(0.0, ratio)
# #     positive_examples_sample = positive_examples.sample(False, sample_ratio, seed=1337)

# #     # Now get an equal number of negative examples
# #     positive_count = positive_examples_sample.count()
# #     negative_count = negative_examples.count()
# #     ratio = min(1.0, positive_count / negative_count)
# #     sample_ratio = max(0.0, ratio)
# #     negative_examples_sample = negative_examples.sample(False, sample_ratio, seed=1337)

# #     final_examples_df = positive_examples_sample.union(negative_examples_sample)

# #     if DEBUG is True:
# #         final_examples_df.show()

# #     # Write the record out as JSON under a directory we will then read in its enrirety
# #     final_examples_df.write.mode('overwrite').json(PATHS['output_jsonl'][PATH_SET].format(TAG_LIMIT, i))

# #     # Free RAM explicitly each loop
# #     del final_examples_df
# #     gc.collect()

