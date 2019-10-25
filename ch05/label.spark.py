# Use Snorkel and PySpark to create weak labels for the data using Label Functions (LFs)

from collections import OrderedDict

import numpy as np

from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
import pyspark.sql.types as T

from snorkel.labeling.apply.spark import SparkLFApplier
<<<<<<< HEAD
from snorkel.labeling import LabelingFunction
from snorkel.types import DataPoint
=======
from snorkel.labeling.lf.nlp_spark import SparkNLPLabelingFunction
>>>>>>> 7476ef5f3d85d8e85ec048499036239025083381


# What limits for tag frequency we're working with
TAG_LIMIT, BAD_LIMIT = 2000, 500

PATHS = {
    'bad_tag_counts': {
        'local': 'data/stackoverflow/TagCounts.Bad.{}.{}.parquet'.format(TAG_LIMIT, BAD_LIMIT),
        's3': 's3://stackoverflow-events/TagCounts.Bad.{}.{}.parquet'.format(TAG_LIMIT, BAD_LIMIT),
    },
    'bad_questions': {
        'local': 'data/stackoverflow/Questions.Bad.{}.{}.parquet'.format(TAG_LIMIT, BAD_LIMIT),
        's3': 's3://stackoverflow-events/Questions.Bad.{}.{}.parquet'.format(TAG_LIMIT, BAD_LIMIT),
    },
    'one_hot': {
        'local': 'data/stackoverflow/Questions.Bad.OneHot.{}.{}.parquet'.format(TAG_LIMIT, BAD_LIMIT),
        's3': 's3://stackoverflow-events/Questions.Bad.OneHot.{}.{}.parquet'.format(TAG_LIMIT, BAD_LIMIT),
    },
    'label_encoded': {
        'local': 'data/stackoverflow/Questions.Bad.LabelEncoded.{}.{}.parquet'.format(TAG_LIMIT, BAD_LIMIT),
        's3': 's3://stackoverflow-events/Questions.Bad.LabelEncoded.{}.{}.parquet'.format(TAG_LIMIT, BAD_LIMIT),
    },
    'weak_labels': 'data/weak_labels.npy',
}

# Define a set of paths for each step for local and S3
PATH_SET = 'local' # 's3'

spark = SparkSession.builder\
    .appName('Deep Products - Create Weak Labels')\
    .config('spark.dynamicAllocation.enabled', True)\
    .config('spark.shuffle.service.enabled', True)\
    .getOrCreate()
sc = spark.sparkContext

bad_questions = spark.read.parquet(
    PATHS['bad_questions'][PATH_SET]
)


#
# Create indexes for each multilabel tag
#
tag_counts_df = spark.read.parquet(PATHS['bad_tag_counts'][PATH_SET])
enumerated_labels = [
    z for z in enumerate(
        sorted(
            tag_counts_df.rdd
            .groupBy(lambda x: 1)
            .flatMap(lambda x: [y.tag for y in x[1]])
            .collect()
        )
    )
]
tag_index = {x: i for i, x in enumerated_labels}
index_tag = {i: x for i, x in enumerated_labels}


#
# Use the indexes to label encode the data
#
def label_encode(x, tag_index):
    """Convert from a list of tags to a label encoded value"""
    for tag in x._Tags:
        yield Row(
            _Body=x._Body,
            _Code=x._Code,
            _Label=tag_index[tag]
        )

label_encoded = bad_questions.rdd.flatMap(
    lambda x: label_encode(x, tag_index)
)
label_encoded_df = label_encoded.toDF()
label_encoded_df.write.mode('overwrite').parquet(PATHS['label_encoded'][PATH_SET])

label_encoded_df = spark.read.parquet(PATHS['label_encoded'][PATH_SET])


#
# Create Label Functions (LFs) for tag search
#
ABSTAIN = -1

def keyword_lookup(x, keywords, label):
    match = any(word in x._Body for word in keywords)
    if match:
        return label
    return ABSTAIN

def make_keyword_lf(keywords, label=ABSTAIN):
    return LabelingFunction(
        name=f"keyword_{keywords}",
        f=keyword_lookup,
        resources=dict(keywords=keywords, label=label),
    )

# A tag split by dashes '-' which aids the search (I think), ex. html-css
keyword_lfs = OrderedDict()
for i, tag in enumerated_labels:
    keyword_lfs[tag] = make_keyword_lf(tag.split('-'), label=i)

#
# Apply labeling functions to get a set of weak labels
#
spark_applier = SparkLFApplier(list(keyword_lfs.values()))
weak_labels = spark_applier.apply(label_encoded)

# Save the weak labels numpy array for use locallys
np.save(
    PATHS['weak_labels'],
    weak_labels
)
