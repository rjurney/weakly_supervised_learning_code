# Seperate out each group of bag tags - documents and their single label as gold standard examples for weak supervision

from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
import pyspark.sql.types as T

PATHS = {
    'bad_questions': {
        'local': 'data/stackoverflow/08-05-2019/Questions.Bad.{}.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Bad.{}.{}.parquet',
    },
    'bad_tag_counts': {
        'local': 'data/stackoverflow/08-05-2019/TagCounts.Bad.{}.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/TagCounts.Bad.{}.{}.parquet',
    },
    'one_hot': {
        'local': 'data/stackoverflow/08-05-2019/Questions.Bad.OneHot.{}.{}.parquet',
        's3': 's3://stackoverflow-events/08-05-2019/Questions.Bad.OneHot.{}.{}.parquet',
    },
    'final_tag_examples': {
        'local': 'data/stackoverflow/08-05-2019/PerTag.Bad.{}.{}.jsonl/{}.{}.jsonl',
        's3': 's3://stackoverflow-events/08-05-2019/PerTag.Bad.{}.{}.jsonl/{}.{}.jsonl',
    },
}
# Define a set of paths for each step for local and S3
PATH_SET = 'local'

spark = SparkSession.builder\
    .appName('Deep Products - Sample JSON')\
    .config('spark.dynamicAllocation.enabled', True)\
    .config('spark.shuffle.service.enabled', True)\
    .getOrCreate()
sc = spark.sparkContext

tag_limit, bad_limit = 2000, 500

# Load the questions with tags occurring between 2000 - 500 times (note: does not include more numerous tags)
bad_df = spark.read.parquet(PATHS['bad_questions'][PATH_SET].format(tag_limit, bad_limit))

#
# Count the instances of each bad tag
#
all_tags = bad_df.rdd.flatMap(lambda x: x['_Tags'])
tag_counts_df = all_tags\
    .groupBy(lambda x: x)\
    .map(lambda x: Row(tag=x[0], total=len(x[1])))\
    .toDF()\
    .select('tag', 'total').orderBy(['total'], ascending=False)

tag_counts_df.write.mode('overwrite').parquet(
    PATHS['bad_tag_counts'][PATH_SET].format(tag_limit, bad_limit)
)
tag_counts_df = spark.read.parquet(
    PATHS['bad_tag_counts'][PATH_SET].format(tag_limit, bad_limit)
)
tag_total = tag_counts_df.count()
tag_counts_df.show()


#
# Create indexes for each multilabel tag
#
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
# One hot encode the questions' tags into a DataFrame
#
def one_hot_encode(tag_list, enumerated_labels):
    """PySpark can't one-hot-encode multilabel data, so we do it ourselves."""

    one_hot_row = []
    for i, label in enumerated_labels:
        if index_tag[i] in tag_list:
            one_hot_row.append(1)
        else:
            one_hot_row.append(0)
    assert(len(one_hot_row) == len(enumerated_labels))
    return one_hot_row

# One hot encode the data using one_hot_encode()
one_hot_questions = bad_df.rdd.map(
    lambda x: Row(_Body=x._Body, _Tags=one_hot_encode(x._Tags, enumerated_labels))
)

# Create a DataFrame out of the one-hot encoded RDD
schema = T.StructType([
    T.StructField("_Body", T.ArrayType(
        T.StringType()
    )),
    T.StructField("_Tags", T.ArrayType(
        T.IntegerType()
    ))
])

one_hot_df = spark.createDataFrame(
    one_hot_questions,
    schema
)
one_hot_df.show()

one_hot_df.write.mode('overwrite').parquet(PATHS['one_hot'][PATH_SET].format(
    tag_limit, bad_limit
))
one_hot_df = spark.read.parquet(PATHS['one_hot'][PATH_SET].format(
    tag_limit, bad_limit
))


#
# Write out a seperate set of records for each label, where that label is positive
#
for i in range(0, tag_total):
    
    tag_str = index_tag[i]
    print(f'\n\nProcessing tag {tag_str} which is {i:,} of {tag_total:,} total tags\n\n')

    # Select records with a positive value for this tag
    positive_examples = one_hot_df.filter(F.col('_Tags')[i] == 1)
    
    # Select the current label column alone
    final_examples = positive_examples.select(
        '_Body',
        F.lit(tag_str).alias('_Tag'),
        F.lit(i).alias('_Index'),
    )

    # Write this tag's examples to a subdirectory as 1 JSON file, so we can load them individually as well as all at 
    # once later
    final_examples.coalesce(1).write.mode('overwrite').json(
        PATHS['final_tag_examples'][PATH_SET].format(tag_limit, bad_limit, i, tag_str)
    )
