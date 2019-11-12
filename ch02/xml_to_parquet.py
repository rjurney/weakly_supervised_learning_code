#
# Convert the Stack Overflow data from XML format to Parquet format for performance reasons.
# Run me with: pyspark/spark-submit --packages com.databricks:spark-xml_2.11:0.7.0
#

from pyspark.sql import SparkSession


spark = SparkSession.builder.appName('Weakly Supervised Learning - Convert XML to Parquet').getOrCreate()
sc = spark.sparkContext

PATH_SET = 's3' # 'local'
PATHS = {
        'posts': {
                'local': 'data/stackoverflow/Posts.xml.bz2',
                's3': 's3://stackoverflow-events/08-05-2019/Posts.xml.bz2',
        },
        'posts_parquet': {
                'local': 'data/stackoverflow/Posts.df.parquet',
                's3': 's3://stackoverflow-events/08-05-2019/Posts.df.parquet',
        },
        'users': {
                'local': 'data/stackoverflow/Users.xml.bz2',
                's3': 's3://stackoverflow-events/08-05-2019/Users.xml.bz2',
        },
        'users_parquet': {
                'local': 'data/stackoverflow/Users.df.parquet',
                's3': 's3://stackoverflow-events/08-05-2019/Users.df.parquet',
        }
}


# Spark-XML DataFrame method
posts_df = spark.read.format('xml').options(rowTag='row').options(rootTag='posts')\
                .load(PATHS['posts'][PATH_SET])
posts_df.write.mode('overwrite')\
        .parquet(PATHS['posts_parquet'][PATH_SET])

users_df = spark.read.format('xml').options(rowTag='row').options(rootTag='users')\
                .load(PATHS['users'][PATH_SET])
users_df.write.mode('overwrite')\
        .parquet(PATHS['users_parquet'][PATH_SET])

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
