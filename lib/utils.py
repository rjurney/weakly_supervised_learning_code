# Utilities for the book's notebooks

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np


# In order to tokenize questions and remove stopwords
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')


def fix_metric_name(name):
    """Remove the trailing _NN, ex. precision_86"""
    if name[-1].isdigit():
        repeat_name = '_'.join(name.split('_')[:-1])
    else:
        repeat_name = name
    return repeat_name


def fix_value(val):
    """Convert from numpy to float"""
    return val.item() if isinstance(val, np.float32) else val


def fix_metric(name, val):
    """Fix Tensorflow/Keras metrics by removing any training _NN and concert numpy.float to python float"""
    repeat_name = fix_metric_name(name)
    py_val = fix_value(val)
    return repeat_name, py_val


def get_indexes(df):
    """Create indexes for each multilabel tag"""
    enumerated_labels = [
        z for z in enumerate(
            sorted(
                df.rdd
                .groupBy(lambda x: 1)
                .flatMap(lambda x: [y.tag for y in x[1]])
                .collect()
            )
        )
    ]
    tag_index = {x: i for i, x in enumerated_labels}
    index_tag = {i: x for i, x in enumerated_labels}
    return tag_index, index_tag, enumerated_labels


def extract_text(x):
    """Extract non-code text from posts (questions/answers)"""
    doc = BeautifulSoup(x, 'lxml')
    codes = doc.find_all('code')
    [code.extract() if code else None for code in codes]
    text = re.sub(r'http\S+', ' ', doc.text)
    tokens = [x for x in tokenizer.tokenize(text) if x not in stop_words]
    padded_tokens = [tokens[i] if len(tokens) > i else PAD_TOKEN for i in range(0, MAX_LEN)]
    return padded_tokens


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


def create_labeled_schema(one_row):
    """Create a schema naming all one-hot encoded fields label_{}"""
    schema_list = [
        T.StructField("_Body", T.ArrayType(
            T.StringType()
        )),
    ]
    for i, val in list(enumerate(one_row._Tags)):
        schema_list.append(
            T.StructField(
                f'label_{i}',
                T.IntegerType()
            )
        )
    return T.StructType(schema_list)


def create_label_row_columns(x):
    """Create a dict keyed with dynamic args to use to create a Row for this record"""
    args = {f'label_{i}': val for i, val in list(enumerate(x._Tags))}
    args['_Body'] = x._Body
    return Row(**args)

