# Utilities for the book's notebooks

import re

import nltk
import numpy as np
import pyspark.sql.functions as F
import pyspark.sql.types as T

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer
from pyspark.sql import Row
from snorkel.analysis import get_label_buckets


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


def extract_text(x, max_len=200, pad_token='__PAD__', stop_words=stop_words):
    """Extract, remove stopwords and tokenize non-code text from posts (questions/answers)"""
    doc = BeautifulSoup(x, 'lxml')
    codes = doc.find_all('code')
    [code.extract() if code else None for code in codes]
    text = re.sub(r'http\S+', ' ', doc.text)
    tokens = [x for x in tokenizer.tokenize(text) if x not in stop_words]

    padded_tokens = []
    if pad_token:
        padded_tokens = [tokens[i] if len(tokens) > i else pad_token for i in range(0, max_len)]
    else:
        padded_tokens = tokens
    return padded_tokens


def extract_text_plain(x):
    """Extract non-code text from posts (questions/answers)"""
    doc = BeautifulSoup(x, 'lxml')
    codes = doc.find_all('code')
    [code.extract() if code else None for code in codes]
    text = re.sub(r'http\S+', ' ', doc.text)
    return text


def extract_code_plain(x):
    """Extract code text from posts (questions/answers)"""
    doc = BeautifulSoup(x, 'lxml')
    codes = doc.find_all('code')
    text = '\n'.join([c.text for c in codes])
    return text


def extract_bert_format(x):
    """Extract text in BERT format"""

    # Parse the sentences from the document
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(x)

    # Write each sentence exactly as it appared to one line each
    for sentence in sentences:
        yield(sentence.encode('unicode-escape').decode().replace('\\\\', '\\'))

    # Add the final document separator
    yield('')


def one_hot_encode(tag_list, enumerated_labels, index_tag):
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
        T.StructField("_Body", T.StringType()),
        T.StructField("_Code", T.StringType()),
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
    args['_Code'] = x._Code
    return Row(**args)


def get_mistakes(df, probs_test, buckets, labels, label_names):
    """Take DataFrame and pair of actual/predicted labels/names and return a DataFrame showing those records."""
    df_fn = df.iloc[buckets[labels]]
    df_fn['probability'] = probs_test[buckets[labels], 1]
    df_fn['true label'] = label_names[0]
    df_fn['predicted label'] = label_names[1]
    return df_fn


def mistakes_df(df, label_model, L_test, y_test):
    """Compute a DataFrame of all the mistakes we've seen."""
    out_dfs = []

    probs_test = label_model.predict_proba(L=L_test)
    preds_test = probs_test >= 0.5

    buckets = get_label_buckets(
        y_test,
        L_test[:, 1]
    )
    print(buckets)

    for (actual, predicted) in buckets.keys():
    
        # Only shot mistakes that we actually voted on
        if actual != predicted:

            actual_name    = number_to_name_dict[actual]
            predicted_name = number_to_name_dict[predicted]

            out_dfs.append(
                get_mistakes(
                    df,
                    probs_test,
                    buckets=buckets,
                    labels=(actual, predicted),
                    label_names=(actual_name, predicted_name)
                )
            )

    if len(out_dfs) > 1:    
        return out_dfs[0].append(
            out_dfs[1:]
        )
    else:
        return out_dfs[0]
