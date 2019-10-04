# Utilities for the book's notebooks

import numpy as np


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
