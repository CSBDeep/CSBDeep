from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import os
import numpy as np
# import keras.backend as K
# from collections import namedtuple
# import argparse
import json
# import keras.models
# from tqdm import tqdm_notebook, tqdm as tqdm_terminal

import collections

# https://www.scivision.co/python-idiomatic-pathlib-use/
try:
    from pathlib import Path
    Path().expanduser()
except (ImportError,AttributeError):
    from pathlib2 import Path



def is_tf_dim():
    import keras.backend as K
    return K.image_data_format() == "channels_last"

def is_tf_back():
    import keras.backend as K
    return K.backend() == "tensorflow"

# # https://stackoverflow.com/a/39662359
# def is_notebook():
#     try:
#         shell = get_ipython().__class__.__name__
#         # print(shell)
#         if shell == 'ZMQInteractiveShell':
#             return True   # Jupyter notebook or qtconsole
#         elif shell == 'TerminalInteractiveShell':
#             return False  # Terminal running IPython
#         else:
#             return False  # Other type (?)
#     except NameError:
#         return False      # Probably standard Python interpreter

# def tqdm(*args, **kwargs):
#     return (tqdm_notebook if is_notebook() else tqdm_terminal)(*args, **kwargs)

# def get_dir_for_file(fname=sys.argv[0]):
#     return os.path.dirname(os.path.realpath(fname))

def moveaxis_if_tf(X,reverse=False):
    if X is None:
        return None
    if reverse:
        return np.moveaxis(X, -1, 1) if not is_tf_dim() else X
    else:
        return np.moveaxis(X, 1, -1) if     is_tf_dim() else X


def to_tensor(x,channel=None,single_sample=True):
    if single_sample:
        x = x[np.newaxis]
    if channel is None:
        return moveaxis_if_tf(np.expand_dims(x,1))
    else:
        return moveaxis_if_tf(np.moveaxis(x,channel,1))

def from_tensor(x,channel=0,single_sample=True):
    return np.moveaxis(
        x[0] if single_sample else x,
        -1 if is_tf_dim() else 1,
        channel)

def tensor_num_channels(x):
    return x.shape[-1 if is_tf_dim() else 1]


def load_json(fpath):
    with open(fpath,'r') as f:
        return json.load(f)

def save_json(data,fpath,**kwargs):
    with open(fpath,'w') as f:
        f.write(json.dumps(data,**kwargs))


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """ TODO """
    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x




def _raise(e):
    raise e

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)

def compose(*funcs):
    return lambda x: reduce(lambda f,g: g(f), funcs, x)

# def pipeline(*steps):
#     return reduce(lambda f,g: g(f), steps)


def shuffle_inplace(*arrs):
    rng = np.random.RandomState()
    state = rng.get_state()
    for a in arrs:
        rng.set_state(state)
        rng.shuffle(a)



def rotate(arr, k=1, axis=1, copy=True):
    """Rotate by 90 degrees around the first 2 axis."""
    if copy:
        arr = arr.copy()

    k = k % 4

    arr = np.rollaxis(arr, axis, arr.ndim)

    if k == 0:
        res = arr
    elif k == 1:
        res = arr[::-1].swapaxes(0, 1)
    elif k == 2:
        res = arr[::-1, ::-1]
    else:
        res = arr.swapaxes(0, 1)[::-1]

    res = np.rollaxis(res, -1, axis)
    return res