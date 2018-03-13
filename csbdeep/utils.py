from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

# import os, sys
import numpy as np
# import keras.backend as K
# from collections import namedtuple
# import argparse
# import json, six
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

def moveaxis_if_tf(X):
    if X is None:
        return None
    if is_tf_dim():
        X = np.moveaxis(X, 1, -1)
    return X

# def to_tensor(x):
#     # print('to_tensor', x.shape)
#     return moveaxis_if_tf(x[np.newaxis, np.newaxis])

# def from_tensor(x,channel=None):
#     # print('from_tensor', x.shape)
#     if channel is None:
#         channel = 0 if tensor_num_channels(x) == 1 else slice(None)
#     if isinstance(channel,slice):
#         return x[0,...,channel] if IS_TF_DIM else moveaxis_if_tf(x[0,channel,...])
#     else:
#         return x[0,...,channel] if IS_TF_DIM else x[0,channel,...]

# def tensor_num_channels(x):
#     return x.shape[-1] if IS_TF_DIM else x.shape[1]

# def load_json(path,fname):
#     with open(os.path.join(path,fname),'r') as f:
#         return json.load(f)

# def save_json(data,path,fname,**kwargs):
#     with open(os.path.join(path,fname),'w') as f:
#         f.write(json.dumps(data,**kwargs))


def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
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
