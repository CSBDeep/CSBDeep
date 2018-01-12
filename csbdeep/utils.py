from __future__ import print_function, unicode_literals, absolute_import, division

# import os, sys
import numpy as np
import keras.backend as K
# from collections import namedtuple
# import argparse
# import json, six
# import keras.models
# from tqdm import tqdm_notebook, tqdm as tqdm_terminal

IS_TF_DIM  = K.image_data_format() == "channels_last"
IS_TF_BACK = K.backend() == "tensorflow"

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
    if IS_TF_DIM:
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


def normalize(x, pmin=3, pmax=99.8, axis = None, clip=False):
    mi = np.percentile(x, pmin, axis = axis, keepdims=True).astype(np.float32)
    ma = np.percentile(x, pmax, axis = axis, keepdims=True).astype(np.float32)

    x = x.astype(np.float32)

    eps = 1.e-20
    try:
        import numexpr
        c_eps = np.float32(eps)
        y = numexpr.evaluate("(x - mi) / (ma - mi + c_eps)")
    except ImportError:
        y = (1. * x - mi) / (ma - mi+eps)

    if clip:
        y = np.clip(y, 0, 1)
    return y
