from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

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

try:
    import tempfile
    tempfile.TemporaryDirectory
except (ImportError,AttributeError):
    from backports import tempfile


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

def move_channel_for_backend(X,channel):
    import keras.backend as K
    assert K.image_data_format() in ('channels_first','channels_last')
    if K.image_data_format() == 'channels_last':
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel,  1)


def moveaxis_if_tf(X,channel=1,reverse=False):
    if X is None:
        return None
    if reverse:
        return np.moveaxis(X, -1, channel) if not is_tf_dim() else X
    else:
        return np.moveaxis(X, channel, -1) if     is_tf_dim() else X


def to_tensor(x,channel=None,single_sample=True):
    if single_sample:
        x = x[np.newaxis]
        if channel is not None and channel >= 0:
            channel += 1
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


def download_and_extract_zip_file(url, provides=None, targetdir='.'):
    if provides is None or not all(map(os.path.exists,provides)):
        import zipfile
        from six.moves.urllib.request import urlretrieve
        try:
            filepath, http_msg = urlretrieve(url)
            with zipfile.ZipFile(filepath,'r') as zipfile:
                zipfile.extractall(targetdir)
        finally:
            os.unlink(filepath)


def axes_dict(axes):
    """
    from axes string to dict
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(a in allowed       or _raise(ValueError("invalid axis '%s'."%a))               for a in axes)
    consume(axes.count(a) <= 1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in allowed)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedtuple('Axes',list(allowed))(*[None if axes.find(a) == -1 else axes.find(a) for a in allowed ])

# def axes_dict_to_str(axes):
#     """
#     from dict to axes string
#     S(ample), T(ime), C(hannel), Z, Y, X
#     """
#     allowed = 'STCZYX'
#     # try:
#     #     axes = axes._asdict()
#     # except:
#     #     pass
#     consume(a in allowed or _raise(ValueError("invalid axis '%s'."%a)) for a in axes)
#     dims = [a for a in axes.values() if a is not None]
#     len(dims) == len(np.unique(dims)) or _raise(ValueError("dimension occurs more than once."))

#     s = ['' for _ in axes]
#     for a in axes:
#         if axes[a] is not None:
#             s[axes[a]] = a
#     s = ''.join(s)
#     return s


def move_image_axes(x, fr, to):
    """
    x: ndarray
    fr,to: axes string (see `axes_dict`)
    """
    isinstance(fr,string_types) and isinstance(to,string_types) or _raise(ValueError())
    fr, to = fr.upper(), to.upper()
    sorted(list(fr)) == sorted(list(to)) or _raise(ValueError())
    len(fr) == x.ndim or _raise(ValueError())
    if fr == to:
        return x
    ax_from, ax_to = axes_dict(fr), axes_dict(to)
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])

# def axes_move(axes,a,p):
#     """move or insert 'a' in 'axes' string to position 'p'."""
#     axes = str(axes).upper()
#     ax
#     ax = axes_dict(axes)
#     if ax[a] is None:
#         return axes
#     else:
