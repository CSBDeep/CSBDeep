# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import numpy as np
from tifffile import imsave
import warnings

from ..utils import axes_check_and_normalize, axes_dict, move_image_axes, move_channel_for_backend, backend_channels_last



def save_tiff_imagej_compatible(file, img, axes, **imsave_kwargs):
    """Save tiff in ImageJ-compatible format."""
    axes = axes_check_and_normalize(axes,img.ndim,disallowed='S')

    # convert to imagej-compatible data type
    t = img.dtype
    if   'float' in t.name: t_new = np.float32
    elif 'uint'  in t.name: t_new = np.uint16 if t.itemsize >= 2 else np.uint8
    elif 'int'   in t.name: t_new = np.int16
    else:                   t_new = t
    img = img.astype(t_new, copy=False)
    if t != t_new:
        warnings.warn("Converting data type from '%s' to ImageJ-compatible '%s'." % (t, np.dtype(t_new)))

    # move axes to correct positions for imagej
    img = move_image_axes(img, axes, 'TZCYX', True)

    imsave_kwargs['imagej'] = True
    imsave(file, img, **imsave_kwargs)



def load_training_data(file, validation_split=0, axes=None, n_images=None):
    """ TODO """
    # print("Loading training data...")
    f = np.load(file)
    X, Y = f['X'], f['Y']
    if axes is None:
        axes = f['axes']
    axes = axes_check_and_normalize(axes)

    assert X.shape == Y.shape
    assert len(axes) == X.ndim
    assert 'C' in axes
    if n_images is None:
        n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
    assert 0 <= validation_split < 1

    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']

    if validation_split > 0:
        n_val   = int(round(n_images * validation_split))
        n_train = n_images - n_val
        assert 0 < n_val and 0 < n_train
        X_t, Y_t = X[-n_val:],  Y[-n_val:]
        X,   Y   = X[:n_train], Y[:n_train]
        assert X.shape[0] == n_train and X_t.shape[0] == n_val
        X_t = move_channel_for_backend(X_t,channel=channel)
        Y_t = move_channel_for_backend(Y_t,channel=channel)

    X = move_channel_for_backend(X,channel=channel)
    Y = move_channel_for_backend(Y,channel=channel)

    axes = axes.replace('C','') # remove channel
    if backend_channels_last():
        axes = axes+'C'
    else:
        axes = axes[:1]+'C'+axes[1:]

    data_val = (X_t,Y_t) if validation_split > 0 else None

    return (X,Y), data_val, axes



def save_training_data(file, X, Y, axes):
    np.savez(file, X=X, Y=Y, axes=axes)
