from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter


from .utils import _raise, consume, normalize_mi_ma, axes_dict, move_channel_for_backend, backend_channels_last
import warnings
import numpy as np


from itertools import product

from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty

# TODO: move Normalizer and Resizer to .data sub-module?

@add_metaclass(ABCMeta)
class Normalizer():
    """Abstract base class for normalization methods."""

    @abstractmethod
    def before(self, img, axes):
        """Normalization of the raw input image (method stub).

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image.
        axes : str
            Axes of ``img``.

        Returns
        -------
        :class:`numpy.ndarray`
            Normalized input image with suitable values for neural network input.
        """
        pass

    @abstractmethod
    def after(self, mean, scale):
        """Possible adjustment of predicted restored image (method stub).

        It is assumed that the image axes are the same as in the call to :func:`before`.

        Parameters
        ----------
        mean : :class:`numpy.ndarray`
            Predicted restored image or per-pixel ``mean`` of Laplace distributions
            for probabilistic model.
        scale: :class:`numpy.ndarray` or None
            Per-pixel ``scale`` of Laplace distributions for probabilistic model (``None`` otherwise.)

        Returns
        -------
        :class:`numpy.ndarray`
            Adjusted restored image.
        """
        pass

    @abstractproperty
    def do_after(self):
        """bool : Flag to indicate whether :func:`after` should be called."""
        pass


class NoNormalizer(Normalizer):
    """No normalization.

    Parameters
    ----------
    do_after : bool
        Flag to indicate whether to undo normalization.

    Raises
    ------
    ValueError
        If :func:`after` is called, but parameter `do_after` was set to ``False`` in the constructor.
    """

    def __init__(self, do_after=False):
        """foo"""
        self._do_after = do_after

    def before(self, img, axes):
        return img

    def after(self, mean, scale):
        self.do_after or _raise(ValueError())
        return mean, scale

    @property
    def do_after(self):
        return self._do_after


class PercentileNormalizer(Normalizer):
    """Percentile-based image normalization.

    Parameters
    ----------
    pmin : float
        Low percentile.
    pmax : float
        High percentile.
    do_after : bool
        Flag to indicate whether to undo normalization (original data type will not be restored).
    dtype : type
        Data type after normalization.
    kwargs : dict
        Keyword arguments for :func:`csbdeep.utils.normalize_mi_ma`.
    """

    def __init__(self, pmin, pmax, do_after=True, dtype=np.float32, **kwargs):
        """TODO."""
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.dtype = dtype
        self.kwargs = kwargs

    def before(self, img, axes):
        """Percentile-based normalization of raw input image.

        See :func:`csbdeep.predict.Normalizer.before` for parameter descriptions.
        Note that percentiles are computed individually for each channel (if present in `axes`).
        """
        len(axes) == img.ndim or _raise(ValueError())
        channel = axes_dict(axes)['C']
        axes = None if channel is None else tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img,self.pmin,axis=axes,keepdims=True).astype(self.dtype,copy=False)
        self.ma = np.percentile(img,self.pmax,axis=axes,keepdims=True).astype(self.dtype,copy=False)
        return normalize_mi_ma(img, self.mi, self.ma, dtype=self.dtype, **self.kwargs)

    def after(self, mean, scale):
        """Undo percentile-based normalization to map restored image to similar range as input image.

        See :func:`csbdeep.predict.Normalizer.before` for parameter descriptions.

        Raises
        ------
        ValueError
            If parameter `do_after` was set to ``False`` in the constructor.

        """
        self.do_after or _raise(ValueError())
        alpha = self.ma - self.mi
        beta  = self.mi
        return (
            ( alpha*mean+beta ).astype(self.dtype,copy=False),
            ( alpha*scale     ).astype(self.dtype,copy=False) if scale is not None else None
        )

    @property
    def do_after(self):
        """``do_after`` parameter from constructor."""
        return self._do_after



@add_metaclass(ABCMeta)
class Resizer():
    """Abstract base class for resizing methods."""

    @abstractmethod
    def before(self, x, div_n, exclude):
        """Resizing of the raw input image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Raw input image.
        div_n : int
            Resized image must be evenly divisible by this value.
        exclude : int or list(int) or None
            Indicates axes to exclude (can be ``None``),
            e.g. channel dimension.

        Returns
        -------
        :class:`numpy.ndarray`
            Resized input image.
        """
        pass

    @abstractmethod
    def after(self, x, exclude):
        """Resizing of the restored image (method stub).

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            Raw input image.
        exclude : int or list(int) or None
            Indicates axes to exclude (can be ``None``),
            e.g. channel dimension.
            Afert ignoring the exlcudes axes,
            note that the shape of x must be same as in :func:`before`.

        Returns
        -------
        :class:`numpy.ndarray`
            Resized restored image image.
        """
        pass


    def _normalize_exclude(self, exclude, n_dim):
        """Return normalized list of excluded axes."""
        if exclude is None:
            return []
        exclude_list = [exclude] if np.isscalar(exclude) else list(exclude)
        exclude_list = [d%n_dim for d in exclude_list]
        len(exclude_list) == len(np.unique(exclude_list)) or _raise(ValueError())
        all(( isinstance(d,int) and 0<=d<n_dim for d in exclude_list )) or _raise(ValueError())
        return exclude_list


class NoResizer(Resizer):
    """No resizing.

    Raises
    ------
    ValueError
        In :func:`before`, if image resizing is necessary.
    """

    def before(self, x, div_n, exclude):
        exclude = self._normalize_exclude(exclude, x.ndim)
        consume ((
            (s%div_n==0) or _raise(ValueError('%d (axis %d) is not divisible by %d.' % (s,i,div_n)))
            for i,s in enumerate(x.shape) if (i not in exclude)
        ))
        return x

    def after(self, x, exclude):
        return x


class PadAndCropResizer(Resizer):
    """Resize image by padding and cropping.

    If necessary, input image is padded before prediction
    and restored image is cropped back to size of input image
    after prediction.

    Parameters
    ----------
    mode : str
        Parameter ``mode`` of :func:`numpy.pad` that
        controls how the image is padded.
    kwargs : dict
        Keyword arguments for :func:`numpy.pad`.
    """

    def __init__(self, mode='reflect', **kwargs):
        """TODO."""
        self.mode = mode
        self.kwargs = kwargs

    def before(self, x, div_n, exclude):
        """Pad input image.

        See :func:`csbdeep.predict.Resizer.before` for parameter descriptions.
        """
        def _split(v):
            a = v // 2
            return a, v-a
        exclude = self._normalize_exclude(exclude, x.ndim)
        self.pad = [_split((div_n-s%div_n)%div_n) if (i not in exclude) else (0,0) for i,s in enumerate(x.shape)]
        # print(self.pad)
        x_pad = np.pad(x, self.pad, mode=self.mode, **self.kwargs)
        for i in exclude:
            del self.pad[i]
        return x_pad

    def after(self, x, exclude):
        """Crop restored image to retain size of input image.

        See :func:`csbdeep.predict.Resizer.after` for parameter descriptions.
        """
        crop = [slice(p[0], -p[1] if p[1]>0 else None) for p in self.pad]
        for i in self._normalize_exclude(exclude, x.ndim):
            crop.insert(i,slice(None))
        len(crop) == x.ndim or _raise(ValueError())
        return x[crop]


# class CropResizer(Resizer):
#     """TODO."""

#     def before(self, x, div_n, exclude):
#         """TODO."""
#         if np.isscalar(div_n):
#             div_n = x.ndim * [div_n]
#         len(div_n) == x.ndim or _raise(ValueError())
#         for i in self._normalize_exclude(exclude, x.ndim):
#             div_n[i] = 1
#         all((s>=i>=1 for s,i in zip(x.shape,div_n))) or _raise(ValueError())
#         if all((i==1 for i in div_n)):
#             return x
#         return x[tuple((slice(0,(s//i)*i) for s,i in zip(x.shape,div_n)))]

#     def after(self, x, exclude):
#         """TODO."""
#         return x


def to_tensor(x,channel=None,single_sample=True):
    if single_sample:
        x = x[np.newaxis]
        if channel is not None and channel >= 0:
            channel += 1
    if channel is None:
        x, channel = np.expand_dims(x,-1), -1
    return move_channel_for_backend(x,channel)



def from_tensor(x,channel=0,single_sample=True):
    return np.moveaxis((x[0] if single_sample else x), (-1 if backend_channels_last() else 1), channel)



def tensor_num_channels(x):
    return x.shape[-1 if backend_channels_last() else 1]



def predict_direct(keras_model,x,channel_in=None,channel_out=0,single_sample=True,**kwargs):
    """TODO."""
    return from_tensor(keras_model.predict(to_tensor(x,channel=channel_in,single_sample=single_sample),**kwargs),
                       channel=channel_out,single_sample=single_sample)



def predict_tiled(keras_model,x,n_tiles,block_size,tile_overlap,channel_in=None,channel_out=0,**kwargs):
    """TODO."""

    # TODO: better check, write an axis normalization function that converts negative indices to positive ones
    channel_in  = (channel_in  + x.ndim) % x.ndim
    channel_out = (channel_out + x.ndim) % x.ndim

    def _remove_and_insert(x,a):
        # remove element at channel_in and insert a at channel_out
        lst = list(x)
        if channel_in is not None:
            del lst[channel_in]
        lst.insert(channel_out,a)
        return tuple(lst)

    # largest axis (that is not channel_in)
    axis = [i for i in np.argsort(x.shape) if i != channel_in][-1]

    if isinstance(n_tiles,(list,tuple)):
        0 < len(n_tiles) <= x.ndim-(0 if channel_in is None else 1) or _raise(ValueError())
        n_tiles, n_tiles_remaining = n_tiles[0], n_tiles[1:]
    else:
        n_tiles_remaining = []

    n_block_overlap = int(np.ceil(tile_overlap / block_size))
    # n_block_overlap += -1
    # n_block_overlap = 10
    # print(tile_overlap,n_block_overlap)

    dst = None
    for tile, s_src, s_dst in tile_iterator(x,axis=axis,n_tiles=n_tiles,block_size=block_size,n_block_overlap=n_block_overlap):

        if len(n_tiles_remaining) == 0:
            pred = predict_direct(keras_model,tile,channel_in=channel_in,channel_out=channel_out,**kwargs)
        else:
            pred = predict_tiled(keras_model,tile,n_tiles_remaining,block_size,tile_overlap,channel_in=channel_in,channel_out=channel_out,**kwargs)

        if dst is None:
            dst_shape = _remove_and_insert(x.shape, pred.shape[channel_out])
            dst = np.empty(dst_shape, dtype=x.dtype)

        s_src = _remove_and_insert(s_src, slice(None))
        s_dst = _remove_and_insert(s_dst, slice(None))

        dst[s_dst] = pred[s_src]

    return dst



def tile_iterator(x,axis,n_tiles,block_size,n_block_overlap):
    """Tile iterator for one dimension of array x.

    Parameters
    ----------
    x : numpy.ndarray
        Input array
    axis : int
        Axis which sould be tiled, all other axis not tiled.
    n_tiles : int
        Number of tiles for axis of x
    block_size : int
        axis of x is assumed to be ebenly divisible by block_size
        all tiles are aligned with the block_size
    n_block_overlap : int
        tiles will overlap a this many blocks

    """
    n = x.shape[axis]

    n % block_size == 0 or _raise(ValueError("'x' must be evenly divisible by 'block_size' along 'axis'"))
    n_blocks = n // block_size

    n_tiles_valid = int(np.clip(n_tiles,1,n_blocks))
    if n_tiles != n_tiles_valid:
        warnings.warn("invalid value (%d) for 'n_tiles', changing to %d" % (n_tiles,n_tiles_valid))
        n_tiles = n_tiles_valid

    s = n_blocks // n_tiles # tile size
    r = n_blocks %  n_tiles # blocks remainder
    assert n_tiles * s + r == n_blocks

    # list of sizes for each tile
    tile_sizes = s*np.ones(n_tiles,int)
    # distribute remaning blocks to tiles at beginning and end
    if r > 0:
        tile_sizes[:r//2]      += 1
        tile_sizes[-(r-r//2):] += 1

    # n_block_overlap = int(np.ceil(92 / block_size))
    # n_block_overlap -= 1
    # print(n_block_overlap)

    # (pre,post) offsets for each tile
    off = [(n_block_overlap if i > 0 else 0, n_block_overlap if i < n_tiles-1 else 0) for i in range(n_tiles)]

    # tile_starts = np.concatenate(([0],np.cumsum(tile_sizes[:-1])))
    # print([(_st-_pre,_st+_sz+_post) for (_st,_sz,(_pre,_post)) in zip(tile_starts,tile_sizes,off)])

    def to_slice(t):
        sl = [slice(None) for _ in x.shape]
        sl[axis] = slice(
            t[0]*block_size,
            t[1]*block_size if t[1]!=0 else None)
        return tuple(sl)

    start = 0
    for i in range(n_tiles):
        off_pre, off_post = off[i]

        # tile starts before block 0 -> adjust off_pre
        if start-off_pre < 0:
            off_pre = start
        # tile end after last block -> adjust off_post
        if start+tile_sizes[i]+off_post > n_blocks:
            off_post = n_blocks-start-tile_sizes[i]

        tile_in   = (start-off_pre,start+tile_sizes[i]+off_post)  # src in input image     / tile
        tile_out  = (start,start+tile_sizes[i])                   # dst in output image    / s_dst
        tile_crop = (off_pre,-off_post)                           # crop of src for output / s_src

        yield x[to_slice(tile_in)], to_slice(tile_crop), to_slice(tile_out)
        start += tile_sizes[i]



def tile_overlap(n_depth, kern_size):
    rf = {(1, 3):   9, (1, 5):  17, (1, 7): 25,
          (2, 3):  22, (2, 5):  43, (2, 7): 62,
          (3, 3):  46, (3, 5):  92, (3, 7): 138,
          (4, 3):  94, (4, 5): 188, (4, 7): 282,
          (5, 3): 190, (5, 5): 380, (5, 7): 570}
    try:
        return rf[n_depth, kern_size]
    except KeyError:
        raise ValueError('tile_overlap value for n_depth=%d and kern_size=%d not available.' % (n_depth, kern_size))

