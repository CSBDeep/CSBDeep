from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter


from .utils import _raise, consume, normalize_mi_ma, from_tensor, to_tensor, tensor_num_channels
import warnings
import numpy as np


from itertools import product

from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty


@add_metaclass(ABCMeta)
class Normalizer():
    """Abstract base class for normalization methods."""

    @abstractmethod
    def before(self, img, channel):
        """Normalization of the raw input image (method stub).

        Parameters
        ----------
        img : :class:`numpy.ndarray`
            Raw input image.
        channel : int or None
            Indicates channel dimension (can be ``None``).

        Returns
        -------
        :class:`numpy.ndarray`
            Normalized input image with suitable values for neural network input.
        """
        pass

    @abstractmethod
    def after(self, mean, scale):
        """Possible adjustment of predicted restored image (method stub).

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
    """Dummy normalizer that does nothing."""

    def __init__(self, do_after=False):
        self._do_after = do_after

    def before(self, img, channel):
        return img

    def after(self, mean, scale):
        assert self.do_after
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
        Flag to indicate whether to undo normalization.
    kwargs : dict
        Keyword arguments for :func:`csbdeep.utils.normalize_mi_ma`.
    """

    def __init__(self, pmin, pmax, do_after=True, **kwargs):
        """TODO."""
        self.pmin = pmin
        self.pmax = pmax
        self._do_after = do_after
        self.kwargs = kwargs

    def before(self, img, channel):
        """Percentile-based normalization of raw input image.

        See :func:`csbdeep.predict.Normalizer.before` for parameter descriptions.
        Note that percentiles are computed individually for each channel
        if `channel` is not ``None``.
        """
        axes = None if channel is None else tuple((d for d in range(img.ndim) if d != channel))
        self.mi = np.percentile(img,self.pmin,axis=axes,keepdims=True)
        self.ma = np.percentile(img,self.pmax,axis=axes,keepdims=True)
        return normalize_mi_ma(img, self.mi, self.ma, **self.kwargs)

    def after(self, mean, scale):
        """Undo percentile-based normalization to map restored image to similar range as input image.

        See :func:`csbdeep.predict.Normalizer.before` for parameter descriptions.
        """
        assert self.do_after
        alpha = self.ma - self.mi
        beta  = self.mi
        return alpha*mean+beta, alpha*scale if scale is not None else None

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
    """Dummy resizer that does not resize the image, only verifies that the size is correct."""

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



def tiled_prediction(predict_function,x,shape_out,channel,n_tiles,block_size):
    """TODO."""
    # from gputools.utils import tile_iterator

    # largest_axis = np.argmax(x.shape)
    # largest axis that is not the channel dimension
    largest_axis = [i for i in np.argsort(x.shape) if i != channel][-1]
    # largest_size = x.shape[largest_axis]

    # blocksize_ideal    = largest_size / n_tiles
    # blocksize_possible = int(np.ceil(blocksize_ideal/block_multiple) * block_multiple)

    # blocksizes = list(x.shape)
    # padsizes   = [0]*x.ndim
    # blocksizes[largest_axis] = blocksize_possible
    # padsizes[largest_axis]   = pad

    dst = np.empty(shape_out, dtype=x.dtype)

    # for padded_tile, s_dst, s_src in tile_iterator(x, blocksize=blocksizes, padsize=padsizes, mode='reflect'):
    for padded_tile, s_src, s_dst in tile_iterator(x,axis=largest_axis,n_tiles=n_tiles,block_size=block_size):
        # remove channel dimension from slices (they disappeared/moved to first dim within predict_function)
        s_src = (slice(None),) + tuple((s for i,s in enumerate(s_src) if i != channel))
        s_dst = (slice(None),) + tuple((s for i,s in enumerate(s_dst) if i != channel))

        # print(s_dst)
        # print(x.shape, s_dst)

        dst[s_dst] = predict_function(padded_tile)[s_src]

    return dst



def tile_iterator(x,axis,n_tiles,block_size,n_block_overlap=6):
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
    1 <= n_tiles <= n_blocks or _raise(ValueError("invalid 'n_tiles' (must be between %s and %s)" % (1,n_blocks)))

    s = n_blocks // n_tiles # tile size
    r = n_blocks %  n_tiles # blocks remainder
    assert n_tiles * s + r == n_blocks

    # list of sizes for each tile
    tile_sizes = s*np.ones(n_tiles,int)
    # distribute remaning blocks to tiles at beginning and end
    if r > 0:
        tile_sizes[:r//2]      += 1
        tile_sizes[-(r-r//2):] += 1

    # (pre,post) offsets for each tile
    off = [(n_block_overlap if i > 0 else 0, n_block_overlap if i < n_tiles-1 else 0) for i in range(n_tiles)]

    # tile_starts = np.concatenate(([0],np.cumsum(tile_sizes[:-1])))
    # print([(_st-_pre,_st+_sz+_post) for (_st,_sz,(_pre,_post)) in zip(tile_starts,tile_sizes,off)])

    def _valid():
        start = 0
        for i in range(n_tiles):
            off_pre, off_post = off[i]
            if (start-off_pre < 0) or (start+tile_sizes[i]+off_post > n_blocks):
                return False
            start += tile_sizes[i]
        return True

    _valid() or _raise(ValueError("'n_tiles' (%d) too large, try reducing it" % n_tiles))


    def to_slice(t):
        sl = [slice(None) for _ in x.shape]
        sl[axis] = slice(t[0]*block_size, t[1]*block_size if t[1]!=0 else None)
        return tuple(sl)


    start = 0
    for i in range(n_tiles):
        off_pre, off_post = off[i]

        tile_in   = (start-off_pre,start+tile_sizes[i]+off_post)  # src in input image     / tile
        tile_out  = (start,start+tile_sizes[i])                   # dst in output image    / s_dst
        tile_crop = (off_pre,-off_post)                           # crop of src for output / s_src

        yield x[to_slice(tile_in)], to_slice(tile_crop), to_slice(tile_out)
        start += tile_sizes[i]



# def tile_iterator_gputools(im, blocksize=(64,64), padsize=(64,64), mode="constant", verbose=False):
#     """Tile iterator (from gputools).

#     iterates over padded tiles of an ND image
#     while keeping track of the slice positions

#     Example:
#     --------
#     im = np.zeros((200,200))
#     res = np.empty_like(im)

#     for padded_tile, s_src, s_dest in tile_iterator(im,
#                               blocksize=(128, 128),
#                               padsize = (64,64),
#                               mode = "wrap"):

#         #do something with the tile, e.g. a convolution
#         res_padded = np.mean(padded_tile)*np.ones_like(padded_tile)

#         # reassemble the result at the correct position
#         res[s_src] = res_padded[s_dest]



#     Parameters
#     ----------
#     im: ndarray
#         the input data (arbitrary dimension)
#     blocksize:
#         the dimension of the blocks to split into
#         e.g. (nz, ny, nx) for a 3d image
#     padsize:
#         the size of left and right pad for each dimension
#     mode:
#         padding mode, like numpy.pad
#         e.g. "wrap", "constant"...

#     Returns
#     -------
#         tile, slice_src, slice_dest

#         tile[slice_dest] is the tile in im[slice_src]

#     """
#     if not(im.ndim == len(blocksize) ==len(padsize)):
#         raise ValueError("im.ndim (%s) != len(blocksize) (%s) != len(padsize) (%s)"
#                          %(im.ndim , len(blocksize) , len(padsize)))

#     subgrids = tuple([int(np.ceil(1.*n/b)) for n,b in zip(im.shape, blocksize)])


#     #if the image dimension are not divible by the blocksize, pad it accordingly
#     pad_mismatch = tuple([(s*b-n) for n,s, b in zip(im.shape,subgrids,blocksize)])

#     if verbose:
#         print("tile padding... ")

#     mode = 'constant'

#     im_pad = np.pad(im,[(p,p+pm) for pm,p in zip(pad_mismatch,padsize)], mode = mode, constant_values=0)

#     # iterates over cartesian product of subgrids
#     for i,index in enumerate(product(*[range(sg) for sg in subgrids])):
#         # the slices
#         # if verbose:
#         #     print("tile %s/%s"%(i+1,np.prod(subgrids)))

#         # dest[s_output] is where we will write to
#         s_input = tuple([slice(i*b,(i+1)*b) for i,b in zip(index, blocksize)])



#         s_output = tuple([slice(p,-p-pm*(i==s-1)) for pm,p,i,s in zip(pad_mismatch,padsize, index, subgrids)])


#         s_output = tuple([slice(p,b+p-pm*(i==s-1)) for b,pm,p,i,s in zip(blocksize,pad_mismatch,padsize, index, subgrids)])


#         s_padinput = tuple([slice(i*b,(i+1)*b+2*p) for i,b,p in zip(index, blocksize, padsize)])
#         padded_block = im_pad[s_padinput]

#         # print im.shape, padded_block.shape, s_output

#         yield padded_block, s_input, s_output