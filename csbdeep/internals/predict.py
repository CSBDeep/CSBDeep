from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from tqdm import tqdm
from ..utils import _raise, consume, move_channel_for_backend, backend_channels_last, axes_check_and_normalize, axes_dict
import warnings
import numpy as np



def to_tensor(x,channel=None,single_sample=True):
    if single_sample:
        x = x[np.newaxis]
        if channel is not None and channel >= 0:
            channel += 1
    if channel is None:
        x, channel = np.expand_dims(x,-1), -1
    return move_channel_for_backend(x,channel)



def from_tensor(x,channel=-1,single_sample=True):
    return np.moveaxis((x[0] if single_sample else x), (-1 if backend_channels_last() else 1), channel)



def tensor_num_channels(x):
    return x.shape[-1 if backend_channels_last() else 1]



def predict_direct(keras_model,x,axes_in,axes_out=None,**kwargs):
    """TODO."""
    if axes_out is None:
        axes_out = axes_in
    ax_in, ax_out = axes_dict(axes_in), axes_dict(axes_out)
    channel_in, channel_out = ax_in['C'], ax_out['C']
    single_sample = ax_in['S'] is None
    len(axes_in) == x.ndim or _raise(ValueError())
    x = to_tensor(x,channel=channel_in,single_sample=single_sample)
    pred = from_tensor(keras_model.predict(x,**kwargs),channel=channel_out,single_sample=single_sample)
    len(axes_out) == pred.ndim or _raise(ValueError())
    return pred



def predict_tiled(keras_model,x,n_tiles,block_sizes,tile_overlaps,axes_in,axes_out=None,pbar=None,**kwargs):
    """TODO."""

    if all(t==1 for t in n_tiles):
        pred = predict_direct(keras_model,x,axes_in,axes_out,**kwargs)
        if pbar is not None:
            pbar.update()
        return pred

    ###

    if axes_out is None:
        axes_out = axes_in
    axes_in, axes_out = axes_check_and_normalize(axes_in,x.ndim), axes_check_and_normalize(axes_out)
    assert 'S' not in axes_in
    assert 'C' in axes_in and 'C' in axes_out
    ax_in, ax_out = axes_dict(axes_in), axes_dict(axes_out)
    channel_in, channel_out = ax_in['C'], ax_out['C']

    assert set(axes_out).issubset(set(axes_in))
    axes_lost = set(axes_in).difference(set(axes_out))

    def _to_axes_out(seq,elem):
        # assumption: prediction size is same as input size along all axes, except for channel (and lost axes)
        assert len(seq) == len(axes_in)
        # 1. re-order 'seq' from axes_in to axes_out semantics
        seq = [seq[ax_in[a]] for a in axes_out]
        # 2. replace value at channel position with 'elem'
        seq[ax_out['C']] = elem
        return tuple(seq)

    ###

    assert x.ndim == len(n_tiles) == len(block_sizes)
    assert n_tiles[channel_in] == 1
    assert all(n_tiles[ax_in[a]] == 1 for a in axes_lost)
    assert all(np.isscalar(t) and 1<=t and int(t)==t for t in n_tiles)

    # first axis > 1
    axis = next(i for i,t in enumerate(n_tiles) if t>1)

    block_size = block_sizes[axis]
    tile_overlap = tile_overlaps[axis]
    n_block_overlap = int(np.ceil(1.* tile_overlap / block_size))

    # print(f"axis={axis},n_tiles={n_tiles[axis]},block_size={block_size},tile_overlap={tile_overlap},n_block_overlap={n_block_overlap}")

    n_tiles_remaining = list(n_tiles)
    n_tiles_remaining[axis] = 1

    dst = None
    for tile, s_src, s_dst in tile_iterator(x,axis=axis,n_tiles=n_tiles[axis],block_size=block_size,n_block_overlap=n_block_overlap):

        pred = predict_tiled(keras_model,tile,n_tiles_remaining,block_sizes,tile_overlaps,axes_in,axes_out,pbar=pbar,**kwargs)

        # if any(t>1 for t in n_tiles_remaining):
        #     pred = predict_tiled(keras_model,tile,n_tiles_remaining,block_sizes,tile_overlaps,axes_in,axes_out,pbar=pbar,**kwargs)
        # else:
        #     # tmp
        #     pred = tile
        #     if pbar is not None:
        #         pbar.update()

        if dst is None:
            dst_shape = _to_axes_out(x.shape, pred.shape[channel_out])
            dst = np.empty(dst_shape, dtype=x.dtype)

        s_src = _to_axes_out(s_src, slice(None))
        s_dst = _to_axes_out(s_dst, slice(None))

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
        axis of x is assumed to be evenly divisible by block_size
        all tiles are aligned with the block_size
    n_block_overlap : int
        tiles will overlap this many blocks

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
        sl = [slice(None)] * x.ndim
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



def tile_overlap(n_depth, kern_size, pool_size=2):
    rf = {(1, 3, 2):    9, (1, 5, 2):   17, (1, 7, 2):   25,
          (2, 3, 2):   22, (2, 5, 2):   43, (2, 7, 2):   62,
          (3, 3, 2):   46, (3, 5, 2):   92, (3, 7, 2):  138,
          (4, 3, 2):   94, (4, 5, 2):  188, (4, 7, 2):  282,
          (5, 3, 2):  190, (5, 5, 2):  380, (5, 7, 2):  570,
          #
          (1, 3, 4):   14, (1, 5, 4):   27, (1, 7, 4):   38,
          (2, 3, 4):   58, (2, 5, 4):  116, (2, 7, 4):  158,
          (3, 3, 4):  234, (3, 5, 4):  468, (3, 7, 4):  638,
          (4, 3, 4):  938, (4, 5, 4): 1876, (4, 7, 4): 2558}
    try:
        return rf[n_depth, kern_size, pool_size]
    except KeyError:
        raise ValueError('tile_overlap value for n_depth=%d, kern_size=%d, pool_size=%d not available.' % (n_depth, kern_size, pool_size))



class Progress(object):
    def __init__(self, total, thr=1):
        self.pbar = None
        self.total = total
        self.thr = thr
    @property
    def total(self):
        return self._total
    @total.setter
    def total(self, total):
        self.close()
        self._total = total
    def update(self):
        if self.total > self.thr:
            if self.pbar is None:
                self.pbar = tqdm(total=self.total)
            self.pbar.update()
            self.pbar.refresh()
    def close(self):
        if self.pbar is not None:
            self.pbar.close()
        self.pbar = None
