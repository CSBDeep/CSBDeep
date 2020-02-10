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
    for tile, s_src, s_dst in tile_iterator_1d(x,axis=axis,n_tiles=n_tiles[axis],block_size=block_size,n_block_overlap=n_block_overlap):

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



class Tile(object):
    def __init__(self, n, size, overlap, prev):
        self.n = int(n)
        self.size = int(size)
        self.overlap = int(overlap)
        if self.n < self.size:
            assert prev is None
            # print("Truncating tile size from %d to %d." % (self.size, self.n))
            self.size = self.n
            self.overlap = 0
        assert self.size > 2*self.overlap
        # assert self.n >= self.size
        if prev is not None:
            assert not prev.at_end, "Previous tile already at end"
        self.prev = prev
        self.read_slice = self._read_slice
        self.write_slice = self._write_slice

    @property
    def at_begin(self):
        return self.prev is None

    @property
    def at_end(self):
        return self.read_slice.stop == self.n

    @property
    def _read_slice(self):
        if self.at_begin:
            start, stop = 0, self.size
        else:
            prev_read_slice = self.prev.read_slice
            start = prev_read_slice.stop - 2*self.overlap
            stop  = start + self.size
            shift = min(0, self.n - stop)
            start, stop = start + shift, stop + shift
            assert start > prev_read_slice.start
        assert start >= 0 and stop <= self.n
        return slice(start, stop)

    @property
    def _write_slice(self):
        if self.at_begin:
            if self.at_end:
                return slice(0, self.n)
            else:
                return slice(0, self.size - 1*self.overlap)
        elif self.at_end:
            s = self.prev.write_slice.stop
            return slice(s, self.n)
        else:
            s = self.prev.write_slice.stop
            return slice(s, s + self.size - 2*self.overlap)

    def __repr__(self):
        s = np.array(list(' '*self.n))
        s[self.read_slice]  = '-'
        s[self.write_slice] = 'x' if (self.at_begin or self.at_end) else 'o'
        return ''.join(s)



class Tiling(object):
    def __init__(self, n, size, overlap):
        self.n = n
        self.size = size
        self.overlap = overlap
        tiles = [Tile(prev=None, **self.__dict__)]
        while not tiles[-1].at_end:
            tiles.append(Tile(prev=tiles[-1], **self.__dict__))
        self.tiles = tiles

    def __len__(self):
        return len(self.tiles)

    def __repr__(self):
        return '\n'.join('{i:3}. {t}'.format(i=i,t=t) for i,t in enumerate(self.tiles,1))

    def slice_generator(self, block_size=1):
        def scale(sl):
            return slice(block_size * sl.start, block_size * sl.stop)
        def crop_slice(read, write):
            stop = write.stop - read.stop
            return slice(write.start - read.start, stop if stop < 0 else None)
        for t in self.tiles:
            read, write = scale(t.read_slice), scale(t.write_slice)
            yield read, write, crop_slice(read, write)

    @staticmethod
    def for_n_tiles(n, n_tiles, overlap):
        smallest_size = 2*overlap + 1
        tile_size = smallest_size # start with smallest posible tile_size
        while len(Tiling(n, tile_size, overlap)) > n_tiles:
            tile_size += 1
        if tile_size == smallest_size:
            return Tiling(n, tile_size, overlap)
        candidates = (
            Tiling(n, tile_size-1, overlap),
            Tiling(n, tile_size,   overlap),
        )
        diffs = [np.abs(len(c) - n_tiles) for c in candidates]
        return candidates[np.argmin(diffs)]



def total_n_tiles(x,n_tiles,block_sizes,n_block_overlaps,guarantee='size'):
    assert x.ndim == len(n_tiles) == len(block_sizes) == len(n_block_overlaps)
    assert guarantee in ('size', 'n_tiles')
    n_tiles_used = 1
    for n, n_tile, block_size, n_block_overlap in zip(x.shape, n_tiles, block_sizes, n_block_overlaps):
        assert n % block_size == 0
        n_blocks = n // block_size
        if guarantee == 'size':
            n_tiles_used *= len(Tiling.for_n_tiles(n_blocks, n_tile, n_block_overlap))
        elif guarantee == 'n_tiles':
            n_tiles_used *= n_tile
    return n_tiles_used



def tile_iterator_1d(x,axis,n_tiles,block_size,n_block_overlap,guarantee='size'):
    """Tile iterator for one dimension of array x.

    Parameters
    ----------
    x : numpy.ndarray
        Input array
    axis : int
        Axis which sould be tiled, all other axis not tiled
    n_tiles : int
        Targeted number of tiles for axis of x (see guarantee below)
    block_size : int
        Axis of x is assumed to be evenly divisible by block_size
        All tiles are aligned with the block_size
    n_block_overlap : int
        Tiles will overlap at least this many blocks (see guarantee below)
    guarantee : str
        Can be either 'size' or 'n_tiles':
        'size':    The size of all tiles is guaranteed to be the same,
                   but the number of tiles can be different and the
                   amount of overlap can be larger than requested.
        'n_tiles': The size of tiles can be different at the beginning and end,
                   but the number of tiles is guarantee to be the one requested.
                   The mount of overlap is also exactly as requested.


    """
    n = x.shape[axis]

    n % block_size == 0 or _raise(ValueError("'x' must be evenly divisible by 'block_size' along 'axis'"))
    n_blocks = n // block_size

    guarantee in ('size', 'n_tiles') or _raise(ValueError("guarantee must be either 'size' or 'n_tiles'"))

    if guarantee == 'size':
        tiling = Tiling.for_n_tiles(n_blocks, n_tiles, n_block_overlap)

        def ndim_slices(t):
            sl = [slice(None)] * x.ndim
            sl[axis] = t
            return tuple(sl)

        for read, write, crop in tiling.slice_generator(block_size):
            tile_in   = read  # src in input image     / tile
            tile_out  = write # dst in output image    / s_dst
            tile_crop = crop  # crop of src for output / s_src
            yield x[ndim_slices(tile_in)], ndim_slices(tile_crop), ndim_slices(tile_out)

    elif guarantee == 'n_tiles':
        n_tiles_valid = int(np.clip(n_tiles,1,n_blocks))
        if n_tiles != n_tiles_valid:
            warnings.warn("invalid value (%d) for 'n_tiles', changing to %d" % (n_tiles,n_tiles_valid))
            n_tiles = n_tiles_valid

        s = n_blocks // n_tiles # tile size
        r = n_blocks %  n_tiles # blocks remainder
        assert n_tiles * s + r == n_blocks

        # list of sizes for each tile
        tile_sizes = s*np.ones(n_tiles,int)
        # distribute remaining blocks to tiles at beginning and end
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

    else:
        assert False



def tile_iterator(x,n_tiles,block_sizes,n_block_overlaps,guarantee='size'):
    """Tile iterator for n-d arrays.

    Yields block-aligned tiles (`block_sizes`) that have at least
    a certain amount of overlapping blocks (`n_block_overlaps`)
    with their neighbors. Also yields slices that allow to map each
    tile back to the original array x.

    Notes
    -----
    - Tiles will not go beyond the array boundary (i.e. no padding).
      This means the shape of x must be evenly divisible by the respective block_size.
    - It is not guaranteed that all tiles have the same size if guarantee is not 'size'.

    Parameters
    ----------
    x : numpy.ndarray
        Input array.
    n_tiles : int or sequence of ints
        Number of tiles for each dimension of x.
    block_sizes : int or sequence of ints
        Block sizes for each dimension of x.
        The shape of x is assumed to be evenly divisible by block_sizes.
        All tiles are aligned with block_sizes.
    n_block_overlaps : int or sequence of ints
        Tiles will at least overlap this many blocks in each dimension.
    guarantee : str
        Can be either 'size' or 'n_tiles':
        'size':    The size of all tiles is guaranteed to be the same,
                   but the number of tiles can be different and the
                   amount of overlap can be larger than requested.
        'n_tiles': The size of tiles can be different at the beginning and end,
                   but the number of tiles is guarantee to be the one requested.
                   The mount of overlap is also exactly as requested.

    Example
    -------

    Duplicate an array tile-by-tile:

    >>> x = np.array(...)
    >>> y = np.empty_like(x)
    >>>
    >>> for tile,s_src,s_dst in tile_iterator(x, n_tiles, block_sizes, n_block_overlaps):
    >>>     y[s_dst] = tile[s_src]
    >>>
    >>> np.allclose(x,y)
    True

    """
    if np.isscalar(n_tiles): n_tiles = (n_tiles,)*x.ndim
    if np.isscalar(block_sizes): block_sizes = (block_sizes,)*x.ndim
    if np.isscalar(n_block_overlaps): n_block_overlaps = (n_block_overlaps,)*x.ndim

    assert x.ndim == len(n_tiles) == len(block_sizes) == len(n_block_overlaps)

    def _accumulate(tile_in,axis,src,dst):
        for tile, s_src, s_dst in tile_iterator_1d(tile_in, axis, n_tiles[axis], block_sizes[axis], n_block_overlaps[axis], guarantee):
            src[axis] = s_src[axis]
            dst[axis] = s_dst[axis]
            if axis+1 == tile_in.ndim:
                # remove None and negative slicing
                src = [slice(s.start, size if s.stop is None else (s.stop if s.stop >= 0 else size + s.stop)) for s,size in zip(src,tile.shape)]
                yield tile, tuple(src), tuple(dst)
            else:
                # yield from _accumulate(tile, axis+1, src, dst)
                for entry in  _accumulate(tile, axis+1, src, dst):
                    yield entry

    return _accumulate(x, 0, [None]*x.ndim, [None]*x.ndim)



def tile_overlap(n_depth, kern_size, pool_size=2):
    rf = {(1, 3, 1):    6, (1, 5, 1):   12, (1, 7, 1):   18,
          (2, 3, 1):   10, (2, 5, 1):   20, (2, 7, 1):   30,
          (3, 3, 1):   14, (3, 5, 1):   28, (3, 7, 1):   42,
          (4, 3, 1):   18, (4, 5, 1):   36, (4, 7, 1):   54,
          (5, 3, 1):   22, (5, 5, 1):   44, (5, 7, 1):   66,
          #
          (1, 3, 2):    9, (1, 5, 2):   17, (1, 7, 2):   25,
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
