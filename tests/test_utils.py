from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

# import warnings
import numpy as np
import pytest
from csbdeep.data import NoNormalizer, PercentileNormalizer, NoResizer, PadAndCropResizer
from csbdeep.utils import normalize_minmse
from csbdeep.internals.predict import tile_iterator_1d, tile_iterator, total_n_tiles
from csbdeep.internals.train import RollingSequence



@pytest.mark.parametrize('axes', ('ZYX', 'XY', 'TXYC'))
def test_resizer(axes):
    rng = np.random.RandomState(42)

    resizer = PadAndCropResizer()
    checker = NoResizer()

    for _ in range(50):

        imdims = list(rng.randint(20,40,size=len(axes)))
        div_by = list(rng.randint(1,20,size=len(axes)))

        u = np.empty(imdims,np.float32)
        if any(s%div_n!=0 for s, div_n in zip(imdims, div_by)):
            with pytest.raises(ValueError):
                checker.before(u, axes, div_by)

        v = resizer.before(u, axes, div_by)
        assert all (
            s_v >= s_u and s_v%div_n==0
            for s_u, s_v, div_n in zip(u.shape, v.shape, div_by)
        )

        w = resizer.after(v, axes)
        assert u.shape == w.shape

        d = rng.choice(len(axes))
        _axes = axes.replace(axes[d],'')
        _u = np.take(u,0,axis=d)
        _v = np.take(v,0,axis=d)
        _w = resizer.after(_v, _axes)
        assert _u.shape == _w.shape



@pytest.mark.parametrize('axes', ('CZYX', 'ZYX', 'XY', 'XCY', 'TXYC'))
def test_normalizer(axes):
    rng = np.random.RandomState(42)

    no_normalizer = NoNormalizer(do_after=False)
    paxis = tuple(d for d,a in enumerate(axes) if a != 'C')
    def _percentile(x,p):
        return np.percentile(x,p,axis=paxis,keepdims=True)

    for _ in range(50):
        pmin = rng.uniform(0,50)
        pmax = rng.uniform(pmin+1,100)
        normalizer = PercentileNormalizer(pmin, pmax, do_after=True)

        imdims = list(rng.randint(10,20,size=len(axes)))
        u = rng.uniform(0,10000,size=imdims).astype(np.float32,copy=False)
        u_pmin, u_pmax = _percentile(u,pmin), _percentile(u,pmax)

        assert np.allclose(u, no_normalizer.before(u, axes))
        with pytest.raises(ValueError):
            no_normalizer.after(u, u, axes)

        v = normalizer.before(u, axes)
        v_pmin, v_pmax = _percentile(v,pmin), _percentile(v,pmax)
        assert np.mean(np.abs(v_pmin-0)) < 1e-5 and np.mean(np.abs(v_pmax-1)) < 1e-5

        w = normalizer.after(v, None, axes)[0]
        w_pmin, w_pmax = _percentile(w,pmin), _percentile(w,pmax)
        assert np.allclose(u_pmin,w_pmin) and np.allclose(u_pmax,w_pmax)



def test_normalize_minmse():
    rng = np.random.RandomState(42)
    for _ in range(50):
        target = rng.uniform(-100,100,size=(32,32,32))
        x = rng.uniform(-500,500)*target + rng.uniform(-500,500)
        assert np.allclose(normalize_minmse(x,target),target)
        x, target = x.astype(np.float32), target.astype(np.float32)
        assert np.max(np.abs(normalize_minmse(x,target)-target)) < 1e-3



@pytest.mark.parametrize('guarantee', ('size', 'n_tiles'))
def test_tile_iterator_1d(guarantee):
    rng = np.random.RandomState(42)
    for _ in range(50):
        n = rng.randint(low=10,high=500)
        block_size = rng.randint(low=1,high=(n-n//3))
        n = block_size * (n // block_size)
        n_blocks = n // block_size
        n_block_overlap = rng.randint(low=0,high=n_blocks+1)
        n_tiles = rng.randint(low=1,high=n_blocks+1)

        x = rng.uniform(size=n)
        y = np.empty_like(x)
        c = 0
        tile_shape = None
        actual_n_tiles = total_n_tiles(x,[n_tiles],[block_size],[n_block_overlap],guarantee=guarantee)
        for tile,s_src,s_dst in tile_iterator_1d(x,0,n_tiles=n_tiles,block_size=block_size,n_block_overlap=n_block_overlap,guarantee=guarantee):
            y[s_dst] = tile[s_src]
            assert tile.shape[0] % block_size == 0
            assert tile[s_src].shape[0] % block_size == 0
            if guarantee == 'size':
                if tile_shape is None: tile_shape = tile.shape
                assert tile_shape == tile.shape
            # TODO: good way to test overlap size?
            c += 1

        assert c == actual_n_tiles
        assert np.allclose(x,y)



@pytest.mark.parametrize('n_dims', (1,2,3))
@pytest.mark.parametrize('guarantee', ('size', 'n_tiles'))
def test_tile_iterator(guarantee, n_dims):
    rng = np.random.RandomState(42)
    for _ in range(10):
        n = rng.randint(low=10,high=300,size=n_dims)
        n_blocks = list(rng.randint(low=1,high=10,size=n_dims))
        block_size = [_n // _n_blocks for _n_blocks,_n in zip(n_blocks,n)]
        n = [_block_size * (_n // _block_size) for _block_size,_n in zip(block_size,n)]
        n_block_overlap = [rng.randint(low=0,high=_n_blocks+1) for _n_blocks in n_blocks]
        n_tiles = [rng.randint(low=1,high=_n_blocks+1) for _n_blocks in n_blocks]

        x = rng.uniform(size=n)
        y = np.empty_like(x)
        c = 0
        actual_n_tiles = total_n_tiles(x,n_tiles,block_size,n_block_overlap,guarantee=guarantee)
        for tile,s_src,s_dst in tile_iterator(x,n_tiles,block_size,n_block_overlap,guarantee):
            y[s_dst] = tile[s_src]
            c += 1

        assert c == actual_n_tiles
        assert np.allclose(x,y)



def test_rolling_sequence():
    rng = np.random.RandomState(42)
    for shuffle in (False,True):
        for data_size in (5,60,123):
            for batch_size in (3,7,32):
                seq = RollingSequence(data_size, batch_size, shuffle=shuffle, rng=rng)

                n_batches = 3 * int(np.ceil(data_size/float(batch_size)))
                perm = np.random.permutation(n_batches)
                batches_perm = [seq[i] for i in perm]
                batches_linear = [seq[i] for i in np.arange(n_batches)]
                assert all(np.all(batches_perm[i]==batches_linear[j]) for i,j in enumerate(perm))

                res = np.concatenate(batches_linear)
                ref = np.concatenate([seq.index_map[i] for i in sorted(seq.index_map.keys())])
                assert np.all(ref[:len(res)] == res)

                counts = np.unique(ref, return_counts=True)[1]
                assert all(counts[0] == c for c in counts)
