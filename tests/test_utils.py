from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

# import warnings
import numpy as np
import pytest
from csbdeep.data import NoNormalizer, PercentileNormalizer, NoResizer, PadAndCropResizer
# from csbdeep.utils import Path, axes_dict, move_image_axes, backend_channels_last



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
