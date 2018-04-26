from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter


from .utils import _raise, consume
import warnings
import numpy as np
from scipy.stats import laplace


class ProbabilisticPrediction(object):
    """Laplace distribution (independently per pixel)."""

    def __init__(self, loc, scale):
        loc.shape == scale.shape or _raise(ValueError())
        #
        self._loc     = loc
        self._scale   = scale
        self._shape   = loc.shape
        # expose methods from laplace object
        _laplace      = self[...]
        self.rvs      = _laplace.rvs
        self.pdf      = _laplace.pdf
        self.logpdf   = _laplace.logpdf
        self.cdf      = _laplace.cdf
        self.logcdf   = _laplace.logcdf
        self.sf       = _laplace.sf
        self.logsf    = _laplace.logsf
        self.ppf      = _laplace.ppf
        self.isf      = _laplace.isf
        self.moment   = _laplace.moment
        self.stats    = _laplace.stats
        self.entropy  = _laplace.entropy
        self.expect   = _laplace.expect
        self.median   = _laplace.median
        self.mean     = _laplace.mean
        self.var      = _laplace.var
        self.std      = _laplace.std
        self.interval = _laplace.interval

    def __getitem__(self, indices):
        return laplace(loc=self._loc[indices],scale=self._scale[indices])

    @property
    def shape(self):
        return self._shape

    def scale(self):
        return self._scale

    def sampling_generator(self,n=None):
        if n is None:
            while True:
                yield self.rvs()
        else:
            for i in range(n):
                yield self.rvs()
