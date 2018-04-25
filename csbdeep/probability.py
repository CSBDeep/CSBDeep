from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter


from .utils import _raise, consume
import warnings
import numpy as np


# from six import add_metaclass
# from abc import ABCMeta, abstractmethod, abstractproperty



class ProbabilisticPrediction():
    """Laplace distribution (independently per pixel)."""

    def __init__(self, mean, scale):
        self._mean = mean
        self._scale = scale

    def mean(self):
        return self._mean

    def scale(self):
        return self._scale

    def variance(self):
        return 2.0*np.square(self._scale)

    def entropy(self):
        return np.log(2*np.e*self._scale)

    def sampling_generator(self):
        while True:
            yield np.random.laplace(self._mean,self._scale)

    def credible_intervals():
        raise NotImplementedError()

    def line_plot():
        raise NotImplementedError()