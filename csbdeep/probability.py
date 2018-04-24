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
        self.mean = mean
        self.scale = scale

    def expectation(self):
        return self.mean

    def variance():
        raise NotImplementedError()

    def sampling_generator():
        raise NotImplementedError()

    def entropy():
        raise NotImplementedError()

    def credible_intervals():
        raise NotImplementedError()

    def line_plot():
        raise NotImplementedError()