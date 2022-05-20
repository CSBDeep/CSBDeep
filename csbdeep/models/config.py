from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
from six import string_types

import numpy as np
import argparse
import warnings

from packaging.version import Version

from ..utils.tf import keras_import
keras = keras_import()
K = keras_import('backend')

from ..utils import _raise, axes_check_and_normalize, axes_dict, backend_channels_last


class BaseConfig(argparse.Namespace):

    def __init__(self, axes='YX', n_channel_in=1, n_channel_out=1, allow_new_parameters=False, **kwargs):

        # parse and check axes
        axes = axes_check_and_normalize(axes)
        ax = axes_dict(axes)
        ax = {a: (ax[a] is not None) for a in ax}

        (ax['X'] and ax['Y']) or _raise(ValueError('lateral axes X and Y must be present.'))
        # not (ax['Z'] and ax['T']) or _raise(ValueError('using Z and T axes together not supported.'))

        axes.startswith('S') or (not ax['S']) or _raise(ValueError('sample axis S must be first.'))
        axes = axes.replace('S','') # remove sample axis if it exists

        n_dim = len(axes.replace('C',''))

        # TODO: Config not independent of backend. Problem?
        # could move things around during train/predict as an alternative... good idea?
        # otherwise, users can choose axes of input image anyhow, so doesn't matter if model is fixed to something else
        if backend_channels_last():
            if ax['C']:
                axes[-1] == 'C' or _raise(ValueError('channel axis must be last for backend (%s).' % K.backend()))
            else:
                axes += 'C'
        else:
            if ax['C']:
                axes[0] == 'C' or _raise(ValueError('channel axis must be first for backend (%s).' % K.backend()))
            else:
                axes = 'C'+axes

        self.n_dim                  = n_dim
        self.axes                   = axes
        self.n_channel_in           = int(max(1,n_channel_in))
        self.n_channel_out          = int(max(1,n_channel_out))

        self.train_checkpoint       = 'weights_best.h5'
        self.train_checkpoint_last  = 'weights_last.h5'
        self.train_checkpoint_epoch = 'weights_now.h5'

        self.update_parameters(allow_new_parameters, **kwargs)


    def is_valid(self, return_invalid=False):
        return (True, tuple()) if return_invalid else True


    def update_parameters(self, allow_new=False, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])

    @classmethod
    def update_loaded_config(cls, config):
        """Called by model to update loaded config dictionary before config object is created

        Can be used to modify or introduce/delete parameters, e.g. to ensure
        backwards compatibility after new parameters have been introduced.

        Parameters
        ----------
        config : dict
            dictionary of config parameters loaded from file

        Returns
        -------
        updated_config: dict
            an updated version of the config parameter dictionary
        """
        return config



class Config(BaseConfig):
    """Default configuration for a CARE model.

    This configuration is meant to be used with :class:`CARE`
    and related models (e.g., :class:`IsotropicCARE`).

    Parameters
    ----------
    axes : str
        Axes of the neural network (channel axis optional).
    n_channel_in : int
        Number of channels of given input image.
    n_channel_out : int
        Number of channels of predicted output image.
    probabilistic : bool
        Probabilistic prediction of per-pixel Laplace distributions or
        typical regression of per-pixel scalar values.
    allow_new_parameters : bool
        Allow adding new configuration attributes (i.e. not listed below).
    kwargs : dict
        Overwrite (or add) configuration attributes (see below).

    Example
    -------
    >>> config = Config('YX', probabilistic=True, unet_n_depth=3)

    Attributes
    ----------
    n_dim : int
        Dimensionality of input images (2 or 3).
    unet_residual : bool
        Parameter `residual` of :func:`csbdeep.nets.common_unet`. Default: ``n_channel_in == n_channel_out``
    unet_n_depth : int
        Parameter `n_depth` of :func:`csbdeep.nets.common_unet`. Default: ``2``
    unet_kern_size : int
        Parameter `kern_size` of :func:`csbdeep.nets.common_unet`. Default: ``5 if n_dim==2 else 3``
    unet_n_first : int
        Parameter `n_first` of :func:`csbdeep.nets.common_unet`. Default: ``32``
    unet_last_activation : str
        Parameter `last_activation` of :func:`csbdeep.nets.common_unet`. Default: ``linear``
    train_loss : str
        Name of training loss. Default: ``'laplace' if probabilistic else 'mae'``
    train_epochs : int
        Number of training epochs. Default: ``100``
    train_steps_per_epoch : int
        Number of parameter update steps per epoch. Default: ``400``
    train_learning_rate : float
        Learning rate for training. Default: ``0.0004``
    train_batch_size : int
        Batch size for training. Default: ``16``
    train_tensorboard : bool
        Enable TensorBoard for monitoring training progress. Default: ``True``
    train_checkpoint : str
        Name of checkpoint file for model weights (only best are saved); set to ``None`` to disable. Default: ``weights_best.h5``
    train_reduce_lr : dict
        Parameter :class:`dict` of ReduceLROnPlateau_ callback; set to ``None`` to disable. Default: ``{'factor': 0.5, 'patience': 10, 'min_delta': 0}``

        .. _ReduceLROnPlateau: https://keras.io/callbacks/#reducelronplateau
    """

    def __init__(self, axes='YX', n_channel_in=1, n_channel_out=1, probabilistic=False, allow_new_parameters=False, **kwargs):
        """See class docstring."""

        super(Config, self).__init__(axes, n_channel_in, n_channel_out)
        not ('Z' in self.axes and 'T' in self.axes) or _raise(ValueError('using Z and T axes together not supported.'))

        self.probabilistic         = bool(probabilistic)

        # default config (can be overwritten by kwargs below)
        self.unet_residual         = self.n_channel_in == self.n_channel_out
        self.unet_n_depth          = 2
        self.unet_kern_size        = 5 if self.n_dim==2 else 3
        self.unet_n_first          = 32
        self.unet_last_activation  = 'linear'
        if backend_channels_last():
            self.unet_input_shape  = self.n_dim*(None,) + (self.n_channel_in,)
        else:
            self.unet_input_shape  = (self.n_channel_in,) + self.n_dim*(None,)

        self.train_loss            = 'laplace' if self.probabilistic else 'mae'
        self.train_epochs          = 100
        self.train_steps_per_epoch = 400
        self.train_learning_rate   = 0.0004
        self.train_batch_size      = 16
        self.train_tensorboard     = True

        # the parameter 'min_delta' was called 'epsilon' for keras<=2.1.5
        min_delta_key = 'epsilon' if Version(keras.__version__)<=Version('2.1.5') else 'min_delta'
        self.train_reduce_lr       = {'factor': 0.5, 'patience': 10, min_delta_key: 0}

        # disallow setting 'n_dim' manually
        try:
            del kwargs['n_dim']
            # warnings.warn("ignoring parameter 'n_dim'")
        except:
            pass

        self.update_parameters(allow_new_parameters, **kwargs)


    def is_valid(self, return_invalid=False):
        """Check if configuration is valid.

        Returns
        -------
        bool
            Flag that indicates whether the current configuration values are valid.
        """
        def _is_int(v,low=None,high=None):
            return (
                isinstance(v,int) and
                (True if low is None else low <= v) and
                (True if high is None else v <= high)
            )

        ok = {}
        ok['n_dim'] = self.n_dim in (2,3)
        try:
            axes_check_and_normalize(self.axes,self.n_dim+1,disallowed='S')
            ok['axes'] = True
        except:
            ok['axes'] = False
        ok['n_channel_in']  = _is_int(self.n_channel_in,1)
        ok['n_channel_out'] = _is_int(self.n_channel_out,1)
        ok['probabilistic'] = isinstance(self.probabilistic,bool)

        ok['unet_residual'] = (
            isinstance(self.unet_residual,bool) and
            (not self.unet_residual or (self.n_channel_in==self.n_channel_out))
        )
        ok['unet_n_depth']         = _is_int(self.unet_n_depth,1)
        ok['unet_kern_size']       = _is_int(self.unet_kern_size,1)
        ok['unet_n_first']         = _is_int(self.unet_n_first,1)
        ok['unet_last_activation'] = self.unet_last_activation in ('linear','relu')
        ok['unet_input_shape'] = (
                isinstance(self.unet_input_shape,(list,tuple))
            and len(self.unet_input_shape) == self.n_dim+1
            and self.unet_input_shape[-1] == self.n_channel_in
            # and all((d is None or (_is_int(d) and d%(2**self.unet_n_depth)==0) for d in self.unet_input_shape[:-1]))
        )
        ok['train_loss'] = (
            (    self.probabilistic and self.train_loss == 'laplace'   ) or
            (not self.probabilistic and self.train_loss in ('mse','mae'))
        )
        ok['train_epochs']          = _is_int(self.train_epochs,1)
        ok['train_steps_per_epoch'] = _is_int(self.train_steps_per_epoch,1)
        ok['train_learning_rate']   = np.isscalar(self.train_learning_rate) and self.train_learning_rate > 0
        ok['train_batch_size']      = _is_int(self.train_batch_size,1)
        ok['train_tensorboard']     = isinstance(self.train_tensorboard,bool)
        ok['train_checkpoint']      = self.train_checkpoint is None or isinstance(self.train_checkpoint,string_types)
        ok['train_reduce_lr']       = self.train_reduce_lr  is None or isinstance(self.train_reduce_lr,dict)

        if return_invalid:
            return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
        else:
            return all(ok.values())
