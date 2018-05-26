from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, move_channel_for_backend, axes_dict, axes_check_and_normalize, backend_channels_last
from ..internals.losses import loss_laplace, loss_mse, loss_mae, loss_thresh_weighted_decay

import numpy as np


import keras.backend as K
from keras.callbacks import Callback, TerminateOnNaN
from keras.utils import Sequence


class ParameterDecayCallback(Callback):
    """ TODO """
    def __init__(self, parameter, decay, name=None, verbose=0):
        self.parameter = parameter
        self.decay = decay
        self.name = name
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        old_val = K.get_value(self.parameter)
        if self.name:
            logs = logs or {}
            logs[self.name] = old_val
        new_val = old_val * (1. / (1. + self.decay * (epoch + 1)))
        K.set_value(self.parameter, new_val)
        if self.verbose:
            print("\n[ParameterDecayCallback] new %s: %s\n" % (self.name if self.name else 'parameter', new_val))


def prepare_model(model, optimizer, loss, metrics=('mse','mae'),
                  loss_bg_thresh=0, loss_bg_decay=0.06, Y=None):
    """ TODO """

    from keras.optimizers import Optimizer
    isinstance(optimizer,Optimizer) or _raise(ValueError())


    loss_standard   = eval('loss_%s()'%loss)
    _metrics        = [eval('loss_%s()'%m) for m in metrics]
    callbacks       = [TerminateOnNaN()]

    # checks
    assert 0 <= loss_bg_thresh <= 1
    assert loss_bg_thresh == 0 or Y is not None
    if loss == 'laplace':
        assert K.image_data_format() == "channels_last", "TODO"
        assert model.output.shape.as_list()[-1] >= 2 and model.output.shape.as_list()[-1] % 2 == 0

    # loss
    if loss_bg_thresh == 0:
        _loss = loss_standard
    else:
        freq = np.mean(Y > loss_bg_thresh)
        # print("class frequency:", freq)
        alpha = K.variable(1.0)
        loss_per_pixel = eval('loss_{loss}(mean=False)'.format(loss=loss))
        _loss = loss_thresh_weighted_decay(loss_per_pixel, loss_bg_thresh,
                                           0.5 / (0.1 + (1 - freq)),
                                           0.5 / (0.1 +      freq),
                                           alpha)
        callbacks.append(ParameterDecayCallback(alpha, loss_bg_decay, name='alpha'))
        if not loss in metrics:
            _metrics.append(loss_standard)


    # compile model
    model.compile(optimizer=optimizer, loss=_loss, metrics=_metrics)

    return callbacks


class DataWrapper(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        return self.X[idx], self.Y[idx]
