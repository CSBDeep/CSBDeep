from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

from ..utils import _raise, move_channel_for_backend, axes_dict, axes_check_and_normalize, backend_channels_last
from ..internals.losses import loss_laplace, loss_mse, loss_mae, loss_thresh_weighted_decay

import numpy as np

from ..utils.tf import keras_import
K = keras_import('backend')
Callback, TerminateOnNaN = keras_import('callbacks', 'Callback', 'TerminateOnNaN')
Sequence = keras_import('utils', 'Sequence')
Optimizer = keras_import('optimizers', 'Optimizer')


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


class RollingSequence(Sequence):
    """Helper class for creating batches for rolling sequence.

    Create batches of size `batch_size` that contain indices in `range(data_size)`.
    To that end, the data indices are repeated (rolling), either in ascending order or
    shuffled if `shuffle=True`. If taking batches sequentially, all data indices will
    appear equally often. All calls to `batch(i)` will return the same batch for same i.
    Parameter `length` will only determine the result of `len`, it has no effect otherwise.
    Note that batch_size is allowed to be larger than data_size.
    """

    def __init__(self, data_size, batch_size, length=None, shuffle=True, rng=None):
        # print(f"### __init__", flush=True)
        if rng is None: rng = np.random
        self.data_size = int(data_size)
        self.batch_size = int(batch_size)
        self.length = 2**63-1 if length is None else int(length) # 2**63-1 is max possible value
        self.shuffle = bool(shuffle)
        self.index_gen = rng.permutation if self.shuffle else np.arange
        self.index_map = {}

    def __len__(self):
        # print(f"### __len__ = {self.length}", flush=True)
        return self.length

    def _index(self, loop):
        if loop in self.index_map:
            return self.index_map[loop]
        else:
            return self.index_map.setdefault(loop, self.index_gen(self.data_size))

    def on_epoch_end(self):
        # print(f"### on_epoch_end", flush=True)
        pass

    def __iter__(self):
        # print(f"### __iter__", flush=True)
        for i in range(len(self)):
            yield self[i]

    def batch(self, i):
        pos      =   i *  self.batch_size
        loop     = pos // self.data_size
        pos_loop = pos %  self.data_size
        sl = slice(pos_loop, pos_loop + self.batch_size)
        index = self._index(loop)
        _loop = loop
        while sl.stop > len(index):
            _loop += 1
            index = np.concatenate((index, self._index(_loop)))
        # print(f"### - batch({i:02}) -> {tuple(index[sl])}", flush=True)
        return index[sl]

    def __getitem__(self, i):
        return self.batch(i)


class DataWrapper(RollingSequence):

    def __init__(self, X, Y, batch_size, length):
        super(DataWrapper, self).__init__(data_size=len(X), batch_size=batch_size, length=length, shuffle=True)
        len(X) == len(Y) or _raise(ValueError("X and Y must have same length"))
        self.X, self.Y = X, Y

    def __getitem__(self, i):
        idx = self.batch(i)
        return self.X[idx], self.Y[idx]
