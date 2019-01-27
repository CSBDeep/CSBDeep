from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Lambda
from keras.layers.merge import Multiply
from keras.activations import softmax

from .care_standard import CARE
from ..utils import _raise, axes_dict, axes_check_and_normalize
from ..internals import nets


class ProjectionCARE(CARE):
    """CARE network for combined image restoration and projection of one dimension."""

    def _get_proj_model_params(self):
        proj_axis    = vars(self.config).get('proj_axis', 'Z')
        proj_n_depth = vars(self.config).get('proj_n_depth', 2)
        proj_n_filt  = vars(self.config).get('proj_n_filt', 8)
        proj_axis    = axes_check_and_normalize(proj_axis,length=1)

        ax = axes_dict(self.config.axes)
        len(self.config.axes) == 4 or _raise(ValueError())
        self.config.axes[-1] == 'C' or _raise(ValueError())
        ax[proj_axis] is not None or _raise(ValueError())
        # proj_axis = ax[proj_axis]

        proj_kern = vars(self.config).get('proj_kern', tuple(3 if d==ax[proj_axis] else 5 for d in range(3)))
        proj_pool = vars(self.config).get('proj_pool', tuple(1 if d==ax[proj_axis] else 4 for d in range(3)))

        return proj_axis, proj_n_depth, proj_n_filt, proj_kern, proj_pool


    def _build(self):
        # get parameters
        proj_axis, proj_n_depth, proj_n_filt, proj_kern, proj_pool = self._get_proj_model_params()
        proj_axis = axes_dict(self.config.axes)[proj_axis]

        # define surface projection network (3D -> 2D)
        inp = u = Input(self.config.unet_input_shape)
        for i in range(proj_n_depth):
            u = Conv3D(proj_n_filt, proj_kern, padding='same', activation='relu')(u)
            u = MaxPooling3D(proj_pool)(u)
        for i in range(proj_n_depth):
            u = Conv3D(proj_n_filt, proj_kern, padding='same', activation='relu')(u)
            u = UpSampling3D(proj_pool)(u)
        u = Conv3D(proj_n_filt, proj_kern, padding='same', activation='relu')(u)
        u = Conv3D(1,           proj_kern, padding='same', activation='linear')(u)
        # convert learned features along Z to surface probabilities
        # (add 1 to proj_axis because of batch dimension in tensorflow)
        u = Lambda(lambda x: softmax(x, axis=1+proj_axis))(u)
        # multiply Z probabilities with Z values in input stack
        u = Multiply()([inp, u])
        # perform surface projection by summing over weighted Z values
        u = Lambda(lambda x: K.sum(x, axis=1+proj_axis))(u)
        model_projection = Model(inp, u)

        # define denoising network (2D -> 2D)
        # (remove projected axis from input_shape)
        input_shape = list(self.config.unet_input_shape)
        del input_shape[proj_axis]
        model_denoising = nets.common_unet(
            n_dim           = self.config.n_dim-1,
            n_channel_out   = self.config.n_channel_out,
            prob_out        = self.config.probabilistic,
            residual        = self.config.unet_residual,
            n_depth         = self.config.unet_n_depth,
            kern_size       = self.config.unet_kern_size,
            n_first         = self.config.unet_n_first,
            last_activation = self.config.unet_last_activation,
        )(tuple(input_shape))

        # chain models together
        return Model(inp, model_denoising(model_projection(inp)))


    def _axes_div_by(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        proj_axis, proj_n_depth, proj_n_filt, proj_kern, proj_pool = self._get_proj_model_params()
        div_by = {
            a : max(a_proj_pool**proj_n_depth, 1 if a==proj_axis else 2**self.config.unet_n_depth)
            for a,a_proj_pool in zip(self.config.axes[:-1],proj_pool)
        }
        return tuple(div_by.get(a,1) for a in query_axes)


    def train(self, X,Y, validation_data, **kwargs):
        proj_axis = self._get_proj_model_params()[0]
        proj_axis = 1+axes_dict(self.config.axes)[proj_axis]
        Y.shape[proj_axis] == 1 or _raise(ValueError())
        Y = np.take(Y,0,axis=proj_axis)
        try:
            X_val, Y_val = validation_data
            # Y_val.shape[proj_axis] == 1 or _raise(ValueError())
            validation_data = X_val, np.take(Y_val,0,axis=proj_axis)
        except:
            pass
        return super(ProjectionCARE, self).train(X,Y, validation_data, **kwargs)


    def _predict_mean_and_scale(self, img, axes, normalizer, resizer, n_tiles=None):
        (n_tiles is None or n_tiles==1 or (isinstance(n_tiles,(list,tuple)) and all(t==1 for t in n_tiles)) or
            _raise(NotImplementedError("tiled prediction not (yet) supported")))
        return super(ProjectionCARE, self)._predict_mean_and_scale(img, axes, normalizer, resizer, n_tiles)


    @property
    def _axes_out(self):
        proj_axis = self._get_proj_model_params()[0]
        return ''.join(a for a in self.config.axes if a != proj_axis)
