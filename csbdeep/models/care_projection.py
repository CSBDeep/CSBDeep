from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from collections import namedtuple
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, Lambda
from keras.layers.merge import Multiply
from keras.activations import softmax

from .care_standard import CARE
from ..utils import _raise, axes_dict, axes_check_and_normalize
from ..internals import nets
from ..internals.predict import tile_overlap


class ProjectionCARE(CARE):
    """CARE network for combined image restoration and projection of one dimension."""

    @property
    def proj_params(self):
        try:
            return self._proj_params
        except AttributeError:
            p = {}
            p['axis']    = vars(self.config).get('proj_axis', 'Z')
            p['n_depth'] = vars(self.config).get('proj_n_depth', 2)
            p['n_filt']  = vars(self.config).get('proj_n_filt', 8)
            p['axis']    = axes_check_and_normalize(p['axis'],length=1)

            ax = axes_dict(self.config.axes)
            len(self.config.axes) == 4 or _raise(ValueError())
            self.config.axes[-1] == 'C' or _raise(ValueError())
            ax[p['axis']] is not None or _raise(ValueError())

            p['kern'] = vars(self.config).get('proj_kern', tuple(3 if d==ax[p['axis']] else 5 for d in range(3)))
            p['pool'] = vars(self.config).get('proj_pool', tuple(1 if d==ax[p['axis']] else 4 for d in range(3)))
            len(self.config.axes)-1 == len(p['pool']) == len(p['kern']) or _raise(ValueError())

            self._proj_params = namedtuple('ProjectionParameters',p.keys())(*p.values())
            return self._proj_params


    def _build(self):
        # get parameters
        proj = self.proj_params
        proj_axis = axes_dict(self.config.axes)[proj.axis]

        # define surface projection network (3D -> 2D)
        inp = u = Input(self.config.unet_input_shape)
        for i in range(proj.n_depth):
            u = Conv3D(proj.n_filt, proj.kern, padding='same', activation='relu')(u)
            u = MaxPooling3D(proj.pool)(u)
        for i in range(proj.n_depth):
            u = Conv3D(proj.n_filt, proj.kern, padding='same', activation='relu')(u)
            u = UpSampling3D(proj.pool)(u)
        u = Conv3D(proj.n_filt, proj.kern, padding='same', activation='relu')(u)
        u = Conv3D(1,           proj.kern, padding='same', activation='linear')(u)
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



    def train(self, X,Y, validation_data, **kwargs):
        proj_axis = self.proj_params.axis
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



    def _axes_div_by(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        proj = self.proj_params
        div_by = {
            a : max(a_proj_pool**proj.n_depth, 1 if a==proj.axis else 2**self.config.unet_n_depth)
            for a,a_proj_pool in zip(self.config.axes.replace('C',''),proj.pool)
        }
        return tuple(div_by.get(a,1) for a in query_axes)



    def _axes_tile_overlap(self, query_axes):
        query_axes = axes_check_and_normalize(query_axes)
        proj = self.proj_params
        unet_overlap = tile_overlap(self.config.unet_n_depth, self.config.unet_kern_size)
        overlap = {
            a : max(tile_overlap(proj.n_depth, a_proj_kern, a_proj_pool), unet_overlap) # approx
            for a,a_proj_pool,a_proj_kern in zip(self.config.axes.replace('C',''),proj.pool,proj.kern)
            if a != proj.axis
        }
        return tuple(overlap.get(a,None) for a in query_axes)



    @property
    def _axes_out(self):
        return ''.join(a for a in self.config.axes if a != self.proj_params.axis)
