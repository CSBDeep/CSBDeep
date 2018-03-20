from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter

import numpy as np
# from .utils import normalize
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
#plt.ioff()

from .utils import normalize


# def plot_foo(*arr,**kwargs):

#     def max_project(a):
#         return np.max(a,axis=1) if a.ndim-2 == 3 else a

#     def color_image(a):
#         return list(map(to_color,a)) if 1 < a.shape[-1] <= 3 else a

#     arr = map(max_project,arr)
#     arr = map(color_image,arr)
#     arr = list(arr)
#     # print(arr)

#     # # max projection
#     # arr = [(np.max(a,axis=1) if a.ndim-2 == 3 else a) for a in arr]
#     # # convert to color image
#     # arr = [([to_color(_) for _ in a] if 1 < a.shape[-1] <= 3 else a) for a in arr]

#     plot_some(arr,**kwargs)


def plot_history(history,*keys,logy=False,**kwargs):
    """ TODO """

    if all(( isinstance(k,str) for k in keys )):
        w, keys = 1, [keys]
    else:
        w = len(keys)

    plt.gcf()
    for i, group in enumerate(keys):
        plt.subplot(1,w,i+1)
        for k in ([group] if isinstance(group,str) else group):
            plt.plot(history.epoch,history.history[k],'.-',label=k,**kwargs)
            if logy:
                plt.gca().set_yscale('log', nonposy='clip')
        plt.xlabel('epoch')
        plt.legend(loc='best')
    # plt.tight_layout()
    plt.show()


def plot_some(*arr, title_list=None, pmin=0, pmax=100, **imshow_kwargs):
    """
    plots a matrix of images

    arr = [ X_1, X_2, ..., X_n]

    where each X_i is a list of images

    :param arr:
    :param title_list:
    :param pmin:
    :param pmax:
    :param imshow_kwargs:
    :return:
    """



    def color_image(a):
        return np.stack(map(to_color,a)) if 1<a.shape[-1]<=3 else np.squeeze(a)
    def max_project(a):
        return np.max(a,axis=1) if (a.ndim==4 and not 1<=a.shape[-1]<=3) else a

    arr = map(color_image,arr)
    arr = map(max_project,arr)
    arr = list(arr)

    h = len(arr)
    w = len(arr[0])
    plt.gcf()
    for i in range(h):
        for j in range(w):
            plt.subplot(h, w, i * w + j + 1)
            try:
                plt.title(title_list[i][j], fontsize=8)
            except:
                pass
            img = arr[i][j]
            if pmin!=0 or pmax!=100:
                img = normalize(img,pmin=pmin,pmax=pmax,clip=True)
            plt.imshow(np.squeeze(img),**imshow_kwargs)
            plt.axis("off")


def to_color(arr, pmin=1, pmax=99.8, gamma=1., colors=((0, 1, 0), (1, 0, 1), (0, 1, 1))):
    """
    converts a 2d or 3d stack to a colored image (maximal 3 channels)

    :param arr: ndarray, 2d or 3d input data
    :param pmin: lower percentile, pass -1 if no lower normalization is required
    :param pmax: upper percentile, pass -1 if no upper normalization is required
    :param gamma: gamma correction
    :param colors: list of colors (r,g,b) for each channel of the input
    :return:
        colored image
    """
    if not arr.ndim in (2,3):
        raise ValueError("only 2d or 3d arrays supported")

    if arr.ndim ==2:
        arr = arr[np.newaxis]

    ind_min = np.argmin(arr.shape)
    arr = np.moveaxis(arr, ind_min, 0).astype(np.float32)

    out = np.zeros(arr.shape[1:] + (3,))

    eps = 1.e-20
    if pmin>=0:
        mi = np.percentile(arr, pmin, axis=(1, 2), keepdims=True)
    else:
        mi = 0

    if pmax>=0:
        ma = np.percentile(arr, pmax, axis=(1, 2), keepdims=True)
    else:
        ma = 1.+eps

    arr_norm = (1. * arr - mi) / (ma - mi + eps)


    for i_stack, col_stack in enumerate(colors):
        if i_stack >= len(arr):
            break
        for j, c in enumerate(col_stack):
            out[..., j] += c * arr_norm[i_stack]

    return np.clip(out, 0, 1)