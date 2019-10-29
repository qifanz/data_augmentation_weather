import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import ndimage as ndi
import numpy as np
import cv2 as cv
import imageio
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage import data, img_as_float, segmentation, color, img_as_ubyte
from skimage.filters import laplace, rank
from skimage.future import graph
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
from skimage.morphology import square, dilation, disk, watershed

import logging
logging.getLogger().setLevel(logging.INFO)

"""
segmentation functions return labels starting from 1
"""


DENOISE_WEIGHT = 0.1

# denoise + kmean
def seg_by_kmean(im, k=10, max_iter=10):
    """
    :param im: file path to input image, or image m*n*3 ndarray
    :param k: number of clusters at most
    :param max_iter: number of iterations before kmean converges
    :return: labels, n
        labels: a mask marks different clusters with 0, 1, ..., n-1.
              for a cluster marked as i, can use im[mask == i]
        n: number of clusters. starting from 1.
    """
    if isinstance(im, str):
        im = imageio.imread(im)

    im = denoise(im, denoising_weight=DENOISE_WEIGHT, multichannel=True)
    labels = segmentation.slic(im, n_segments=k, max_iter=max_iter)
    # mask labels starting from 1
    labels = labels + 1

    return labels, np.unique(labels)


# kmean + threshold
def seg_by_kmean_threshold(im, k=40, edge_thresh=15):
    """
    :param im: file path to input image, or image m*n*3 ndarray
    :param k: number of clusters for kmean
    :param cuts: number of regions merged from k clusters
    :return: labels, n
        labels: a mask marks different clusters with 0, 1, ..., n-1.
              for a cluster marked as i, can use im[mask == i]
        n: number of clusters. starting from 1.
    """


    labels1 = segmentation.slic(im, compactness=10, n_segments=k)
    # out1 = color.label2rgb(labels1, im, kind='avg')

    g = graph.rag_mean_color(im, labels1)
    labels2 = graph.cut_threshold(labels1, g, edge_thresh)
    # labels value staring from 1
    labels2 = labels2 + 1

    # out2 = color.label2rgb(labels2, im, kind='avg')
    # logging.info(np.unique(labels2))

    return labels2, np.unique(labels2)


# denoise + filter significant edges + water threshold
def seg_by_edge_watershred(im, lowest_gradient=50):
    """
    :param im: file path to input image, or image m*n*3 ndarray
    :param lowest_gradient: gradients below this values not considered as edges
    :return: labels, n
        labels: a mask marks different clusters with 0, 1, ..., n-1.
              for a cluster marked as i, can use im[mask == i]
        n: number of clusters. starting from 1.
    """

    if isinstance(im, str):
        im = imageio.imread(im)

    # denoise rgb image
    im = denoise(im)

    gray_in = color.rgb2gray(im)
    gray = img_as_ubyte(gray_in)

    # denoise gray image
    denoised = rank.median(gray, disk(5))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) > lowest_gradient
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))

    # process the watershed
    labels = watershed(gradient, markers)

    return labels, np.unique(labels)


def denoise(im, denoising_weight=DENOISE_WEIGHT, multichannel=True):
    im = img_as_float(im)
    im = denoise_tv_chambolle(im, weight=denoising_weight, multichannel=multichannel)
    return im
