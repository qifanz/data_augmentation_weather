import numpy as np
import imageio
from skimage import color

import logging

from segmentation.segment import seg_by_kmean, seg_by_kmean_threshold, seg_by_edge_watershred

logging.getLogger().setLevel(logging.INFO)


from matplotlib import pyplot as plt

path = '../images/e.jpg'
im = imageio.imread(path)

# set 10 clusters at most, 12 iterations
markers_1, n_1 = seg_by_kmean(im, k=12, max_iter=15)
# plot segment regions, plot every region with mean pixel
out_1 = color.label2rgb(markers_1, im, kind='avg')
logging.info(np.unique(markers_1))
plt.imshow(out_1)
plt.show()

# set 40 clusters for kmean, and combine clusters with edges gradient value below 10
markers_2, n_2 = seg_by_kmean_threshold(im, k=30, edge_thresh=20)
out_2 = color.label2rgb(markers_2, im, kind='avg')
logging.info(np.unique(markers_2))
plt.imshow(out_2)
plt.show()


# gradient value below 60 will not be considered edge for segmentation
markers_3, n_3 = seg_by_edge_watershred(im, lowest_gradient=80)
out_3 = color.label2rgb(markers_3, im, kind='avg')
logging.info(np.unique(markers_3))
plt.imshow(out_3)
plt.show()

fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(im)
axes[0, 0].set_title('original')

axes[0, 1].imshow(out_1)
axes[0, 1].set_title('kmean(k=12, max_iter=15)')

axes[1, 0].imshow(out_2)
axes[1, 0].set_title('kmean & threshold (k=30, edge_thresh=20)')

axes[1, 1].imshow(out_3)
axes[1, 1].set_title('edge & water threshold (lowest_gradient=80)')
plt.show()
