from pathlib import Path

import numpy as np
import imageio
from skimage import feature, img_as_float, exposure, color
from skimage.filters import gaussian
from skimage.restoration import denoise_tv_chambolle
from skimage.transform import resize
import matplotlib.pyplot as plt

from segmentation.segment import seg_by_kmean


np.random.seed(0)


count = 24
AREA_RATE = 9 / 10
WEATHER_RATE = 9 / 10
SNOWY_FILE = '../data/weather/s1.jpg'
FOGGY_FILE = '../data/weather/f1.jpg'

def segment_markers():
    for i in range(1, count):
        input = '../data/roads/{}.jpg'.format(i)
        if Path(input).is_file():
            output = '../output/markers/{}'.format(i)
            im = imageio.imread(input)

            markers, n_1 = seg_by_kmean(im, k=15, max_iter=15)
            print(np.unique(markers))

            np.save(output, markers)


def visual_markers():
    for i in range(1, count):
        road_file = f'../data/roads/{i}.jpg'
        marker_file = f'../output/markers/{i}.npy'
        if Path(road_file).is_file() and Path(marker_file).is_file():
            output = '../output/markers/v_{}.png'.format(i)
            marker_im = np.load(marker_file).astype(float)

            road_im = imageio.imread(road_file)
            seg_im = color.label2rgb(marker_im, road_im, kind='avg')
            imageio.imwrite(output, seg_im)
            # plt.imshow(seg_im)
            # plt.show()

def img_synthesize(road_file, marker_file, weather_file, weather, output_file):
    marker_im = np.load(marker_file).astype(float)

    # array dtype bool
    edges = feature.canny(marker_im, sigma=1)
    edges_x, edges_y = np.where(edges)
    edge_ratio = 1 / 3
    edge_samples = np.sort(np.random.choice(np.arange(edges.sum()),
                            size=int(edges.sum()*edge_ratio), replace=False))

    edge_x_sample, edge_y_sample = edges_x[edge_samples], edges_y[edge_samples]
    edges_keep = np.zeros_like(edges, dtype=bool)
    edges_keep[edge_x_sample, edge_y_sample] = True
    edges = edges_keep

    w_mask = np.zeros_like(marker_im, dtype=bool)

    # uint8
    road_im = imageio.imread(road_file)
    road_im = img_as_float(road_im)

    # adjust_gamma to make image lighter or darker
    gamma = 0
    if weather == 's':
        gamma = 0.2
    elif weather == 'f':
        gamma = 0.5
    road_im_adj = exposure.adjust_gamma(road_im, gamma)
    m, n, _ = road_im.shape

    area_ratio = AREA_RATE
    weather_ratio = WEATHER_RATE

    # sample areas of image to be covered by weather condition
    samples = np.sort(np.random.choice(np.arange(m*n), size=int(m*n*area_ratio), replace=False))
    s_row = (samples / n).astype(int)
    s_col = (samples % n).astype(int)
    w_mask[s_row, s_col] = True
    w_mask[edges] = False

    # resize weather image to the same size as road image
    weat_im = imageio.imread(weather_file)
    weat_im = resize(weat_im, road_im.shape[:2])

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(road_im)

    # synthesize image
    syn_im = np.array(road_im_adj)
    syn_im[w_mask] = road_im_adj[w_mask] * (1 - weather_ratio) + weat_im[w_mask] * weather_ratio

    # smooth synthesis image
    syn_im_den = gaussian(syn_im, sigma=1.2, multichannel=True)
    # secondary smooth for foggy weather
    if weather == 'f':
        syn_im_den = denoise_tv_chambolle(syn_im_den, weight=0.2, multichannel=True)

    imageio.imwrite(output_file + '.png', syn_im_den)
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(syn_im_den)

    plt.savefig(output_file + '_plt')
    # plt.show()


def synthesize_all():
    snow_file = SNOWY_FILE
    fog_file = FOGGY_FILE

    for i in range(1, count):
        road_file = '../data/roads/{}.jpg'.format(i)
        marker_file = '../output/markers/{}.npy'.format(i)
        if Path(road_file).is_file() and Path(marker_file).is_file():
            output_file = '../output/syntheses/{}'.format(i)
            img_synthesize(road_file, marker_file, snow_file, 's', output_file + '_s')
            img_synthesize(road_file, marker_file, fog_file, 'f', output_file + '_f')


# segment_markers()
visual_markers()
synthesize_all()
