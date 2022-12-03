import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import scipy.ndimage as filters
import sys



def q1A():
    uint_img = np.array(np.random.normal(10, 5, [100, 100]) * 255).astype('uint8')
    gray_Image = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    plt.imshow(gray_Image, cmap='gray')
    plt.title("Question 1 A")
    plt.show()


def q1B():
    count, bins, ignored = plt.hist(np.random.normal(10, 5, [100, 100]), 50, density=True)
    # plots the y=f(x) , x=bins, y=densityFuncOFGaussian
    plt.plot(bins, 1 / (5 * np.sqrt(2 * np.pi)) * np.exp(- (bins - 10) ** 2 / (2 * 5 ** 2)),
             linewidth=3, color='black')
    plt.title("Question 1 B")
    plt.show()


def q1C():
    rgb_Image = cv2.imread('lena.png', cv2.IMREAD_COLOR)[..., ::-1]
    gray_Image = cv2.cvtColor(rgb_Image, cv2.COLOR_RGB2GRAY)

    figure, (plotRgb, plotGray) = plt.subplots(1, 2)
    plotRgb.set_title("Question 1 C - RGB")
    plotGray.set_title("Question 1 C - Gray")

    plotRgb.imshow(rgb_Image)
    plotGray.imshow(gray_Image, cmap='gray')

    plt.show()


def q1D():
    rgb_Image = cv2.imread('lena.png', cv2.IMREAD_COLOR)[..., ::-1]
    gray_Image = cv2.cvtColor(rgb_Image, cv2.COLOR_RGB2GRAY)

    first_Edged = cv2.Canny(gray_Image, 0, 100)
    second_Edged = cv2.Canny(gray_Image, 100, 200)
    third_Edged = cv2.Canny(gray_Image, 200, 300)

    figure, (plot_first_Edged, plot_second_Edged, plot_third_Edged) = plt.subplots(1, 3)
    plot_first_Edged.set_title("threshold 0-100")
    plot_second_Edged.set_title("threshold 100-200")
    plot_third_Edged.set_title("threshold 200-300")

    plot_first_Edged.imshow(first_Edged, cmap='gray')
    plot_second_Edged.imshow(second_Edged, cmap='gray')
    plot_third_Edged.imshow(third_Edged, cmap='gray')

    plt.show()


def q1E():
    rgb_Image = cv2.imread('lena.png', cv2.IMREAD_COLOR)[..., ::-1]
    gray_Image = np.float32(cv2.cvtColor(rgb_Image, cv2.COLOR_RGB2GRAY))

    figure, (plotOriginal, plotCorners1, plotCorners2) = plt.subplots(1, 3)
    plotOriginal.set_title("Original")
    plotCorners1.set_title("blockSize=2, ksize=3, k=0.2")
    plotCorners2.set_title("blockSize=3, ksize=7, k=0.02")

    plotOriginal.imshow(rgb_Image, cmap='gray')

    cornersAfterK1Param = get_corners_by_k_param(rgb_Image, gray_Image, 2, 3, 0.2)
    plotCorners1.imshow(cornersAfterK1Param, cmap='gray')

    cornersAfterK2Param = get_corners_by_k_param(rgb_Image, gray_Image, 3, 7, 0.02)
    plotCorners2.imshow(cornersAfterK2Param, cmap='gray')

    plt.show()


def get_corners_by_k_param (rgb_image, gray_image, blockSize, ksize, k):
    cornersAfterKfirstParam = cv2.cornerHarris(gray_image, blockSize, ksize, k)
    cornersAfterKfirstParam = cv2.dilate(cornersAfterKfirstParam, None)
    thresh = 0.01 * cornersAfterKfirstParam.max()
    rgb_image[cornersAfterKfirstParam > thresh] = [0, 255, 0]

    return rgb_image


def log_filt(ksize, sig):
    std2 = float(sig ** 2)
    x = np.arange(-(ksize - 1) / 2, (ksize - 1) / 2 + 1, 1)
    y = np.arange(-(ksize - 1) / 2, (ksize - 1) / 2 + 1, 1)
    X, Y = np.meshgrid(x, y)
    arg = -(X * X + Y * Y) / (2 * std2);

    h = np.exp(arg);
    eps = sys.float_info.epsilon
    h[h < eps * np.max(h)] = 0;

    sumh = np.sum(h)
    if sumh != 0:
        h = h / sumh

        # now calculate Laplacian
    h1 = h * (X * X + Y * Y - 2 * std2) / (std2 ** 2);
    h = h1 - np.sum(h1) / (ksize * ksize)  # make the filter sum to zero

    return h

def q2(image_name):
    rgb_Image = cv2.imread(image_name, cv2.IMREAD_COLOR)[..., ::-1]
    gray_Image = cv2.cvtColor(rgb_Image, cv2.COLOR_RGB2GRAY)

    num_pyramids = 10
    max_min_threshold = 15
    h, w = gray_Image.shape
    sigma = 2
    k = 2 ** (0.25)

    """ create filters array """
    filters_array = []
    sigma_array = []
    filter_size_array = []
    current_sigma = sigma
    for i in range(num_pyramids):
        filter_size = 2 * np.ceil(3 * current_sigma) + 1  # filter size
        filt = log_filt(ksize=filter_size, sig=current_sigma)
        filt *= current_sigma**2
        filter_size_array.append(filter_size)
        filters_array.append(filt)
        sigma_array.append(current_sigma)
        current_sigma *= k

    """ create pyramids hXwXn , n is num of level"""
    pyramids = np.zeros((h, w, num_pyramids), dtype=float)
    for i, filt in enumerate(filters_array):
        pyramids[:, :, i] = convolve2d(in1=gray_Image, in2=filt, mode='same')

    """ find max locations """
    suppression_diameter = np.median(sigma_array)
    data_max = filters.maximum_filter(pyramids, suppression_diameter)
    maxima_mask = np.logical_and((pyramids == data_max), data_max > max_min_threshold)
    true_max_locations = np.where(maxima_mask)

    """ draw blobs with scales on color image """
    fig, ax = plt.subplots()
    ax.imshow(rgb_Image, interpolation='nearest', cmap="gray")

    for maxima_locations_x, maximum_location_y, mask_ind in zip(true_max_locations[1], true_max_locations[0],
                                                                true_max_locations[2]):
        c = plt.Circle((maxima_locations_x, maximum_location_y), int(np.ceil(sigma_array[mask_ind])), color='red', linewidth=1.5, fill=False)
        ax.add_patch(c)

    ax.plot()
    plt.show()


if __name__ == '__main__':
    q1A()
    q1B()
    q1C()
    q1D()
    q1E()
    q2('butterfly.jpg')
    q2('einstein.jpg')
    q2('fishes.jpg')
    q2('sunflowers.jpg')
    q2('balls.jpg')
    q2('lena.png')





