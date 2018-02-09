import math
import os

import numpy as np
from scipy.misc import imread
from skimage.feature import local_binary_pattern


def gaussian(x, mu, sigma):
    """Calculates the gaussian normal distribution function for given numpy array x"""
    return ((1 / np.sqrt(2 * np.pi * np.power(sigma, 2))) * np.exp((((-((np.power((x - mu), 2))/(2 * np.power(sigma, 2))))))))


def log_gaussian(x, mu, sigma):
    """Calculates the logarithmic gaussian normal for given numpy array x"""
    return (1 / (sigma * x * np.sqrt(2 * np.pi))) * np.exp((-(np.power((np.log(x) - mu), 2) / (2 * np.power(sigma, 2)))))


def extract_features(paths):
    """Extract the local binary pattern from images"""
    radius = 1
    n_points = 3 * radius

    data = []
    labels = []
    for n, path in enumerate(paths):
        for file in os.listdir(path):
            lbp = local_binary_pattern(imread(path + file), n_points, radius)
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp, bins = n_bins, range=(0,n_bins))
            data.append(hist)
            labels.append(n)

    return np.array(data), np.array(labels)
