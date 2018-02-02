import math

import numpy as np


def gaussian(x, mu, sigma):

    return ((1/np.sqrt(2*np.pi*np.power(sigma, 2))) * np.exp((((-1/(2*math.pow(sigma, 2))*(np.power(x-mu,2)))))))


def log_gaussian(x, mu, sigma):

    return np.log(gaussian(x, mu, sigma))
