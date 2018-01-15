import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Need to add to make projection = "3d" work
import numpy as np


def normalize_data(data):

    normalized = np.zeros(np.shape(data))
    for i in range(np.shape(data)[1]):
        normalized[:, i:i+1] = (data[0:, i:i+1] - np.mean(data[0:, i:i+1])) / np.std(data[0:, i:i+1])
    return normalized


def plot_data_3d(x, y, z):

    ax = plt.subplot(1, 1, 1, projection = "3d")
    plt.plot(x, y, z)
    plt.show()


def plot_data_2d(x, y):

    plt.plot(x, y)
    plt.show()


def main():

    data = np.loadtxt("locationData.csv")
    print("{:d} x {:d}\n".format(np.shape(data)[0],np.shape(data)[1]))

    plot_data_2d(data[0:, 0], data[0:, 1])
    plot_data_3d(data[0:, 0], data[0:, 1], data[0:, 2])

    X_norm = normalize_data(data)

    print(np.mean(X_norm, axis=0), "\n")
    print(np.std(X_norm, axis=0), "\n")


if __name__ == "__main__":
    main()
