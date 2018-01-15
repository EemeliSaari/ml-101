import numpy as np
import matplotlib.pyplot as plt

from scipy import misc
from scipy.io import loadmat


def load_data(file):

    with open(file, 'r') as f:
        data = [np.fromstring(x.rstrip(), dtype=np.float64, sep=" ") for x in f.readlines()]
    return np.array(data)


def plot_mat(file):

    mat = loadmat(file)
    X = mat["X"]
    y = mat["y"].ravel()

    class_0 = X[y == 0, :]
    class_1 = X[y == 1, :]

    plt.scatter(class_0[:,0], class_0[:,1], c='red')
    plt.scatter(class_1[:,0], class_1[:,1], c='blue')
    plt.show()


def fix_image(file):

    img = misc.imread(file)

    plt.imshow(img, cmap='gray')
    plt.title('Image shape is {:d}x{:d}'.format(img.shape[1], img.shape[0]))
    plt.show()
    # Create the X-Y coordinate pairs in a matrix
    X, Y = np.meshgrid(range(1300), range(1030))
    Z = img

    x = X.ravel()
    y = Y.ravel()
    z = Z.ravel()

    # Create data matrix
    H = np.column_stack((x*x, y*y, x*y, x, y, np.ones(x.shape)))
    # Solve coefficients
    c = np.dot(np.dot(np.linalg.inv(np.dot(H.T, H)), H.T), z)

    # Predict  
    z_pred = np.dot(H, c)
    Z_pred = np.reshape(z_pred, X.shape)
    plt.imshow(Z_pred, cmap='gray')
    plt.show()
    # Subtract & show
    S = Z - Z_pred
    plt.imshow(S, cmap='gray')
    plt.show()


def main():

    x = np.load("x.npy")
    y = np.load("y.npy")
    a, b = np.polyfit(x,y, 1)
    print("a={}\nb={}\n".format(a,b))

    file_name = "locationData.csv"
    data1 = np.loadtxt(file_name)
    data2 = load_data(file_name)

    if np.array_equal(data1, data2):
        print("Datas are equal")

    plot_mat("twoClassData.mat")
    fix_image("uneven_illumination.jpg")

if __name__ == '__main__':
    main()