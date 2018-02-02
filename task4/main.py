import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
from skimage.feature import local_binary_pattern
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from utils import gaussian, log_gaussian

warnings.filterwarnings('ignore', category=UserWarning)


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


def test_classifiers(X, y):
    """Test different classifiers"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, test_size=.2)

    models = [KNeighborsClassifier(), LinearDiscriminantAnalysis(), SVC()]
    model_names = ['KNN', 'LDA', 'SVC']

    for n, model in enumerate(models):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        print(model_names[n], 'score:', score)

def test_utils():

    x = np.linspace(-5, 5)
    plt.plot(gaussian(x, 0, 1))
    plt.plot(log_gaussian(x, 0, 1))
    plt.show()

# Entrypoint
if __name__ == '__main__':

    paths = ['resources/class1/', 'resources/class2/']
    labels = ['first order', 'speed sign']

    X, y = extract_features(paths)

    test_classifiers(X, y)

    test_utils()