from time import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.cross_validation as cross #TODO replace this one to model_selection
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def estimate_freq(x):
    """Estimate the frequency for given x"""
    scores = []
    frequencies = []
    for f in np.linspace(0, 0.5, 1000):
        score = np.abs(np.dot(x, np.exp(-2 * np.pi * 1j * f * np.arange(100))))
        scores.append(score)
        frequencies.append(f)
    return frequencies[np.argmax(scores)]


def split_data(plot=False):
    """Split the sklearn load_digits dataset to 80% training, 20% test"""
    digits = load_digits()
    print('sklearn digits dataset keys:', digits.keys())
    if plot:
        plt.imshow(digits.images[0], cmap='gray')
        plt.show()
    return cross.train_test_split(digits.data, digits.target, test_size=0.2, train_size=0.8)


def gen_sample_signal(f, n, plot=False):
    """
    Generates n-long synthetic test signal from the model:
        x[i] = sin(2 * pi * f{0} * i) + w[i], i = 0...n
        w[n] ~ N(0, 0.25)
    """
    w = np.sqrt(0.25) * np.random.randn(n)
    data = [(np.sin(2*np.pi*f*i)+w[i]) for i in range(n)]
    if plot:
        plt.plot(data)
        plt.show()
    return np.array(data)


def test_estimate(n, plot=True):
    """Test the estimated signal n tiems"""
    measurements = []
    for i in range(n):
        measurements.append(estimate_freq(gen_sample_signal(f=0.017, n=100)))

    data = np.array(measurements)
    print('Mean after {:d} runs: {:.5f}'.format(n, np.mean(data)))
    print('STD after {:d} runs: {:.5f}\n'.format(n, np.std(data)))
    if plot:
        plt.scatter(y=data, x=np.arange(n))
        plt.show()


def classifier_process(**kwargs):
    """Classify dataset using scipy KNN classifier."""

    X_train, X_test, y_train, y_test = split_data(plot=True)

    clf = KNeighborsClassifier(**kwargs)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    print('Score for KNN classifier: {:.5f}'.format(score))


if __name__ == '__main__':

    print("Task one process...\n")
    
    t = time()
    samples = gen_sample_signal(f=0.017, n=100, plot=True)

    test_estimate(100, plot=True)

    classifier_process()
    print("\n...Completed in {:.4f} seconds".format(time()-t))
