import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.svm import SVC

sys.path.append('..')

from log_loss_template import test_log_loss
from task4.utils import extract_features

style.use('ggplot')

def get_data():
    """Loads the data for GTSRB"""
    paths = ['resources/class1/', 'resources/class2/']

    X, y = extract_features(paths)
    return train_test_split(X, y, train_size=.8, test_size=.2)


def select_params():
    """Tests the classifiers for best params for penalty and C"""
    X_train, X_test, y_train, y_test = get_data()
    
    clf_list = [LogisticRegression(), SVC()]
    clf_name = ['LR', 'SVC']

    c_size = 100

    data = {}

    for clf, name in zip(clf_list, clf_name):

        c_range = np.linspace(1, 0.00001, num=c_size, endpoint=False,dtype=np.float)
        meta = {}
        meta['c'] = c_range        
        for penalty in ['l1', 'l2']:
            meta[penalty] = []
            X_train_n = normalize(X_train, norm=penalty)
            X_test_n = normalize(X_test, norm=penalty)
            for c in c_range:
                clf.C = c
                clf.penalty = penalty
                clf.fit(X_train_n, y_train)
                y_pred = clf.predict(X_test_n)
                score = accuracy_score(y_test, y_pred)
                meta[penalty].append(score)

        data[name] = meta
    
    # Plots all the data
    for key in data.keys():
        
        x = data[key]['c']
        y1 = data[key]['l1']
        y2 = data[key]['l2']

        plt.plot(x,y1, label='{:s}-{:s}'.format(key, 'l1'))
        plt.plot(x,y2, label='{:s}-{:s}'.format(key, 'l2'))
    
    plt.ylim(ymax=1, ymin=0)
    plt.xlim(xmax=1, xmin=0)
    
    plt.yticks(rotation=90)
    
    plt.xlabel('c')
    plt.ylabel('acc')

    plt.title('C test size: {:d}'.format(c_size))

    plt.legend(loc='best')
    plt.show()
    

def test_RFC():
    """Test different ensembled classifiers"""
    X_train, X_test, y_train, y_test = get_data()

    clf_list = [RandomForestClassifier(), AdaBoostClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier()]
    clf_names = ['RFC', 'ABC', 'ETC', 'GBC']

    data = {}
    for clf, name in zip(clf_list, clf_names):

        clf.n_estimator = 100

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        data[name] = accuracy_score(y_test, y_pred)
    
    plt.bar(data.keys(), data.values(), width=.2)
    plt.title('Ensemble method test')
    plt.show()


if __name__ == '__main__':

    X = pd.read_csv('resources/X.csv').as_matrix()
    y = pd.read_csv('resources/y.csv').as_matrix()

    test_log_loss(X, y)

    select_params()
    test_RFC()
