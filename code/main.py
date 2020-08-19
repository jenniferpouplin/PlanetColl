import numpy as np
import os
import argparse
from time import time
from sklearn.model_selection import train_test_split
from optimise import optimise, PARAMS, CLASSIFIERS
from utils import print_conf_mat, describe_data
from dataloader import load_data, DATALOADERS
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataloader', default='cambioni', type=str, choices=sorted(list(DATALOADERS.keys())))
    parser.add_argument('--Data', default='../data/', type=str)
    parser.add_argument('--outDir', default= '../experiments/', type=str)
    parser.add_argument('--model', default='decision_tree', type=str, choices=sorted(list(PARAMS.keys())))
    parser.add_argument('--optim', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    opts = get_args()

    # get data
    df, class_name, y, X = load_data(dataloader=opts.dataloader, data_path=opts.Data)

    # describe data
    describe_data(df)

    # spit into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    t0 = time()
    # get params of model (optimised using a grid search)
    if opts.optim:
        estimator = optimise(X_train, y_train, clf=opts.model, verbose=0, cross_val=5)
        opt_params = estimator.get_params()
        opt_params = {key: opt_params['estimator__' + key] for key in PARAMS[opts.model]}
        print(opt_params)
    else:
        estimator = CLASSIFIERS[opts.model]()
        estimator.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    # evaluation
    t0 = time()
    y_pred = estimator.predict(X_test)
    test_time = time() - t0
    print("predicted in %fs" %(test_time))
    score = accuracy_score(y_test, y_pred)
    print("accuracy:   %0.3f" % score)
    # confusion matrix
    print_conf_mat(y_test, y_pred, class_name.values())

