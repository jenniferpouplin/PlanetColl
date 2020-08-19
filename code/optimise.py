import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import warnings

warnings.filterwarnings('ignore')

# Dictionnary of available classifiers

CLASSIFIERS = {
    'adaboost': AdaBoostClassifier,
    'decision_tree': DecisionTreeClassifier,
    'knn': KNeighborsClassifier,
    'linear_svm': LinearSVC,
    'logistic_regression': LogisticRegression,
    'naive_bayes': GaussianNB,
    'random_forest': RandomForestClassifier,
    'svm': SVC,
    "MLP": MLPClassifier
}


PARAMS = {
    'adaboost': {
        'n_estimators': np.linspace(10, 100, 5).astype(int),
        'learning_rate': np.logspace(-4, 0, 5)
    },
    'decision_tree': {
        'min_samples_leaf': np.logspace(-4, np.log10(0.5), 10),
        'max_features': np.logspace(-4, 0, 10),
        'class_weight': [None, 'balanced']
    },
    'knn': {
        'n_neighbors': np.linspace(1, 25, 25).astype(int),
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': np.linspace(1, 5, 1).astype(int),
    },
    'linear_svm': {
        'penalty': ['l1', 'l2'],
        'loss': ['hinge', 'squared_hinge'],
        'dual': [False, True],
        'C': np.logspace(-4, 3, 10),
        'max_iter': [1000]
    },
    'logistic_regression': {
        'penalty': ['l1', 'l2'],
        'C': np.logspace(-4, 3, 10),
        'class_weight': [None, 'balanced']
    },
    'naive_bayes': {
        'var_smoothing': np.logspace(-14, -7, 10)
    },
    'random_forest': {
        'n_estimators': np.linspace(10, 100, 5).astype(int),
        'min_samples_leaf': np.logspace(-4, np.log10(0.5), 5),
        'class_weight': [None, 'balanced']
    },
    'svm': {
        'C': np.logspace(-4, 3, 10),
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'degree': np.linspace(2, 10, 1).astype(int),
        'gamma': np.logspace(-4, 3, 10),
    },
    'MLP': {
        'solver': ['lbfgs', 'adam', 'sgd'],
        'alpha': [1e-5],
        'random_state': [1],
    }
}

# Optimisation function
def optimise(x, y, clf='decision_tree', verbose=0, cross_val=5, optimise_by='accuracy'):
    classifier = CLASSIFIERS[clf]()
    params = PARAMS[clf]
    new_classifier = GridSearchCV(estimator=classifier, param_grid=params, verbose=verbose, cv=cross_val,
                                  scoring=optimise_by, error_score=-10**6)
    new_classifier.fit(x, y)
    return new_classifier


if __name__ == '__main__':
    MODEL = 'linear_svm'
    iris = datasets.load_iris()
    X, Y = iris.data, iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
    clf = optimise(X_train, Y_train, clf=MODEL, verbose=0, cross_val=5)
    opt_params = clf.get_params()
    opt_params = {key: opt_params['estimator__' + key] for key in PARAMS[MODEL]}

    Y_pred = clf.predict(X_test)
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print(opt_params)
    print(conf_matrix)
