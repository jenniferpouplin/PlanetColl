import numpy as np
import pandas as pd

# describe data
import seaborn as sns
import matplotlib.colors as mcl
import matplotlib.pyplot as plt

# sklearn
from sklearn import manifold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


def print_conf_mat(y_test, y_pred, class_name):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = pd.DataFrame(cm_norm, columns=class_name, index=class_name)
    cm_norm = cm_norm.round(2)
    print(cm_norm.to_string())


def describe_data(df):
    # pair distribtion
    sns.set(style="ticks")
    sns.pairplot(df, hue="class")
    # pair density
    g = sns.PairGrid(df, diag_sharey=False)
    g.map_lower(sns.kdeplot)
    g.map_upper(sns.scatterplot)
    g.map_diag(sns.kdeplot, lw=3)
    plt.show()


def scatter_data(X, y):
    figure = plt.figure()

    # just plot the dataset first
    cm_bright = mcl.ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax = plt.subplot(1, 1, 1)
    ax.set_title("Input data")

    # tsne
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    _X = tsne.fit_transform(X)
    ax.scatter(_X[:, 0], _X[:, 1], c=y, cmap=cm_bright)
    ax.set_xticks(())
    ax.set_yticks(())
    plt.tight_layout()
    plt.show()
