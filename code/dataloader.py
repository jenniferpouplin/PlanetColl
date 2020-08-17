import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
import os

def get_cambioni(data_path):
    # get data
    df = pd.read_csv(data_path)
    class_name = {1: 'hit-and-run', 2: 'graze-and-merge', 3: 'merging', 4: 'disruption'}
    y = df['class'].values # target
    X = df[['Mass', 'Gamma', 'Angle', 'Velocity']].values # data
    df['class'] = df['class'].map(class_name)
    return df, class_name, y, X

def get_pouplin(data_path):
    # get data
    df = pd.read_csv(data_path)
    class_name = {1: 'hit-and-run', 2: 'merger', 3: 'disruption', 4: 'supercatastrophic'}
    y = df['class'].values # target
    # Mass: Mass of target
    # Gamma : Ratio impactor mass / target mass
    # b : impact parameter
    # Velocity : Velocity of impact

    X = df[['Mass', 'Gamma', 'b', 'Velocity']].values # data
    df['class'] = df['class'].map(class_name)
    return df, class_name, y, X

def get_iris(*args, **kwargs):
    # get data
    data = load_iris()
    df = pd.DataFrame(data= np.c_[data['data'], data['target']], columns= data['feature_names'] + ['class'])
    class_name = dict(enumerate(data.target_names))
    df['class'] = df['class'].map(class_name)
    X = data.data # target
    y = data.target
    return df, class_name, y, X


DATALOADERS = {
    'cambioni': get_cambioni,
    'iris': get_iris,
    'pouplin':get_pouplin
}


DATAPATH = {
    'iris': '',
    'cambioni': 'cambioni.csv',
    'pouplin': 'pouplin.csv'
}


def load_data(dataloader, data_path=None):
    dl = DATALOADERS[dataloader]
    file_path = data_path
    if file_path is not None:
        file_path = os.path.join(file_path, DATAPATH[dataloader])
    else:
        file_path = DATAPATH[dataloader]
    return dl(data_path=file_path)
