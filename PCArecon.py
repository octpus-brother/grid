import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import getdata as gd
import os
import pickle


def get_dataset(args, filepath, svrname):
    _, seq_keywise = gd.read_log_keywise(filepath, svrname)
    seqs = pd.DataFrame(seq_keywise)
    return seqs


class PCArecon:
    def __init__(self, explained_var=0.9):
        super(PCArecon, self).__init__()
        self.explained_var = explained_var
        self.model = None
        self.scaler = None

    def fit(self, x):
        x.bfill(inplace=True)
        x.ffill(inplace=True)
        data = x.values
        self.scaler = StandardScaler()

        data_scaled = self.scaler.fit_transform(data)

        self.model = PCA(n_components=self.explained_var)
        print("Fitting PCA")
        self.model.fit(data_scaled)
        print("Done fitting PCA. ncomponents for explained variance {} = {}".format(self.explained_var,
                                                                                    self.model.n_components_))

    def test(self, x):
        x.bfill(inplace=True)
        x.ffill(inplace=True)
        data = x.values
        data_scaled = self.scaler.transform(data)
        recons_tc = np.dot(self.model.transform(data_scaled), self.model.components_)
        error_tc = (data_scaled - recons_tc) ** 2
        # print(self.model.transform(data_scaled).shape, self.model.components_.shape)
        # print(recons_tc.shape)
        error = np.sum(error_tc, axis=0)
        return error


def train(args, device, filepath, svrname):
    model = PCArecon(args["explained_var"])
    train_data = get_dataset({}, filepath, svrname)
    model.fit(train_data)
    train_error = model.test(train_data)

    modelnum = args["modelnum"]
    if not os.path.exists('./models/'):
        os.mkdir('./models/')
    f = open('./models/PCAreconmodel{}.pkl'.format(modelnum), 'wb')
    pickle.dump(model, f)
    pickle.dump(train_error, f)
    f.close()
    f = open('./models/PCAreconargs{}.pkl'.format(modelnum), 'wb')
    pickle.dump(args, f)
    f.close()


def test(test_args, device, testfilepath, testsvrname):
    modelnum = test_args["modelnum"]
    threshold = test_args["threshold"]
    f = open('./models/PCAreconargs{}.pkl'.format(modelnum), 'rb')
    args = pickle.load(f)
    f.close()
    f = open('./models/PCAreconmodel{}.pkl'.format(modelnum), 'rb')
    model = pickle.load(f)
    train_error = pickle.load(f)
    f.close()

    test_data = get_dataset({}, testfilepath, testsvrname)
    test_error = model.test(test_data)

    sth_wrong = 0
    for i in range(test_error.shape[0]):
        if test_error[i] > threshold * train_error[i]:
            sth_wrong = 1

    if sth_wrong == 1:
        print("Anomaly detected.")
    else:
        print("No anomaly detected.")

