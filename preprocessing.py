#!/usr/bin/env python3

import numpy as np

import torch
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier

from autoencoder import train_autoencoder


def preprocess(X, y, X_val, test_data, verbose=True):
    if verbose:
        print("## Autoencoder")
        print("### Train...", end=" ", flush=True)
        autoencoder = train_autoencoder(X, size=32, epochs=30, verbose=1)
    else:
        autoencoder = train_autoencoder(X, size=32, epochs=30, verbose=0)
    if verbose:
        print("done.")
        print("### Evaluate...", end=" ", flush=True)
    autoencoder.eval()
    X_ae = autoencoder.layer1(Variable(torch.Tensor(X))).data
    X = np.c_[X, X_ae]
    X_val_ae = autoencoder.layer1(Variable(torch.Tensor(X_val))).data
    X_val = np.c_[X_val, X_val_ae]
    test_data_ae = autoencoder.layer1(Variable(torch.Tensor(test_data))).data
    test_data = np.c_[test_data, test_data_ae]
    if verbose:
        print("done.")

    if verbose:
        print("## Quadratic Discriminant Analysis...", end=" ", flush=True)
    qda = QuadraticDiscriminantAnalysis(reg_param=0.02)
    qda.fit(X, y)
    X_qda = qda.predict_proba(X)
    X = np.c_[X, X_qda[:, 1]]
    X_val_qda = qda.predict_proba(X_val)
    X_val = np.c_[X_val, X_val_qda[:, 1]]
    test_data_qda = qda.predict_proba(test_data)
    test_data = np.c_[test_data, test_data_qda[:, 1]]
    if verbose:
        print("done.")

    # print("## K-Nearest Neighbours...", end=" ", flush=True)
    # knn = KNeighborsClassifier(n_neighbors=10, p=2, n_jobs=-1)
    # knn.fit(X, y)
    # X_knn = knn.predict_proba(X)
    # X = np.c_[X, X_knn[:, 1]]
    # X_val_knn = knn.predict_proba(X_val)
    # X_val = np.c_[X_val, X_val_knn[:, 1]]
    # test_data_knn = knn.predict_proba(test_data)
    # test_data = np.c_[test_data, test_data_knn[:, 1]]
    # print("done.")

    # print("## XGBoost...", end=" ", flush=True)
    # xgb = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=1000,
    #                     gamma=10, min_child_weight=10,
    #                     objective='binary:logistic', n_jobs=4)
    # xgb.fit(X, y)
    # X_xgb = xgb.predict_proba(X)
    # X_val_xgb = xgb.predict_proba(X_val)
    # X = np.c_[X, X_xgb[:, 1]]
    # X_val = np.c_[X_val, X_val_xgb[:, 1]]
    # print("done.")

    # print("## Add polynomial features...", end=" ", flush=True)
    # poly = PolynomialFeatures(degree=2, include_bias=False)
    # X = poly.fit_transform(X)
    # X_val = poly.transform(X_val)
    # print("done.")

    if verbose:
        print("## Scaling...", end=" ", flush=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_val = scaler.transform(X_val)
    test_data = scaler.transform(test_data)
    if verbose:
        print("done.")

    return X, y, X_val, test_data
