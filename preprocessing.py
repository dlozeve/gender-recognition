#!/usr/bin/env python3

import numpy as np

import torch
from torch.autograd import Variable

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from autoencoder import train_autoencoder


def preprocess(X, y, X_val, test_data, verbose=True, scale=True,
               autoencoder=True, qda=True, knn=False, xgb=False):
    if autoencoder:
        if verbose:
            print("## Autoencoder")
            print("### Train...", end=" ", flush=True)
            ae = train_autoencoder(X, size=32, epochs=20, verbose=1)
        else:
            ae = train_autoencoder(X, size=32, epochs=20, verbose=0)
        if verbose:
            print("done.")
            print("### Evaluate...", end=" ", flush=True)
        ae.eval()
        X_ae = ae.layer1(Variable(torch.Tensor(X))).data
        X = np.c_[X, X_ae]
        X_val_ae = ae.layer1(Variable(torch.Tensor(X_val))).data
        X_val = np.c_[X_val, X_val_ae]
        test_data_ae = ae.layer1(Variable(torch.Tensor(test_data))).data
        test_data = np.c_[test_data, test_data_ae]
        if verbose:
            print("done.")

    if qda:
        if verbose:
            print("## Quadratic Discriminant Analysis...", end=" ", flush=True)
        qdaclf = QuadraticDiscriminantAnalysis(reg_param=0.02)
        qdaclf.fit(X, y)
        X_qda = qdaclf.predict_proba(X)
        X = np.c_[X, X_qda[:, 1]]
        X_val_qda = qdaclf.predict_proba(X_val)
        X_val = np.c_[X_val, X_val_qda[:, 1]]
        test_data_qda = qdaclf.predict_proba(test_data)
        test_data = np.c_[test_data, test_data_qda[:, 1]]
        if verbose:
            print("done.")

    if knn:
        print("## K-Nearest Neighbours...", end=" ", flush=True)
        knnclf = KNeighborsClassifier(n_neighbors=10, p=2, n_jobs=-1)
        knnclf.fit(X, y)
        X_knn = knnclf.predict_proba(X)
        X = np.c_[X, X_knn[:, 1]]
        X_val_knn = knnclf.predict_proba(X_val)
        X_val = np.c_[X_val, X_val_knn[:, 1]]
        test_data_knn = knnclf.predict_proba(test_data)
        test_data = np.c_[test_data, test_data_knn[:, 1]]
        print("done.")

    if xgb:
        print("## XGBoost...", end=" ", flush=True)
        xgbclf = XGBClassifier(max_depth=3, learning_rate=0.1,
                               n_estimators=1000,
                               gamma=10, min_child_weight=10,
                               objective='binary:logistic', n_jobs=4)
        xgbclf.fit(X, y)
        X_xgb = xgbclf.predict_proba(X)
        X_val_xgb = xgbclf.predict_proba(X_val)
        X = np.c_[X, X_xgb[:, 1]]
        X_val = np.c_[X_val, X_val_xgb[:, 1]]
        test_data_xgb = xgbclf.predict_proba(test_data)
        test_data = np.c_[test_data, test_data_xgb[:, 1]]
        print("done.")

    if scale:
        if verbose:
            print("## Scaling...", end=" ", flush=True)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_val = scaler.transform(X_val)
        test_data = scaler.transform(test_data)
        if verbose:
            print("done.")

    return X, y, X_val, test_data
