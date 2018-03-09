#!/usr/bin/env python3

import time

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torchvision

from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE
# from xgboost import XGBClassifier

from autoencoder import Autoencoder


GPU = False

start_time = time.time()

print("# Loading data...", end=" ", flush=True)
X = pd.read_csv("data/train.data.csv")
y = pd.read_csv("data/train.labels.csv")
test_data = pd.read_csv("data/test.data.csv")
X = X.values
y = y.values.ravel()
test_data = test_data.values
print("done.")

print("# Preprocessing")

# print("# t-SNE...", end=" ", flush=True)
# proj = TSNE(n_components=2, n_jobs=4, n_iter=1000,
#             perplexity=10, early_exaggeration=50, learning_rate=100)
# X_proj = proj.fit_transform(X)
# X = np.c_[X, X_proj]
# print("done.")

X, X_val, y, y_val = train_test_split(X, y, test_size=0.2,
                                      stratify=y)

print("## Autoencoder")
print("### Train...", end=" ", flush=True)
ae_trainset = TensorDataset(torch.Tensor(X), torch.Tensor(X))
ae_trainloader = DataLoader(ae_trainset, batch_size=256,
                            shuffle=True, num_workers=2)
autoencoder = Autoencoder(X.shape[1], 32)
criterion = nn.MSELoss()
optimizer = optim.Adadelta(autoencoder.parameters(),
                           lr=1.0, rho=0.95, weight_decay=1e-5)
print("epoch #", end=" ", flush=True)
for epoch in range(20):
    running_loss = 0.0
    for i, data in enumerate(ae_trainloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = autoencoder(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # running_loss += loss.data[0]
        # if i % 50 == 49:
        #     print(f"[{epoch:3},{i+1:3}] Loss: {running_loss/50:.3f}")
        #     running_loss = 0.0
    print(epoch+1, end=" ", flush=True)
print("done.")
print("### Evaluate...", end=" ", flush=True)
autoencoder.eval()
X_ae = autoencoder.layer1(Variable(torch.Tensor(X))).data
X = np.c_[X, X_ae]
X_val_ae = autoencoder.layer1(Variable(torch.Tensor(X_val))).data
X_val = np.c_[X_val, X_val_ae]
test_data_ae = autoencoder.layer1(Variable(torch.Tensor(test_data))).data
test_data = np.c_[test_data, test_data_ae]
print("done.")


print("## Quadratic Discriminant Analysis...", end=" ", flush=True)
qda = QuadraticDiscriminantAnalysis(reg_param=0.02)
qda.fit(X, y)
X_qda = qda.predict_proba(X)
X = np.c_[X, X_qda[:, 1]]
X_val_qda = qda.predict_proba(X_val)
X_val = np.c_[X_val, X_val_qda[:, 1]]
test_data_qda = qda.predict_proba(test_data)
test_data = np.c_[test_data, test_data_qda[:, 1]]
print("done.")

# print("## K-Nearest Neighbours...", end=" ", flush=True)
# knn = KNeighborsClassifier(n_neighbors=10, p=2, n_jobs=-1)
# knn.fit(X, y)
# X_knn = knn.predict_proba(X)
# X = np.c_[X, X_knn[:, 1]]
# X_val_knn = knn.predict_proba(X_val)
# X_val = np.c_[X_val, X_val_knn[:, 1]]
# # test_data_knn = knn.predict_proba(test_data)
# # test_data = np.c_[test_data, test_data_knn[:, 1]]
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

print("## Scaling...", end=" ", flush=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)
print("done.")


# Neural net definition
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 150),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        # self.layer2 = nn.Sequential(
        #     nn.Linear(70, 200),
        #     nn.Dropout(0.5),
        #     nn.ReLU()
        # )
        self.output = nn.Sequential(
            nn.Linear(150, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.output(x)
        return x.view(-1)


# Training
print("# Training the Neural Networks...", flush=True)
nets = list(range(20))
losses = np.zeros(len(nets))
skf = StratifiedShuffleSplit(n_splits=len(nets), test_size=0.15)
for k, (train, test) in enumerate(skf.split(X, y)):
    X_train = X[train, :]
    y_train = y[train]
    X_test = X[test, :]
    y_test = y[test]

    print(f"## Training neural net {k}")
    trainset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    trainloader = DataLoader(trainset, batch_size=300,
                             shuffle=True, num_workers=2)
    testset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    testloader = DataLoader(testset, batch_size=300,
                            shuffle=True, num_workers=2)

    net = Net(X_train.shape[1])
    if GPU:
        net.cuda()
    # print(net)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-3)
    if GPU:
        criterion.cuda()

    for epoch in range(20):
        running_loss = 0.0
        running_correct = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if GPU:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            pred = F.sigmoid(outputs).cpu().data.numpy() > .5
            running_correct += np.sum(pred == labels.cpu().data.numpy())
            # if i % 20 == 19:
            #     print(f"[{epoch:2},{i+1:3}] Loss: {running_loss/20:.3f}, "
            #           f"Accuracy: "
            #           f"{100*running_correct/(20*len(outputs)):.1f}%")
            #     running_loss = 0.0
            #     running_correct = 0
        if epoch % 10 == 9:
            print(f"   [{epoch+1:2}] "
                  f"Loss: {running_loss/((i+1)):.3f}, Accuracy: "
                  f"{100*running_correct/((i+1)*trainloader.batch_size):.1f}%")
            running_loss = 0.0
            running_correct = 0
    val_loss = 0
    correct = 0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        if GPU:
            inputs.cuda()
            labels.cuda()
        net.eval()
        output = net(inputs)
        val_loss += F.binary_cross_entropy_with_logits(
            output, labels.float()).cpu().data[0]
        sigma_output = F.sigmoid(output)
        pred = sigma_output.cpu().data.numpy() > .5
        correct += np.sum(pred == labels.cpu().data.numpy())
    val_loss /= len(testloader)
    print(f"   -> Test set: Average loss: {val_loss:.4f}, "
          f"Accuracy: {correct}/{((i+1)*testloader.batch_size)} "
          f"({100. * correct/((i+1)*testloader.batch_size):.1f}%)")

    nets[k] = net
    losses[k] = val_loss

# Validation
X_val, y_val = Variable(torch.Tensor(X_val)), Variable(torch.Tensor(y_val))
if GPU:
    X_val.cuda()
    y_val.cuda()
output = 0
for k, net in enumerate(nets):
    net.eval()
    output += net(X_val) * 1/losses[k]
output /= np.sum(1/losses)
val_loss = F.binary_cross_entropy_with_logits(output, y_val).cpu().data[0]
sigma_output = F.sigmoid(output)
pred = sigma_output.cpu().data.numpy() > .5
correct = np.sum(pred == y_val.cpu().data.numpy())
auc = roc_auc_score(y_val.data, sigma_output.data)
print(f"\n=> Validation set: Average loss: {val_loss:.4f}, "
      f"ROC AUC: {auc:.4f}, "
      f"Accuracy: {correct}/{len(y_val)} "
      f"({100. * correct/len(y_val):.1f}%)\n")

test_data = Variable(torch.Tensor(test_data))
if GPU:
    test_data.cuda()
output = 0
for net in nets:
    output += net(test_data)
output /= len(nets)
sigma_output = F.sigmoid(output)
print(sigma_output.cpu().data)

submission = pd.DataFrame({'Id': range(1, 15001),
                           'ProbFemale': sigma_output.cpu().data})
submission = submission[['Id', 'ProbFemale']]
submission.to_csv("submission.csv", index=False)

time_elapsed = time.time() - start_time
print(time.strftime("Timing: %Hh %Mm %Ss", time.gmtime(time_elapsed)))
