import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(130, 256),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.layer1(x)
        # x = self.layer2(x)
        x = self.output(x)
        return x


print("Loading data...", end=" ", flush=True)
X = pd.read_csv("data/train.data.csv")
y = pd.read_csv("data/train.labels.csv")
test_data = pd.read_csv("data/test.data.csv")
X = X.values
y = y.values.ravel()
test_data = test_data.values
print("done.")

# print("t-SNE...", end=" ", flush=True)
# proj = TSNE(n_components=2, verbose=0, perplexity=30)
# X_proj = proj.fit_transform(X)
# X = np.c_[X, X_proj]
# print("done.")

X, X_val, y, y_val = train_test_split(X, y, test_size=0.1)

print("Quadratic Discriminant Analysis...", end=" ", flush=True)
qda = QuadraticDiscriminantAnalysis(reg_param=0.025)
qda.fit(X, y)
X_qda = qda.predict_proba(X)
X = np.c_[X, X_qda[:, 1]]
X_val_qda = qda.predict_proba(X_val)
X_val = np.c_[X_val, X_val_qda[:, 1]]
test_data_qda = qda.predict_proba(test_data)
test_data = np.c_[test_data, test_data_qda[:, 1]]
print("done.")

print("K-Nearest Neighbours...", end=" ", flush=True)
knn = KNeighborsClassifier(n_neighbors=10, p=2, n_jobs=-1)
knn.fit(X, y)
X_knn = knn.predict_proba(X)
X = np.c_[X, X_knn[:, 1]]
X_val_knn = knn.predict_proba(X_val)
X_val = np.c_[X_val, X_val_knn[:, 1]]
test_data_knn = knn.predict_proba(test_data)
test_data = np.c_[test_data, test_data_knn[:, 1]]
print("done.")

# Scaling
print("Scaling...", end=" ", flush=True)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)
test_data = scaler.transform(test_data)
print("done.")

# Training
print("Training the Neural Network...", flush=True)
SPLITS = 10
sss = StratifiedShuffleSplit(n_splits=SPLITS, test_size=0.2)
net = list(range(SPLITS))
for k, (train, test) in enumerate(sss.split(X, y)):
    X_train = X[train, :]
    y_train = y[train]
    X_test = X[test, :]
    y_test = y[test]

    trainset = torch.utils.data.TensorDataset(torch.Tensor(X_train),
                                              torch.LongTensor(y_train))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=300,
                                              shuffle=True, num_workers=2)
    testset = torch.utils.data.TensorDataset(torch.Tensor(X_test),
                                             torch.LongTensor(y_test))
    testloader = torch.utils.data.DataLoader(testset, batch_size=300,
                                             shuffle=True, num_workers=2)

    if k == 0:
        net[k] = Net()
    else:
        net[k] = net[k-1]
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net[k].parameters(), lr=0.001, weight_decay=1e-3)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net[k](inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    net[k].eval()
    data = Variable(torch.Tensor(X_test))
    target = Variable(torch.LongTensor(y_test))
    output = net[k](data)
    test_loss = log_loss(y_test.data, output.data)
    pred = output.data.max(1, keepdim=True)[1]
    correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
    auc = roc_auc_score(y_test.data, output.data[:, 1])

    print(f"[{k}] Test set: Average loss: {test_loss:.4f}, "
          f"ROC AUC: {auc:.4f}, "
          f"Accuracy: {correct}/{len(testloader.dataset)} "
          f"({100. * correct/len(testloader.dataset):.1f}%)")


# Validation
X_val, y_val = Variable(torch.Tensor(X_val)), Variable(torch.LongTensor(y_val))
output = net[SPLITS-1](X_val)
# print(output)
# val_loss = F.cross_entropy(output, y_val).data[0]
val_loss = log_loss(y_val.data, output.data)
pred = output.data.max(1, keepdim=True)[1]
correct = pred.eq(y_val.data.view_as(pred)).long().cpu().sum()
auc = roc_auc_score(y_val.data, output.data[:, 1])
print(f"\nValidation set: Average loss: {val_loss:.4f}, "
      f"ROC AUC: {auc:.4f}, "
      f"Accuracy: {correct}/{len(y_val)} "
      f"({100. * correct/len(y_val):.1f}%)\n")

# test_data = Variable(torch.Tensor(test_data))
# test_data_pred = net[SPLITS-1](test_data)
# print(test_data_pred.data[:, 1])

# submission = pd.DataFrame({'Id': range(1, 15001),
#                            'ProbFemale': test_data_pred.data[:, 1]})
# submission = submission[['Id', 'ProbFemale']]
# submission.to_csv("submission.csv", index=False)
