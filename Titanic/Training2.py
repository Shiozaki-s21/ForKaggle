import os
import sys

sys.path.append(os.pardir)
import numpy as np
import pandas as pd
from Titanic.two_layer_net import TwoLayerNet

# Data processing
dataset_train = pd.read_csv('../Titanic/titanic_data/train.csv')
dataset_test = pd.read_csv('../Titanic/titanic_data/test.csv')

t_train = dataset_train[0: 600].get('Survived')
t_test = dataset_train[600: dataset_train.shape[0]].get('Survived')

dataset_train.drop(['Cabin', 'Name', 'PassengerId', 'Survived', 'Ticket', 'Fare'], axis=1, inplace=True)

dataset_train.replace({'male': 0, 'female': 1}, inplace=True)
dataset_train['male'] = 0
dataset_train['female'] = 0

for i in dataset_train.index:
    dataset_train.loc[dataset_train.index[i], 'male'] = [1, 0][dataset_train.Sex[i]]
    dataset_train.loc[dataset_train.index[i], 'female'] = [0, 1][dataset_train.Sex[i]]

dataset_train.Embarked.replace({'C': 0, 'Q': 1, 'S': 2}, inplace=True)
dataset_train['C'] = 0
dataset_train['Q'] = 0
dataset_train['S'] = 0

# filled NaN
dataset_train = dataset_train.fillna({'Age': dataset_train['Age'].median()})
dataset_train = dataset_train.fillna({'Embarked': 1})

for i in dataset_train.index:
    embarked = int(dataset_train.loc[dataset_train.index[i], 'Embarked'])
    dataset_train.loc[dataset_train.index[i], 'C'] = [1, 0, 0][embarked]
    dataset_train.loc[dataset_train.index[i], 'Q'] = [0, 1, 0][embarked]
    dataset_train.loc[dataset_train.index[i], 'S'] = [0, 0, 1][embarked]

dataset_train.drop(['Sex', 'Embarked', 'Age'], inplace=True, axis=1)

x_train = dataset_train[0: 600].values
x_test = dataset_train[600: dataset_train.shape[0]].values

network = TwoLayerNet(input_size=dataset_train.columns.shape[0], output_size=2, hidden_size=200, weight_init_std=0.1)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.04

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配
    grad = network.gradient(x_batch, t_batch)

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % 100 == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('Tern: ' + str(i) + 'train accuracy: ' + str(train_acc) + ' test accuracy: ' + str(test_acc))
        # print('Tern: ' + str(i) + ' train accuracy: ' + str(train_acc))

pid = dataset_test['PassengerId']
dataset_test.drop(['Cabin', 'Name', 'PassengerId', 'Ticket', 'Fare'], axis=1, inplace=True)
dataset_test.replace({'male': 0, 'female': 1}, inplace=True)
dataset_test['male'] = 0
dataset_test['female'] = 0

for i in dataset_test.index:
    dataset_test.loc[dataset_test.index[i], 'male'] = [1, 0][dataset_test.Sex[i]]
    dataset_test.loc[dataset_test.index[i], 'female'] = [0, 1][dataset_test.Sex[i]]

dataset_test.Embarked.replace({'C': 0, 'Q': 1, 'S': 2}, inplace=True)

dataset_test['C'] = 0
dataset_test['Q'] = 0
dataset_test['S'] = 0

# filled NaN
dataset_test = dataset_test.fillna({'Age': dataset_test['Age'].median()})
dataset_test = dataset_test.fillna({'Embarked': 1})

for i in dataset_test.index:
    embarked = int(dataset_test.loc[dataset_test.index[i], 'Embarked'])
    dataset_test.loc[dataset_test.index[i], 'C'] = [1, 0, 0][embarked]
    dataset_test.loc[dataset_test.index[i], 'Q'] = [0, 1, 0][embarked]
    dataset_test.loc[dataset_test.index[i], 'S'] = [0, 0, 1][embarked]

dataset_test.drop(['Sex', 'Embarked', 'Age'], inplace=True, axis=1)

x_test = dataset_test.values
result = network.predict(x_test)
survived = []

for i in result:
    survived.append(0) if i[0] > i[1] else survived.append(1)

survivedDf = pd.DataFrame(survived, index=pid, columns=['Survived'])

survivedDf.to_csv('../Titanic/result.csv')
