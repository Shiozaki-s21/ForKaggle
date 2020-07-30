import os
import sys

sys.path.append(os.pardir)
import numpy as np
import pandas as pd

from Titanic.two_layer_net import TwoLayerNet
from Titanic.common.multi_layer_net_extend import MultiLayerNetExtend
from Titanic.common.optimizer import SGD, Adam


# get data from csv
dataset_train = pd.read_csv('../Titanic/titanic_data/train.csv')
dataset_test = pd.read_csv('../Titanic/titanic_data/test.csv')

# Data processing
training_csv = dataset_train.copy(deep=True)
training_csv.drop(['SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Name', 'PassengerId', 'Survived'], inplace=True, axis=1)

# fill NaN of Age by mean
training_csv['Age'].fillna(training_csv['Age'].mean(), inplace=True)
# print(training_csv.isna().any())

# translate Pclass, Sex to one-hot
# Pclass
training_csv['Pclass'].replace({1: 0, 2: 1, 3: 2}, inplace=True)
training_csv['Pclass_1'] = 0
training_csv['Pclass_2'] = 0
training_csv['Pclass_3'] = 0

for i in training_csv.index:
    training_csv.loc[i, 'Pclass_1'] = [1, 0, 0][training_csv.Pclass[i]]
    training_csv.loc[i, 'Pclass_2'] = [0, 1, 0][training_csv.Pclass[i]]
    training_csv.loc[i, 'Pclass_3'] = [0, 0, 1][training_csv.Pclass[i]]

# Sex
training_csv['Sex'].replace({'male': 0, 'female': 1}, inplace=True)

# drop Pclass and Sex
training_csv.drop({'Pclass'}, inplace=True, axis=1)

# x_train
x_train = training_csv[0: 600].values

# t_train
t_train = dataset_train[0: 600].get('Survived').values

# x_test
x_test = training_csv[600: dataset_train.shape[0]].values

# t_test
t_test = dataset_train[600: dataset_train.shape[0]].get('Survived').values

# training
network = MultiLayerNetExtend(input_size=training_csv.columns.shape[0], hidden_size_list=[100, 100, 100, 100, 100], output_size=2,
                                 weight_init_std=0.1, use_batchnorm=True)

iters_num = 9000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.001

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

# output as a csv
# data processing
test_csv = dataset_test.copy(deep=True)
test_csv.drop(['SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'Name', 'PassengerId'], inplace=True, axis=1)

# fill NaN of Age by mean
test_csv['Age'].fillna(test_csv['Age'].mean(), inplace=True)
# print(test_csv.isna().any())

# translate Pclass, Sex to one-hot
# Pclass
test_csv['Pclass'].replace({1: 0, 2: 1, 3: 2}, inplace=True)
test_csv['Pclass_1'] = 0
test_csv['Pclass_2'] = 0
test_csv['Pclass_3'] = 0

for i in test_csv.index:
    test_csv.loc[i, 'Pclass_1'] = [1, 0, 0][test_csv.Pclass[i]]
    test_csv.loc[i, 'Pclass_2'] = [0, 1, 0][test_csv.Pclass[i]]
    test_csv.loc[i, 'Pclass_3'] = [0, 0, 1][test_csv.Pclass[i]]

# Sex
test_csv['Sex'].replace({'male': 0, 'female': 1}, inplace=True)

# drop Pclass and Sex
test_csv.drop({'Pclass'}, inplace=True, axis=1)

test_values = test_csv.values
result = network.predict(test_values)
survived = []
pid = dataset_test['PassengerId']

for i in result:
    survived.append(0) if i[0] > i[1] else survived.append(1)

survivedDf = pd.DataFrame(survived, index=pid, columns=['Survived'])

survivedDf.to_csv('../Titanic/result05330731.csv')


print('fin')
