import os
import sys

sys.path.append(os.pardir)
import numpy as np
import pandas as pd
from Titanic.two_layer_net import TwoLayerNet


# Data processing
dataset_train = pd.read_csv('../Titanic/titanic_data/train.csv')
dataset_test = pd.read_csv('../Titanic/titanic_data/test.csv')
dataset_ans = pd.read_csv('../Titanic/titanic_data/gender_submission.csv')

# create t_train
dataset_train = dataset_train.replace({'C': 1, 'Q': 2, 'S': 3})
dataset_train = dataset_train.replace({'male': 0, 'female': 1})
dataset_train.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
lowData = dataset_train.get('Survived')

# filled NaN by median number of age
dataset_train = dataset_train.fillna(dataset_train['Age'].mean())

t_train = []

for i in range(lowData.size):
    t_train.append([[0, 1], [1, 0]][lowData[i]])

t_train = np.array(t_train)

# create x_train
dataset_train.drop(['Survived'], axis=1, inplace=True)
x_train = dataset_train.values

# create t_test
t_test = dataset_ans.values

# create x_test
dataset_test = dataset_test.replace({'C': 1, 'Q': 2, 'S': 3})
dataset_test = dataset_test.replace({'male': 0, 'female': 1})
dataset_test.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
# filled NaN by median number of Age
dataset_test = dataset_test.fillna(dataset_test['Age'].mean())

x_test = dataset_test.values

network = TwoLayerNet(input_size=7, output_size=2, hidden_size=100, weight_init_std=0.001)
iters_num = 5000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

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
        print('train accuracy: ' + str(train_acc) + ' test accuracy: ' + str(test_acc))

#
# batch_mask = np.random.choice(train_size, batch_size)
# lowResult = network.predict(x_train[range(419)])
#
# # for i in range(lowResult.shape[0]):
# #
# #     tmp = lowResult[i]
# #     print(lowResult[i])
# #     result = 0 if tmp[0] > tmp[1] else 1
# #     print([0, 1][result])
