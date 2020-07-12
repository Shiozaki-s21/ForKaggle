from Titanic.train_model import TrainModel
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn

from Titanic.two_layer_net import TwoLayerNet
from Titanic.common.multi_layer_net_extend import MultiLayerNetExtend
from Titanic.common.optimizer import SGD, Adam

sys.path.append(os.pardir)

# get data from csv
dataset_train = pd.read_csv('../Titanic/titanic_data/train.csv')
dataset_test = pd.read_csv('../Titanic/titanic_data/test.csv')

# Data processing
training_csv = dataset_train.copy(deep=True)
test_csv = dataset_test.copy(deep=True)
all_csv = pd.concat([training_csv, test_csv], sort=False)
all_csv['Age'] = all_csv['Age'].fillna(all_csv['Age'].median())
all_csv['Fare'] = all_csv['Fare'].fillna(all_csv['Fare'].median())
# test = seaborn.catplot(x='Embarked', kind='count', data=all_csv)
# plt.plot(data=all_csv['Embarked'])
# plt.show()

all_csv['Embarked'] = all_csv['Embarked'].fillna('S')

all_csv.loc[all_csv['Age'] <= 16, 'Age'] = 0
all_csv.loc[(all_csv['Age'] > 16) & (all_csv['Age'] <= 32), 'Age'] = 1
all_csv.loc[(all_csv['Age'] > 32) & (all_csv['Age'] <= 48), 'Age'] = 2
all_csv.loc[(all_csv['Age'] > 48) & (all_csv['Age'] <= 64), 'Age'] = 3
all_csv.loc[all_csv['Age'] > 64, 'Age'] = 4

all_csv['Age'].replace({1: '1', 2: '2', 3: '3', 4: '4'}, inplace=True)
#
# all_csv.loc[all_csv['Fare'] <= 50, 'Fare'] = 1
# all_csv.loc[(all_csv['Fare'] > 50) & (all_csv['Fare'] <= 100), 'Fare'] = 2
# all_csv.loc[all_csv['Fare'] > 100, 'Fare'] = 3
# #
# all_csv['Fare'].replace({1: '1', 2: '2', 3: '3'}, inplace=True)


# Title
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+\.)', name)

    if title_search:
        return title_search.group(1)
    return ""

all_csv['Title'] = all_csv['Name'].apply(get_title)
all_csv['Title'] = all_csv['Title'].replace(['Capt.', 'Dr.', 'Major.', 'Rev.'], 'Officer.')
all_csv['Title'] = all_csv['Title'].replace(['Lady.', 'Countess.', 'Don.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Royal.')
all_csv['Title'] = all_csv['Title'].replace(['Mlle.', 'Ms.'], 'Miss.')
all_csv['Title'] = all_csv['Title'].replace(['Mme.'], 'Mrs.')

all_csv['Cabin'] = all_csv['Cabin'].fillna('Missing')
all_csv['Cabin'] = all_csv['Cabin'].str[0]

all_csv['Family_Size'] = all_csv['SibSp'] + all_csv['Parch'] + 1
all_csv['IsAlone'] = 0
all_csv.loc[all_csv['Family_Size']==1, 'IsAlone'] = 1

all_csv['Pclass'].replace({1: '1', 2: '2', 3:'3'}, inplace=True)

all_csv_1 = all_csv.drop(['Name', 'Ticket', 'SibSp', 'Parch', 'Family_Size'], axis=1)
all_csv_dummies = pd.get_dummies(all_csv_1, drop_first=True)
all_csv_train = all_csv_dummies[all_csv_dummies['Survived'].notna()]

all_csv_test = all_csv_dummies[all_csv_dummies['Survived'].isna()]

ans = all_csv_train.get({'PassengerId', 'Survived'})

all_csv_train = all_csv_train.drop(['Survived', 'PassengerId'], axis=1)
all_csv_test = all_csv_test.drop(['PassengerId'], axis=1)

x_train = all_csv_train[0: 600].values
t_train = ans[0: 600].values
x_test = all_csv_train[600: 891].values
t_test = ans[600: 891].values

network = MultiLayerNetExtend(input_size=all_csv_train.columns.shape[0], hidden_size_list=[100, 100, 100, 100, 100], output_size=2,
                                 weight_init_std=0.1, use_batchnorm=True)

train_model = TrainModel(network=network, iter_num=10000)
train_model.train(x_train=x_train,  t_train=t_train, x_test=x_test, t_test=t_test)

all_csv_test = all_csv_test.drop(['Survived'], axis=1)
test_csv = all_csv_test.values

network = train_model.network
result = network.predict(test_csv)
# result = network.predict(test_csv)
survived = []
pid = dataset_test['PassengerId']

for i in result:
    survived.append(0) if i[0] > i[1] else survived.append(1)

survivedDf = pd.DataFrame(survived, index=pid, columns=['Survived'])
survivedDf.to_csv('../Titanic/result.csv')

print('fin')

