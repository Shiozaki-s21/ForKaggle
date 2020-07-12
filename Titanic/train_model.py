import numpy as np


class TrainModel:
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    def __init__(self, network, iter_num=10000, batch_size=100, learning_rate=0.01):
        self.iters_num = iter_num
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.network = network

    def train(self, x_train, t_train, x_test, t_test):
        for i in range(self.iters_num):
            train_size = x_train.shape[0]
            batch_mask = np.random.choice(train_size, self.batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # 勾配
            grad = self.network.gradient(x_batch, t_batch)

            # 更新
            for key in ('W1', 'b1', 'W2', 'b2'):
                self.network.params[key] -= self.learning_rate * grad[key]

            loss = self.network.loss(x_batch, t_batch)
            self.train_loss_list.append(loss)

            if i % 100 == 0:
                train_acc = self.network.accuracy(x_train, t_train)
                test_acc = self.network.accuracy(x_test, t_test)
                self.train_acc_list.append(train_acc)
                self.test_acc_list.append(test_acc)
                print('Tern: ' + str(i) + 'train accuracy: ' + str(train_acc) + ' test accuracy: ' + str(test_acc))
