from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import sgd
import numpy as np
import pandas as pd

'''
使用keras+loss函数
Actor Critic中的A2C算法，引入Advantages函数
'''

class Actor:
    def __init__(self, n_features, n_actions, lr):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.model = None

        a = Input(shape=(self.n_features,))
        b = Dense(128, activation='relu', kernel_initializer='random_uniform')(a)
        c = Dropout(0.5)(b)
        d = Dense(64, activation='relu', kernel_initializer='random_uniform')(c)
        f = Dropout(0.5)(d)
        g = Dense(32, activation='relu', kernel_initializer='random_uniform')(f)
        h = Dropout(0.5)(g)
        i = Dense(self.n_actions, activation='softmax', kernel_initializer='random_uniform')(h)
        self.model = Model(inputs=a, outputs=i)

        self.model.compile(loss='categorical_crossentropy', optimizer=sgd(lr=self.lr), metrics=['mse', 'accuracy'])

    def learn(self, s, q):
        self.model.train_on_batch(s, q)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        actions = self.model.predict(s)
        choice = np.random.choice(self.n_actions, p=actions.flatten())
        return choice

    def random_action(self):
        choice = np.random.choice(self.n_actions)
        return choice

class Critic:
    def __init__(self, n_features, lr):
        self.n_features = n_features
        self.lr = lr

        a = Input(shape=(self.n_features,))
        b = Dense(128, activation='relu', kernel_initializer='random_uniform')(a)
        c = Dropout(0.5)(b)
        d = Dense(64, activation='relu', kernel_initializer='random_uniform')(c)
        f = Dropout(0.5)(d)
        g = Dense(32, activation='relu', kernel_initializer='random_uniform')(f)
        h = Dropout(0.5)(g)
        i = Dense(1)(h)
        self.model = Model(inputs=a, outputs=i)
        self.model.compile(loss='mse', optimizer=sgd(lr=self.lr), metrics=['mse', 'accuracy'])

    def learn(self, s, td):
        self.model.train_on_batch(s, td)

    def eval(self, s):
        return self.model.predict(s)

class Memory:
    def __init__(self, capacity, dims, nb_feature):
        self.nb_feature = nb_feature
        self.capacity = capacity
        self.memory = np.zeros((self.capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        index = self.pointer % self.capacity
        self.memory[index, :] = np.hstack((s, a, r, s_))
        self.pointer += 1

    def persist_memory(self):
        self.memory.to_csv('./memory/m.csv', index=False)

    def load_memory(self):
        self.memory = pd.read_csv('./memory/m.csv')

    def sample(self, n):
        indices = np.random.choice(self.memory.shape[0], n)
        samples = self.memory[indices, :]

        s = samples[:, 0: self.nb_feature]
        a = samples[:, self.nb_feature + 1]
        r = samples[:, self.nb_feature + 2]
        s_ = samples[:, :self.nb_feature]

        return s, a, r, s_

