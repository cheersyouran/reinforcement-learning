from keras.models import Model
from keras.layers import Input, Dense, Add, Dropout
from keras.optimizers import sgd
import numpy as np
import pandas as pd
from keras import backend as K

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
        b = Dense(20, activation='relu', kernel_initializer='random_uniform')(a)
        d = Dense(self.n_actions, activation='softmax', kernel_initializer='random_uniform')(b)
        self.model = Model(inputs=a, outputs=d)

        # def mycrossentropy(y_true, y_pred):
        #     return - K.log(y_pred) * y_true
        # self.model.compile(loss=mycrossentropy, optimizer=sgd(lr=self.lr), metrics=['mse', 'accuracy'])
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd(lr=self.lr), metrics=['mse', 'accuracy'])

    def learn(self, s, q):
        s = s[np.newaxis, :]
        self.model.train_on_batch(s, q)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        actions = self.model.predict(s)
        choice = np.random.choice(self.n_actions, p=actions.flatten())
        return choice

class Critic:
    def __init__(self, n_features, lr):
        self.n_features = n_features
        self.lr = lr

        a = Input(shape=(self.n_features,))
        b = Dense(20, activation='relu')(a)
        d = Dense(1)(b)
        self.model = Model(inputs=a, outputs=d)
        self.model.compile(loss='mse', optimizer=sgd(lr=self.lr), metrics=['mse', 'accuracy'])

    def learn(self, s, td):
        s = s[np.newaxis, :]
        self.model.train_on_batch(s, td)

    def eval(self, s):
        s = s[np.newaxis, :]
        return self.model.predict(s)