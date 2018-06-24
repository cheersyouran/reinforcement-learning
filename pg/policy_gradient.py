from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import sgd
import numpy as np
import pandas as pd
np.random.seed(0)

'''
Policy Gradient中的REINIFORCE算法，每回合更新一次
'''

class PG:
    def __init__(self, n_features, n_actions, lr):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.model = None

        a = Input(shape=(self.n_features,))
        b = Dense(20, activation='relu')(a)
        d = Dense(self.n_actions, activation='softmax')(b)
        self.model = Model(inputs=a, outputs=d)

        self.model.compile(loss='categorical_crossentropy', optimizer=sgd(lr=self.lr), metrics=['mse', 'accuracy'])

    def learn(self, s, q):
        s = s[np.newaxis, :]
        q = q[np.newaxis, :]
        self.model.train_on_batch(s, q)

    def choose_action(self, s):
        s = s[np.newaxis, :]
        actions = self.model.predict(s)
        choice = np.random.choice(self.n_actions, p=actions.flatten())
        return choice
