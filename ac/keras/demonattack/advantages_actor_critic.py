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
        self.update_steps = 500
        self.step_counter = 1

        a = Input(shape=(self.n_features,))
        b = Dense(128, activation='relu', kernel_initializer='random_uniform')(a)
        c = Dropout(0.5)(b)
        d = Dense(64, activation='relu', kernel_initializer='random_uniform')(c)
        e = Dropout(0.5)(d)
        f = Dense(32, activation='relu', kernel_initializer='random_uniform')(e)
        g = Dropout(0.5)(f)
        h = Dense(16, activation='relu', kernel_initializer='random_uniform')(g)
        j = Dropout(0.5)(h)
        o = Dense(self.n_actions, activation='softmax', kernel_initializer='random_uniform')(j)

        self.eval_model = Model(inputs=a, outputs=o)
        self.target_model = Model(inputs=a, outputs=o)
        self.eval_model.compile(loss='categorical_crossentropy', optimizer=sgd(lr=self.lr), metrics=['mse', 'accuracy'])

    def learn(self, s, q):
        self.eval_model.train_on_batch(s, q)
        if self.update_steps % self.step_counter == 1:
            self.target_model.set_weights(self.eval_model.get_weights())
            self.step_counter = 1
        else:
            self.step_counter += 1

    def eval_choose_action(self, s):
        s = s[np.newaxis, :]
        actions = self.eval_model.predict(s)
        if (actions.flatten() > 0.5).any():
            s
        choice = np.random.choice(self.n_actions, p=actions.flatten())
        return choice

    def target_choose_action(self, s):
        s = s[np.newaxis, :]
        actions = self.target_model.predict(s)
        choice = np.random.choice(self.n_actions, p=actions.flatten())
        return choice

    def random_action(self):
        choice = np.random.choice(self.n_actions)
        return choice

class Critic:
    def __init__(self, n_features, lr):
        self.n_features = n_features
        self.lr = lr
        self.update_steps = 500
        self.step_counter = 1

        a = Input(shape=(self.n_features,))
        b = Dense(128, activation='relu', kernel_initializer='random_uniform')(a)
        c = Dropout(0.5)(b)
        d = Dense(64, activation='relu', kernel_initializer='random_uniform')(c)
        e = Dropout(0.5)(d)
        f = Dense(32, activation='relu', kernel_initializer='random_uniform')(e)
        g = Dropout(0.5)(f)
        h = Dense(16, activation='relu', kernel_initializer='random_uniform')(g)
        j = Dropout(0.5)(h)
        o = Dense(1)(j)

        self.eval_model = Model(inputs=a, outputs=o)
        self.target_model = Model(inputs=a, outputs=o)
        self.eval_model.compile(loss='mse', optimizer=sgd(lr=self.lr), metrics=['mse', 'accuracy'])

    def learn(self, s, td):
        self.eval_model.train_on_batch(s, td)
        if self.update_steps % self.step_counter == 1:
            self.target_model.set_weights(self.eval_model.get_weights())
            self.step_counter = 1
        else:
            self.step_counter += 1

    def eval_predit(self, s):
        return self.eval_model.predict(s)

    def target_predict(self, s):
        return self.target_model.predict(s)

class Memory:
    def __init__(self, capacity, dims, nb_feature):
        self.nb_feature = nb_feature
        self.capacity = capacity
        self.memory = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_, epo_steps):
        index = self.pointer % self.capacity
        self.memory[index, :] = np.hstack((s, a, r, epo_steps, s_))
        self.pointer += 1

    def sample(self, backward_length, gamma):
        count = self.pointer if self.pointer < self.capacity else self.capacity
        indices = np.random.choice(count)
        states = []
        rewards = [0]
        states_ = []
        actions = []
        for i in range(backward_length):
            sample = self.memory[indices - i, :]
            states.append(sample[0: self.nb_feature])
            actions.append(sample[self.nb_feature])
            rewards.append(sample[self.nb_feature + 1] + gamma * rewards[-1])
            states_.append(sample[:self.nb_feature])
            if sample[self.nb_feature + 2] == 0:
                break
        rewards.pop(0)
        return np.array(states), np.array(actions)[:, np.newaxis], np.array(rewards)[:, np.newaxis], np.array(states_)

    def persist_memory(self):
        self.memory.to_csv('./memory/m.csv', index=False)

    def load_memory(self):
        self.memory = pd.read_csv('./memory/m.csv')

