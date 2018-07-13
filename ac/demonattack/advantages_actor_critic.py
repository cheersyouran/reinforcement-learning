from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras import optimizers, regularizers
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd

'''
使用keras+loss函数
Actor Critic中的A2C算法，引入Advantages函数
'''

class Actor:
    def __init__(self, n_features, n_actions, lr, load=False):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.update_steps = 500
        self.step_counter = 1

        if load:
            self.eval_model = load_model('./models/actor.m')
            self.target_model = load_model('./models/actor.m')
        else:
            self.eval_model = self.build_model()
            self.target_model = self.build_model()
        sgd = optimizers.SGD(lr=self.lr, decay=1e-7, momentum=0.9, nesterov=True)
        self.eval_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mse', 'accuracy'])

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=self.n_features, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(self.n_actions, activation='softmax', kernel_initializer='random_uniform'))
        return model

    def save_model(self):
        self.eval_model.save('./models/actor.m')

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
    def __init__(self, n_features, lr, load=False):
        self.n_features = n_features
        self.lr = lr
        self.update_steps = 500
        self.step_counter = 1

        if load:
            self.eval_model = load_model('./models/critic.m')
            self.target_model = load_model('./models/critic.m')
        else:
            self.eval_model = self.build_model()
            self.target_model = self.build_model()
            sgd = optimizers.SGD(lr=self.lr, decay=1e-7, momentum=0.9, nesterov=True)
            self.eval_model.compile(loss='mse', optimizer=sgd, metrics=['mse', 'accuracy'])

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=self.n_features, kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='relu', kernel_initializer='random_uniform', kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dropout(0.3))
        model.add(Dense(1))
        return model

    def learn(self, s, td):
        self.eval_model.train_on_batch(s, td)
        if self.update_steps % self.step_counter == 1:
            self.target_model.set_weights(self.eval_model.get_weights())
            self.step_counter = 1
        else:
            self.step_counter += 1

    def save_model(self):
        self.eval_model.save('./models/critic.m')

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

