from keras.layers import Dense, Dropout
from keras import optimizers, regularizers
from keras.models import Sequential, load_model
import numpy as np

class Actor:
    def __init__(self, n_features, n_actions, lr, load=False):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        if load:
            self.eval_model = load_model('./models/actor.m')
        else:
            self.eval_model = self.build_model()
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

    def choose_action(self, s):
        s = s[np.newaxis, :]
        actions = self.eval_model.predict(s)
        choice = np.random.choice(self.n_actions, p=actions.flatten())
        return choice

    def random_action(self):
        choice = np.random.choice(self.n_actions)
        return choice

class Critic:
    def __init__(self, n_features, lr, load=False):
        self.n_features = n_features
        self.lr = lr

        if load:
            self.eval_model = load_model('./models/critic.m')
        else:
            self.eval_model = self.build_model()
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

    def save_model(self):
        self.eval_model.save('./models/critic.m')

    def predict(self, s):
        return self.eval_model.predict(s)
