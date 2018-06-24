from keras.models import Model
from keras.layers import Input, Dense, Add
from keras.optimizers import sgd
import tensorflow as tf
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
        b = Dense(20, activation='relu')(a)
        d = Dense(self.n_actions, activation='softmax')(b)
        self.model = Model(inputs=a, outputs=d)

        def mycrossentropy(y_true, y_pred):
            return - K.log(y_pred) * y_true
        self.model.compile(loss=mycrossentropy, optimizer=sgd(lr=self.lr), metrics=['mse', 'accuracy'])

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

class Memory:
    def __init__(self):
        self.memory = pd.DataFrame(columns=['s', 'a', 'r', 's_'])

    def store_transition(self, s, a, r, s_):
        self.memory = self.memory.append({'s': s, 'a': a, 'r': r, 's_': s_}, ignore_index=True)

    def sample(self, n):
        indices = np.random.choice(self.memory.shape[0], n)
        samples = self.memory.iloc[indices, :]

        s = np.vstack(samples['s'])
        r = np.vstack(samples['r'])
        a = np.vstack(samples['a'])
        s_ = np.vstack(samples['s_'])

        return s, r, a, s_

class Actor_tf():

    def __init__(self, n_features, n_actions, lr, sess):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.sess = sess
        self.model = None

        K.set_session(sess)

        a = Input(shape=(self.n_features,))
        b = Dense(20, activation='relu')(a)
        d = Dense(self.n_actions, activation='softmax')(b)
        self.model = Model(inputs=a, outputs=d)

        grads = tf.gradients(d, self.model.weights)
        self.q = tf.placeholder(tf.float32, [None, n_actions])
        self.train_op = tf.train.AdamOptimizer(0.001).apply_gradients(zip(grads, self.model.weights))
        self.sess.run(tf.initialize_all_variables())

    def learn(self, s, q):
        self.sess.run(self.optimize, feed_dict={
            self.state: s,
            self.q: q
        })

    def choose_action(self, s):
        s = s[np.newaxis, :]
        actions = self.model.predict(s)
        choice = np.random.choice(self.n_actions, p=actions.flatten())
        return choice
