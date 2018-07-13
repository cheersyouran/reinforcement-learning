from keras.models import Model
from keras.layers import Input, Dense, Add
from keras.optimizers import sgd
import tensorflow as tf
import numpy as np
from keras import backend as K

'''
使用tf.gradients计算梯度，避免loss函数。
Actor Critic中的A2C算法，引入Advantages函数.
'''
class Actor():

    def __init__(self, n_features, n_actions, lr, sess):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.sess = sess
        self.model = None

        K.set_session(sess)

        inputs = Input(shape=(self.n_features,))
        h1 = Dense(20, activation='relu')(inputs)
        outputs = Dense(self.n_actions, activation='softmax')(h1)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.s = inputs
        self.q = tf.placeholder(tf.float32, [None, n_actions])
        grads = tf.gradients(-tf.log(outputs)*self.q, self.model.weights)
        self.train_op = tf.train.AdamOptimizer(0.001).apply_gradients(zip(grads, self.model.weights))
        self.sess.run(tf.initialize_all_variables())

    def learn(self, s, q):
        s = s[np.newaxis, :]
        self.sess.run(self.train_op, feed_dict={
            self.s: s,
            self.q: q
        })

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
