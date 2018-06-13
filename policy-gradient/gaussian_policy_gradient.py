import numpy as np
import random
import tensorflow as tf

# 用神经网络拟合一个mean

def build_model(inputs):
    net = tf.layers.dense(inputs, units=hidden_size, activation=tf.nn.relu)
    net = tf.layers.dense(net, units=hidden_size, activation=tf.nn.relu)
    mean = tf.layers.dense(net, 1)
    mean = tf.squeeze(mean, -1)
    std = tf.layers.dense(net, 1)
    std = tf.nn.softplus(std) * 10
    # std = tf.exp(std)
    std = tf.squeeze(std, -1)
    normal = tf.distributions.Normal(loc=mean, scale=std)
    return normal