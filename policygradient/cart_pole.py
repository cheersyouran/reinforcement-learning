import gym
import numpy as np
from keras.utils import to_categorical

from policygradient.advantages_actor_critic import Actor, Critic, Memory

env = gym.make('CartPole-v0')
# env = env.unwrapped
np.random.seed(0)

n_freatures = env.observation_space.shape[0]
n_actions = env.action_space.n

actor = Actor(n_features=n_freatures, lr=0.01, n_actions=n_actions)
critic = Critic(n_features=n_freatures, lr=0.01)
memory = Memory()

for e in range(10000):
    obs = env.reset()
    total_reward = 0
    while True:
        env.render()
        a = actor.choose_action(obs)
        obs_, r, done, info = env.step(a)
        if done:
            r = 0
        total_reward += r
        td = r + 0.9 * critic.eval(obs_)
        actor.learn(obs,  to_categorical(a, num_classes=2) * (td - critic.eval(obs)))
        critic.learn(obs, td)
        obs = obs_
        if done:
            print('[episode]:', e, ' [rewards]: ', total_reward)
            break

print('Finish!')