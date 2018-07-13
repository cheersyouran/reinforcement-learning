import gym
import numpy as np
from keras.utils import to_categorical
from ac.demonattack2.advantages_actor_critic import Actor, Critic

env = gym.make('DemonAttack-ram-v0')

n_features = env.observation_space.shape[0]
n_actions = env.action_space.n
max_episode = 99999
random_episodes = 50

actor = Actor(n_features=n_features, lr=0.001, n_actions=n_actions, load=False)
critic = Critic(n_features=n_features, lr=0.001, load=False)

all_reward = 0

for e in range(1, max_episode):
    obs = env.reset()
    epo_reward = 0
    lives_cnt = 4
    steps_of_last_reward = 0

    states = []
    states_ = []
    actions = []

    while True:
        env.render()
        a = actor.random_action() if e <= random_episodes else actor.choose_action(obs)
        obs_, r, done, info = env.step(a)
        if lives_cnt - info['ale.lives'] == 1 or steps_of_last_reward > 500:
            r = -30
            done = True
        lives_cnt = info['ale.lives']

        states.append(obs)
        actions.append(a)
        states_.append(obs_)

        if r != 0:
            rewards = r * np.logspace(0, len(states) - 1, len(states), base=0.96)
            td = rewards[:, np.newaxis] + 0.95 * critic.predict(np.array(states_))
            actor.learn(np.array(states), to_categorical(actions, num_classes=n_actions) * (td - critic.predict(np.array(states))))
            critic.learn(np.array(states), td)

        obs = obs_
        epo_reward += r
        if done:
            all_reward = all_reward * 0.95 + epo_reward * 0.05
            print('[回合]:', e, ' [回合回报]:', epo_reward, '[累计回报]:', all_reward, "[局面估计]：", critic.predict(obs[np.newaxis, :]),
                  '动作：', np.max(actor.eval_model.predict(obs_[np.newaxis, :])))
            break