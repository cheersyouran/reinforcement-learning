import gym
import numpy as np
from keras.utils import to_categorical
from ac.demonattack.advantages_actor_critic import Actor, Critic, Memory

env = gym.make('DemonAttack-ram-v0')

n_features = env.observation_space.shape[0]
n_actions = env.action_space.n
max_episode = 99999
random_episodes = 50
memory_size = 50000
actor = Actor(n_features=n_features, lr=0.001, n_actions=n_actions, load=False)
critic = Critic(n_features=n_features, lr=0.001, load=False)
memory = Memory(memory_size, 2 * n_features + 2 + 1, n_features)

all_reward = 0
for e in range(1, max_episode):
    obs = env.reset()
    epo_reward = 0
    lives_cnt = 4
    steps_of_last_reward = 0

    for epo_steps in range(99999999):
        env.render()
        a = actor.random_action() if e <= random_episodes else actor.eval_choose_action(obs)
        obs_, r, done, info = env.step(a)
        if lives_cnt - info['ale.lives'] == 1 or steps_of_last_reward > 500:
            r = -30
            done = True
        lives_cnt = info['ale.lives']
        steps_of_last_reward = steps_of_last_reward + 1 if r == 0 else 0

        memory.store_transition(obs, a, r/10, obs_, epo_steps)
        obs = obs_
        epo_reward += r

        if e > random_episodes:
            states, actions, rewards, states_ = memory.sample(backward_length=16, gamma=0.95)
            for s in range(50):
                state, action, reward, state_ = memory.sample(backward_length=16, gamma=0.95)
                states = np.vstack((states, state))
                actions = np.vstack((actions, action))
                rewards = np.vstack((rewards, reward))
                states_ = np.vstack((states_, state_))

            td = rewards + 0.95 * critic.target_predict(states_)
            actor.learn(states, to_categorical(actions, num_classes=n_actions) * (td - critic.eval_predit(states)))
            critic.learn(states, td)

        if done:
            all_reward = all_reward * 0.9 + epo_reward * 0.1
            print('[回合]:', e, ' [回合回报]:', epo_reward, '[累计回报]:', all_reward, "[局面估计]：", critic.target_predict(obs[np.newaxis, :]),
                  '动作：',np.max(actor.target_model.predict(obs_[np.newaxis, :])))
            break