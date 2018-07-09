import gym
import numpy as np
from keras.utils import to_categorical
from ac.keras.breakout.advantages_actor_critic import Actor, Critic, Memory

env = gym.make('DemonAttack-ram-v0')

n_features = env.observation_space.shape[0]
n_actions = env.action_space.n
max_episode = 99999

actor = Actor(n_features=n_features, lr=0.0002, n_actions=n_actions)
critic = Critic(n_features=n_features, lr=0.0002)
memory = Memory(50000, 2 * n_features + 2, n_features)

all_reward = 0
for e in range(max_episode):
    obs = env.reset()
    epo_reward = 0
    pos = 0
    neg = 0
    lives_cnt = 4
    while True:
        # env.render()
        a = actor.random_action() if e < 5 else actor.choose_action(obs)
        obs_, r, done, info = env.step(a)

        if lives_cnt - info['ale.lives'] == 1:
            r = - 20
        lives_cnt = info['ale.lives']

        memory.store_transition(obs, a, r / 10, obs_)

        state, action, reward, state_ = memory.sample(64)
        td = r + 0.95 * critic.eval(state_)
        '''
        更新actor时
          若td_error恒>=0，则所有更新会增加pi_(s,a)的选择概率，只是不同的a增加的幅度不同;
          若td_error恒<=0，则所有更新会降低pi_(s,a)的选择概率，只是不同的a减小的幅度不同;
          设计r done时，尽量保证E[td_error] = 0，收敛会更快更稳定，这也符合advantages函数的思想，
          否者很容易过快收敛到局部最优，比如r=-100，这种情况下可以适当降低学习速度来防止过快收敛。
        '''
        pos += np.sum(td - critic.eval(state_) > 0)
        neg += np.sum(td - critic.eval(state_) <= 0)

        actor.learn(state,  to_categorical(a, num_classes=n_actions) * (td - critic.eval(state)))
        critic.learn(state, td)

        obs = obs_
        epo_reward += r

        if done:
            all_reward = all_reward * 0.9 + epo_reward * 0.1
            print('[e]:', e, ' [epo-r]:', epo_reward, '[all-r]:', all_reward)
            # print('pos:', pos, ' neg:', neg)
            break