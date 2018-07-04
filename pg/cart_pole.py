import gym
from keras.utils import to_categorical
from pg.policy_gradient import PG

env = gym.make('CartPole-v0')

n_freatures = env.observation_space.shape[0]
n_actions = env.action_space.n
max_step = 9999
max_episode = 99999

# 注意V_t的算法，需要reversed
def calcu_vt(step_reward, t):
    all = 0
    for r in reversed(step_reward[t:]):
        all = r + 0.9 * all
    return all

pg = PG(n_features=n_freatures, lr=0.005, n_actions=n_actions)

for e in range(max_episode):
    obs = env.reset()
    step_reward = []
    step_obs = []
    step_action = []
    for t in range(max_step):
        env.render()
        a = pg.choose_action(obs)
        obs_, r, done, info = env.step(a)
        if done:
            r = -10
        '''
        更新policy时
          若vt恒>=0，则所有更新会增加pi_(s,a)的选择概率，只是不同的a增加的幅度不同;
          若vt恒<=0，则所有更新会降低pi_(s,a)的选择概率，只是不同的a减小的幅度不同;
          设计r done时，尽量保证E[vt] = 0，收敛会更快更稳定，这也符合advantages函数的思想。
        '''
        step_reward.append(r)
        step_obs.append(obs)
        step_action.append(a)
        obs = obs_
        if done:
            for i in range(len(step_reward)):
                v_t = calcu_vt(step_reward, i)
                obs = step_obs[i]
                a = step_action[i]
                pg.learn(obs, to_categorical(a, num_classes=n_actions) * v_t)
            print('[episode]:', e, ' [rewards]: ', t)
            break