import gym
from keras.utils import to_categorical
from ac.advantages_actor_critic import Actor, Critic, Memory

env = gym.make('CartPole-v0')

n_freatures = env.observation_space.shape[0]
n_actions = env.action_space.n
max_episode = 99999

actor = Actor(n_features=n_freatures, lr=0.001, n_actions=n_actions)
critic = Critic(n_features=n_freatures, lr=0.001)
memory = Memory()

for e in range(max_episode):
    obs = env.reset()
    total_reward = 0
    pos = 0
    neg = 0
    while True:
        env.render()
        a = actor.choose_action(obs)
        obs_, r, done, info = env.step(a)
        if done:
            r = -100 # 关于done时r的设计，参见下面注释。
        total_reward += r
        td = r + 0.9 * critic.eval(obs_)
        '''
        更新actor时
          若td_error恒>=0，则所有更新会增加pi_(s,a)的选择概率，只是不同的a增加的幅度不同;
          若td_error恒<=0，则所有更新会降低pi_(s,a)的选择概率，只是不同的a减小的幅度不同;
          设计r done时，尽量保证E[td_error] = 0，收敛会更快更稳定，这也符合advantages函数的思想，
          否者很容易过快收敛到局部最优，比如r=-100，这种情况下可以适当降低学习速度来防止过快收敛。
        '''
        if (td - critic.eval(obs)) > 0:
            pos += 1
        else:
            neg += 1

        actor.learn(obs,  to_categorical(a, num_classes=2) * (td - critic.eval(obs)))
        critic.learn(obs, td)
        obs = obs_
        if done:
            print('[episode]:', e, ' [rewards]: ', total_reward)
            print('pos:', pos, ' neg:', neg)
            break