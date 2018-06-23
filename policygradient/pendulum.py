import gym
from policygradient.advanced_actor_critic import Actor, Critic

env = gym.make('Pendulum-v0')
env.reset()

n_freatures = env.observation_space.shape[0]
n_actions = env.action_space.n

actor = Actor(n_features=n_freatures, lr=0.001, n_actions=n_actions)
critic = Critic(n_features=n_freatures, lr=0.01)


for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())


