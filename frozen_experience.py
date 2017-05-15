import argparse
from time import time
import gym
from gym import spaces
import numpy as np
from agents import QNetworkAgent, QAgent

def run_episode(env, agent, max_step, epsilon, update=True):
    observation = env.reset()
    total_reward = 0
    for t in range(max_step):
        env.render()
        rand = np.random.rand()
        if rand < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.next_action(observation)
        next_observation, reward, done, _ = env.step(action)
        if update:
            agent.update(observation, action, next_observation, reward)
        observation = next_observation
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    return total_reward

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:] / n

parser = argparse.ArgumentParser()
parser.add_argument('--large_lake',  action='store_true', default=False)
parser.add_argument('--max_step', type=int, default=100)
parser.add_argument('--n_episode', type=int, default=2000)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--n_hidden', type=int, default=20)
parser.add_argument('--l_rate', type=float, default=0.5)
parser.add_argument('--emb_dim', type=int, default=10)
parser.add_argument('--neural',  action='store_true', default=False)

if __name__ == "__main__":
    args = parser.parse_args()
    large = args.large_lake
    max_step = args.max_step
    epsilon = args.epsilon
    emb_dim = args.emb_dim
    n_hidden = args.n_hidden
    l_rate = args.l_rate

    env = gym.make('FrozenLake8x8-v0' if large else 'FrozenLake-v0')
    n_state = env.observation_space.n
    n_action = env.action_space.n
    if args.neural:
        agent = QNetworkAgent(n_state, n_action,
                             n_hidden, emb_dim=emb_dim, l_rate=l_rate)
    else:
        agent = QAgent(n_state, n_action, l_rate=l_rate)
    n_episode = args.n_episode
    np.random.seed(int(time()))

    results = []
    for _ in range(10):
        all_rewards = np.zeros(n_episode)
        for i in range(n_episode):
            total_reward = run_episode(env, agent, max_step, epsilon, update=True)
            if (i+1)%100==0:
                epsilon /= 10

            all_rewards[i] = total_reward

        print(np.mean(all_rewards))
        results.append(moving_average(all_rewards, 10))
