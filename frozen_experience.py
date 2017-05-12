import argparse
from time import time
import gym
from gym import spaces
import numpy as np
from agents import QNetworkAgent, QAgent

def run_episode(env, agent, max_step, epsilon, update=True):
    observation = env.reset()
    tmp_m = []
    for t in range(max_step):
        env.render()
        rand = np.random.rand()
        if rand < epsilon:
            action = env.action_space.sample()
        else:
            action = agent.next_action(observation)
        next_observation, reward, done, info = env.step(action)
        if update:
            agent.update(observation, action, next_observation, reward)
        observation = next_observation
        tmp_m.append(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    total_reward = np.sum(np.array(tmp_m))
    return total_reward

parser = argparse.ArgumentParser()
parser.add_argument('--max_step', type=int, default=100)
parser.add_argument('--n_episode', type=int, default=2000)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--n_hidden', type=int, default=10)
parser.add_argument('--emb_dim', type=int, default=10)
parser.add_argument('--neural', type=bool, default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    max_step = args.max_step
    epsilon = args.epsilon
    emb_dim = args.emb_dim
    n_hidden = args.n_hidden

    env = gym.make('FrozenLake8x8-v0')
    n_state = env.observation_space.n
    n_action = env.action_space.n
    if args.neural:
        agent = QNetworkAgent(n_state, n_action, n_hidden, emb_dim=emb_dim)
    else:
        agent = QAgent(n_state, n_action)
    n_episode = args.n_episode
    np.random.seed(int(time()))
    all_rewards = np.zeros(n_episode)

    for i in range(n_episode):
        tmp_reward = run_episode(env, agent, max_step, epsilon/(i+1), update=True)
        # if (i+1)%10==0:
        #     for j in range(10):
        #         total_rewards[(i+1)//10-1] += run_episode(env, agent, max_step, 0., update=False)
        # epsilon = 1./((i+50) + 10)
        # if (i+1)%25==0:
            # epsilon *= 0.9

        all_rewards[i] = tmp_reward

    print(np.mean(all_rewards))
