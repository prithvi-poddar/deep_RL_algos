import gym
import numpy as np
import matplotlib.pyplot as plt 
from agent import Agent

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    agent = Agent(lr=0.001, gamma=0.9, eps_start=1.0, eps_end=0.01, eps_dec=0.9999995,
                    n_actions=4, n_states=16)
    
    scores = []
    win_pct_list = []
    n_games = 1000000

    for i in range(n_games):
        done = False
        obs = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, obs_)
            score += reward
            obs = obs_
        scores.append(score)

        if i%100 == 0:
            win_pct = np.mean(scores[-100:])
            win_pct_list.append(win_pct)
            if i%1000==0:
                print('episode ', i, ' win pct %.2f '%win_pct, ' epsilon %.2f'%agent.epsilon)

    plt.plot(win_pct_list)
    plt.savefig("sample.jpg")