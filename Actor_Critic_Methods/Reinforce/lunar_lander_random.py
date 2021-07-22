import gym

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    # env = gym.wrappers.Monitor(env, './video/',video_callable=lambda episode_id: True,
    #                         force = True)


    n_games = 100

    for i in range(n_games):
        obs = env.reset()
        score = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs_, reward, done, info = env.step(action)
            score += reward
            #env.render
        
        print('episode ', i, ' score %.1f'%score)