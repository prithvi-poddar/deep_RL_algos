from typing import DefaultDict
import numpy as np
import agents as Agents
from utils import *
import argparse, os
from gym import wrappers

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q Learning Algorithms for Atari')
    parser.add_argument('-n_games', type=int, default=1,
                        help='Number of games to play')
    parser.add_argument('-lr', type=float, default=0.0001,
                        help='Learning rate for optimizer')
    parser.add_argument('-eps_min', type=float, default=0.1,
                        help='Minimum value for epsilon')
    parser.add_argument('-gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('-eps_dec', type=float, default=1e-5,
                        help='Linear factor for decreasing epsilon')
    parser.add_argument('-eps', type=float, default=1.0,
                        help='Starting value of epsilon')
    parser.add_argument('-max_mem', type=int, default=50000,
                        help='Size fo memory buffer')
    parser.add_argument('-repeat', type=int, default=4,
                        help='Number of frames to repeat & stack for environment')
    parser.add_argument('-bs', type=int, default=32,
                        help='Batch size')
    parser.add_argument('-replace', type=int, default=1000,
                        help='interval for replacing target network')
    parser.add_argument('-env', type=str, default='PongNoFrameskip-v4',
                        help='Atari Environment\nPongNoFrameskip-v4\n \
                            BreakoutNoFrameskip-v4\n \
                            SpaceInvadersNoFrameskip-v4\n \
                            EnduroNoFrameskip-v4\n \
                            AtlantisNoFrameskip-v4')
    parser.add_argument('-gpu', type=str, default='0',
                        help='GPU: 0 or 1')
    parser.add_argument('-load_checkpoint', type=bool, default=False,
                        help='load model checkpoint')
    parser.add_argument('-path', type=str, default='tmp/',
                        help='path for model saving/loading')
    parser.add_argument('-algo', type=str, default='DQNAgent',
                        help='DQNAgent / DDQNAgent / DuelingDQNAgent / DuelingDDQNAgent')
    parser.add_argument('-clip_rewards', type=bool, default=False,
                        help='Clip rewards to range -1 to 1')
    parser.add_argument('-no_ops', type=int, default=0,
                        help='Max number of no ops for testing')
    parser.add_argument('-fire_first', type=bool, default=False,
                        help='Set first action of episode to fire')
    parser.add_argument('-render', type=bool, default=False,
                        help='Set True to generate renders')
    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    env = make_env(env_name=args.env, repeat=args.repeat,
                    clip_rewards=args.clip_rewards, no_ops=args.no_ops,
                    fire_first=args.fire_first)

    best_score = -np.inf
    agent_ = getattr(Agents, args.algo)
    agent = agent_(gamma=args.gamma,
                    epsilon=args.eps,
                    lr=args.lr,
                    input_dims=env.observation_space.shape,
                    n_actions=env.action_space.n,
                    mem_size=args.max_mem,
                    eps_min=args.eps_min,
                    batch_size=args.bs,
                    replace=args.replace,
                    eps_dec=args.eps_dec,
                    chkpt_dir=args.path,
                    algo=args.algo,
                    env_name=args.env)

    if args.load_checkpoint:
        agent.load_models()

    if args.render:
        try:
            os.makedirs('renders', exist_ok = True)
        except OSError as error:
            print("Directory '%s' can not be created")
        env = wrappers(env, 'renders',
                        video_callable=lambda episode_id:True,
                        force=True)

    fname = args.algo+'_'+args.env+'_lr'+str(args.lr)+'__'+str(args.n_games)+'games'
    try:
        os.makedirs('plots', exist_ok = True)
    except OSError as error:
        print("Directory '%s' can not be created")
    figure_file = 'plots/'+fname+'.png'
    n_steps=0
    scores, eps_history, steps_array = [], [], []

    for i in range(args.n_games):
        done = False
        score = 0
        observation = env.reset()

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not args.load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, 
                                        int(done))
                agent.learn()

            observation = observation_
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])

        print('episode', i, ' score:', score, 
                ' average score %.1f best score %.1f epsilon %.2f'%
                (avg_score, best_score, agent.epsilon), ' steps:', str(n_steps))

        if avg_score > best_score:
            if not args.load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    plot_learning_curve(steps_array, scores, eps_history, figure_file)