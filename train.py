import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
from env_utils import get_env_params, create_env
from utils import make_dir
from logger import Logger
from video import VideoRecorder
import time
import json

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""

def launch(args):
    # create the ddpg_agent
    # env = gym.make(args.env_name)
    env = create_env(args)
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env, args)
    # set up everything for logging
    ts = time.gmtime()
    ts = time.strftime("%m-%d", ts) + "-" + time.strftime("%T", ts)
    env_name = args.env_name
    exp_name = env_name + '-' + ts + '-' + str(args.seed)
    args.work_dir = args.work_dir + '/' + exp_name
    make_dir(args.work_dir)
    #store arguments
    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    args.video_dir = make_dir(os.path.join(args.work_dir, 'video'))
    args.video = VideoRecorder(args.video_dir)
    L = Logger(args.work_dir, use_tb = args.save_tb)
    # create the ddpg agent to interact with the environment
    ddpg_trainer = ddpg_agent(args, env, env_params, L)
    ddpg_trainer.learn()

if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
