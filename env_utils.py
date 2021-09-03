import numpy as np
import gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch
import dmc2gym
from collections import deque

def get_state(env):
    ##Gets the current state of the environment to allow for true reward calculations - will need to be changed depending on the env/task
    name = env.unwrapped.spec.id[:10]
    if name == 'dmc_reache':
        state = env.physics.named.data.geom_xpos['finger', :2]
        state2 = env.physics.named.data.geom_xpos['arm', :2]
        toret = [state[0], state[1], state2[0], state2[1]]
    elif name == 'dmc_point_':
        state = env.physics.named.data.geom_xpos['pointmass', :2]
        toret = [state[0], state[1]]
    elif name == 'FetchReach':
        state = env.sim.get_state()
    return np.array(toret)

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)

class GoalBasedPixelObservationsDMControl(gym.Wrapper):
    ##This is for the dm control, with single frames
    def __init__(self, env, state_based):
        gym.Wrapper.__init__(self, env)
        self.env = env #maybe unneccessary?
        self.desired_goal = None
        self.desired_goal_state = None
        self._max_episode_steps = env._max_episode_steps
        self.state_based = state_based

    def reset(self):
        self.desired_goal = self.env.reset()
        self.desired_goal_state = get_state(self.env)
        obs = self.env.reset()
        obs_state = get_state(self.env)
        achieved_goal = obs
        achieved_goal_state = obs_state
        if not self.state_based:
            self.last_obs =  {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy(),
            'observation_state': obs_state.copy(),
            'achieved_goal_state': achieved_goal_state.copy(),
            'desired_goal_state': self.desired_goal_state.copy()
            }
        else:
            self.last_obs =  {
            'observation': obs_state.copy(),
            'achieved_goal': achieved_goal_state.copy(),
            'desired_goal': self.desired_goal_state.copy(),
            'observation_state': obs_state.copy(),
            'achieved_goal_state': achieved_goal_state.copy(),
            'desired_goal_state': self.desired_goal_state.copy()
            }
        return self.last_obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_state = get_state(self.env)
        achieved_goal = obs
        achieved_goal_state = obs_state
        info['is_success'] = np.sqrt(((obs_state - self.desired_goal_state) ** 2).sum()) < 0.03
        if not self.state_based:
            self.last_obs = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy(),
            'observation_state': obs_state.copy(),
            'achieved_goal_state': achieved_goal_state.copy(),
            'desired_goal_state': self.desired_goal_state.copy()
            }
        else:
            self.last_obs =  {
            'observation': obs_state.copy(),
            'achieved_goal': achieved_goal_state.copy(),
            'desired_goal': self.desired_goal_state.copy(),
            'observation_state': obs_state.copy(),
            'achieved_goal_state': achieved_goal_state.copy(),
            'desired_goal_state': self.desired_goal_state.copy()
            }

        return self.last_obs, reward, done, info

class GoalBasedPixelObservationsOpenAIGym(gym.Wrapper):
    def __init__(self, env, state_based):
        gym.Wrapper.__init__(self, env)
        self.env = env #maybe unneccessary?
        self.desired_goal = None
        self.desired_goal_state = None
        self._max_episode_steps = env._max_episode_steps
        self.state_based = state_based
        name = env.unwrapped.spec.id[:10]
        if name == 'FetchReach':
            payload = torch.load('/home/aaron_putterman/project1/hindsight-experience-replay-master/fetchreachgoals/goals3.pt')
            self.states = payload[0] ## 1000 x 10
            self.renderings = payload[1] ## 1000 x 100 x 100 x 3

    def reset(self):
        idx = np.random.randint(0, self.renderings.shape[0])
        self.desired_goal = self.renderings[idx].numpy()
        self.desired_goal = np.swapaxes(self.desired_goal, 1, 2)
        self.desired_goal = np.swapaxes(self.desired_goal, 0, 1)
        self.desired_goal_state = self.states[idx].numpy()
        obs_state = self.env.reset()
        obs_state = obs_state['observation']
        self.env.env._get_viewer(mode = 'rgb_array').render(width = 100, height = 100, camera_id = 3)
        rendered_obs3 = self.env.env._get_viewer(mode = 'rgb_array').read_pixels(width = 100, height = 100, depth = False)
        obs = rendered_obs3[::-1, :, :]
        # obs = self.env.render(mode = 'rgb_array', height = 100, width = 100)
        obs = np.swapaxes(obs, 1, 2)
        obs = np.swapaxes(obs, 0, 1)
        achieved_goal = obs
        achieved_goal_state = obs_state
        if not self.state_based:
            self.last_obs =  {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy(),
            'observation_state': obs_state.copy(),
            'achieved_goal_state': achieved_goal_state.copy(),
            'desired_goal_state': self.desired_goal_state.copy()
            }
        else:
            self.last_obs =  {
            'observation': obs_state.copy(),
            'achieved_goal': achieved_goal_state.copy(),
            'desired_goal': self.desired_goal_state.copy(),
            'observation_state': obs_state.copy(),
            'achieved_goal_state': achieved_goal_state.copy(),
            'desired_goal_state': self.desired_goal_state.copy()
            }
        return self.last_obs

    def step(self, action):
        obs_state, reward, done, info = self.env.step(action)
        obs_state = obs_state['observation']
        # obs = self.env.render(mode = 'rgb_array', height = 100, width = 100)
        self.env.env._get_viewer(mode = 'rgb_array').render(width = 100, height = 100, camera_id = 3)
        rendered_obs3 = self.env.env._get_viewer(mode = 'rgb_array').read_pixels(width = 100, height = 100, depth = False)
        obs = rendered_obs3[::-1, :, :]
        obs = np.swapaxes(obs, 1, 2)
        obs = np.swapaxes(obs, 0, 1)
        achieved_goal = obs
        achieved_goal_state = obs_state
        info['is_success'] = np.sqrt(((obs_state - self.desired_goal_state) ** 2).sum()) < 0.03
        if not self.state_based:
            self.last_obs = {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.desired_goal.copy(),
            'observation_state': obs_state.copy(),
            'achieved_goal_state': achieved_goal_state.copy(),
            'desired_goal_state': self.desired_goal_state.copy()
            }
        else:
            self.last_obs =  {
            'observation': obs_state.copy(),
            'achieved_goal': achieved_goal_state.copy(),
            'desired_goal': self.desired_goal_state.copy(),
            'observation_state': obs_state.copy(),
            'achieved_goal_state': achieved_goal_state.copy(),
            'desired_goal_state': self.desired_goal_state.copy()
            }

        return self.last_obs, reward, done, info




def get_env_params(env, args):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape,
            'goal': obs['desired_goal'].shape,
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params

def create_env(args):
    ##TODO add support for openai gym environments
    if args.env_name == 'reacher' or args.env_name == 'point_mass':
        task_name = 'easy'
        env = dmc2gym.make(
        domain_name = args.env_name,
        task_name = task_name,
        seed = args.seed,
        visualize_reward = False,
        from_pixels = (args.encoder_type == 'pixel'),
        height = args.pre_transform_image_size,
        width = args.pre_transform_image_size,
        frame_skip = args.action_repeat,
        )
        # if args.encoder_type == 'pixel':
        env = FrameStack(env, k = args.frame_stack)
        env = GoalBasedPixelObservationsDMControl(env, args.encoder_type != 'pixel')
        return env
    elif args.env_name == 'FetchReach-v1':
        env = gym.make(args.env_name)
        env = GoalBasedPixelObservationsOpenAIGym(env, args.encoder_type != 'pixel')
        return env
    else:
        print("not yet implemented in env_utils.py - create_env")
