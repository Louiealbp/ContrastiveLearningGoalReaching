import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic, contrastive_learner, unchanging_contrastive_learner
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from utils import calculate_reward_func, center_translate

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, args, env, env_params, L):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.L = L
        self.step = 0
        # create the network
        self.actor_network = actor(env_params, args)
        self.critic_network = critic(env_params, args)
        if args.encoder_type == 'pixel':
            self.actor_network.encoder.copy_conv_weights_from(self.critic_network.encoder)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params, args)
        self.critic_target_network = critic(env_params, args)
        if args.encoder_type == 'pixel':
            self.actor_target_network.encoder.copy_conv_weights_from(self.critic_target_network.encoder)
            self.contrastive_learner = contrastive_learner(env_params, args, self.critic_network, self.critic_target_network)
        if args.encoder_type == 'pixel' and args.load_reward_curl:
            tempcritic1 = critic(env_params, args)
            tempcritic2 = critic(env_params, args)
            self.reward_contrastive = unchanging_contrastive_learner(env_params, args, tempcritic1, tempcritic2)
            self.reward_contrastive.load_state_dict(
                torch.load('%s/curl_%s.pt' % (args.load_reward_curl_path, args.load_reward_curl_step))
            )
            if self.args.cuda:
                self.reward_contrastive.cuda()
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
            if self.args.encoder_type == 'pixel':
                self.contrastive_learner.cuda()
        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        if self.args.encoder_type == 'pixel':
            self.contrastive_learner_optim = torch.optim.Adam(self.contrastive_learner.parameters(), lr=self.args.lr_contrastive)
            self.encoder_optim = torch.optim.Adam(self.critic_network.encoder.parameters(), lr = self.args.encoder_lr)
        # her sampler
        if self.args.env_name == 'reacher' or self.args.env_name == 'point_mass' or self.args.env_name == 'FetchReach-v1':
            reward_func = calculate_reward_func(self)
        else:
            reward_func = self.env.compute_reward
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, reward_func)
        # create the replay buffer
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # create the normalizer
        if self.args.encoder_type != 'pixel':
            self.o_norm = normalizer(size=env_params['obs'][0], default_clip_range=self.args.clip_range)
            self.g_norm = normalizer(size=env_params['goal'][0], default_clip_range=self.args.clip_range)
        # create the dict for store the model
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)
            # path to save the model
            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
                for _ in range(self.args.num_rollouts_per_mpi):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            if self.args.encoder_type != 'pixel':
                                input_tensor = self._preproc_inputs(obs, g)
                            else:
                                input_tensor = self._preproc_inputs_pixel(obs, g)
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        # feed the actions into the environment
                        observation_new, _, _, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # append rollouts
                        ep_obs.append(obs.copy())
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action.copy())
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # convert them into arrays
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                if self.args.encoder_type != 'pixel':
                    self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0:
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                self.L.log('eval/success_rate', success_rate, step = self.step)
                if self.args.encoder_type != 'pixel':
                    torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict()], \
                            self.model_path + '/model.pt')
                else:
                    torch.save(
                    [self.actor_network.state_dict()],
                    self.model_path + '/model.pt',
                            )

    # pre_process the inputs
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs

    #pre_process the inputs if they are pixel-based
    def _preproc_inputs_pixel(self, obs, g):
        #TODO maybe add more preprocessing steps? unsure
        obs = center_translate(obs[None], self.args.image_size)[0]
        g = center_translate(g[None], self.args.image_size)[0]
        inputs = np.concatenate([obs, g])
        inputs = torch.tensor(inputs, dtype = torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs



    # this function will choose action for the agent and do the exploration
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action

    # update the normalizer
    def _update_normalizer(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        mb_obs_next = mb_obs[:, 1:, :]
        mb_ag_next = mb_ag[:, 1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[1]
        # create the new buffer to store them
        buffer_temp = {'obs': mb_obs,
                       'ag': mb_ag,
                       'g': mb_g,
                       'actions': mb_actions,
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        # recompute the stats
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g
    #pre_process the inputs if they are pixel-based
    def _preproc_og_pixel(self, obs, g):
        #TODO maybe add more preprocessing steps? unsure
        obs = center_translate(obs, self.args.image_size)
        g = center_translate(g, self.args.image_size)
        inputs = np.concatenate([obs, g], axis = 1)
        inputs = torch.tensor(inputs, dtype = torch.float32)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        if self.args.encoder_type != 'pixel':
            o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
            transitions['obs'], transitions['g'] = self._preproc_og(o, g)
            transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
            # start to do the update
            obs_norm = self.o_norm.normalize(transitions['obs'])
            g_norm = self.g_norm.normalize(transitions['g'])
            inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
            obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
            g_next_norm = self.g_norm.normalize(transitions['g_next'])
            inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
            # transfer them into the tensor
            inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
            inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
            actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
            r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)
            # print("rewards", r_tensor[:10])
            # print("ag_next", transitions['ag_next'][:10])
            # print("g", transitions['g'][:10])
        else:
            o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
            inputs_norm = self._preproc_og_pixel(o, g)
            inputs_next_norm = self._preproc_og_pixel(transitions['obs_next'], g)

            inputs_norm_tensor = inputs_norm.float()
            inputs_next_norm_tensor = inputs_next_norm.float()
            actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
            r_tensor = torch.tensor(transitions['r'], dtype=torch.float32)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

        # the actor loss
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # start to update the network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        if self.step % self.args.log_frequency == 0:
            self.L.log('train_actor/loss', actor_loss, step = self.step)
            self.L.log('train_critic/loss', critic_loss, step = self.step)


        if self.args.encoder_type == 'pixel':
            self.contrastive_learner_optim.zero_grad()
            self.encoder_optim.zero_grad()
            contrastive_loss = self.contrastive_learner.get_loss(o)
            if self.step % self.args.log_frequency == 0:
                self.L.log('train_CURL/loss', contrastive_loss, step = self.step)
            contrastive_loss.backward()
            sync_grads(self.contrastive_learner)
            self.contrastive_learner_optim.step()
            self.encoder_optim.step()
        self.step += 1

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        for k in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            self.args.video.init(enabled = (k == 0))
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    if self.args.encoder_type != 'pixel':
                        input_tensor = self._preproc_inputs(obs, g)
                    else:
                        input_tensor = self._preproc_inputs_pixel(obs, g)
                    pi = self.actor_network(input_tensor)
                    # convert the actions
                    actions = pi.detach().cpu().numpy().squeeze()
                observation_new, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                self.args.video.record(self.env)
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            if self.args.save_video:
                self.args.video.save('%d.mp4' % self.step)
                if k == 0:
                    path = os.path.join(self.args.video.dir_name, str(self.step) + '_goal.pt')
                    payload = [
                        g
                    ]
                    torch.save(payload, path)
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
