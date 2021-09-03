import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_modules.encoder import make_encoder
from utils import center_translate, random_translate

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params, args):
        super(actor, self).__init__()
        self.args = args
        self.max_action = env_params['action_max']
        if args.encoder_type == 'pixel':
            self.fc1 = nn.Linear(args.encoder_feature_dim * 2, 256) ## 1 * encoder_feature_dim for obs/goal
        else:
            self.fc1 = nn.Linear(env_params['obs'][0] + env_params['goal'][0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        if args.encoder_type == 'pixel':
            self.encoder = make_encoder(
                args.encoder_type,
                3 * args.frame_stack,
                args.image_size,
                args.encoder_feature_dim,
                args.num_layers,
                args.num_filters,
            )

    def forward(self, x):
        ##Find out how they are concatenated or if it is just an array
        if self.args.encoder_type == 'pixel':
            x = self.encode(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions

    def encode(self, x):
        ##Make x channel concat
        obs, goal = torch.chunk(x, 2, dim = 1)
        if self.args.dont_detach_actor:
            obs_enc = self.encoder(obs)
        else:
            with torch.no_grad():
                obs_enc = self.encoder(obs)
        with torch.no_grad():
            goal_enc = self.encoder(goal)
        x = torch.cat([obs_enc, goal_enc], dim = 1)
        return x

class critic(nn.Module):
    def __init__(self, env_params, args):
        super(critic, self).__init__()
        self.args = args
        self.max_action = env_params['action_max']
        if args.encoder_type == 'pixel':
            self.fc1 = nn.Linear(args.encoder_feature_dim * 2 + env_params['action'], 256)
        else:
            self.fc1 = nn.Linear(env_params['obs'][0] + env_params['goal'][0] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)
        if args.encoder_type == 'pixel':
            self.encoder = make_encoder(
                args.encoder_type,
                3 * args.frame_stack,
                args.image_size,
                args.encoder_feature_dim,
                args.num_layers,
                args.num_filters,
            )

    def forward(self, x, actions):
        if self.args.encoder_type == 'pixel':
            og = self.encode(x)
        else:
            og = x
        x = torch.cat([og, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value

    def encode(self, x):
        ##Make x channel concat
        obs, goal = torch.chunk(x, 2, dim = 1)
        if not self.args.dont_detach_critic:
            with torch.no_grad():
                obs_enc = self.encoder(obs)
        else:
            obs_enc = self.encoder(obs)
        with torch.no_grad():
            goal_enc = self.encoder(goal)
        x = torch.cat([obs_enc, goal_enc], dim = 1)
        return x


class contrastive_learner(nn.Module):
    def __init__(self, env_params, args, critic, critic_target):
        super(contrastive_learner, self).__init__()
        self.args = args
        self.encoder = critic.encoder
        self.encoder_target = [critic_target.encoder] #so it doesnt show up as a parameter and mess up mpi
        self.W = nn.Parameter(torch.rand(args.encoder_feature_dim, args.encoder_feature_dim))
        if self.args.cuda:
            self.W.cuda()
            self.encoder.cuda()

    def encode(self, x, detach = False, ema = False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if self.args.cuda:
            x = x.cuda()
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target[0](x)
        else:
            z_out = self.encoder(x)
        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, observations):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        obs_pos, obs_anchor = self.random_augment(observations)
        z_pos = self.encode(obs_pos, ema = True)
        z_a = self.encode(obs_anchor)
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def random_augment(self, observations):
        """
        Performs a random translation of the pixel observation (most basic augmentation currently)
        """
        obs_pos, obs_anchor = random_translate(observations, self.args.image_size), random_translate(observations, self.args.image_size)
        obs_pos = torch.as_tensor(obs_pos).float()
        obs_anchor = torch.as_tensor(obs_anchor).float()
        if self.args.cuda:
            obs_pos.cuda()
            obs_anchor.cuda()
        return obs_pos, obs_anchor

    def get_loss(self, observations):
        """
        Uses cross entropy loss over computed logits to get loss
        """
        logits = self.compute_logits(observations)
        labels = torch.arange(logits.shape[0]).long()
        if self.args.cuda:
            labels = labels.cuda()
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

class unchanging_contrastive_learner(nn.Module):
    def __init__(self, env_params, args, critic, critic_target):
        super(unchanging_contrastive_learner, self).__init__()
        self.args = args
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder #so it doesnt show up as a parameter and mess up mpi
        self.W = nn.Parameter(torch.rand(args.encoder_feature_dim, args.encoder_feature_dim))
        if self.args.cuda:
            self.W.cuda()
            self.encoder.cuda()

    def encode(self, x, detach = False, ema = False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if self.args.cuda:
            x = x.cuda()
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)
        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, observations):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        obs_pos, obs_anchor = self.random_augment(observations)
        z_pos = self.encode(obs_pos, ema = True)
        z_a = self.encode(obs_anchor)
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def random_augment(self, observations):
        """
        Performs a random translation of the pixel observation (most basic augmentation currently)
        """
        obs_pos, obs_anchor = random_translate(observations, self.args.image_size), random_translate(observations, self.args.image_size)
        obs_pos = torch.as_tensor(obs_pos).float()
        obs_anchor = torch.as_tensor(obs_anchor).float()
        if self.args.cuda:
            obs_pos.cuda()
            obs_anchor.cuda()
        return obs_pos, obs_anchor

    def get_loss(self, observations):
        """
        Uses cross entropy loss over computed logits to get loss
        """
        logits = self.compute_logits(observations)
        labels = torch.arange(logits.shape[0]).long()
        if self.args.cuda:
            labels = labels.cuda()
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss
