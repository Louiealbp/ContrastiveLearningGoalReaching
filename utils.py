import numpy as np
import torch
from collections import deque
import os

def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path

def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs

def center_translate(imgs, size):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, :, h1:h1 + h, w1:w1 + w] = imgs
    return outs

def random_translate(imgs, size, return_random_idxs=False, h1s=None, w1s=None):
    n, c, h, w = imgs.shape
    assert size >= h and size >= w
    outs = np.zeros((n, c, size, size), dtype=imgs.dtype)
    h1s = np.random.randint(0, size - h + 1, n) if h1s is None else h1s
    w1s = np.random.randint(0, size - w + 1, n) if w1s is None else w1s
    for out, img, h1, w1 in zip(outs, imgs, h1s, w1s):
        out[:, h1:h1 + h, w1:w1 + w] = img
    if return_random_idxs:  # So can do the same to another set of imgs.
        return outs, dict(h1s=h1s, w1s=w1s)
    return outs

def calculate_reward_func(agent):
    def reward_func(ag_next, g, third_argument):
        if agent.args.encoder_type == 'pixel':
            ag_next = torch.as_tensor(center_translate(ag_next, agent.args.image_size)).float()
            g = torch.as_tensor(center_translate(g, agent.args.image_size)).float()
            if agent.args.cuda:
                ag_next.cuda()
                g.cuda()
            if not agent.args.load_reward_curl:
                ag_next_enc = agent.contrastive_learner.encode(ag_next, ema = True)
                g_enc = agent.contrastive_learner.encode(g, ema = True)
            else:
                ag_next_enc = agent.reward_contrastive.encode(ag_next, ema = True)
                g_enc = agent.reward_contrastive.encode(g, ema = True)
            if agent.args.cosine_similarity:
                distances = (ag_next_enc * g_enc).sum(dim = 1) / (torch.norm(ag_next_enc, dim = 1) * torch.norm(g_enc, dim = 1))
                if agent.args.not_sparse_reward:
                    rewards = distances.cpu().numpy()
                else:
                    rewards = (distances > agent.args.cosine_cutoff).cpu().numpy()
            else:
                distances = torch.sqrt(((ag_next_enc - g_enc) **2).sum(dim = 1))
                if agent.args.not_sparse_reward:
                    rewards = torch.exp(-distances).cpu().numpy()
                else:
                    rewards = (distances < 1).cpu().numpy()
            if agent.args.zero_one_reward:
                return rewards
            else:
                return rewards - 1
        else:
            ag_next = torch.as_tensor(ag_next).float()
            g = torch.as_tensor(g).float()
            if agent.args.cuda:
                ag_next.cuda()
                g.cuda()
            distances = torch.sqrt(((ag_next - g) **2).sum(dim = 1))
            rewards = (distances < 0.03) .cpu().numpy()
            if agent.args.zero_one_reward:
                return rewards
            else:
                return rewards - 1
    return reward_func
