import copy
import random
import numpy as np
import torch
from torch.nn import functional as F
from gym.spaces.discrete import Discrete

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sample(model, critic_model, state, obs, sample=False, actions=None, rtgs=None,
           timesteps=None, stgs=None, available_actions=None):
    if torch.cuda.is_available():
        block_size = model.module.get_block_size()
    else:
        block_size = model.get_block_size()
    model.eval()
    critic_model.eval()

    context_length = block_size // 5
    obs_cond = obs if obs.size(1) <= context_length else obs[:, -context_length:]
    state_cond = state if state.size(1) <= context_length else state[:, -context_length:]
    if actions is not None:
        actions = actions if actions.size(1) <= context_length else actions[:, -context_length:]
    rtgs = rtgs if rtgs.size(1) <= context_length else rtgs[:, -context_length:]
    timesteps = timesteps if timesteps.size(1) <= context_length else timesteps[:, -context_length:]
    stgs = stgs if stgs.size(1) <= context_length else stgs[:, -context_length:]

    logits = model(obs_cond, pre_actions=actions, rtgs=rtgs, timesteps=timesteps, stgs=stgs)
    logits = logits[:, -1, :]
    if available_actions is not None:
        logits[available_actions == 0] = -1e10
    probs = F.softmax(logits, dim=-1)

    if sample:
        a = torch.multinomial(probs, num_samples=1)
    else:
        _, a = torch.topk(probs, k=1, dim=-1)

    v = critic_model(state_cond, pre_actions=actions, rtgs=rtgs, timesteps=timesteps, stgs=stgs).detach()
    v = v[:, -1, :]

    return a, v

def get_dim_from_space(space):
    if isinstance(space[0], Discrete):
        return space[0].n
    elif isinstance(space[0], list):
        return space[0][0]

def padding_obs(obs, target_dim):
    len_obs = np.shape(obs)[-1]
    if len_obs > target_dim:
        print("target_dim (%s) too small, obs dim is %s." % (target_dim, len(obs)))
        raise NotImplementedError
    elif len_obs < target_dim:
        padding_size = target_dim - len_obs
        if isinstance(obs, list):
            obs = np.array(copy.deepcopy(obs))
            padding = np.zeros(padding_size)
            obs = np.concatenate((obs, padding), axis=-1).tolist()
        elif isinstance(obs, np.ndarray):
            obs = copy.deepcopy(obs)
            shape = np.shape(obs)
            padding = np.zeros((shape[0], shape[1], padding_size))
            obs = np.concatenate((obs, padding), axis=-1)
        else:
            print("unknwon type %s." % type(obs))
            raise NotImplementedError
    return obs


def padding_ava(ava, target_dim):
    len_ava = np.shape(ava)[-1]
    if len_ava > target_dim:
        print("target_dim (%s) too small, ava dim is %s." % (target_dim, len(ava)))
        raise NotImplementedError
    elif len_ava < target_dim:
        padding_size = target_dim - len_ava
        if isinstance(ava, list):
            ava = np.array(copy.deepcopy(ava), dtype=np.int64)
            padding = np.zeros(padding_size, dtype=np.int64)
            ava = np.concatenate((ava, padding), axis=-1).tolist()
        elif isinstance(ava, np.ndarray):
            ava = copy.deepcopy(ava)
            shape = np.shape(ava)
            padding = np.zeros((shape[0], shape[1], padding_size), dtype=np.long)
            ava = np.concatenate((ava, padding), axis=-1)
        else:
            print("unknwon type %s." % type(ava))
            raise NotImplementedError
    return ava
