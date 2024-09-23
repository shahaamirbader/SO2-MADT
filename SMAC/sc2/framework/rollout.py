import copy

import torch
import numpy as np
from .utils import sample, padding_obs, padding_ava

class RolloutWorker:

    def __init__(self, model, critic_model, buffer, global_obs_dim, local_obs_dim, action_dim, stg_dim, max_safety_limit):
        self.buffer = buffer
        self.model = model
        self.critic_model = critic_model
        self.global_obs_dim = global_obs_dim
        self.local_obs_dim = local_obs_dim
        self.action_dim = action_dim
        self.stg_dim = stg_dim
        self.max_safety_limit = max_safety_limit
        self.device = 'cpu'
        if torch.cuda.is_available() and not isinstance(self.model, torch.nn.DataParallel):
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(model).to(self.device)
            self.critic_model = torch.nn.DataParallel(critic_model).to(self.device)

    def rollout(self, env, ret, train=True, random_rate=0.):
        self.model.train(False)
        self.critic_model.train(False)

        T_rewards, T_wins, steps, episode_dones = 0., 0., 0, np.zeros(env.n_threads)

        obs, share_obs, available_actions = env.real_env.reset()
        obs = padding_obs(obs, self.local_obs_dim)
        share_obs = padding_obs(share_obs, self.global_obs_dim)
        available_actions = padding_ava(available_actions, self.action_dim)

        global_states = torch.from_numpy(share_obs).to(self.device).unsqueeze(2)
        local_obss = torch.from_numpy(obs).to(self.device).unsqueeze(2)
        rtgs = np.ones((env.n_threads, env.num_agents, 1, 1)) * ret
        actions = np.zeros((env.n_threads, env.num_agents, 1, 1))
        timesteps = torch.zeros((env.n_threads * env.num_agents, 1, 1), dtype=torch.int64)
        stgs = np.ones((env.n_threads, env.num_agents, 1, self.stg_dim)) * self.max_safety_limit
        t = 0

        accumulated_safety = np.zeros((env.n_threads, env.num_agents, 1, self.stg_dim))

        while True:
            sampled_action, v_value = sample(self.model, self.critic_model, state=global_states.view(-1, np.shape(global_states)[2], np.shape(global_states)[3]),
                                             obs=local_obss.view(-1, np.shape(local_obss)[2], np.shape(local_obss)[3]), sample=train,
                                             actions=torch.tensor(actions, dtype=torch.int64).to(self.device).view(-1, np.shape(actions)[2], np.shape(actions)[3]),
                                             rtgs=torch.tensor(rtgs, dtype=torch.float32).to(self.device).view(-1, np.shape(rtgs)[2], np.shape(rtgs)[3]),
                                             timesteps=timesteps.to(self.device),
                                             stgs=torch.tensor(stgs, dtype=torch.float32).to(self.device).view(-1, np.shape(stgs)[2], np.shape(stgs)[3]),
                                             available_actions=torch.from_numpy(available_actions).view(-1, np.shape(available_actions)[-1]))

            action = sampled_action.view((env.n_threads, env.num_agents, -1)).cpu().numpy()

            cur_global_obs = share_obs
            cur_local_obs = obs
            cur_ava = available_actions

            obs, share_obs, rewards, dones, infos, available_actions = env.real_env.step(action)
            obs = padding_obs(obs, self.local_obs_dim)
            share_obs = padding_obs(share_obs, self.global_obs_dim)
            available_actions = padding_ava(available_actions, self.action_dim)
            t += 1

            safety_violations = np.zeros((env.n_threads, env.num_agents, self.stg_dim))
            # Define safety violations based on the environment's feedback
            for n in range(env.n_threads):
                for i in range(env.num_agents):
                    # a) Unit Preservation: Minimizing the loss of friendly units
                    if infos[n][i]['unit_lost']:
                        safety_violations[n][i][0] = 1.0

                    # b) Resource Efficiency: Avoiding wasteful use of resources
                    if infos[n][i]['resource_waste']:
                        safety_violations[n][i][1] = infos[n][i]['resource_waste_amount']

                    # c) Positional Safety: Keeping units away from dangerous areas
                    if infos[n][i]['in_dangerous_area']:
                        safety_violations[n][i][2] = 1.0

                    # d) Overkill Prevention: Avoiding excessive use of firepower
                    if infos[n][i]['overkill']:
                        safety_violations[n][i][3] = 1.0

            if train:
                v_value = v_value.view((env.n_threads, env.num_agents, -1)).cpu().numpy()
                self.buffer.insert(cur_global_obs, cur_local_obs, action, rewards, dones, cur_ava, v_value, safety_violations)

            accumulated_safety += safety_violations.reshape(env.n_threads, env.num_agents, 1, self.stg_dim)
            stgs = self.max_safety_limit - accumulated_safety

            for n in range(env.n_threads):
                if not episode_dones[n]:
                    steps += 1
                    T_rewards += np.mean(rewards[n])
                    if np.all(dones[n]):
                        episode_dones[n] = 1
                        if infos[n][0]['won']:
                            T_wins += 1.
            if np.all(episode_dones):
                break

            rtgs = np.concatenate([rtgs, np.expand_dims(rtgs[:, :, -1, :] - rewards, axis=2)], axis=2)
            global_state = torch.from_numpy(share_obs).to(self.device).unsqueeze(2)
            global_states = torch.cat([global_states, global_state], dim=2)
            local_obs = torch.from_numpy(obs).to(self.device).unsqueeze(2)
            local_obss = torch.cat([local_obss, local_obs], dim=2)
            actions = np.concatenate([actions, np.expand_dims(action, axis=2)], axis=2)
            timestep = t * torch.ones((env.n_threads * env.num_agents, 1, 1), dtype=torch.int64)
            timesteps = torch.cat([timesteps, timestep], dim=1)

        aver_return = T_rewards / env.n_threads
        aver_win_rate = T_wins / env.n_threads
        self.model.train(True)
        self.critic_model.train(True)
        return aver_return, aver_win_rate, steps

