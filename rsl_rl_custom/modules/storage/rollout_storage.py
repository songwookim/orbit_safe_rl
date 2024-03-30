#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories
from typing import Generator, Optional, Union, NamedTuple, Dict, Tuple

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self, 
        num_envs, 
        num_transitions_per_env, 
        obs_shape, 
        privileged_obs_shape, 
        actions_shape, 
        device="cpu"
        ):
        
        self.device = device
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0 # Truncated GAE
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float() # done 이면 0, 아니면 1
            delta     = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step] # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            advantage = delta + next_is_not_terminal * gamma * lam * advantage                              # advantage = delta + gamma * lamda * advantage 이전 단계 advantage 를 더함
            self.returns[step] = advantage + self.values[step]  # Q = V + A

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values            # advantage = returns - values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8) # advantage 정규화

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device) # 0~n 사이의 정수를 무작위로 섞어서 텐서 만들기 => 인덱스 생성

        observations = self.observations.flatten(0, 1) # env_per_step by num of envs 를 flattent => [env_per_step * num of envs, mini_batch_size] 크기 텐서로 변환
        if self.privileged_observations is not None:    # critic 에서 사용하는 
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None
                
   
class CollisionRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            
            # ==== Safety Term ====
            self.collision_prob: torch.Tensor = None # type: ignore = col_prob_preds
            self.collision_prob_policy: torch.Tensor = None # type: ignore  (= col_prob_pi)
            self.collision_rewards: torch.Tensor = None # type: ignore

        def clear(self):
            self.__init__()

    def __init__(
        self, 
        num_envs, 
        num_transitions_per_env, 
        obs_shape, 
        privileged_obs_shape, 
        actions_shape, 
        device="cpu",
        
        # ==== Safety Term ====
        collision_reward: int = -10,
        ):
        
        self.device = device
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0
        
        # ==== Safety Term ====
        buffer_size = 1024
        self.collision_prob = torch.zeros(buffer_size, num_envs, 1, device=self.device) # estimated probability of collision P(x,a)
        self.collision_prob_policy = torch.zeros(buffer_size, num_envs, 1, device=self.device) # estimated probability of collisions by sampling from policy P(x)
        self.collision_panalty = collision_reward 
        self.min_reward = 0
        self.min_value = torch.tensor(0).cuda() # type: ignore
        self.collision_prob_target = torch.zeros(buffer_size, num_envs, device=self.device) # type: ignore
        self.collision_rewards = torch.zeros(buffer_size, num_envs, device=self.device)  # type: ignore
        

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1
        
        # ==== Safety Term ====
        self.collision_prob[self.step].copy_(transition.collision_prob)
        self.collision_prob_policy[self.step].copy_(transition.collision_prob_policy)
        self.collision_rewards[self.step] = torch.Tensor(transition.rewards  < 0.99 * self.collision_panalty).cuda() # type: ignore
        
    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i]) # type: ignore

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, 
                        safe_multi: bool = False, 
                        last_collision_prob_value: Optional[torch.Tensor] = None,
                        last_collision_prob_policy: Optional[torch.Tensor] = None,
                        gamma_col_net: float = 1.0,):
        # 1. 클램핑을 위해, collision 이 일어나지 않은 minimum 값을 가져온다.
        self.min_reward = min(self.min_reward, torch.min(self.rewards[self.rewards > self.collision_panalty])) # type: ignore
        
        # 2. 충돌 확률에 대한 목표값 정한다.
        # 타겟 충돌 확률을 계산한다.
        self.collision_prob_target = self.compute_safety_critic_targets(last_collision_prob_value, gamma_col_net, lam)

        # 3. 
        if safe_multi:
            self.returns, self.qs = self.compute_critic_targets(last_values, gamma, lam, safe_multi)
        # else:  
            # self.returns, self.qs = self.compute_critic_targets(last_values, gamma, lam, safe_multi)
            batch_min_value = torch.min(torch.min(self.qs), torch.min(self.values))
            self.min_value = torch.min(self.min_value, batch_min_value)
        
        # Modes:
        # V1a:  A = [r + y * (V(x') - Vmin) * P(x'))] - (V(x) - Vmin) * P(x)
        advantage = 0 # Truncated GAE
        
        # last_gae_lam = 0 
        # ==== safety term ====
        min_value_init = 0
        diff_col_now_next = torch.zeros_like(self.collision_prob)      
                
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
                # ==== safety term ====
                next_col_prob = last_collision_prob_value.flatten()
                next_collision_prob_policy = last_collision_prob_policy.flatten()
            else:
                next_values = self.values[step + 1]
                
            # ==== safety term ====
            next_non_col = torch.abs(self.rewards[step] - self.collision_panalty) > 0.001  # -> v1b, v1c 에 사용
            
            # Reward Clipping:
            if safe_multi:
                # gae version
                reward = torch.clip(self.rewards[step], min=self.min_reward, max=torch.inf).flatten() # type: ignㅐㄱㄷ
            else :
                reward = self.rewards[step]       
                
            if safe_multi:
                next_values = next_values.flatten()
                next_is_not_terminal = 1.0 - self.dones[step].float().flatten() # done 이면 0, 아니면 1
                f_next = next_is_not_terminal * (next_values - self.min_value) * (1 - next_collision_prob_policy.flatten())
                f = (self.values[step].flatten() - self.min_value) * (1 - self.collision_prob_policy[step].flatten())
                delta = (reward + gamma * f_next) - f
                advantage = delta + next_is_not_terminal * gamma * lam * advantage # advantage = delta + gamma * lamda * advantage 이전 단계 advantage 를 더함
                self.returns[step] = advantage + self.values[step].flatten()  # Q = V + A
            else : 
                next_is_not_terminal = 1.0 - self.dones[step].float() # done 이면 0, 아니면 1
                # delta = self.qs[step] - self.values[step].flatten() # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
                delta = reward + next_is_not_terminal * gamma * next_values - self.values[step]
                advantage = delta + next_is_not_terminal * gamma * lam * advantage # advantage = delta + gamma * lamda * advantage 이전 단계 advantage 를 더함
                self.returns[step] = advantage + self.values[step]  # Q = V + A
            
            
            
            # # Log difference in safety to see what model is learning
            # if mode == "old":  # A = [Q * P(x',a')] - V(x) * P(x,a)
            #     diff_col_now_next[step] = col_prob_preds[step] - next_col_prob
            # elif mode == "V2b":  # A = Q(x,a) * P(x,a) - V(x) * P(x)
            #     diff_col_now_next[step] = col_prob_pi[step] - col_prob_preds[step]
            # else:  # A = [r + y * (V(x') * P(x'))] - V(x) * P(x)
            diff_col_now_next[step] = self.collision_prob_policy[step] - self.collision_prob_policy[step+1]

        # Compute and normalize the advantages
        if safe_multi:
            self.advantages = self.returns - self.values.reshape(self.num_transitions_per_env, self.num_envs)            # advantage = returns - values
        else :
            self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8) # advantage 정규화
        
    # def compute_returns(self, last_values, gamma, lam, safe_multi: bool = False, 
    #                      last_collision_prob_value: Optional[torch.Tensor] = None,
    #                      last_colllision_prob_policy: Optional[torch.Tensor] = None,
    #                      gamma_col_net: float = 1.0,):
    #     advantage = 0 # Truncated GAE
    #     for step in reversed(range(self.num_transitions_per_env)):
    #         if step == self.num_transitions_per_env - 1:
    #             next_values = last_values
    #         else:
    #             next_values = self.values[step + 1]
    #         next_is_not_terminal = 1.0 - self.dones[step].float() # done 이면 0, 아니면 1
    #         delta     = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step] # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
    #         advantage = delta + next_is_not_terminal * gamma * lam * advantage                              # advantage = delta + gamma * lamda * advantage 이전 단계 advantage 를 더함
    #         self.returns[step] = advantage + self.values[step]  # Q = V + A

    #     # Compute and normalize the advantages
    #     # 16 x 4096 x 1
    #     self.advantages = self.returns - self.values            # advantage = returns - values
    #     self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8) # advantage 정규화

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device) # 0~n 사이의 정수를 무작위로 섞어서 텐서 만들기 => 인덱스 생성

        observations = self.observations.flatten(0, 1) # env_per_step by num of envs 를 flattent => [env_per_step * num of envs, mini_batch_size] 크기 텐서로 변환
        if self.privileged_observations is not None:    # critic 에서 사용하는 
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)  # 65536 x 1
        values = self.values.flatten(0, 1)  
        returns = self.returns.flatten(0, 1) 
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        
        # ==== Safety Term ====
        collision_prob_target = self.collision_prob_target.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                
                # ==== Safety term ====
                collision_prob_target_batch = collision_prob_target[batch_idx]
                
                
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None, collision_prob_target_batch
                
    # ==== Safety Term ====
    def compute_safety_critic_targets(self, last_collision_prob_values, gamma_col_net, gae_lammda):
        last_safety_lam = 0

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                # next_non_terminal = 1.0 - self.dones
                next_col_prob = last_collision_prob_values.flatten()
            else:
                next_col_prob = self.collision_prob[step + 1].flatten()
                
            next_is_not_terminal = 1.0 - self.dones[step].float().flatten() # done 이면 0, 아니면 1
            # TD(Lambda) Safety Critic
            q_safety = self.collision_rewards[step] + gamma_col_net * next_col_prob * next_is_not_terminal
            delta_safety = q_safety - self.collision_prob[step].flatten()
            last_safety_lam = delta_safety + gamma_col_net * gae_lammda * next_is_not_terminal * last_safety_lam
            self.collision_prob_target[step] = last_safety_lam + self.collision_prob[step].flatten()

        return self.collision_prob_target
    
    def compute_critic_targets(self, last_values, gamma_col_net, gae_lammda, safe_multi=False):
        returns = torch.zeros(self.num_transitions_per_env, self.num_envs, device=self.device)
        qs = torch.zeros(self.num_transitions_per_env, self.num_envs, device=self.device)
        last_val_lam = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                # next_non_terminal = 1.0 - dones
                next_values = last_values.clone().flatten()
            else:
                next_values = self.values[step + 1].flatten()

            if safe_multi:
                reward = torch.clamp(self.rewards[step], min=self.min_reward, max=torch.tensor(torch.inf).cuda()).flatten()
                # reward = torch.clip(self.rewards[step], min=self.min_reward, max=torch.tensor(torch.inf))
            else:
                reward = self.rewards[step].flatten()

            next_is_not_terminal = 1.0 - self.dones[step].float().flatten() # done 이면 0, 아니면 1
            # TD Lambda Critic
            qs[step] = reward + gamma_col_net * next_values * next_is_not_terminal   # rt + γV (st+1)
            delta_val = qs[step] - self.values[step].flatten()                           # delta
            last_val_lam = delta_val + gamma_col_net * gae_lammda * next_is_not_terminal * last_val_lam  # hat{A}_{t}
            returns[step] = last_val_lam + self.values[step].flatten()                   # Q_sa = A_hat + V_s
        return returns, qs
                        
    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj          
                
        
                            
class CollisionRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_col_probs: torch.Tensor
    col_prob_targets: torch.Tensor
    col_rewards: torch.Tensor
    diff_col_now_next: torch.Tensor
    next_observations: torch.Tensor