#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

import os 
import sys
from rsl_rl_custom.modules.algorithms.actor_critic import ActorCritic, SafetyCritic, Actor, Critic
# from rsl_rl.storage import RolloutStorage
from rsl_rl_custom.modules.storage import RolloutStorage, CollisionRolloutStorage
from torch.nn.functional import binary_cross_entropy

class PPO:
    # actor_critic: ActorCritic
    actor: Actor
    critic: Critic
    safety_critic: SafetyCritic
    
    def __init__(
        self,
        # actor_critic,
        actor,
        critic,
        safety_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        
        # ==== safe_rl ====
        safe_largrange : bool=False,
        l_multiplier_init=0.1,
        n_lagrange_samples=1,
        lagrange_penalty_mean=[],
        lagrange_penalty_var=[],
        collision_reward : int = -10,
        gamma_col_net : float = 1.0,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        # self.actor_critic = actor_critic 
        # self.actor_critic.to(self.device)
        self.actor = actor
        self.critic = critic
        self.storage = None  # initialized later
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # ==== safety term ====
        self.safety_critic = safety_critic
        self.safety_critic.to(self.device)
        self.optimizer_safety = optim.Adam(self.safety_critic.parameters(), lr=learning_rate)
        
        self.transition = CollisionRolloutStorage.Transition()
        
        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma  # gamma for GAE
        self.lam = lam      # lambda for GAE  => compute returns (advantage 계산 함수쪽에서 인자로 사용됨)
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        # ==== safe_rl ====
        self.safe_largrange = safe_largrange
        self.col_net = None
        self.l_multiplier = l_multiplier_init
        self.n_lagrange_samples = n_lagrange_samples
        self.collision_reward = collision_reward
        self.gamma_col_net = gamma_col_net
        self.kl_early_stop =0  # for maximum allowed kl divergence tuning
        self.lr_schedule_step = 4

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape): 
        # 현재 state로 부터 다음 state value를 최대화하는 action을 찾기 위해 다음 action을  eval

        self.storage = CollisionRolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device, self.collision_reward
        )

        # self.storage = RolloutStorage(
        #     num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        # )

    # def test_mode(self):
    #     self.actor.test()
    #     self.critic.test()

    def train_mode(self):
        self.actor.train()
        self.critic.train()
        self.safety_critic.train()

    def act(self, obs, critic_obs):
            
        # Compute the actions and values 
        self.transition.actions = self.actor.act(obs).detach() #  actor : action에 대해 샘플링 진행, 이후 값을 detach하여 그래프 연산을 분리합니다. (상수취급)
        self.transition.values = self.critic.evaluate(critic_obs).detach()  # critic : action에 대해 value를 계산, 이후 값을 detach하여 그래프 연산을 분리합니다. (상수취급)
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(self.transition.actions).detach()  # action 분포에 대한 log_prob 계산, 이후 값을 detach하여 그래프 연산을 분리합니다. (상수취급)
        self.transition.action_mean = self.actor.action_mean.detach()     # action 분포의 평균값 계산, 이후 값을 detach하여 그래프 연산을 분리합니다. (상수취급)
        self.transition.action_sigma = self.actor.action_std.detach()     # action 분포의 표준편차 계산, 이후 값을 detach하여 그래프 연산을 분리합니다. (상수취급)
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs  # transition에 현재 state(obs) 저장
        self.transition.critic_observations = critic_obs # transition에 현재 state(critic_obs) 저장
        return self.transition.actions
        
    # 원래 있거에서 추가 대체, collision(safety term) 관련 추가
    def process_env_step_with_safety(self, rewards, dones, infos, 
                                     collision_prob: torch.Tensor,
                                     collision_prob_policy: torch.Tensor,
                                     ):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.collision_prob = collision_prob.clone()
        self.transition.collision_prob_policy = collision_prob_policy.clone()
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition) # transition 저장 후, 리셋
        self.transition.clear()
        
    def compute_returns(self, last_critic_obs, last_actions, last_colllision_prob_policy):
        last_values = self.critic.evaluate(last_critic_obs).detach()
        
        # ==== Safety Term ====
        # with torch.no_grad():
        last_collision_prob_values = self.safety_critic.compute_safety_critic(last_critic_obs, last_actions).detach()
        self.storage.compute_returns(last_values, 
                                     self.gamma, 
                                     self.lam, 
                                     self.safe_largrange, 
                                     last_collision_prob_values,
                                     last_colllision_prob_policy,
                                     self.gamma_col_net
                                     )

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_collision_loss = 0 # ==== safe_rl ====
        
        
        mean_surrogate_loss, mean_value_loss, mean_collision_loss = self.optimize_actor(mean_value_loss, mean_surrogate_loss, mean_collision_loss)
        mean_value_loss = self.optimize_critic(mean_value_loss)
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        num_updates2 = self.num_learning_epochs * self.num_mini_batches * 2
        mean_value_loss /= (num_updates+num_updates2)
        mean_surrogate_loss /= num_updates
        
        # ==== safe_rl ====
        mean_collision_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_collision_loss
    
    def optimize_actor(self, mean_value_loss, mean_surrogate_loss, mean_collision_loss) -> tuple[float, float, float]:
        generator_actor = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs) 
        early_stop_flag = False
        for ( # type: ignore
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            
            # ==== safe_rl ====
            col_prob_targets_batch,
            
        ) in generator_actor:
            if early_stop_flag:
                break
            
            self.actor.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
            value_batch = self.critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor.action_mean
            sigma_batch = self.actor.action_std
            entropy_batch = self.actor.entropy  # ##########################################

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)) # actions_log_prob_batch = log_prob //  torch.squeeze(old_actions_log_prob_batch) = rollout_data.old_log_prob
            surrogate = -torch.squeeze(advantages_batch) * ratio # policy_loss_1 = torch.sqeeze(advantages_batch) 
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) # policy_loss_1
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean() # policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
            
            # Value function loss
            value_loss = (returns_batch - value_batch).pow(2).mean()
                                
            # Collision loss
            col_loss = torch.tensor(0.0, device=self.device)
            # Minimize the probability of crashing
            # Deciding how many samples to take to approximate Lagrange Expectation Gradient
            if self.safe_largrange:
                if self.n_lagrange_samples == 1:
                    action_sample = self.actor.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0]) # a, _, _ = self.alg.forward(obs_batch)
                    c = torch.cat(self.safety_critic.forward_safety_critic(obs_batch, action_sample), dim=1) # num env*mini_batch_size x critic 갯수
                    c, _ = torch.max(c, dim=1, keepdim=True)
                # else :
                #     c, a = self.forward_sample_col_prob(rollout_data.observations, self.n_lagrange_samples, "mean")

                col_loss_log = self.l_multiplier * (c - 0.1) # Tensor         0.1: c_max
                col_loss = col_loss_log.mean() # type: ignore
                # lagrange_penalty_mean.append(col_loss.item())
                # lagrange_penalty_var.append(col_loss_log.var().item())

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean() + col_loss
            
            
            with torch.no_grad():
                log_ratio = actions_log_prob_batch - old_actions_log_prob_batch
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                # approx_kl_divs.append(approx_kl_div)
                    
            # KL (learning rate 조정)
            optimizer_param_groups = [self.optimizer_actor.param_groups, self.optimizer_critic.param_groups]
            if self.desired_kl is not None and self.schedule == "adaptive":
                if self.safe_largrange and False :
                    if approx_kl_div > self.desired_kl:
                        self.kl_early_stop += 1
                        if self.kl_early_stop >= self.lr_schedule_step:
                            if self.learning_rate > 1e-5:
                                self.learning_rate *= 0.5
                                for optimizer_param_group in optimizer_param_groups:
                                    for param_group in optimizer_param_group:
                                        param_group["lr"] = self.learning_rate  
                                self.kl_early_stop = 0
                        early_stop_flag = True
                        print(f"Early stopping at step __ due to reaching max kl: {approx_kl_div:.2f}")

                else :
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                            / (2.0 * torch.square(sigma_batch))
                            - 0.5,
                            axis=-1,
                        ) # type: ignore
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for optimizer_param_group in optimizer_param_groups:
                            for param_group in optimizer_param_group:
                                param_group["lr"] = self.learning_rate    
            
            
                                
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_collision_loss += col_loss.item()
        
            
            # print("SL:", surrogate_loss.item(), "VL:", (self.value_loss_coef * value_loss).item(), "EL:", (self.entropy_coef * entropy_batch.mean()).item(), "CL:", col_loss.item(), "TOTAL:", loss.item())
            # else :
            #     loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            # loss = policy_loss    + ent_coef * entropy_loss           + self.vf_coef * value_loss + col_loss 
            
            # policy_loss: L^{CLIP}
            # ent_coef * entropy_loss: S[pi](s)
            # vf_coef * value_loss: L^{VF}
            # Gradient step
            
            # ==== safe_rl ====
            # 라그랑주 상수 업데이트하지만, 0으로 되지는 않도록 하기            
            if self.safe_largrange:
                self.l_multiplier = max(0.05, self.l_multiplier + 1e-4 * (c.mean().item() - 0.1))
            
            self.optimizer_actor.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer_actor.step()
   
        return mean_surrogate_loss, mean_value_loss, mean_collision_loss
    
    def optimize_critic(self, mean_value_loss) -> None:
        generator_critic = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs * 2) 
        
        for ( # type: ignore
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            
            # ==== safe_rl ====
            col_prob_targets_batch,
            
        ) in generator_critic:
            
            value_batch = self.critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            # Value function loss
            returns_batch = returns_batch.requires_grad_()
            value_batch = value_batch.requires_grad_()
            value_loss = (returns_batch - value_batch).pow(2).mean()
            
            mean_value_loss += value_loss.item()
                      
            # Value function loss
            self.optimizer_critic.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer_critic.step()     
                                  
            # ==== safe_rl ====
            current_col_prob = self.safety_critic.forward_safety_critic(obs_batch, actions_batch)
            target = col_prob_targets_batch.view(-1, 1) # 텐서 크기 (?,1) 로 변환
            col_net_loss = 0.5 * sum([weighted_bce(pred, target) for pred in current_col_prob])
            # col_net_losses.append(col_net_loss.item())
            self.optimizer_safety.zero_grad()
            col_net_loss.backward()
            self.optimizer_safety.step()
        return mean_value_loss
            
def weighted_bce(pred: torch.Tensor,   # safety critic 계산값.. safety 확률 얻음
                 target: torch.Tensor, 
                 p_threshold=0.5) -> torch.Tensor :
    bce = binary_cross_entropy(pred, target, reduction='none')
    n = 1
    for n_elements in target.shape:
        n *= n_elements
    # n = target.shape[0]
    n_positives = target[target >= p_threshold].shape[0]
    n_negatives = n - n_positives

    if n_positives == 0:  # No collision in targets
        positive_weight = 1
        negative_weight = 0.1
    elif n_negatives == 0:
        positive_weight = 0.1
        negative_weight = 1
    else:  # taken from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        positive_weight = 1 / n_positives * n / 2
        negative_weight = 1 / n_negatives * n / 2

    weight = (target >= p_threshold).type(torch.int32) * positive_weight \
             + (target < p_threshold).type(torch.int32) * negative_weight

    return torch.mean(bce * weight)