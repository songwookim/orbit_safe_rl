#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticRecurrent, EmpiricalNormalization # ActorCritic
from rsl_rl.utils import store_code_state

from rsl_rl_custom.modules.algorithms import PPO, ActorCritic, SafetyCritic, Critic, Actor
# from rsl_rl.algorithms import PPO
# from rsl_rl_custom.modules.algorithms.ppo import PPO
from typing import Union

class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"] # 알고리즘 관련 cfg
        self.policy_cfg = train_cfg["policy"] # 정책 관련 cfg
        self.value_cfg = train_cfg["value"] # 가치 관련 cfg
        self.safety_cfg = train_cfg["safety_critic"] # Safety Critic 관련 cfg
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]   # num of state space
        if "critic" in extras["observations"]:                          # critic을 위한 추가 obs가 있는 경우
            num_critic_obs = extras["observations"]["critic"].shape[1]  # 설정한대로 critic obs의 수를 가져옴
        else:
            num_critic_obs = num_obs
        actor_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic 생성자 [Policy 클래스]
        critic_class = eval(self.value_cfg.pop("class_name"))  # ActorCritic 생성자 [Policy 클래스]
        
        actor: Actor = actor_class(  # Actor 인스턴스 생성 (obs, critic_obs, action_space, **policy_cfg=정책관련 cfg)
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        critic: Critic = critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.value_cfg
        ).to(self.device)
        
        # ==== Safety Term ====
        safety_critic_class = eval(self.safety_cfg.pop("class_name"))
        safety_critic: SafetyCritic = safety_critic_class(  # ActorCritic 인스턴스 생성 (obs, critic_obs, action_space, **policy_cfg=정책관련 cfg)
            num_obs, self.env.num_actions, **self.safety_cfg
        ).to(self.device)
        
        # ==== PPO Term ====
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO 생성자 [알고리즘 클래스]
        self.alg: PPO = alg_class(actor, critic, safety_critic, device=self.device, **self.alg_cfg) # PPO 인스턴스 생성 (policy, **alg_cfg=알고리즘관련 cfg)
        
        self.num_steps_per_env = self.cfg["num_steps_per_env"]  # env마다 수행할 step 수
        self.save_interval = self.cfg["save_interval"]          # 저장 간격
        self.empirical_normalization = self.cfg["empirical_normalization"] # Emprical 정규화 여부.. 
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env, 
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],  # 
        )

        # > Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        
        # ==== Safety Term ====
        self.n_col_value_samples = self.cfg.get("n_col_value_samples", 10) # 충돌 확률 샘플링 수

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            # Launch either Tensorboard or Neptune & Tensorboard summary writer(s), default: Tensorboard.
            # Logger 설정 (기본으론 Tensorboard)
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()
            
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len: # 랜덤한 episode 길이로 초기화
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs, extras = self.env.get_observations()
        critic_obs = extras["observations"].get("critic", obs)               # critic을 위한 추가 obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.train_mode()  # switch to train mode (for dropout for example) [dropout : 뉴런들 사이에서 끊기 (overfitting 방지)]

        ep_infos = []
        rewbuffer = deque(maxlen=100) # reward replay buffer
        lenbuffer = deque(maxlen=100) # episode length replay buffer
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)     # 현재 reward 합
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) # 현재 episode 길이

        start_iter = self.current_learning_iteration     # 기본값 = 0,  0번째 iteration부터 시작
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):           # 0 ~ 150번
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):    # num_env를 num_steps_per_env만큼 수행
                    actions = self.alg.act(obs, critic_obs) # [action : num_env x 1]     PPO에서의 action 샘플링 [self.alg.transition.{action,value} 만 업데이트]
                    collision_prob = self.alg.safety_critic.compute_safety_critic(critic_obs, actions, minimum=False) # safety # self.alg.actor_critic.compute_safety_critic(torch.zeros_like(critic_obs), torch.zeros_like(actions)-1)
                    obs, rewards, dones, infos = self.env.step(actions)
                    obs = self.obs_normalizer(obs)  # obs 정규화
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"]) # critic obs 정규화
                    else:
                        critic_obs = obs
                    obs, critic_obs, rewards, dones = ( # 텐서 gpu device로 변환
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    
                    # Add collision probability of sampled actions to get V(x) = E_a[ Q(x,a) ]
                    # 샘플된 액션에 대해 충돌 확률 계산 후 V(x) = E_a[ Q(x,a) ] 얻기
                    with torch.no_grad():
                        collision_prob_policy, _ = self.forward_sample_col_prob(critic_obs, self.n_col_value_samples, reduction="mean")
                
                    # (safety term 추가)self.alg.process_env_step(rewards, dones, infos) # self.transition.{reward, dones} 저장 후 reset
                    self.alg.process_env_step_with_safety(rewards, dones, infos, collision_prob, collision_prob_policy) # self.transition.{reward, dones} 저장 후 reset
                    # self.alg.process_env_step_with_safety(rewards, dones, infos, collision_prob, collision_prob_policy) # self.transition.{reward, dones} 저장 후 reset

                    if self.log_dir is not None:
                        # Book keeping
                        # note: we changed logging to use "log" instead of "episode" to avoid confusion with
                        # different types of logging data (rewards, curriculum, etc.)
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                last_colllision_prob_policy, _ = self.forward_sample_col_prob(critic_obs, self.n_col_value_samples, reduction="mean")
                self.alg.compute_returns(critic_obs, actions, last_colllision_prob_policy) # 마지막 value 계산

            mean_value_loss, mean_surrogate_loss, mean_collision_loss = self.alg.update() # !TODO: 얘가 핵심
            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()
            # if it == start_iter:
            #     # obtain all the diff files
            #     git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
            #     # if possible store them to wandb
            #     if self.logger_type in ["wandb", "neptune"] and git_file_paths:
            #         for path in git_file_paths:
            #             self.writer.save_file(path)

        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))
        
        # ==== Safety Term ====
        self.writer.add_scalar("Loss/collision", locs["mean_collision_loss"], locs["it"])
        # =====================
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        # ==== Safety Term ====
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'-' * int(width/2)} About Safety Term {'-' * int(width/2)}\n"""
            f"""{'Mean Collision Loss:':>{pad}} {locs['mean_collision_loss']:.4f}\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "actor_state_dict": self.alg.actor.state_dict(),
            "critic_state_dict": self.alg.critic.state_dict(),
            "optimizer_actor_state_dict": self.alg.optimizer_actor.state_dict(),
            "optimizer_critic_state_dict": self.alg.optimizer_critic.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor.load_state_dict(loaded_dict["actor_state_dict"])
        self.alg.critic.load_state_dict(loaded_dict["critic_state_dict"])
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer_actor.load_state_dict(loaded_dict["optimizer_actor_state_dict"])
            self.alg.optimizer_critic.load_state_dict(loaded_dict["optimizer_critic_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor.to(device)
            self.alg.critic.to(device)
        policy = self.alg.actor.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor.train() # actor_critic을 train mode로 변경, 각각의 뉴럴넷 학습
        self.alg.critic.train() # actor_critic을 train mode로 변경, 각각의 뉴럴넷 학습
        self.alg.safety_critic.train() # actor_critic을 train mode로 변경, 각각의 뉴럴넷 학습
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()
            

    def eval_mode(self):
        self.alg.actor.eval()
        self.alg.critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    # ==== Safety Critic ====
    def forward_sample_col_prob(self, obs: torch.Tensor, n_samples=10, reduction=None):
        """
        action을 n_samples만큼 샘플링하고 그 action에 대한 충돌 확률을 계산 with 해당 policy
        collisions by sampling from policy P(x) 
        
        @param obs: Observation
        @param n_samples: Number of actions to sample
        @param reduction: None: return all collision probabilities, Mean: Reduce to mean collision probability
        @return: Collision probabilities of sampled actions and sampled actions
        """
        # features = self.policy.extract_features(obs)
        original_batch_size = obs.shape[0] # type: ignore
        # hook_actor = self.alg.actor_critic.regist_hook(2, self.alg.actor_critic.actor)
        # hook_value = self.alg.actor_critic.regist_hook(2, self.alg.actor_critic.critic)
        
             
        # latent_pi, latent_vf = self.alg.actor_critic.mlp_extractor(obs)
        # actions = self.alg.actor_critic.actor(obs)  # 4 x 1
        # latent_policy = self.alg.actor_critic.get_latent_vector() # 4 x 32
        
        # _ = self.alg.actor_critic.critic(obs)
        # latent_value = self.alg.actor_critic.get_latent_vector()
        
        # # hook 제거
        # hook_actor.remove()
        # hook_value.remove()
        
        # latent_sde = latent_policy

        # If in has shape [batch_size, ...] out has shape [batch_size * n_samples, ...]
        # x_actor = actions.repeat((n_samples, *tuple(1 for d in obs.shape[1:]))) # type: ignore [80 x 32]
        col_state_samples = obs.repeat((n_samples, *tuple(1 for d in obs.shape[1:]))) # type: ignore

        # distribution = self.alg.actor_critic._get_action_dist_from_latent(x_actor, latent_sde=latent_sde)
        # actions = distribution.get_actions(deterministic=False)
        actions = self.alg.actor.get_action_samples(col_state_samples)

        # qvalue_input = torch.cat([col_state_samples, actions], dim=1)
        predictions = None
        
        qvalue_input = torch.cat([col_state_samples, actions], dim=-1)
        out = self.alg.safety_critic.compute_safety_critic(col_state_samples, actions)
        
        # out = out.reshape((n_samples, original_batch_size, *tuple(out.shape[1:])))
        # predictions = out if predictions is None else torch.cat((predictions, out), dim=0)
        for collsion_net in self.alg.safety_critic.safety_networks:
            out = collsion_net(qvalue_input)
            # From [batch_size * n_passes, ...] to [n_passes, batch_size, ...]
            out = out.reshape((n_samples, original_batch_size, *tuple(out.shape[1:])))
            predictions = out if predictions is None else torch.cat((predictions, out), dim=0)

        if reduction == "mean" or reduction == "Mean":
            predictions = predictions.mean(dim=0)
        return predictions, actions