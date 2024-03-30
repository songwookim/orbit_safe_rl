import warnings
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from safe.common.on_policy_algorithm import OnPolicyAlgorithm
from safe.common.policies import BasePolicy
# from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from safe.common.policies import CustomActorCriticPolicy as ActorCriticPolicy
from safe.common.util import weighted_bce

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, update_learning_rate

SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    # policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
    #     "MlpPolicy": ActorCriticPolicy,
    #     "CnnPolicy": ActorCriticCnnPolicy,
    #     "MultiInputPolicy": MultiInputActorCriticPolicy,
    # }
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        # "CnnPolicy": ActorCriticCnnPolicy,
        # "MultiInputPolicy": MultiInputActorCriticPolicy,
    }
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        # stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 1,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "cpu",
        _init_setup_model: bool = True, 

        # Safe rl specific
        safe_mult: bool = False,     # safe multiplicative 사용 여부
        gamma_col_net: float = 0.6,  # gamma for the 충돌 관련
        col_reward: float = -1, 
        safe_lagrange: bool = False,
        tau: float = 0.005,
        lr_schedule_step: int = 4,
        td_lambda_col_target: float = 0.98,
        use_bayes: bool = False,
        l_multiplier_init: float = 0.1,
        auxiliary_tasks: Tuple[str, ...] = None, # type: ignore
        legacy_advantage: bool = False,
        n_epochs_value_multiplier: int = 2,
        advantage_mode: str = "V1a",
        n_lagrange_samples: int = 1,
        n_col_value_samples: int = 20,
        optimize_gamma_col_net: bool = False,
        gamma_col_net_target: int = 0.8 # type: ignore
    ):
        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            # stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(spaces.Box,spaces.Discrete,spaces.MultiDiscrete,spaces.MultiBinary), # type : ignore

            # # SafeRL specific
            safe_mult=safe_mult,
            # col_reward=col_reward,
            # td_lambda_col_target=td_lambda_col_target,
            # gamma_col_net=gamma_col_net,
            # safe_lagrange=safe_lagrange,
            # use_bayes=use_bayes,
            # legacy_advantage=legacy_advantage,
            # advantage_mode=advantage_mode,
            # n_col_value_samples=n_col_value_samples,
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # rollout 버퍼 크기가 여러 개의 미니-배치 크기인지 확인
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.learning_rate = 0.0003

        # ================= Safe RL specific =================
        # Safe rl specific
        self.gamma_col_net = gamma_col_net
        self.gamma_col_net_init = gamma_col_net
        self.tau = tau
        # max min Lagrange Optimization
        self.l_multiplier = l_multiplier_init
        self.n_lagrange_samples = n_lagrange_samples

        # Early stopping and learning rate adaption if KL is too big
        self.lr_schedule_step = lr_schedule_step
        self.kl_early_stop = 0
        
        self.auxiliary_tasks = [] if auxiliary_tasks is None else auxiliary_tasks
        self.n_epochs_value_multiplier = n_epochs_value_multiplier
        self.optimize_gamma_col_net = optimize_gamma_col_net
        self.gamma_col_net_target = gamma_col_net_target

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules(얼마나 남았는지 1~0으로 매핑해줌) for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        # ================ Safe RL specific ================
        # Initialize schedules for the learning rate
        self.ent_coef = get_schedule_fn(self.ent_coef)
        if self.target_kl is not None:
            self.target_kl = get_schedule_fn(self.target_kl)
        if self.lr_schedule_step is None or self.target_kl is None:
            self.learning_rate = get_schedule_fn(self.learning_rate)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # ================ Safe RL specific ================
        # 피하기 standard variance 가 터지는 거 in the log_std of the policy:
        if hasattr(self.policy, "log_std") and not self.use_sde:
            if th.any(self.policy.log_std.data > self.policy.log_std_init + 0.05).item(): # type: ignore
                diff = self.policy.log_std[0].item() - self.policy.log_std_init           # type
                self.policy.log_std.data -= diff
                print("Exploding std. Resetting to init")

        # when KL reduction , 즉 KL을 목표로 하는 값에 줄이지 않는다면
        # if self.lr_schedule_step is None or self.target_kl is None:
        #     learning_rate = self.learning_rate(self._current_progress_remaining)
        #     update_learning_rate(self.policy.action_net.optimizer, learning_rate) # type: ignore
        #     update_learning_rate(self.policy.value_net.optimizer, learning_rate) # type: ignore
        #     self.logger.record("train/learning_rate", learning_rate)
        # else:
        
        self.logger.record("train/learning_rate", self.learning_rate)

        if self.n_epochs_value_multiplier > 1:
            self.optimize_actor()    # 1) current 
            self.optimize_critics()
        else:
            self.optimize_jointly()

        self._n_updates += self.n_epochs
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def optimize_actor(self) -> None:
        pg_losses, value_losses = [], [] # 추가

        clip_range = self.clip_range(self._current_progress_remaining) # compute current clip range => 0~2 사이 값으로 조정
        ent_coef = self.ent_coef(self._current_progress_remaining)     # Entropy coefficient for the loss calculation #  => 사이 값으로 조정

        target_kl = None if self.target_kl is None else self.target_kl(self._current_progress_remaining) # => 사이 값으로 조정
        continue_training = True


        clip_fractions = []
        entropy_losses = [] # value loss는 critic Optimize에서 계산
        pg_losses = []

        # ================= Safe RL specific =================
        p_col_now_next_mean = []  # Mult: p_col(x') - p_col(x), Add: w * p_col(actor(x))
        p_col_now_next_var = []  
        lagrange_penalty_mean = [] # Lagrange penalty mean
        lagrange_penalty_var = []  # Lagrange penalty variance
        auxiliary_losses = []     # hidden state 중간에 loss를 적용하는 방식 (auxiliary loss)
        # ====================================================

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            '''
            Optimize Actor
            '''
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()  # Convert from float to long
    
                # log_std가 바껴서, noise-matrix를 re-sample 해준다.
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                '''
                Compute Advantage
                '''
                values_pred, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values_pred = values_pred.flatten()
                advantages = rollout_data.advantages

                if self.safe_mult:
                    p_col_now_next_mean.append(rollout_data.diff_col_now_next.mean().item())
                    p_col_now_next_var.append(rollout_data.diff_col_now_next.var().item())

                # Normalize advantage
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ### ================= Safe RL specific =================
                # Auxiliary loss
                auxiliary_loss = self.get_auxiliary_loss(rollout_data)

                if len(self.auxiliary_tasks) > 0: #  Loss that is added to actor loss. Mostly used to extract better features
                    auxiliary_losses.append(auxiliary_loss.item())


                '''
                Compute loss for policy
                '''
                # 이전 policy와 비교해서 ratio를 계산
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # actor-critic에서는 clipped surrogate loss를 사용
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # (Value loss using the TD(gae_lambda) target 부분은 critic에서 계산)
                # 피처 추출을 위한 value loss
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # ================= Safe RL specific =================
                # => col_loss + auxiliary_loss 계산함
                col_loss = 0
                # Minimize the probability of crashing
                if self.safe_lagrange:
                    # Deciding how many samples to take to approximate Lagrange Expectation Gradient
                    if self.n_lagrange_samples == 1:
                        a, _, _ = self.policy.forward(rollout_data.observations)
                        c = th.cat(self.col_net(rollout_data.observations, a), dim=1)
                        c, _ = th.max(c, dim=1, keepdim=True)
                    else:
                        c, a = self.forward_sample_col_prob(rollout_data.observations, self.n_lagrange_samples, "mean")

                    col_loss_log = self.l_multiplier * (c - 0.1)
                    # col_loss = col_loss_log[col_loss_log > 0].mean()
                    col_loss = col_loss_log.mean()
                    lagrange_penalty_mean.append(col_loss.item())
                    lagrange_penalty_var.append(col_loss_log.var().item())

                    # Auxiliary Action penalty
                    if "action_penalty" in self.auxiliary_tasks:
                        auxiliary_loss += 50 * (a[:, 1] ** 2).mean()
                        auxiliary_losses[-1] = auxiliary_loss.item()
                # ====================================================
                        
                loss = policy_loss + ent_coef      * entropy_loss + self.vf_coef * value_loss + col_loss + auxiliary_loss
              # loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                '''
                KL divergence for learning rate adaption and early stopping
                (Learning rate 와 early 을 위한 KL divergence 계산)
                '''

                # KL Divergence를 근사하여 early stopping을 위한 것 (http://joschu.net/blog/kl-approx.html) 
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                # ================= Safe RL specific (수정) =================
                if target_kl is not None and approx_kl_div > target_kl:
                    # Learning rate를 줄여서 policy가 불안정해지는 것을 방지                     
                    if self.lr_schedule_step is not None:
                        self.kl_early_stop += 1
                        # if self.kl_early_stop >= self.lr_schedule_step:
                            # # Only reduce learning rate if it is not too small
                            # if self.learning_rate > 1e-6:
                            #     self.learning_rate *= 0.5
                            #     update_learning_rate(self.policy.action_net.optimizer, self.learning_rate)
                            #     update_learning_rate(self.policy.value_net.optimizer, self.learning_rate)
                            #     self.logger.record("train/learning_rate", self.learning_rate)
                            # self.kl_early_stop = 0
                            # if hasattr(self.policy, "log_std"):
                            #     self.policy.log_std.data -= 0.1
                            
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                # ====================================================

                '''
                Optimization step Actor Critic
                '''
                # Optimize Lagrange Multiplier but don't let it go to zero
                if self.safe_lagrange:
                    # self.l_multiplier = max(0.05, self.l_multiplier + 1e-3 * (c.mean().item() - 0.1))
                    self.l_multiplier = max(0.05, self.l_multiplier + 1e-4 * (c.mean().item() - 0.1))

                # Optimization step
                self.policy.action_net.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.action_net.parameters(), self.max_grad_norm)
                self.policy.action_net.optimizer.step()

            if not continue_training:  # If KL Target is exceeded
                break
        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/ent_coef", ent_coef)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        if len(self.auxiliary_tasks) > 0:
            self.logger.record("train/auxiliary_loss", np.mean(auxiliary_losses))

        if self.safe_mult:
            self.logger.record("train/diff_col_now_next", np.mean(p_col_now_next_mean))
            self.logger.record("train/diff_col_now_next_var", np.mean(p_col_now_next_var))
        if self.safe_lagrange:
            self.logger.record("train/multiplier", self.l_multiplier)
            self.logger.record("train/col_lagrange_penalty", np.mean(lagrange_penalty_mean))
            self.logger.record("train/col_lagrange_penalty_var", np.mean(lagrange_penalty_var))

    def optimize_critics(self) -> None:
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        value_losses = []
        col_net_losses = []

        for epoch in range(self.n_epochs_value_multiplier * self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                '''
                Optimize Value Network
                '''
                # Prediction
                values, _, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                if self.clip_range_vf is None:  # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value. NOTE: This depends on the reward scaling
                    values_pred = rollout_data.old_values \
                                  + th.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)

                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Optimize Value Network
                self.policy.value_net.optimizer.zero_grad()
                value_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.value_net.parameters(), self.max_grad_norm)
                self.policy.value_net.optimizer.step()

                '''
                Optimize Collision Network
                '''
                current_col_prob = self.col_net.forward(rollout_data.observations, actions)
                target = rollout_data.col_prob_targets.view(-1, 1)
                col_net_loss = 0.5 * sum([weighted_bce(pred, target) for pred in current_col_prob])
                col_net_losses.append(col_net_loss.item())
                self.col_net.optimizer.zero_grad()
                col_net_loss.backward()
                self.col_net.optimizer.step()

        '''
        Optional: Optimize Gamma of collision net
        '''
        # Increase Gamma col net, when policy becomes more safe, else decrease
        # Gamma col net을 증가시킴, policy가 더 안전해지면, 그렇지 않으면 감소시킴
        if self.optimize_gamma_col_net:
            target = 0.1 - self.rollout_buffer.col_prob_preds.mean().item()
            alpha = 0.001 if target > 0 else 0.0005
            self.gamma_col_net = min(self.gamma_col_net_target, self.gamma_col_net + alpha * target)
            self.gamma_col_net = max(self.gamma_col_net_init, self.gamma_col_net)
            self.rollout_buffer.gamma_col_net = self.gamma_col_net

        '''
        Logging
        '''

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/explained_variance", explained_var)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        self.logger.record("train/col_net_loss", np.mean(col_net_losses))
        self.logger.record("train/gamma_col_net", self.gamma_col_net)

    def optimize_jointly(self):
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        ent_coef = self.ent_coef(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        p_col_now_next_mean = []  # Mult: p_col(x') - p_col(x), Add: w * p_col(actor(x))
        p_col_now_next_var = []
        lagrange_penalty_mean = []
        lagrange_penalty_var = []
        col_net_losses = []
        auxiliary_losses = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                '''
                Compute Advantage
                '''
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                if self.safe_mult:
                    p_col_now_next_mean.append(rollout_data.diff_col_now_next.mean().item())
                    p_col_now_next_var.append(rollout_data.diff_col_now_next.var().item())

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                '''
                Compute Value Loss
                '''

                if self.clip_range_vf is None:  # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                '''
                Minimize Probability of Crashing via Lagrange Optimization
                '''
                col_loss = 0
                if self.safe_lagrange:
                    # Deciding how many samples to take to approximate Lagrange Expectation Gradient
                    if self.n_lagrange_samples == 1:
                        a, _, _ = self.policy.forward(rollout_data.observations)
                        c = th.cat(self.col_net(rollout_data.observations, a), dim=1)
                        c, _ = th.max(c, dim=1, keepdim=True)
                    else:
                        c, a = self.forward_sample_col_prob(rollout_data.observations, self.n_lagrange_samples, "mean")

                    col_loss_log = self.l_multiplier * (c - 0.1)
                    # col_loss = col_loss_log[col_loss_log > 0].mean()
                    col_loss = col_loss_log.mean()
                    lagrange_penalty_mean.append(col_loss.item())
                    lagrange_penalty_var.append(col_loss_log.var().item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                # Auxiliary loss
                auxiliary_loss = self.get_auxiliary_loss(rollout_data)

                if len(self.auxiliary_tasks) > 0:
                    auxiliary_losses.append(auxiliary_loss.item())

                loss = policy_loss + ent_coef * entropy_loss + self.vf_coef * value_loss + col_loss + auxiliary_loss

                '''
                KL divergence for learning rate adaption and early stopping
                '''
                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > self.target_kl:
                    # Decrease Learning Rate to prevent policy to deviate and become unstable
                    # Taken from https://github.com/zhejz/carla-roach/blob/main/agents/rl_birdview/models/ppo.py
                    if self.lr_schedule_step is not None:
                        self.kl_early_stop += 1
                        if self.kl_early_stop >= self.lr_schedule_step:
                            self.learning_rate *= 0.5
                            update_learning_rate(self.policy.action_net.optimizer, self.learning_rate)
                            update_learning_rate(self.policy.value_net.optimizer, self.learning_rate)
                            self.logger.record("train/learning_rate", self.learning_rate)
                            self.kl_early_stop = 0
                            if hasattr(self.policy, "log_std"):
                                self.policy.log_std.data -= 0.1
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break
                '''
                Optimization step Actor Critic
                '''
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                '''
                Optimization Safety Critic
                '''
                # Optimize Lagrange Multiplier but don't let it go to zero
                if self.safe_lagrange:
                    self.l_multiplier = max(0.05, self.l_multiplier + 1e-4 * (c.mean().item() - 0.1))

                current_col_prob = self.col_net.forward(rollout_data.observations, actions)
                target = rollout_data.col_prob_targets.view(-1, 1)
                col_net_loss = 0.5 * sum([weighted_bce(pred, target) for pred in current_col_prob])
                col_net_losses.append(col_net_loss.item())
                self.col_net.optimizer.zero_grad()
                col_net_loss.backward()
                self.col_net.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/ent_coef", ent_coef)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        # self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        if len(self.auxiliary_tasks) > 0:
            self.logger.record("train/auxiliary_loss", np.mean(auxiliary_losses))

        self.logger.record("train/col_net_loss", np.mean(col_net_losses))
        if self.safe_mult:
            self.logger.record("train/diff_col_now_next", np.mean(p_col_now_next_mean))
            self.logger.record("train/diff_col_now_next_var", np.mean(p_col_now_next_var))
        if self.safe_lagrange:
            self.logger.record("train/multiplier", self.l_multiplier)
            self.logger.record("train/col_lagrange_penalty", np.mean(lagrange_penalty_mean))
            self.logger.record("train/col_lagrange_penalty_var", np.mean(lagrange_penalty_var))

    def get_auxiliary_loss(self, rollout_data):
        auxiliary_loss = th.zeros(1, device="cuda")
        for task in self.auxiliary_tasks:
            if "action_penalty" in self.auxiliary_tasks:
                continue

            x = rollout_data.observations["img"] if isinstance(rollout_data.observations, dict) \
                else rollout_data.observations
            if task == "autoencoder":
                w = 10
                x_hat = self.policy.features_extractor.forward_autoencoder(x)
                x = x / 255  # divide by 255 since models outputs number between 0 and 1
                obstacle_loss = (x[:, 0] - x_hat[:, 0]) ** 2
                state_loss = (x[:, 1] - x_hat[:, 1]) ** 2
                weights = th.ones_like(state_loss) * 0.5
                weights[x[:, 1] > 0.35] = 10
                auxiliary_loss = auxiliary_loss + w * (weights * state_loss).mean()
                # auxiliary_loss = auxiliary_loss + w * (0.01 * obstacle_loss + weights * state_loss).mean()

            if task == "local_oc_grid":
                w = 1
                x_local_hat = self.policy.features_extractor.forward_local_oc_grid(x)
                x_local = get_local_oc_grid(x / 255)
                weights = th.ones_like(x_local) * 0.5
                weights[x_local > 0.1] = 10
                # Cells outside oc grid are marked with -1 in local oc_grid. Disregard those for loss
                weights[x_local < 0] = 0
                reconstruction_loss = weights * (x_local - x_local_hat) ** 2
                auxiliary_loss = auxiliary_loss + w * reconstruction_loss.mean()

            if task == "autoencoder_local":
                w = 10
                x = rollout_data.observations["img"] if isinstance(rollout_data.observations, dict) \
                    else rollout_data.observations
                x_hat = self.policy.features_extractor.forward_autoencoder(x)
                x = x[:, [0]] / 255  # divide by 255 since models outputs number between 0 and 1
                obstacle_loss = weighted_bce(x_hat, x)
                auxiliary_loss = auxiliary_loss + w * obstacle_loss

            if task == "autoencoder_car_racing":
                w = 10
                x = rollout_data.observations["img"]
                x_hat = self.policy.features_extractor.forward_autoencoder(x)
                x = x[:, [0]] / 255
                auxiliary_loss = auxiliary_loss + w * F.mse_loss(x, x_hat)

            if task == "depth":
                w = 10
                x = rollout_data.observations["img"] if isinstance(rollout_data.observations, dict) \
                    else rollout_data.observations

                batch_size = x.shape[0]
                idx = th.where(x > 250)
                idx = th.stack(idx, dim=0).cpu().numpy()[[0, 2, 3]]

                # find the min distances
                min_distances = []
                start = 0
                end = 0
                min_distances = th.zeros(batch_size, dtype=float, device="cuda")
                for i in range(batch_size):  # First row, last element is last batch
                    # Find how many elements correspond to batch
                    for k in range(start, idx.shape[1]):
                        if idx[0, k] != i:
                            end = k
                            break
                        elif k == idx.shape[1] - 1:  # For last element
                            end = k
                    # Distance calculation
                    distances = np.linalg.norm(idx[1:, start:end].T - np.array([5, 5]), axis=1)
                    if distances.shape[0] == 0:  # No obstacle
                        min_distances[i] = 1
                    else:
                        min_distances[i] = np.min(distances) / 7  # 7 is max distance for 11x11 grid
                    # Update search space
                    start = end

                # Loss calculation
                depth_pred = self.policy.features_extractor.forward_depth(x)
                auxiliary_loss = auxiliary_loss + w * ((depth_pred - min_distances.unsqueeze(1)) ** 2).mean()

            if task == "predict_next_state":
                w = 10
                x = rollout_data.observations
                a = rollout_data.actions
                x_next = rollout_data.next_observations
                x_robot = x[:, 0:7] - x_next[:, 0:7]  # Only learn difference in states
                x_lidar = x[:, 7:] - x_next[:, 7:]
                x_robot_pred, x_lidar_pred = self.policy.features_extractor.forward_next_state_prediction(x, a)
                robot_loss = ((x_robot - x_robot_pred) ** 2).mean()
                # Lidar has 720 rays. Only predict 24 of those, so down sample
                # down_sample_rate = 30 if x_lidar.shape[1] == 720 else 5
                # lidar_loss = ((x_lidar[:, ::down_sample_rate] - x_lidar_pred) ** 2).mean()
                lidar_loss = 0
                auxiliary_loss = auxiliary_loss + w * (robot_loss + 0.1 * lidar_loss)

        return auxiliary_loss
       
    # def train(self) -> None:
    #     """
    #     Update policy using the currently gathered rollout buffer.
    #     """
    #     # Switch to train mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(True)

    #     # Update optimizer learning rate
    #     self._update_learning_rate(self.policy.optimizer)
    #     # Compute current clip range
    #     clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
    #     # Optional: clip range for the value function
    #     if self.clip_range_vf is not None:
    #         clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

    #     entropy_losses = []
    #     pg_losses, value_losses = [], []
    #     clip_fractions = []

    #     continue_training = True
    #     # train for n_epochs epochs
    #     for epoch in range(self.n_epochs):
    #         approx_kl_divs = []
    #         # Do a complete pass on the rollout buffer
    #         for rollout_data in self.rollout_buffer.get(self.batch_size):
    #             actions = rollout_data.actions
    #             if isinstance(self.action_space, spaces.Discrete):
    #                 # Convert discrete action from float to long
    #                 actions = rollout_data.actions.long().flatten()

    #             # Re-sample the noise matrix because the log_std has changed
    #             if self.use_sde:
    #                 self.policy.reset_noise(self.batch_size)

    #             values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
    #             values = values.flatten()
    #             # Normalize advantage
    #             advantages = rollout_data.advantages
    #             # Normalization does not make sense if mini batchsize == 1, see GH issue #325
    #             if self.normalize_advantage and len(advantages) > 1:
    #                 advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    #             # ratio between old and new policy, should be one at the first iteration
    #             ratio = th.exp(log_prob - rollout_data.old_log_prob)

    #             # clipped surrogate loss
    #             policy_loss_1 = advantages * ratio
    #             policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
    #             policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

    #             # Logging
    #             pg_losses.append(policy_loss.item())
    #             clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
    #             clip_fractions.append(clip_fraction)

    #             if self.clip_range_vf is None:
    #                 # No clipping
    #                 values_pred = values
    #             else:
    #                 # Clip the difference between old and new value
    #                 # NOTE: this depends on the reward scaling
    #                 values_pred = rollout_data.old_values + th.clamp(
    #                     values - rollout_data.old_values, -clip_range_vf, clip_range_vf
    #                 )
    #             # Value loss using the TD(gae_lambda) target
    #             value_loss = F.mse_loss(rollout_data.returns, values_pred)
    #             value_losses.append(value_loss.item())

    #             # Entropy loss favor exploration
    #             if entropy is None:
    #                 # Approximate entropy when no analytical form
    #                 entropy_loss = -th.mean(-log_prob)
    #             else:
    #                 entropy_loss = -th.mean(entropy)

    #             entropy_losses.append(entropy_loss.item())

    #             loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

    #             # Calculate approximate form of reverse KL Divergence for early stopping
    #             # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
    #             # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
    #             # and Schulman blog: http://joschu.net/blog/kl-approx.html
    #             with th.no_grad():
    #                 log_ratio = log_prob - rollout_data.old_log_prob
    #                 approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
    #                 approx_kl_divs.append(approx_kl_div)

    #             if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
    #                 continue_training = False
    #                 if self.verbose >= 1:
    #                     print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
    #                 break

    #             # Optimization step
    #             self.policy.optimizer.zero_grad()
    #             loss.backward()
    #             # Clip grad norm
    #             th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    #             self.policy.optimizer.step()

    #         self._n_updates += 1
    #         if not continue_training:
    #             break

    #     explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    #     # Logs
    #     self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    #     self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    #     self.logger.record("train/value_loss", np.mean(value_losses))
    #     self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    #     self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    #     self.logger.record("train/loss", loss.item())
    #     self.logger.record("train/explained_variance", explained_var)
    #     if hasattr(self.policy, "log_std"):
    #         self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    #     self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    #     self.logger.record("train/clip_range", clip_range)
    #     if self.clip_range_vf is not None:
    #         self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        # progress_bar: bool = False,
        
        # ====== Safe RL specific ======
        # eval_env: Optional[GymEnv] = None,
        # eval_freq: int = -1,
        # n_eval_episodes: int = 5,
        # eval_log_path: Optional[str] = None,
    ) -> SelfPPO:
        return super(PPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            # progress_bar=progress_bar,
        
            # total_timesteps=total_timesteps,
            # callback=callback,
            # log_interval=log_interval,
            # eval_env=eval_env,
            # eval_freq=eval_freq,
            # n_eval_episodes=n_eval_episodes,
            # # tb_log_name=tb_log_name,
            # eval_log_path=eval_log_path,
            # reset_num_timesteps=reset_num_timesteps,
        ) # type: ignore
