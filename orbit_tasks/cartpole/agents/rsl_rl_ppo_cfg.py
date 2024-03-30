# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass
from rsl_rl_custom.modules.runners.rsl_rl_cfgs import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

# from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlPpoActorCriticCfg,
#     RslRlPpoAlgorithmCfg,
# )


@configclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 16
    max_iterations = 300
    save_interval = 50
    experiment_name = "cartpole"
    empirical_normalization = False
    # ==== Safety ====
    n_col_value_samples = 20
    
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
        
        # ==== Safety ====
        n_critics = 2, # 2
        safety_critic_hidden_dims=[256, 256],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        
        # ==== Safety ====
        safe_largrange = True,
        l_multiplier_init = 1.0,
        n_lagrange_samples = 1,
        lagrange_penalty_mean=[],
        lagrange_penalty_var=[],
        collision_reward = -10,
        gamma_col_net = 0.6
        
    )
