# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass

from rsl_rl_custom.modules.runners.rsl_rl_cfgs import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlPpoSafetyCriticCfg, RslRlPpoActorCfg, RslRlPpoCriticCfg
# from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import (
#     RslRlOnPolicyRunnerCfg,
#     RslRlPpoActorCriticCfg,
#     RslRlPpoAlgorithmCfg,
# )


@configclass
class UR10ReachPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "reach_obs_ur10"
    run_name = ""
    resume = False
    empirical_normalization = False
    policy = RslRlPpoActorCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128,64],
        activation="elu",
    )
    
    value = RslRlPpoCriticCfg(
        init_noise_std=1.0,
        critic_hidden_dims=[256, 128,64],
        activation="elu",
    )
    # ==== Safety ====
    safety_critic = RslRlPpoSafetyCriticCfg(
        activation="relu",
        safety_critic_hidden_dims=[256, 128, 64],
        n_critics = 2, # 2        
    )
        
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        
        # ==== Safety ====
        safe_largrange = True,
        l_multiplier_init = 1.0,
        n_lagrange_samples = 1,
        lagrange_penalty_mean=[],
        lagrange_penalty_var=[],
        collision_reward = -1,
        gamma_col_net = 0.95      
    )
