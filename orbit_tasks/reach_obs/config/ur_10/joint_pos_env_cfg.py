# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

from omni.isaac.orbit.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.orbit.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from omni.isaac.orbit.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.orbit.utils import configclass
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm

import os
import sys

from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import orbit_tasks.reach_obs.mdp as mdp
from orbit_tasks.reach_obs.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
# from omni.isaac.orbit_assets import UR10_CFG  # isort: skip
from orbit_assets import UR10_CFG_contact  # isort: skip


##
# Environment configuration
##


@configclass
class UR10ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to ur10
        self.scene.robot = UR10_CFG_contact.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
        # override events
        # self.events.reset_robot_joints.params["position_range"] = (0.75, 1.25)
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]
        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along x-direction
        self.commands.ee_pose.body_name = "ee_link"
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        
@configclass
class UR10ReachEnvCfg_PLAY(UR10ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)
