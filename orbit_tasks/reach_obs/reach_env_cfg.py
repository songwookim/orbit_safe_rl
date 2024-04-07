# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from omni.isaac.orbit.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import ActionTermCfg as ActionTerm
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import EventTermCfg as EventTerm
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RewardTermCfg as RewTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.orbit.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.noise import AdditiveUniformNoiseCfg as Unoise

# import omni.isaac.orbit_tasks.manipulation.reach.mdp as mdp
import orbit_tasks.reach_obs.mdp as mdp

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    
    # objects
    object: RigidObjectCfg = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, -0.1, 0.255], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(5, 0.1, 5),
                rigid_props=RigidBodyPropertiesCfg(
                    # solver_position_iteration_count=16,
                    # solver_velocity_iteration_count=1,
                    # max_angular_velocity=1000.0,
                    # max_linear_velocity=1000.0,
                    # max_depenetration_velocity=5.0,
                    # disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
        )
    
    # sensor
    contact_forces= ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # pos_x=(0.35, 0.65),
            # pos_y=(-0.2, 0.2),
            # pos_z=(0.15, 0.5),
            # roll=(0.0, 0.0),
            # pitch = MISSING,  # depends on end-effector axis
            # yaw=(-3.14, 3.14),

            pos_x=(0.35, 0.35),
            pos_y=(-0.2, -0.2),
            pos_z=(0.35, 0.35),
            roll=(0.0, 0.0),
            pitch = MISSING,  # depends on end-effector axis
            yaw=(3.14, 3.14),            
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            # "position_range":  (-1.5, 1.5),  
            "position_range": (1.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    
    # reset_obstacle_pose = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
            
    #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)},
    #         # "position_range":  (-1.5, 1.5),  
    #         # "velocity_range": (0.0, 0.0),
    #         "velocity_range": {"roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    
    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        # weight=-0.2,
        weight=-1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time_out : DoneTerm = None # type: ignore
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact, 
        params={"threshold": 50, "sensor_cfg": SceneEntityCfg("contact_forces")}
        )

    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"), "threshold": 1.0},
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(RLTaskEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0
        
        # self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        # self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        # self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        # self.sim.physx.friction_correlation_distance = 0.00625
        # added
        # self.sim.physics_material = self.scene.object.physics_material
        self.sim.disable_contact_processing = True
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
