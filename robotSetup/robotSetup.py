import argparse
import torch
import cv2
import numpy as np
from pathlib import Path

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="UR5e with integrated RealSense D455 demo.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always enable cameras from code
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------------------
# Isaac Lab Imports
# ------------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors.camera import CameraCfg

# ------------------------------------------------------------------------------
# Robot Configuration
# ------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

UR5E_USD = sim_utils.UsdFileCfg(usd_path=f"{ROOT}/robotSetup/UR5E_RSD455.usd")

ROBOT_CFG = ArticulationCfg(
    spawn=UR5E_USD,
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 1.57,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 1.57,
            "wrist_3_joint": 0.0,
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            stiffness=4000.0,
            damping=80.0,
        ),
    },
)

# ------------------------------------------------------------------------------
# Scene Configuration
# ------------------------------------------------------------------------------
class UR5eSceneCfg(InteractiveSceneCfg):
    """Scene with ground plane, lighting, UR5e robot and integrated RealSense D455."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    Robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Use the color camera inside the D455 sensor already attached to the robot
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rsd455/RSD455/Camera_OmniVision_OV9782_Color",
        spawn=None,
        height=480,
        width=640,
        data_types=["rgb"],
    )

# ------------------------------------------------------------------------------
# UR5e Controller
# ------------------------------------------------------------------------------
class UR5eController:
    """Wrapper to control the UR5e using Differential Inverse Kinematics."""

    def __init__(self, sim, scene, ee_link="wrist_3_link"):
        self.sim = sim
        self.scene = scene
        self.robot = scene["Robot"]

        # Differential IK controller
        cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.controller = DifferentialIKController(cfg, num_envs=scene.num_envs, device=sim.device)

        # Scene entity: all joints, specify end-effector body
        self.entity = SceneEntityCfg("Robot", joint_names=[".*"], body_names=[ee_link])
        self.entity.resolve(scene)

        # End-effector index
        self.ee_idx = self.entity.body_ids[0] - 1 if self.robot.is_fixed_base else self.entity.body_ids[0]

    def move_to(self, position, orientation=(1.0, 0.0, 0.0, 0.0)):
        """Set a target pose for the end-effector."""
        goal = torch.tensor([[*position, *orientation]], device=self.sim.device, dtype=torch.float32)
        self.controller.set_command(goal)

    def step(self):
        """Perform one IK step and apply joint targets."""
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_idx, :, self.entity.joint_ids]
        ee_pose_w = self.robot.data.body_pose_w[:, self.entity.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w
        joint_pos = self.robot.data.joint_pos[:, self.entity.joint_ids]

        # Compute end-effector pose relative to base
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )

        # IK compute and set joint targets
        joint_pos_des = self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.entity.joint_ids)

# ------------------------------------------------------------------------------
# Simulation Runner
# ------------------------------------------------------------------------------
def run_simulation(sim, scene, ur5e, camera):
    """
    Run the simulation loop. The function yields RGB frames from the robot camera.

    Args:
        sim: SimulationContext
        scene: InteractiveScene
        ur5e: UR5eController
        camera: Camera object

    Yields:
        rgb_frame (np.ndarray): Current RGB image from the RealSense camera as (H, W, 3) uint8.
    """
    sim_dt = sim.get_physics_dt()

    while simulation_app.is_running():
        # Robot control
        # ur5e.step()

        # Simulation step
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # Camera image
        rgb_frame = None
        if "rgb" in camera.data.output:
            rgb_tensor = camera.data.output["rgb"][0]  # (H, W, 4), float32 [0,1]
            rgb_frame = (rgb_tensor.cpu().numpy() * 255).astype("uint8")
            rgb_frame = rgb_frame[:, :, :3]  # drop alpha channel

            #cv2.imshow("Isaac Lab Camera", rgb_frame)

            #if cv2.waitKey(1) & 0xFF == ord("q"):
                #break

        
        yield rgb_frame
    
    cv2.destroyAllWindows()

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    # Create simulation and scene
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0, 0, 0])

    scene_cfg = UR5eSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Robot controller and camera
    ur5e = UR5eController(sim, scene, ee_link="wrist_3_link")
    camera = scene["camera"]

    # Example: set initial goal
    #ur5e.move_to((0.5, 0.0, 0.3))

    # Run loop and handle frames
    for rgb_frame in run_simulation(sim, scene, ur5e, camera):
        if rgb_frame is not None:
            # Example: print shape, or pass to AI
            print("Frame shape:", rgb_frame.shape)
            print("frame: ", rgb_frame)
            # later: process with AI instead of printing

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    simulation_app.close()

    
