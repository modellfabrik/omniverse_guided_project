import argparse
import cv2
from pathlib import Path

from isaaclab.app import AppLauncher

# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(
    description="UR5e with integrated RealSense D455 demo (camera + YOLO detection + IK move_to)."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always enable cameras
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------- #
# Isaac Lab Imports
# ----------------------------------------------------------------------------- #
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg

from ultralytics import YOLO


# ----------------------------------------------------------------------------- #
# Robot & Environment Configuration
# ----------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ROBOT_USD = sim_utils.UsdFileCfg(
    usd_path=f"{ROOT}/robotSetup/components/UR5e_OnRobot_Realsense.usd"
)
ARENA_USD = sim_utils.UsdFileCfg(
    usd_path=f"{ROOT}/robotSetup/components/Environment.usd"
)

ROBOT_CFG = ArticulationCfg(
    spawn=ROBOT_USD,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.75, -0.35, 0.2451),
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
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_finger_joint",
                "right_finger_joint",
            ],
            stiffness=4000.0,
            damping=80.0,
        ),
    },
)

# ----------------------------------------------------------------------------- #
# YOLO Setup
# ----------------------------------------------------------------------------- #
model = YOLO("yolov8n.pt")


def run_detection(frame):
    """Run YOLO detection and return annotated image + results."""
    results = model(frame)
    return results[0].plot(), results[0]


# ----------------------------------------------------------------------------- #
# Scene Configuration
# ----------------------------------------------------------------------------- #
class SceneCfg(InteractiveSceneCfg):
    """Scene with arena, lighting, UR5e robot, and integrated RealSense D455."""

    dome_light = AssetBaseCfg(
        prim_path="/World/Lights/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 0.95, 0.85)),
    )

    arena = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/NewArena",
        spawn=ARENA_USD,
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0, 0, 0, 1)),
    )

    Robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rsd455/RSD455/Camera_OmniVision_OV9782_Color",
        spawn=None,
        height=480,
        width=640,
        data_types=["rgb"],
    )


# ----------------------------------------------------------------------------- #
# Simulation Loop
# ----------------------------------------------------------------------------- #
def run_simulation(sim, scene, camera):
    sim_dt = sim.get_physics_dt()

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        if "rgb" in camera.data.output:
            rgb_tensor = camera.data.output["rgb"][0]
            rgb_frame = rgb_tensor.cpu().numpy().astype("uint8")

            annotated, results = run_detection(rgb_frame)
            cv2.imshow("UR5e Camera", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                print(f"Detected {model.names[cls_id]} with {conf:.2f} at {xyxy}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0, 0, 0])

    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    sim.step()
    scene.update(sim.get_physics_dt())

    robot = scene["Robot"]
    camera = scene["camera"]

    current_q = robot.data.joint_pos.clone()
    robot.set_joint_position_target(current_q)

    run_simulation(sim, scene, camera)
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
    simulation_app.close()
