import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                    

TEST = sim_utils.UsdFileCfg(usd_path=f"{ROOT}/robotSetup/UR5E_2F85_RSD455.usd")

ROBOT_CFG = ArticulationCfg(
    spawn = TEST,
    init_state = ArticulationCfg.InitialStateCfg(),
    actuators = {

    }
)

def define_camera() -> Camera:
    camera_cfg = CameraCfg(
        prim_path="/World/envs/env_0/Robot/rsd455/RSD455/Camera_OmniVision_OV9782_Color",
        spawn=None,
        update_period=0,
        height=800,
        width=1280,
        data_types=[
            "rgb",
        ]
    )

    return Camera(cfg=camera_cfg)

class NewRobotsSceneCfg(InteractiveSceneCfg):
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
