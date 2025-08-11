# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimalbeispiel: Intel RealSense D455 als digitale Simulation in Isaac Lab
→ Mit Live-Stream im OpenCV-Fenster, keine Speicherung auf Festplatte
"""

import argparse
from isaaclab.app import AppLauncher

# CLI-Argumente
parser = argparse.ArgumentParser(description="Intel RealSense D455 Kamera Demo.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

# App starten
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import random
import cv2

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import omni
from pxr import PhysxSchema, UsdPhysics

def disable_rigidbody(prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        print(f"Prim {prim_path} nicht gefunden.")
        return

    rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    if rigid_body_api:
        rigid_body_api.CreateRigidBodyEnabledAttr(False)
        print(f"RigidBody bei {prim_path} deaktiviert.")

    physx_rigid_api = PhysxSchema.PhysxRigidBodyAPI(prim)
    if physx_rigid_api:
        physx_rigid_api.CreateDisableGravityAttr(True)
        print(f"PhysX Gravity bei {prim_path} deaktiviert.")

def define_sensor() -> Camera:
    """Lädt die RealSense D455 als digitalen Zwilling und erstellt den Farbsensor."""
    prim_utils.create_prim(
        "/World/RealSense_D455",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Sensors/Intel/RealSense/rsd455.usd",
        translation=(-1.0, 0.5, 0.5),
    )

    disable_rigidbody("/World/RealSense_D455/RSD455")

    camera_cfg = CameraCfg(
        prim_path="/World/RealSense_D455/RSD455/Camera_OmniVision_OV9782_Color",
        spawn=None,
        update_period=0,
        height=800,
        width=1280,
        data_types=[
            "rgb",
        ]
    )

    return Camera(cfg=camera_cfg)

def define_sensor_left_eye() -> Camera:
    # Farbsensor auswählen
    camera_cfg = CameraCfg(
        prim_path="/World/RealSense_D455/RSD455/Camera_OmniVision_OV9782_Left",
        spawn=None,
        update_period=0,
        height=800,
        width=1280,
        data_types=[
            "rgb",
        ]
    )

    return Camera(cfg=camera_cfg)

def define_sensor_right_eye() -> Camera:
    # Farbsensor auswählen
    camera_cfg = CameraCfg(
        prim_path="/World/RealSense_D455/RSD455/Camera_OmniVision_OV9782_Right",
        spawn=None,
        update_period=0,
        height=800,
        width=1280,
        data_types=[
            "rgba",
        ]
    )

    return Camera(cfg=camera_cfg)

def define_sensor_depth() -> Camera:
    # Farbsensor auswählen
    camera_cfg = CameraCfg(
        prim_path="/World/RealSense_D455/RSD455/Camera_Pseudo_Depth",
        spawn=None,
        update_period=0,
        height=800,
        width=1280,
        data_types=[
            "depth",
        ]
    )

    return Camera(cfg=camera_cfg)

def design_scene() -> dict:
    """Erstellt die Szene."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    scene_entities = {}
    prim_utils.create_prim("/World/Objects", "Xform")

    for i in range(8):
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        color = (random.random(), random.random(), random.random())
        prim_type = random.choice(["Cube", "Cone", "Cylinder"])
        common_properties = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=5.0),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=color, metallic=0.5),
            "semantic_tags": [("class", prim_type)],
        }
        if prim_type == "Cube":
            shape_cfg = sim_utils.CuboidCfg(size=(0.25, 0.25, 0.25), **common_properties)
        elif prim_type == "Cone":
            shape_cfg = sim_utils.ConeCfg(radius=0.1, height=0.25, **common_properties)
        else:
            shape_cfg = sim_utils.CylinderCfg(radius=0.25, height=0.25, **common_properties)

        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Objects/Obj_{i:02d}",
            spawn=shape_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        )
        scene_entities[f"rigid_object{i}"] = RigidObject(cfg=obj_cfg)

    camera = define_sensor()
    camera_left = define_sensor_left_eye()
    scene_entities["camera"] = camera_left
    return scene_entities

def run_simulator(sim: sim_utils.SimulationContext, scene_entities: dict):
    """Führt die Simulation mit Live-Stream aus."""
    camera: Camera = scene_entities["camera"]

    if sim.has_gui() and args_cli.draw:
        cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/CameraPointCloud")
        cfg.markers["hit"].radius = 0.002
        pc_markers = VisualizationMarkers(cfg)

    while simulation_app.is_running():
        sim.step()
        camera.update(dt=sim.get_physics_dt())

        if "rgb" in camera.data.output.keys():
            rgb_frame = camera.data.output["rgb"][0].cpu().numpy()  # (H, W, 4) --> Getting Data from the camera
            rgb_frame = (rgb_frame * 255).astype(np.uint8)
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGBA2BGR)
            cv2.imshow("RealSense D455 Stream", rgb_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.0])
    scene_entities = design_scene()
    sim.reset()
    print("[INFO]: Szene mit RealSense D455 geladen...")
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
