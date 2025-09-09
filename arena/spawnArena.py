import os
from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import omni.replicator.core as rep
import omni.usd
from pxr import Sdf, Gf


def run_custom_scene():
    # USD-Datei laden
    script_dir = os.path.dirname(os.path.abspath(__file__))
    usd_file_path = os.path.join(script_dir, "RoboArena.usd")
    
    omni.usd.get_context().new_stage()
    omni.usd.get_context().open_stage(usd_file_path)
    rep.orchestrator.set_capture_on_play(False)
    
    stage = omni.usd.get_context().get_stage()
    
    if stage is None:
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
    
    
    # Robot aus Robo.usd laden
    robo_usd_path = os.path.join(script_dir, "Robo.usd")
    
    print(f"Lade Robot aus: {robo_usd_path}")
    
    if not os.path.exists(robo_usd_path):
        print(f"FEHLER: Robo.usd nicht gefunden in {script_dir}")
        return
    
    # Temporäre Stage für Robo.usd erstellen
    temp_context = omni.usd.create_context("temp_robot_context")
    temp_success = temp_context.open_stage(robo_usd_path)
    
    if not temp_success:
        print("FEHLER: Konnte Robo.usd nicht öffnen")
        temp_context.close_stage()
        return
    
    temp_stage = temp_context.get_stage()
    
    # Robot Prim in der temp Stage finden
 
    robot_source_path = "/World/Robot"
    robot_source_prim = temp_stage.GetPrimAtPath(robot_source_path)

    if not robot_source_prim or not robot_source_prim.IsValid():
        print("FEHLER: Robot nicht bei /World/Robot gefunden")
        temp_context.close_stage()
        return
    print(f"Robot gefunden: {robot_source_path}")
    
    # Robot in Hauptszene kopieren
    robot_target_path = "/World/ImportedRobot"
    robot_prim = stage.DefinePrim(robot_target_path, "Xform")
    
    # Robot als Referenz hinzufügen
    try:
        robot_prim.GetReferences().AddReference(robo_usd_path, robot_source_path)
        print("Robot erfolgreich geladen")
    except Exception as e:
        print(f"FEHLER beim Laden: {e}")
        # Fallback: Kopiere Layer-Inhalt
        try:
            root_layer = stage.GetRootLayer()
            temp_layer = temp_stage.GetRootLayer()
            Sdf.CopySpec(temp_layer, robot_source_path, root_layer, robot_target_path)
            print("Robot per CopySpec geladen")
        except Exception as e2:
            print(f"FEHLER: {e2}")
            temp_context.close_stage()
            return
    
    # Temp Stage schließen
    temp_context.close_stage()
    
    # Robot positionieren
    robot_prim.CreateAttribute("xformOp:translate", Sdf.ValueTypeNames.Double3).Set((180.0, 3.0, -41.0))
    robot_prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Double3).Set((-90.0, 0.0, 0.0))
    robot_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3).Set((100.0, 100.0, 100.0))
    robot_prim.CreateAttribute("xformOpOrder", Sdf.ValueTypeNames.TokenArray).Set(["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"])
    
    # Beleuchtung hinzufügen falls nicht vorhanden
    if not stage.GetPrimAtPath("/World/DomeLight"):
        dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
        dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)
    
    # Render Setup
    rp = rep.create.render_product("/OmniverseKit_Persp", (1024, 768))
    
    writer = rep.writers.get("BasicWriter")
    out_dir = os.path.join(os.getcwd(), "_out_robo_arena")
    
    writer.initialize(output_dir=out_dir, rgb=True)
    writer.attach(rp)
    
    # Bilder generieren
    num_frames = 5
    
    for i in range(num_frames):
        print(f"Frame {i+1}/{num_frames}")
        
        # Robot leicht bewegen
        new_position = (180.0 + i*0.2, 3.0, -41.0)
        robot_prim.GetAttribute("xformOp:translate").Set(new_position)
        
        rep.orchestrator.step()
    
    # Cleanup
    writer.detach()
    rp.destroy()
    rep.orchestrator.wait_until_complete()
    
    print(f"Fertig! {num_frames} Bilder gespeichert in: {out_dir}")
    input("Enter zum Schließen...")


if __name__ == "__main__":
    run_custom_scene()
    
    while simulation_app.is_running():
        simulation_app.update()
    
    simulation_app.close()