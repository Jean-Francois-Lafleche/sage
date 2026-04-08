# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main USD post-processing pipeline for SimReady compliance.

This module transforms generated USD scenes into simulation-ready format by:
1. Adding proper physics properties (collision meshes, mass, friction)
2. Standardizing materials to UsdPreviewSurface
3. Creating proper USD prim hierarchy
4. Adding semantic labels and metadata

SimReady Guidelines Reference:
https://docs.omniverse.nvidia.com/simready/latest/overview.html
"""

from __future__ import annotations
import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Try to import USD libraries
try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
    HAS_USD = True
except ImportError:
    HAS_USD = False
    print("Warning: USD (pxr) libraries not available. SimReady processing disabled.")


@dataclass
class PhysicsProperties:
    """Physics properties for a SimReady asset."""
    mass: float  # kg
    density: float = 1000.0  # kg/m³
    friction_static: float = 0.5
    friction_dynamic: float = 0.4
    restitution: float = 0.3
    collision_type: str = "convex"  # convex, mesh, or box


@dataclass
class SemanticLabel:
    """Semantic label for a SimReady asset."""
    category: str  # e.g., "furniture", "appliance", "decor"
    subcategory: str = ""  # e.g., "chair", "table", "lamp"
    purpose: str = ""  # e.g., "seating", "storage", "lighting"
    material_class: str = ""  # e.g., "wood", "metal", "fabric"


# Default physics properties by object category
DEFAULT_PHYSICS = {
    "furniture": PhysicsProperties(mass=20.0, friction_static=0.6, collision_type="convex"),
    "appliance": PhysicsProperties(mass=10.0, friction_static=0.5, collision_type="box"),
    "decor": PhysicsProperties(mass=1.0, friction_static=0.4, collision_type="convex"),
    "lighting": PhysicsProperties(mass=2.0, friction_static=0.4, collision_type="convex"),
    "storage": PhysicsProperties(mass=15.0, friction_static=0.6, collision_type="convex"),
    "textile": PhysicsProperties(mass=0.5, friction_static=0.8, collision_type="mesh"),
    "default": PhysicsProperties(mass=5.0, friction_static=0.5, collision_type="convex"),
}


class SimReadyProcessor:
    """Processes USD stages to make them SimReady compliant.
    
    Usage:
        processor = SimReadyProcessor()
        processor.process_file("input.usda", "output_simready.usda")
    """

    def __init__(
        self,
        up_axis: str = "Y",
        meters_per_unit: float = 1.0,
        add_physics: bool = True,
        add_semantics: bool = True,
        add_materials: bool = True,
    ):
        """Initialize the processor.
        
        Args:
            up_axis: Up axis for the scene (Y or Z)
            meters_per_unit: Scale factor (1.0 = meters)
            add_physics: Whether to add physics properties
            add_semantics: Whether to add semantic labels
            add_materials: Whether to standardize materials
        """
        if not HAS_USD:
            raise RuntimeError("USD (pxr) libraries required for SimReady processing")
        
        self.up_axis = up_axis
        self.meters_per_unit = meters_per_unit
        self.add_physics = add_physics
        self.add_semantics = add_semantics
        self.add_materials = add_materials

    def process_file(
        self,
        input_path: str,
        output_path: str,
        object_metadata: Optional[Dict[str, Dict]] = None,
    ) -> bool:
        """Process a USD file to make it SimReady.
        
        Args:
            input_path: Path to input USD file
            output_path: Path for output SimReady USD file
            object_metadata: Optional dict mapping prim paths to metadata
                            e.g., {"/World/Table": {"category": "furniture", "mass": 25.0}}
        
        Returns:
            True if successful, False otherwise
        """
        # Open the stage
        stage = Usd.Stage.Open(input_path)
        if not stage:
            print(f"Error: Could not open USD stage: {input_path}")
            return False
        
        # Set stage metadata
        self._set_stage_metadata(stage)
        
        # Process all mesh prims
        object_metadata = object_metadata or {}
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                self._process_mesh_prim(stage, prim, object_metadata.get(str(prim.GetPath()), {}))
        
        # Create physics scene if not exists
        if self.add_physics:
            self._ensure_physics_scene(stage)
        
        # Save the processed stage
        stage.Export(output_path)
        print(f"SimReady USD saved to: {output_path}")
        return True

    def _set_stage_metadata(self, stage: "Usd.Stage") -> None:
        """Set stage-level metadata for SimReady compliance."""
        # Set up axis
        if self.up_axis == "Y":
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        else:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        
        # Set meters per unit
        UsdGeom.SetStageMetersPerUnit(stage, self.meters_per_unit)
        
        # Set documentation
        stage.GetRootLayer().documentation = (
            "SimReady USD scene generated by SAGE. "
            "Compliant with NVIDIA SimReady guidelines."
        )

    def _process_mesh_prim(
        self,
        stage: "Usd.Stage",
        prim: "Usd.Prim",
        metadata: Dict[str, Any],
    ) -> None:
        """Process a single mesh prim for SimReady compliance."""
        prim_path = prim.GetPath()
        
        # Get or infer category
        category = metadata.get("category", "default")
        
        # Add physics
        if self.add_physics:
            physics_props = DEFAULT_PHYSICS.get(category, DEFAULT_PHYSICS["default"])
            if "mass" in metadata:
                physics_props = PhysicsProperties(
                    mass=metadata["mass"],
                    friction_static=physics_props.friction_static,
                    friction_dynamic=physics_props.friction_dynamic,
                    restitution=physics_props.restitution,
                    collision_type=physics_props.collision_type,
                )
            self._add_physics_to_prim(stage, prim, physics_props)
        
        # Add semantic label
        if self.add_semantics:
            semantic = SemanticLabel(
                category=category,
                subcategory=metadata.get("subcategory", ""),
                purpose=metadata.get("purpose", ""),
                material_class=metadata.get("material", ""),
            )
            self._add_semantic_label(prim, semantic)
        
        # Ensure material binding
        if self.add_materials:
            self._ensure_material_binding(stage, prim)

    def _add_physics_to_prim(
        self,
        stage: "Usd.Stage",
        prim: "Usd.Prim",
        props: PhysicsProperties,
    ) -> None:
        """Add physics properties to a prim."""
        prim_path = prim.GetPath()
        
        # Add rigid body API
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
        
        # Add mass API
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(props.mass)
        mass_api.CreateDensityAttr(props.density)
        
        # Add collision API
        collision = UsdPhysics.CollisionAPI.Apply(prim)
        
        # Create collision mesh (simplified for now - just use the mesh itself)
        # In a full implementation, we'd compute convex hulls or simplified meshes
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        if props.collision_type == "convex":
            mesh_collision.CreateApproximationAttr("convexHull")
        elif props.collision_type == "mesh":
            mesh_collision.CreateApproximationAttr("none")
        else:  # box
            mesh_collision.CreateApproximationAttr("boundingCube")
        
        # Create physics material
        material_path = prim_path.AppendChild("PhysicsMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(
            stage.DefinePrim(material_path, "PhysicsMaterial")
        )
        physics_material.CreateStaticFrictionAttr(props.friction_static)
        physics_material.CreateDynamicFrictionAttr(props.friction_dynamic)
        physics_material.CreateRestitutionAttr(props.restitution)
        
        # Bind material to collision
        # Note: Material binding would go here in a full implementation

    def _add_semantic_label(self, prim: "Usd.Prim", label: SemanticLabel) -> None:
        """Add semantic labels to a prim as custom attributes."""
        prim.CreateAttribute("semantic:category", Sdf.ValueTypeNames.String).Set(label.category)
        if label.subcategory:
            prim.CreateAttribute("semantic:subcategory", Sdf.ValueTypeNames.String).Set(label.subcategory)
        if label.purpose:
            prim.CreateAttribute("semantic:purpose", Sdf.ValueTypeNames.String).Set(label.purpose)
        if label.material_class:
            prim.CreateAttribute("semantic:material", Sdf.ValueTypeNames.String).Set(label.material_class)

    def _ensure_material_binding(self, stage: "Usd.Stage", prim: "Usd.Prim") -> None:
        """Ensure the prim has a valid material binding."""
        # Check if material already bound
        material_binding = UsdShade.MaterialBindingAPI(prim)
        bound_material = material_binding.ComputeBoundMaterial()
        
        if bound_material[0]:
            # Material exists, optionally convert to UsdPreviewSurface
            return
        
        # Create a default material if none exists
        prim_path = prim.GetPath()
        material_path = prim_path.AppendChild("DefaultMaterial")
        
        material = UsdShade.Material.Define(stage, material_path)
        shader_path = material_path.AppendChild("Shader")
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        # Set default shader properties
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        
        # Connect shader output to material surface
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        
        # Bind material to prim
        material_binding.Bind(material)

    def _ensure_physics_scene(self, stage: "Usd.Stage") -> None:
        """Ensure a physics scene exists at the stage level."""
        physics_scene_path = "/PhysicsScene"
        physics_scene = stage.GetPrimAtPath(physics_scene_path)
        
        if not physics_scene:
            physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
            physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, -1, 0))
            physics_scene.CreateGravityMagnitudeAttr(9.81)

    def process_scene_layout(
        self,
        layout_json: Dict,
        output_path: str,
    ) -> bool:
        """Create a SimReady USD from a SAGE layout JSON.
        
        This creates a fresh USD stage from a scene layout rather than
        processing an existing file.
        
        Args:
            layout_json: SAGE scene layout dictionary
            output_path: Path for output USD file
            
        Returns:
            True if successful
        """
        # Create new stage
        stage = Usd.Stage.CreateNew(output_path)
        
        # Set stage metadata
        self._set_stage_metadata(stage)
        
        # Define root xform
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        
        # Process rooms
        for room in layout_json.get("rooms", []):
            room_path = f"/World/{room.get('id', 'Room')}"
            room_xform = UsdGeom.Xform.Define(stage, room_path)
            
            # Process objects in room
            for obj in room.get("objects", []):
                obj_path = f"{room_path}/{obj.get('id', 'Object')}"
                # Note: In a full implementation, we'd load the actual mesh here
                obj_xform = UsdGeom.Xform.Define(stage, obj_path)
                
                # Set transform
                if "position" in obj:
                    pos = obj["position"]
                    obj_xform.AddTranslateOp().Set(Gf.Vec3d(pos["x"], pos["y"], pos["z"]))
                
                if "rotation" in obj:
                    rot = obj["rotation"]
                    # Convert euler to quaternion or use euler ops
                    obj_xform.AddRotateXYZOp().Set(Gf.Vec3f(rot.get("x", 0), rot.get("y", 0), rot.get("z", 0)))
        
        # Add physics scene
        if self.add_physics:
            self._ensure_physics_scene(stage)
        
        # Save
        stage.Save()
        print(f"SimReady USD created: {output_path}")
        return True


def process_scene_to_simready(
    input_path: str,
    output_path: str,
    object_metadata: Optional[Dict] = None,
) -> bool:
    """Convenience function to process a scene to SimReady format.
    
    Args:
        input_path: Path to input USD file
        output_path: Path for output SimReady USD file
        object_metadata: Optional metadata for objects
        
    Returns:
        True if successful
    """
    processor = SimReadyProcessor()
    return processor.process_file(input_path, output_path, object_metadata)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        success = process_scene_to_simready(input_file, output_file)
        sys.exit(0 if success else 1)
    else:
        print("Usage: python usd_processor.py <input.usd> <output_simready.usd>")
        sys.exit(1)
