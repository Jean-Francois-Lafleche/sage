#!/usr/bin/env python3
"""
Build USD scene with proper textured floor, walls, and lighting.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, '/tmp/sage-repo/server')


def create_procedural_wood_texture(output_path, width=512, height=512):
    """Create a simple wood grain texture."""
    from PIL import Image
    import numpy as np
    
    # Base wood color
    base_color = np.array([0.55, 0.35, 0.2])  # brown
    
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            # Wood grain pattern
            noise = np.sin(x * 0.1 + np.sin(y * 0.05) * 3) * 0.1
            noise += np.sin(y * 0.02) * 0.05
            
            color = base_color + noise
            color = np.clip(color, 0, 1)
            img[y, x] = color
    
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_pil.save(output_path)
    return output_path


def create_procedural_brick_texture(output_path, width=512, height=512):
    """Create a simple brick texture."""
    from PIL import Image
    import numpy as np
    
    # Brick and mortar colors
    brick_color = np.array([0.65, 0.35, 0.25])  # reddish brown
    mortar_color = np.array([0.8, 0.75, 0.7])   # light gray
    
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    brick_h = 32  # brick height in pixels
    brick_w = 64  # brick width in pixels
    mortar = 4    # mortar thickness
    
    for y in range(height):
        for x in range(width):
            row = y // brick_h
            # Offset every other row
            offset = (brick_w // 2) if row % 2 else 0
            col = (x + offset) // brick_w
            
            # Check if we're in mortar
            y_in_brick = y % brick_h
            x_in_brick = (x + offset) % brick_w
            
            if y_in_brick < mortar or x_in_brick < mortar:
                img[y, x] = mortar_color
            else:
                # Add some variation to bricks
                variation = np.random.uniform(-0.05, 0.05, 3)
                img[y, x] = np.clip(brick_color + variation, 0, 1)
    
    img_pil = Image.fromarray((img * 255).astype(np.uint8))
    img_pil.save(output_path)
    return output_path


def build_scene_with_materials(layout_path: str, output_dir: str) -> str:
    """Build USD scene with proper materials for floor and walls."""
    
    with open(layout_path) as f:
        layout = json.load(f)
    
    room = layout['rooms'][0]
    dims = room['dimensions']
    w, l, h = dims['width'], dims['length'], dims['height']
    walls = room.get('walls', [])
    env = layout.get('environment', {})
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    tex_dir = os.path.join(output_dir, 'textures')
    os.makedirs(tex_dir, exist_ok=True)
    mesh_dir = os.path.join(output_dir, 'meshes')
    
    # Generate textures
    wood_tex = os.path.join(tex_dir, 'wood_floor.png')
    brick_tex = os.path.join(tex_dir, 'brick_wall.png')
    
    print("Generating floor texture...")
    create_procedural_wood_texture(wood_tex)
    
    print("Generating brick texture...")
    create_procedural_brick_texture(brick_tex)
    
    # Build scene USD
    scene_path = os.path.join(output_dir, 'scene.usda')
    
    with open(scene_path, 'w') as f:
        # Header
        f.write('#usda 1.0\n')
        f.write('(\n')
        f.write('    defaultPrim = "World"\n')
        f.write('    metersPerUnit = 1\n')
        f.write('    upAxis = "Z"\n')
        f.write(')\n\n')
        
        f.write('def Xform "World"\n')
        f.write('{\n')
        
        # =====================
        # ROOM GEOMETRY
        # =====================
        f.write('    def Xform "Room"\n')
        f.write('    {\n')
        
        # Floor with wood texture
        f.write('        def Mesh "Floor"\n')
        f.write('        {\n')
        f.write('            int[] faceVertexCounts = [4]\n')
        f.write('            int[] faceVertexIndices = [0, 1, 2, 3]\n')
        f.write(f'            point3f[] points = [(0, 0, 0), ({w}, 0, 0), ({w}, {l}, 0), (0, {l}, 0)]\n')
        f.write(f'            texCoord2f[] primvars:st = [(0, 0), ({w}, 0), ({w}, {l}), (0, {l})] (\n')
        f.write('                interpolation = "vertex"\n')
        f.write('            )\n')
        f.write('            rel material:binding = <FloorMaterial>\n')
        f.write('\n')
        f.write('            def Material "FloorMaterial"\n')
        f.write('            {\n')
        f.write('                token outputs:surface.connect = <PBRShader.outputs:surface>\n')
        f.write('\n')
        f.write('                def Shader "PBRShader"\n')
        f.write('                {\n')
        f.write('                    uniform token info:id = "UsdPreviewSurface"\n')
        f.write('                    color3f inputs:diffuseColor.connect = <../DiffuseTexture.outputs:rgb>\n')
        f.write('                    float inputs:roughness = 0.4\n')
        f.write('                    token outputs:surface\n')
        f.write('                }\n')
        f.write('\n')
        f.write('                def Shader "DiffuseTexture"\n')
        f.write('                {\n')
        f.write('                    uniform token info:id = "UsdUVTexture"\n')
        f.write('                    asset inputs:file = @./textures/wood_floor.png@\n')
        f.write('                    float2 inputs:st.connect = <../PrimvarReader.outputs:result>\n')
        f.write('                    token inputs:wrapS = "repeat"\n')
        f.write('                    token inputs:wrapT = "repeat"\n')
        f.write('                    float3 outputs:rgb\n')
        f.write('                }\n')
        f.write('\n')
        f.write('                def Shader "PrimvarReader"\n')
        f.write('                {\n')
        f.write('                    uniform token info:id = "UsdPrimvarReader_float2"\n')
        f.write('                    token inputs:varname = "st"\n')
        f.write('                    float2 outputs:result\n')
        f.write('                }\n')
        f.write('            }\n')
        f.write('        }\n')
        
        # Ceiling (white)
        f.write('\n        def Mesh "Ceiling"\n')
        f.write('        {\n')
        f.write('            int[] faceVertexCounts = [4]\n')
        f.write('            int[] faceVertexIndices = [0, 1, 2, 3]\n')
        f.write(f'            point3f[] points = [(0, 0, {h}), ({w}, 0, {h}), ({w}, {l}, {h}), (0, {l}, {h})]\n')
        f.write('            color3f[] primvars:displayColor = [(0.95, 0.95, 0.95)]\n')
        f.write('        }\n')
        
        # Walls with brick texture
        for i, wall in enumerate(walls):
            sp = wall['start_point']
            ep = wall['end_point']
            wh = wall['height']
            
            # Create wall as a quad mesh
            f.write(f'\n        def Mesh "Wall_{i}"\n')
            f.write('        {\n')
            f.write('            int[] faceVertexCounts = [4]\n')
            f.write('            int[] faceVertexIndices = [0, 1, 2, 3]\n')
            
            # Wall vertices: bottom-left, bottom-right, top-right, top-left
            f.write(f'            point3f[] points = [({sp["x"]}, {sp["y"]}, 0), ({ep["x"]}, {ep["y"]}, 0), ({ep["x"]}, {ep["y"]}, {wh}), ({sp["x"]}, {sp["y"]}, {wh})]\n')
            
            # UV coords for tiling
            wall_len = np.sqrt((ep['x'] - sp['x'])**2 + (ep['y'] - sp['y'])**2)
            f.write(f'            texCoord2f[] primvars:st = [(0, 0), ({wall_len}, 0), ({wall_len}, {wh}), (0, {wh})] (\n')
            f.write('                interpolation = "vertex"\n')
            f.write('            )\n')
            f.write(f'            rel material:binding = <WallMaterial_{i}>\n')
            f.write('\n')
            f.write(f'            def Material "WallMaterial_{i}"\n')
            f.write('            {\n')
            f.write('                token outputs:surface.connect = <PBRShader.outputs:surface>\n')
            f.write('\n')
            f.write('                def Shader "PBRShader"\n')
            f.write('                {\n')
            f.write('                    uniform token info:id = "UsdPreviewSurface"\n')
            f.write('                    color3f inputs:diffuseColor.connect = <../DiffuseTexture.outputs:rgb>\n')
            f.write('                    float inputs:roughness = 0.8\n')
            f.write('                    token outputs:surface\n')
            f.write('                }\n')
            f.write('\n')
            f.write('                def Shader "DiffuseTexture"\n')
            f.write('                {\n')
            f.write('                    uniform token info:id = "UsdUVTexture"\n')
            f.write('                    asset inputs:file = @./textures/brick_wall.png@\n')
            f.write('                    float2 inputs:st.connect = <../PrimvarReader.outputs:result>\n')
            f.write('                    token inputs:wrapS = "repeat"\n')
            f.write('                    token inputs:wrapT = "repeat"\n')
            f.write('                    float3 outputs:rgb\n')
            f.write('                }\n')
            f.write('\n')
            f.write('                def Shader "PrimvarReader"\n')
            f.write('                {\n')
            f.write('                    uniform token info:id = "UsdPrimvarReader_float2"\n')
            f.write('                    token inputs:varname = "st"\n')
            f.write('                    float2 outputs:result\n')
            f.write('                }\n')
            f.write('            }\n')
            f.write('        }\n')
        
        f.write('    }\n')  # End Room
        
        # =====================
        # LIGHTING
        # =====================
        f.write('\n    def Xform "Lights"\n')
        f.write('    {\n')
        
        # Main area lights (4 ceiling lights)
        light_positions = [
            (w * 0.25, l * 0.25),
            (w * 0.75, l * 0.25),
            (w * 0.25, l * 0.75),
            (w * 0.75, l * 0.75),
        ]
        for i, (lx, ly) in enumerate(light_positions):
            f.write(f'        def RectLight "CeilingLight_{i}"\n')
            f.write('        {\n')
            f.write('            float inputs:intensity = 30000\n')
            f.write('            float inputs:width = 0.6\n')
            f.write('            float inputs:height = 0.6\n')
            f.write('            color3f inputs:color = (1.0, 0.98, 0.95)\n')
            f.write(f'            double3 xformOp:translate = ({lx}, {ly}, {h - 0.05})\n')
            f.write('            float3 xformOp:rotateXYZ = (180, 0, 0)\n')
            f.write('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]\n')
            f.write('        }\n')
        
        # Add dome light for ambient
        f.write('\n        def DomeLight "AmbientDome"\n')
        f.write('        {\n')
        f.write('            float inputs:intensity = 500\n')
        f.write('            color3f inputs:color = (0.9, 0.95, 1.0)\n')
        f.write('        }\n')
        
        f.write('    }\n')
        
        # =====================
        # OBJECTS
        # =====================
        f.write('\n    def Xform "Objects"\n')
        f.write('    {\n')
        
        for i, obj in enumerate(room.get('objects', [])):
            source_id = obj.get('source_id')
            if not source_id:
                continue
            
            mesh_path = f'./meshes/{source_id}.usda'
            if not os.path.exists(os.path.join(mesh_dir, f'{source_id}.usda')):
                continue
            
            pos = obj.get('position', {})
            rot = obj.get('rotation', {})
            scale = obj.get('scale', 1.0)
            
            x = pos.get('x', 0)
            y = pos.get('y', 0)
            z = pos.get('z', 0)
            
            rx = rot.get('x', 0)
            ry = rot.get('y', 0)
            rz = rot.get('z', 0)
            
            obj_name = f'obj_{i}_{source_id[:8]}'
            
            f.write(f'        def Mesh "{obj_name}" (\n')
            f.write(f'            references = @{mesh_path}@</Mesh>\n')
            f.write('        )\n')
            f.write('        {\n')
            f.write(f'            double3 xformOp:translate = ({x}, {y}, {z})\n')
            f.write(f'            float3 xformOp:rotateXYZ = ({rx}, {ry}, {rz})\n')
            f.write(f'            float3 xformOp:scale = ({scale}, {scale}, {scale})\n')
            f.write('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]\n')
            f.write('        }\n')
        
        f.write('    }\n')  # End Objects
        
        # =====================
        # CAMERA
        # =====================
        f.write('\n    def Xform "Cameras"\n')
        f.write('    {\n')
        
        # Camera positioned at corner looking into room
        cam_x = -0.5
        cam_y = -0.5
        cam_z = 1.6  # eye height
        target_x = w / 2
        target_y = l / 2
        target_z = h / 3
        
        f.write('        def Camera "MainCamera"\n')
        f.write('        {\n')
        f.write(f'            double3 xformOp:translate = ({cam_x}, {cam_y}, {cam_z})\n')
        # Calculate rotation to look at target
        import math
        dx = target_x - cam_x
        dy = target_y - cam_y
        dz = target_z - cam_z
        yaw = math.degrees(math.atan2(dy, dx))
        pitch = math.degrees(math.atan2(-dz, math.sqrt(dx*dx + dy*dy)))
        f.write(f'            float3 xformOp:rotateXYZ = ({90 + pitch}, 0, {90 + yaw})\n')
        f.write('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]\n')
        f.write('            float focalLength = 24\n')
        f.write('            float horizontalAperture = 36\n')
        f.write('        }\n')
        
        f.write('    }\n')
        
        f.write('}\n')  # End World
    
    print(f"Built scene USD: {scene_path}")
    return scene_path


if __name__ == '__main__':
    layout_path = '/tmp/sage-repo/server/results/layout_image3_v2/layout_image3_v2.json'
    output_dir = '/tmp/sage-repo/server/results/layout_image3_v2/final_usd'
    
    build_scene_with_materials(layout_path, output_dir)
