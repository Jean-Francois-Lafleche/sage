#!/usr/bin/env python3
"""
PLY to USD converter for TRELLIS-generated meshes.
Converts PLY mesh files with textures to USD format compatible with Isaac Sim.
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from plyfile import PlyData


def ply_to_usd(ply_path: str, usd_path: str, texture_path: str = None, texture_rel_path: str = None) -> bool:
    """
    Convert a PLY mesh file to USD format.
    
    Args:
        ply_path: Path to input PLY file
        usd_path: Path to output USD file
        texture_path: Optional path to texture file (will be copied to output dir)
        texture_rel_path: Relative path to texture from USD file location
        
    Returns:
        True if conversion successful, False otherwise
    """
    if not os.path.exists(ply_path):
        print(f"PLY file not found: {ply_path}")
        return False
    
    try:
        plydata = PlyData.read(ply_path)
    except Exception as e:
        print(f"Failed to read PLY: {e}")
        return False
    
    # Extract vertices
    vertex_data = plydata['vertex']
    num_vertices = len(vertex_data)
    
    vertices = np.zeros((num_vertices, 3), dtype=np.float32)
    vertices[:, 0] = vertex_data['x']
    vertices[:, 1] = vertex_data['y']
    vertices[:, 2] = vertex_data['z']
    
    # Extract faces
    try:
        face_data = plydata['face']
        faces = np.vstack(face_data['vertex_indices'])
    except Exception as e:
        print(f"Failed to extract faces: {e}")
        return False
    
    num_faces = len(faces)
    
    # Extract UVs if available
    uvs = None
    has_uvs = 's' in vertex_data.data.dtype.names and 't' in vertex_data.data.dtype.names
    if has_uvs:
        uvs = np.zeros((num_vertices, 2), dtype=np.float32)
        uvs[:, 0] = vertex_data['s']
        uvs[:, 1] = vertex_data['t']
    
    # Extract vertex colors if available
    colors = None
    has_colors = all(c in vertex_data.data.dtype.names for c in ['red', 'green', 'blue'])
    if has_colors:
        colors = np.zeros((num_vertices, 3), dtype=np.float32)
        colors[:, 0] = vertex_data['red'] / 255.0
        colors[:, 1] = vertex_data['green'] / 255.0
        colors[:, 2] = vertex_data['blue'] / 255.0
    
    # Check for texture
    has_texture = texture_path and os.path.exists(texture_path) and texture_rel_path
    
    # Write USD file
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)
    
    with open(usd_path, 'w') as f:
        # Header
        f.write('#usda 1.0\n')
        f.write('(\n')
        f.write('    defaultPrim = "Mesh"\n')
        f.write('    metersPerUnit = 1\n')
        f.write('    upAxis = "Z"\n')
        f.write(')\n\n')
        
        # Mesh definition
        f.write('def Mesh "Mesh"\n')
        f.write('{\n')
        
        # Face vertex counts (all triangles = 3)
        f.write(f'    int[] faceVertexCounts = [{", ".join(["3"] * num_faces)}]\n')
        
        # Face vertex indices (flattened)
        indices = faces.flatten().tolist()
        f.write(f'    int[] faceVertexIndices = [{", ".join(str(i) for i in indices)}]\n')
        
        # Points
        points_str = ', '.join(f'({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})' for v in vertices)
        f.write(f'    point3f[] points = [{points_str}]\n')
        
        # Normals (compute face normals, then vertex normals by averaging)
        # For now, skip normals - USD can compute them
        
        # UVs
        if uvs is not None:
            uvs_str = ', '.join(f'({uv[0]:.6f}, {uv[1]:.6f})' for uv in uvs)
            f.write(f'    texCoord2f[] primvars:st = [{uvs_str}] (\n')
            f.write('        interpolation = "vertex"\n')
            f.write('    )\n')
        
        # Vertex colors (as displayColor)
        if colors is not None and not has_texture:
            colors_str = ', '.join(f'({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})' for c in colors)
            f.write(f'    color3f[] primvars:displayColor = [{colors_str}] (\n')
            f.write('        interpolation = "vertex"\n')
            f.write('    )\n')
        
        # Material binding
        if has_texture:
            f.write('    rel material:binding = </Mesh/Material>\n')
        
        f.write('}\n')
        
        # Material definition with texture
        if has_texture:
            f.write('\n')
            f.write('def Material "Material"\n')
            f.write('{\n')
            f.write('    token outputs:surface.connect = </Material/PBRShader.outputs:surface>\n')
            f.write('\n')
            f.write('    def Shader "PBRShader"\n')
            f.write('    {\n')
            f.write('        uniform token info:id = "UsdPreviewSurface"\n')
            f.write('        color3f inputs:diffuseColor.connect = </Material/DiffuseTexture.outputs:rgb>\n')
            f.write('        float inputs:metallic = 0.0\n')
            f.write('        float inputs:roughness = 0.5\n')
            f.write('        token outputs:surface\n')
            f.write('    }\n')
            f.write('\n')
            f.write('    def Shader "DiffuseTexture"\n')
            f.write('    {\n')
            f.write('        uniform token info:id = "UsdUVTexture"\n')
            f.write(f'        asset inputs:file = @{texture_rel_path}@\n')
            f.write('        float2 inputs:st.connect = </Material/PrimvarReader.outputs:result>\n')
            f.write('        token inputs:wrapS = "repeat"\n')
            f.write('        token inputs:wrapT = "repeat"\n')
            f.write('        float3 outputs:rgb\n')
            f.write('    }\n')
            f.write('\n')
            f.write('    def Shader "PrimvarReader"\n')
            f.write('    {\n')
            f.write('        uniform token info:id = "UsdPrimvarReader_float2"\n')
            f.write('        token inputs:varname = "st"\n')
            f.write('        float2 outputs:result\n')
            f.write('    }\n')
            f.write('}\n')
    
    print(f"Converted {ply_path} -> {usd_path}")
    print(f"  Vertices: {num_vertices}, Faces: {num_faces}, UVs: {has_uvs}, Texture: {has_texture}")
    
    return True


def convert_layout_meshes(layout_path: str, generation_dir: str, output_dir: str) -> dict:
    """
    Convert all meshes referenced in a layout JSON to USD format.
    
    Args:
        layout_path: Path to layout JSON file
        generation_dir: Directory containing PLY and texture files
        output_dir: Output directory for USD files
        
    Returns:
        Dict mapping source_id to USD path
    """
    with open(layout_path) as f:
        layout = json.load(f)
    
    mesh_dir = os.path.join(output_dir, 'meshes')
    tex_dir = os.path.join(output_dir, 'textures')
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)
    
    # Collect unique source_ids
    source_ids = set()
    for room in layout.get('rooms', []):
        for obj in room.get('objects', []):
            sid = obj.get('source_id')
            if sid:
                source_ids.add(sid)
    
    print(f"Converting {len(source_ids)} unique meshes...")
    
    results = {}
    for source_id in source_ids:
        ply_path = os.path.join(generation_dir, f'{source_id}.ply')
        tex_path = os.path.join(generation_dir, f'{source_id}_texture.png')
        usd_path = os.path.join(mesh_dir, f'{source_id}.usda')
        
        # Copy texture if exists
        tex_rel_path = None
        if os.path.exists(tex_path):
            tex_dest = os.path.join(tex_dir, f'{source_id}_texture.png')
            shutil.copy(tex_path, tex_dest)
            tex_rel_path = f'../textures/{source_id}_texture.png'
        
        if ply_to_usd(ply_path, usd_path, tex_path, tex_rel_path):
            results[source_id] = usd_path
    
    print(f"Successfully converted {len(results)} meshes")
    return results


def build_scene_usd(layout_path: str, output_path: str, mesh_dir: str) -> bool:
    """
    Build a complete scene USD file from a layout JSON.
    
    Args:
        layout_path: Path to layout JSON
        output_path: Path for output scene USD
        mesh_dir: Directory containing converted mesh USDs
        
    Returns:
        True if successful
    """
    with open(layout_path) as f:
        layout = json.load(f)
    
    room = layout['rooms'][0]
    room_id = room['id']
    dims = room['dimensions']
    w, l, h = dims['width'], dims['length'], dims['height']
    
    with open(output_path, 'w') as f:
        # Header
        f.write('#usda 1.0\n')
        f.write('(\n')
        f.write('    defaultPrim = "World"\n')
        f.write('    metersPerUnit = 1\n')
        f.write('    upAxis = "Z"\n')
        f.write(')\n\n')
        
        f.write('def Xform "World"\n')
        f.write('{\n')
        
        # Room geometry
        f.write('    def Xform "Room"\n')
        f.write('    {\n')
        
        # Floor
        f.write('        def Cube "Floor"\n')
        f.write('        {\n')
        f.write('            double size = 1\n')
        f.write(f'            float3 xformOp:scale = ({w}, {l}, 0.1)\n')
        f.write(f'            double3 xformOp:translate = ({w/2}, {l/2}, -0.05)\n')
        f.write('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n')
        f.write('            color3f[] primvars:displayColor = [(0.6, 0.5, 0.4)]\n')
        f.write('        }\n')
        
        # Ceiling
        f.write('        def Cube "Ceiling"\n')
        f.write('        {\n')
        f.write('            double size = 1\n')
        f.write(f'            float3 xformOp:scale = ({w}, {l}, 0.1)\n')
        f.write(f'            double3 xformOp:translate = ({w/2}, {l/2}, {h + 0.05})\n')
        f.write('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n')
        f.write('            color3f[] primvars:displayColor = [(0.95, 0.95, 0.95)]\n')
        f.write('        }\n')
        
        # Walls
        walls = room.get('walls', [])
        for i, wall in enumerate(walls):
            sp = wall['start_point']
            ep = wall['end_point']
            wh = wall['height']
            
            # Calculate wall center and dimensions
            cx = (sp['x'] + ep['x']) / 2
            cy = (sp['y'] + ep['y']) / 2
            cz = wh / 2
            
            wall_len = np.sqrt((ep['x'] - sp['x'])**2 + (ep['y'] - sp['y'])**2)
            wall_angle = np.degrees(np.arctan2(ep['y'] - sp['y'], ep['x'] - sp['x']))
            
            f.write(f'        def Cube "Wall_{i}"\n')
            f.write('        {\n')
            f.write('            double size = 1\n')
            f.write(f'            float3 xformOp:scale = ({wall_len}, 0.1, {wh})\n')
            f.write(f'            double3 xformOp:translate = ({cx}, {cy}, {cz})\n')
            f.write(f'            float xformOp:rotateZ = {wall_angle}\n')
            f.write('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]\n')
            f.write('            color3f[] primvars:displayColor = [(0.9, 0.9, 0.88)]\n')
            f.write('        }\n')
        
        f.write('    }\n')  # End Room
        
        # Lights
        f.write('\n    def Xform "Lights"\n')
        f.write('    {\n')
        light_positions = [
            (w * 0.25, l * 0.25),
            (w * 0.75, l * 0.25),
            (w * 0.25, l * 0.75),
            (w * 0.75, l * 0.75),
        ]
        for i, (lx, ly) in enumerate(light_positions):
            f.write(f'        def RectLight "Light_{i}"\n')
            f.write('        {\n')
            f.write('            float inputs:intensity = 30000\n')
            f.write('            float inputs:width = 0.5\n')
            f.write('            float inputs:height = 0.5\n')
            f.write('            color3f inputs:color = (1.0, 0.98, 0.95)\n')
            f.write(f'            double3 xformOp:translate = ({lx}, {ly}, {h - 0.1})\n')
            f.write('            float3 xformOp:rotateXYZ = (180, 0, 0)\n')
            f.write('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]\n')
            f.write('        }\n')
        f.write('    }\n')
        
        # Objects
        f.write('\n    def Xform "Objects"\n')
        f.write('    {\n')
        
        for i, obj in enumerate(room.get('objects', [])):
            source_id = obj.get('source_id')
            if not source_id:
                continue
            
            mesh_path = f'./meshes/{source_id}.usda'
            if not os.path.exists(os.path.join(os.path.dirname(output_path), 'meshes', f'{source_id}.usda')):
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
            
            # Reference the mesh USD with proper prim path
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
        f.write('}\n')  # End World
    
    print(f"Built scene USD: {output_path}")
    return True


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ply_to_usd.py <layout_json> [output_dir]")
        sys.exit(1)
    
    layout_path = sys.argv[1]
    
    # Derive paths
    result_dir = os.path.dirname(layout_path)
    gen_dir = os.path.join(result_dir, 'generation')
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(result_dir, 'final_usd')
    
    # Convert meshes
    convert_layout_meshes(layout_path, gen_dir, output_dir)
    
    # Build scene
    scene_path = os.path.join(output_dir, 'scene.usda')
    build_scene_usd(layout_path, scene_path, os.path.join(output_dir, 'meshes'))
