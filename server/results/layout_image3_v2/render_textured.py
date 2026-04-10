#!/usr/bin/env python3
"""
Render playroom scene with textures using nvdiffrast.
Properly samples textures instead of just using vertex colors.
"""

import sys
import os
import json
import math
import numpy as np
from PIL import Image

sys.path.insert(0, '/tmp/sage-repo/server')

RESULT_DIR = '/tmp/sage-repo/server/results/layout_image3_v2'
LAYOUT_FILE = f'{RESULT_DIR}/layout_image3_v2.json'
GEN_DIR = f'{RESULT_DIR}/generation'
OUTPUT_DIR = f'{RESULT_DIR}/renders_textured'


def load_mesh_with_texture(source_id, gen_dir):
    """Load mesh with UV coords and texture."""
    from plyfile import PlyData
    
    ply_path = f'{gen_dir}/{source_id}.ply'
    tex_path = f'{gen_dir}/{source_id}_texture.png'
    
    if not os.path.exists(ply_path):
        return None
    
    plydata = PlyData.read(ply_path)
    
    vertices = np.vstack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ]).T.astype(np.float32)
    
    try:
        faces = np.vstack(plydata['face']['vertex_indices']).astype(np.int32)
    except:
        return None
    
    if len(faces) == 0:
        return None
    
    # Get UVs
    uvs = None
    try:
        uvs = np.vstack([
            plydata['vertex']['s'],
            plydata['vertex']['t']
        ]).T.astype(np.float32)
    except:
        pass
    
    # Load texture
    texture = None
    if os.path.exists(tex_path):
        try:
            img = Image.open(tex_path).convert('RGB')
            texture = np.array(img).astype(np.float32) / 255.0
        except:
            pass
    
    # Get vertex colors as fallback
    colors = None
    try:
        colors = np.vstack([
            plydata['vertex']['red'],
            plydata['vertex']['green'],
            plydata['vertex']['blue']
        ]).T.astype(np.float32) / 255.0
    except:
        colors = np.ones((len(vertices), 3), dtype=np.float32) * 0.7
    
    return {
        'vertices': vertices,
        'faces': faces,
        'uvs': uvs,
        'colors': colors,
        'texture': texture
    }


def create_floor_mesh(dims, color=[0.55, 0.4, 0.28]):
    """Create floor quad with hardwood color."""
    w, l = dims['width'], dims['length']
    verts = np.array([
        [0, 0, 0],
        [w, 0, 0],
        [w, l, 0],
        [0, l, 0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    colors = np.array([color] * 4, dtype=np.float32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    return {'vertices': verts, 'faces': faces, 'colors': colors, 'uvs': uvs, 'texture': None}


def create_wall_mesh(start, end, height, color=[0.7, 0.5, 0.4]):
    """Create wall quad with brick color."""
    sx, sy = start['x'], start['y']
    ex, ey = end['x'], end['y']
    h = height
    
    verts = np.array([
        [sx, sy, 0],
        [ex, ey, 0],
        [ex, ey, h],
        [sx, sy, h],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    colors = np.array([color] * 4, dtype=np.float32)
    return {'vertices': verts, 'faces': faces, 'colors': colors, 'uvs': None, 'texture': None}


def sample_texture(uvs, texture):
    """Sample texture at UV coordinates."""
    if texture is None or uvs is None:
        return None
    
    h, w = texture.shape[:2]
    
    # Clamp UVs to [0, 1]
    u = np.clip(uvs[:, 0], 0, 1)
    v = np.clip(uvs[:, 1], 0, 1)
    
    # Convert to pixel coords (flip V for image coords)
    px = (u * (w - 1)).astype(np.int32)
    py = ((1 - v) * (h - 1)).astype(np.int32)
    
    # Sample
    colors = texture[py, px]
    return colors


def transform_vertices(verts, pos, rot, scale):
    """Apply transform to vertices."""
    # Scale
    verts = verts * scale
    
    # Rotate around Z
    rz = np.radians(rot.get('z', 0))
    cos_z, sin_z = np.cos(rz), np.sin(rz)
    rot_mat = np.array([
        [cos_z, -sin_z, 0],
        [sin_z, cos_z, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    verts = verts @ rot_mat.T
    
    # Translate
    verts[:, 0] += pos.get('x', 0)
    verts[:, 1] += pos.get('y', 0)
    verts[:, 2] += pos.get('z', 0)
    
    return verts


def build_mvp(eye, target, fov=60, aspect=1.0, near=0.1, far=100.0):
    """Build Model-View-Projection matrix."""
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 0.001:
        right = np.array([1, 0, 0], dtype=np.float32)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    view = np.eye(4, dtype=np.float32)
    view[:3, 0] = right
    view[:3, 1] = up
    view[:3, 2] = -forward
    view[:3, 3] = eye
    view = np.linalg.inv(view)
    
    f = 1.0 / np.tan(np.radians(fov) / 2)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = 2 * far * near / (near - far)
    proj[3, 2] = -1
    
    return proj @ view


def render_scene():
    """Render the playroom scene with textures."""
    import torch
    import nvdiffrast.torch as dr
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading layout...")
    with open(LAYOUT_FILE) as f:
        layout = json.load(f)
    
    room = layout['rooms'][0]
    objects = room['objects']
    dims = room['dimensions']
    walls = room.get('walls', [])
    
    env = layout.get('environment', {})
    floor_color = [0.55, 0.4, 0.28]  # hardwood
    wall_color = [0.7, 0.5, 0.4]     # brick
    
    print(f"Room: {dims['width']}m x {dims['length']}m x {dims['height']}m")
    print(f"Objects: {len(objects)}")
    
    # Collect all meshes
    all_meshes = []
    
    # Add floor
    floor = create_floor_mesh(dims, floor_color)
    all_meshes.append(floor)
    
    # Add walls
    for wall in walls:
        wall_mesh = create_wall_mesh(
            wall['start_point'], wall['end_point'], wall['height'], wall_color
        )
        all_meshes.append(wall_mesh)
    
    print(f"Added floor + {len(walls)} walls")
    
    # Load objects
    obj_count = 0
    for obj in objects:
        source_id = obj.get('source_id')
        if not source_id:
            continue
        
        mesh = load_mesh_with_texture(source_id, GEN_DIR)
        if mesh is None:
            continue
        
        pos = obj.get('position', {})
        rot = obj.get('rotation', {})
        scale = obj.get('scale', 1.0)
        
        mesh['vertices'] = transform_vertices(mesh['vertices'], pos, rot, scale)
        
        # Sample texture to get vertex colors
        if mesh['texture'] is not None and mesh['uvs'] is not None:
            tex_colors = sample_texture(mesh['uvs'], mesh['texture'])
            if tex_colors is not None:
                mesh['colors'] = tex_colors
        
        all_meshes.append(mesh)
        obj_count += 1
    
    print(f"Loaded {obj_count} objects with textures")
    
    # Combine meshes
    all_vertices = []
    all_faces = []
    all_colors = []
    vertex_offset = 0
    
    for mesh in all_meshes:
        all_vertices.append(mesh['vertices'])
        all_faces.append(mesh['faces'] + vertex_offset)
        all_colors.append(mesh['colors'])
        vertex_offset += len(mesh['vertices'])
    
    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    
    print(f"Total: {len(vertices)} vertices, {len(faces)} faces")
    
    # Setup rendering
    device = 'cuda'
    glctx = dr.RasterizeCudaContext()
    
    vertices_gpu = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_gpu = torch.tensor(faces, dtype=torch.int32, device=device)
    colors_gpu = torch.tensor(colors, dtype=torch.float32, device=device)
    
    # Camera setup
    cx, cy = dims['width'] / 2, dims['length'] / 2
    height = dims['height']
    
    # Render from multiple angles
    res = 1920
    angles = [
        ("corner_sw", (-0.5, -0.5, 1.8), (cx, cy, height/3)),
        ("corner_se", (dims['width']+0.5, -0.5, 1.8), (cx, cy, height/3)),
        ("front", (cx, -1.5, 1.5), (cx, cy, height/3)),
        ("top", (cx, cy, 6), (cx, cy, 0)),
    ]
    
    for name, eye, target in angles:
        print(f"Rendering {name}...")
        
        mvp = build_mvp(eye, target, fov=60, aspect=res/1080)
        mvp_gpu = torch.tensor(mvp, dtype=torch.float32, device=device)
        
        # Transform vertices
        v_homo = torch.cat([vertices_gpu, torch.ones(len(vertices), 1, device=device)], dim=1)
        v_clip = v_homo @ mvp_gpu.T
        
        # Rasterize
        rast, _ = dr.rasterize(glctx, v_clip[None, ...], faces_gpu, resolution=[1080, res])
        
        # Interpolate colors
        color_out, _ = dr.interpolate(colors_gpu[None, ...], rast, faces_gpu)
        
        # Simple lighting (brighten)
        color_out = color_out[0].cpu().numpy()
        color_out = np.clip(color_out * 1.3, 0, 1)
        
        # Add background
        mask = rast[0, :, :, 3:4].cpu().numpy() > 0
        bg = np.array([0.9, 0.92, 0.95])
        color_out = color_out * mask + bg * (1 - mask)
        
        # Save
        img = Image.fromarray((color_out * 255).astype(np.uint8))
        out_path = f'{OUTPUT_DIR}/playroom_{name}.png'
        img.save(out_path)
        print(f"  Saved: {out_path}")
    
    print("\nRendering complete!")
    return f'{OUTPUT_DIR}/playroom_corner_sw.png'


if __name__ == '__main__':
    render_scene()
