#!/usr/bin/env python3
"""
Generic shelf placement script for 3D scenes.
Places products on shelving units with realistic grocery-store appearance.
"""

import json
import math
import random
import struct
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional


def detect_shelf_tiers_from_ply(ply_path: str, min_peak_height: float = 50) -> List[float]:
    """
    Detect shelf tier heights from PLY mesh using vertex Z-histogram.
    
    Args:
        ply_path: Path to shelf PLY file
        min_peak_height: Minimum histogram count to consider a peak
        
    Returns:
        List of Z heights where shelf surfaces exist
    """
    # Read PLY vertices
    with open(ply_path, 'rb') as f:
        # Parse header
        nv = 0
        props = []
        in_vertex = False
        while True:
            line = f.readline().decode('ascii', errors='replace').strip()
            if line.startswith('element vertex'):
                nv = int(line.split()[-1])
                in_vertex = True
            elif line.startswith('element ') and in_vertex:
                in_vertex = False
            elif in_vertex and line.startswith('property'):
                parts = line.split()
                props.append(parts[1])
            elif line == 'end_header':
                break
        
        # Calculate bytes per vertex
        type_sizes = {'float': 4, 'double': 8, 'uchar': 1, 'int': 4, 'uint8': 1}
        bytes_per_vertex = sum(type_sizes.get(p, 4) for p in props)
        
        # Read Z values
        z_values = []
        for _ in range(nv):
            data = f.read(bytes_per_vertex)
            if len(data) >= 12:
                x, y, z = struct.unpack('<fff', data[:12])
                z_values.append(z)
    
    if not z_values:
        return []
    
    # Histogram to find shelf levels
    import numpy as np
    z_arr = np.array(z_values)
    hist, bin_edges = np.histogram(z_arr, bins=200)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peaks (many vertices at same height = shelf surface)
    threshold = np.percentile(hist, 85)
    peak_indices = np.where(hist > threshold)[0]
    
    # Cluster adjacent peaks
    tiers = []
    if len(peak_indices) > 0:
        current_cluster = [bin_centers[peak_indices[0]]]
        for i in range(1, len(peak_indices)):
            if peak_indices[i] - peak_indices[i-1] <= 3:
                current_cluster.append(bin_centers[peak_indices[i]])
            else:
                tiers.append(np.mean(current_cluster))
                current_cluster = [bin_centers[peak_indices[i]]]
        if current_cluster:
            tiers.append(np.mean(current_cluster))
    
    return sorted(tiers)


def local_to_world(local_x: float, local_y: float, local_z: float, 
                   shelf_tf: Dict[str, float]) -> Tuple[float, float, float]:
    """Convert shelf-local coordinates to world space."""
    rot_rad = math.radians(shelf_tf['rot_z'])
    wx = local_x * math.cos(rot_rad) - local_y * math.sin(rot_rad)
    wy = local_x * math.sin(rot_rad) + local_y * math.cos(rot_rad)
    return wx + shelf_tf['x'], wy + shelf_tf['y'], local_z + shelf_tf['z']


def fill_shelf_realistic(
    products: List[Tuple[Dict, Dict]],
    shelf_tf: Dict[str, float],
    tiers: List[float],
    shelf_width: float,
    shelf_depth: float,
    max_products: int = 50,
    clearance: float = 0.02,
    front_margin: float = 0.015,
    side_margin: float = 0.015,
    pos_jitter: float = 0.005,
    rot_jitter: float = 5.0,
    depth_gap_range: Tuple[float, float] = (0.005, 0.015),
    col_gap_range: Tuple[float, float] = (0.01, 0.025),
    seed: Optional[int] = None
) -> List[Dict]:
    """
    Fill a shelf with products using realistic placement.
    
    Args:
        products: List of (product_dict, bbox_dict) tuples
                  bbox_dict must have 'w', 'h', 'd' keys for width, height, depth
        shelf_tf: Shelf transform with 'x', 'y', 'z', 'rot_z' keys
        tiers: List of Z heights for shelf surfaces
        shelf_width: Usable width of shelf
        shelf_depth: Usable depth of shelf
        max_products: Maximum products to place
        clearance: Headroom below next tier (meters)
        front_margin: Distance from front edge (meters)
        side_margin: Distance from side edges (meters)
        pos_jitter: Position randomness ± (meters)
        rot_jitter: Rotation randomness ± (degrees)
        depth_gap_range: (min, max) gap between depth-stacked items
        col_gap_range: (min, max) gap before next product type
        seed: Random seed for reproducibility
        
    Returns:
        List of placement dicts with 'template', 'pos', 'rot_z' keys
    """
    if seed is not None:
        random.seed(seed)
    
    placements = []
    usable_width = shelf_width - 2 * side_margin
    usable_depth = shelf_depth - front_margin - 0.01
    
    # Calculate clearances for each tier
    tier_clearances = []
    for i, z in enumerate(tiers):
        if i < len(tiers) - 1:
            tier_clearances.append(tiers[i+1] - z - clearance)
        else:
            tier_clearances.append(0.40)  # Top tier default
    
    placed = 0
    
    for tier_idx, tier_z in enumerate(tiers):
        if placed >= max_products:
            break
        
        max_height = tier_clearances[tier_idx]
        
        # Filter products that fit this tier
        fitting = [(p, bb) for p, bb in products if bb['h'] <= max_height]
        if not fitting:
            continue
        
        current_x = -usable_width / 2 + side_margin
        
        while current_x < usable_width / 2 - side_margin and placed < max_products:
            # Random product selection
            prod_template, bbox = random.choice(fitting)
            pw, pd, ph = bbox['w'], bbox['d'], bbox['h']
            
            if current_x + pw > usable_width / 2 - side_margin:
                break
            
            # Depth stacking (2-4 deep)
            max_depth_count = int(usable_depth / pd)
            num_deep = min(random.randint(2, 4), max_depth_count)
            
            # Columns of same product (1-3)
            max_cols = int((usable_width / 2 - current_x) / (pw + 0.01))
            num_cols = random.randint(1, min(3, max_cols))
            
            for col in range(num_cols):
                col_x = current_x + col * (pw + random.uniform(*depth_gap_range))
                
                if col_x + pw > usable_width / 2 - side_margin:
                    break
                
                for d in range(num_deep):
                    if placed >= max_products:
                        break
                    
                    depth_gap = random.uniform(*depth_gap_range) if d > 0 else 0
                    
                    # Position with jitter
                    local_x = col_x + pw / 2 + random.uniform(-pos_jitter, pos_jitter)
                    local_y = -usable_depth / 2 + front_margin + d * (pd + depth_gap) + pd / 2
                    local_z = tier_z + ph / 2
                    
                    if local_y + pd / 2 > usable_depth / 2:
                        break
                    
                    # Rotation with jitter
                    rot_z = shelf_tf['rot_z'] + random.uniform(-rot_jitter, rot_jitter)
                    
                    wx, wy, wz = local_to_world(local_x, local_y, local_z, shelf_tf)
                    
                    placements.append({
                        'template': prod_template,
                        'pos': {'x': wx, 'y': wy, 'z': wz},
                        'rot_z': rot_z
                    })
                    placed += 1
            
            # Move to next product type
            current_x += num_cols * (pw + 0.01) + random.uniform(*col_gap_range)
    
    return placements


def main():
    parser = argparse.ArgumentParser(description='Fill shelves with products')
    parser.add_argument('--layout', required=True, help='Input layout JSON')
    parser.add_argument('--mesh-bboxes', required=True, help='Product bounding boxes JSON')
    parser.add_argument('--shelf-mesh', help='Shelf PLY for tier detection')
    parser.add_argument('--tiers', help='Comma-separated tier Z heights (alternative to --shelf-mesh)')
    parser.add_argument('--output', required=True, help='Output layout JSON')
    parser.add_argument('--max-per-shelf', type=int, default=45, help='Max products per shelf')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Load data
    with open(args.layout) as f:
        layout = json.load(f)
    
    with open(args.mesh_bboxes) as f:
        mesh_bboxes = json.load(f)
    
    # Detect or parse tiers
    if args.tiers:
        tiers = [float(t) for t in args.tiers.split(',')]
    elif args.shelf_mesh:
        tiers = detect_shelf_tiers_from_ply(args.shelf_mesh)
        print(f"Detected {len(tiers)} shelf tiers: {[f'{t:.3f}' for t in tiers]}")
    else:
        raise ValueError("Must provide either --shelf-mesh or --tiers")
    
    # Collect products and shelf info
    room = layout['rooms'][0]
    products = []
    shelves = []
    other_objects = []
    
    for obj in room['objects']:
        if obj['type'] == 'shelving':
            shelves.append(obj)
        elif obj.get('source_id') in mesh_bboxes:
            products.append((obj, mesh_bboxes[obj['source_id']]))
        else:
            other_objects.append(obj)
    
    print(f"Found {len(shelves)} shelves, {len(products)} product types")
    
    # Fill each shelf
    import copy
    new_objects = []
    total_products = 0
    random.seed(args.seed)
    
    for shelf in shelves:
        new_objects.append(shelf)
        
        shelf_tf = {
            'x': shelf['position']['x'],
            'y': shelf['position']['y'],
            'z': shelf['position']['z'],
            'rot_z': shelf['rotation']['z']
        }
        
        # Get shelf dimensions from layout or use defaults
        shelf_width = shelf['dimensions'].get('width', 1.12)
        shelf_depth = shelf['dimensions'].get('length', 0.43)
        
        placements = fill_shelf_realistic(
            products=products,
            shelf_tf=shelf_tf,
            tiers=tiers,
            shelf_width=shelf_width,
            shelf_depth=shelf_depth,
            max_products=args.max_per_shelf
        )
        
        for j, p in enumerate(placements):
            prod = copy.deepcopy(p['template'])
            prod['id'] = f"prod_{shelf['id']}_{j}"
            prod['position'] = p['pos']
            prod['rotation'] = {'x': 0, 'y': 0, 'z': p['rot_z']}
            prod['place_id'] = shelf['id']
            new_objects.append(prod)
            total_products += 1
    
    # Keep other objects
    new_objects.extend(other_objects)
    
    room['objects'] = new_objects
    
    with open(args.output, 'w') as f:
        json.dump(layout, f, indent=2)
    
    print(f"Placed {total_products} products across {len(shelves)} shelves")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
