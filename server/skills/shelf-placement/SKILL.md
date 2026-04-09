---
name: shelf-placement
description: Place products on shelving units in 3D scenes with realistic grocery-store appearance. Use when populating shelves, racks, gondolas, or display units with products. Handles tier detection from mesh geometry, depth stacking, height clearance, position/rotation jitter, and product variety. Works with any shelf mesh that has horizontal surfaces.
---

# Shelf Placement

Place products realistically on shelving units in 3D scenes.

## Quick Start

```python
from shelf_placement import fill_shelf_realistic, detect_shelf_tiers

# 1. Detect shelf tiers from mesh
tiers = detect_shelf_tiers(shelf_ply_path)  # Returns list of Z heights

# 2. Place products
placements = fill_shelf_realistic(
    products=[(obj, bbox), ...],  # Product templates with bounding boxes
    shelf_transform={'x': 0.5, 'y': 7.5, 'z': 0, 'rot_z': 270},
    tiers=tiers,
    shelf_width=1.12,
    shelf_depth=0.43
)
```

## Core Principles

### 1. Tier Detection
Analyze shelf mesh to find horizontal surfaces:
- **Vertex clustering**: Histogram Z-values, find peaks where many vertices cluster
- **Normal-based** (more accurate): Find faces with upward normals (Z > 0.9), cluster by height
- Skip top cap (usually not usable shelf surface)

### 2. Height Clearance
Products must fit between tiers with clearance:
```python
clearance = next_tier_z - current_tier_z - 0.02  # 2cm headroom
valid_products = [p for p in products if p.height <= clearance]
```

### 3. Depth Stacking
Real shelves have 2-4 identical items front-to-back:
```python
num_deep = min(random.randint(2, 4), int(shelf_depth / product_depth))
for d in range(num_deep):
    place_product(x, front_margin + d * (depth + gap), z)
```

### 4. Position & Rotation Jitter
Avoid perfect grid alignment:
- Position jitter: ±5mm in X and Y
- Rotation jitter: ±5° around Z axis
- Gap variation: 5-20mm between items

### 5. Product Variety
- **Random selection** from valid products (not strict cycling)
- **Column grouping**: 1-3 columns of same product before switching
- **Category zones**: Group similar products (cereals together, cans together)

## Algorithm

```
for each tier (bottom to top):
    filter products by height clearance
    current_x = left_margin
    
    while current_x < shelf_width - right_margin:
        product = random_choice(valid_products)
        num_columns = random(1, 3)
        num_deep = random(2, 4)
        
        for col in range(num_columns):
            for depth in range(num_deep):
                place with jitter:
                    x = current_x + col * (width + gap) + random(±5mm)
                    y = front + depth * (depth_size + random(5-15mm))
                    z = tier_z + height/2
                    rot_z = shelf_rot + random(±5°)
        
        current_x += num_columns * width + random(10-25mm)
```

## World Transform

Convert shelf-local coordinates to world space:
```python
def local_to_world(local_x, local_y, local_z, shelf):
    rot = radians(shelf['rot_z'])
    wx = local_x * cos(rot) - local_y * sin(rot) + shelf['x']
    wy = local_x * sin(rot) + local_y * cos(rot) + shelf['y']
    wz = local_z + shelf['z']
    return wx, wy, wz
```

## Common Shelf Rotations
- **270°**: Shelf faces +X (left wall of aisle)
- **90°**: Shelf faces -X (right wall of aisle)
- **0°**: Shelf faces +Y
- **180°**: Shelf faces -Y

## Script Usage

See `scripts/shelf_placer.py` for complete implementation:

```bash
python scripts/shelf_placer.py \
    --layout layout.json \
    --mesh-bboxes bboxes.json \
    --shelf-mesh shelf.ply \
    --output layout_filled.json
```

## Tunable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clearance` | 0.02m | Headroom below next tier |
| `front_margin` | 0.015m | Distance from shelf front edge |
| `side_margin` | 0.015m | Distance from shelf sides |
| `pos_jitter` | 0.005m | Position randomness (±) |
| `rot_jitter` | 5° | Rotation randomness (±) |
| `depth_gap_min` | 0.005m | Min gap between depth-stacked items |
| `depth_gap_max` | 0.015m | Max gap between depth-stacked items |
| `col_gap_min` | 0.01m | Min gap before next product type |
| `col_gap_max` | 0.025m | Max gap before next product type |
