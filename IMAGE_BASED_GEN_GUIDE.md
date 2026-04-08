# Image-Based Scene Generation: Research Survey & Implementation Guide

## Key Skills & Learnings

*Created: 2026-04-08 | Branch: `image-based-gen` | Author: SDGClaw (AI Agent)*

---

## 1. Research Survey: Image-Conditioned Scene Generation

### 1.1 Problem Statement

Given one or more reference images of a room/environment, generate a complete 3D simulation-ready scene that:
- Faithfully reproduces visible portions of the scene
- Infers unseen portions (behind camera, occluded areas)
- Preserves spatial relationships between objects
- Maintains the overall "vibe" (style, clutter level, era, function)
- Produces USD assets ready for physics simulation

### 1.2 Existing Approaches

#### A. Holistic 3D Reconstruction (Geometry-First)

**Methods: NeRF/3DGS → Mesh Extraction**
- Neural Radiance Fields, 3D Gaussian Splatting (3DGS)
- Pros: Photorealistic appearance from images
- Cons: Not sim-ready, no object decomposition, single-view input insufficient
- Examples: InstantNGP, Nerfstudio, GaussianEditor

**Methods: Monocular Depth + Layout Estimation**
- Single-image room layout estimation (RoomNet, LayoutNet, PanoContext)
- Monocular depth prediction (MiDAS, Depth Anything, Marigold)
- Pros: Works from single image, gives 3D structure
- Cons: Scale ambiguity, no individual object meshes

#### B. Object-Centric Scene Understanding → Retrieval/Generation

**Methods: Detect → Retrieve/Generate → Place**
- Object detection + instance segmentation (SAM, GroundingDINO, YOLO-World)
- 3D object generation from image crops (TRELLIS, Shap-E, TripoSR, InstantMesh)
- Spatial layout inference from image (monocular 3D layout estimation)
- Pros: Produces individual manipulable objects, natural for simulation
- Cons: Requires robust detection pipeline, challenging reconstruction from partial views

**Key Papers & Systems:**
1. **Holodeck (Allen AI, 2023)** — Text-to-scene using LLMs + Objaverse retrieval. SAGE builds on this approach.
2. **SceneComplete / HoloScene** — Layout completion and augmentation for embodied AI.
3. **SceneTex (2023)** — Generating consistent textures for 3D indoor scenes.
4. **RoomDreamer (2023)** — Text-driven room generation with diffusion-based inpainting.
5. **PhotoScene (2022)** — Image-conditioned material estimation for indoor scenes.
6. **ProcTHOR (2022)** — Procedural generation of training environments, configurable via parameters.
7. **Total3DUnderstanding (2020)** — Holistic scene understanding from a single image (room layout + object meshes + camera pose).
8. **SceneFormer (2021)** — Transformer-based indoor scene generation conditioned on partial observations.
9. **ATISS (2021)** — Autoregressive transformer for generating furniture layouts.
10. **DiffuScene (2024)** — Diffusion-based indoor scene synthesis with layout priors.
11. **CommonScenes (2024)** — Generating common-sense 3D scenes from text.
12. **SAGE (2026)** — This project. Agentic framework using LLMs/VLMs to generate sim-ready scenes from text descriptions.

#### C. VLM-Agentic Approaches (SAGE Category)

**SAGE's Current Approach:**
- Client VLM (Qwen3-VL-32B-Thinking) orchestrates scene generation
- MCP server provides tools: room layout generation, object selection, object placement
- LLM-driven room structure → LLM-driven object selection → constraint-based placement
- 3D assets generated via TRELLIS or retrieved from Objaverse
- Already supports `--input_images` but only for VLM context (no structured image analysis pipeline)

**Gap Analysis — What's Missing for Image-Conditioned Generation:**
1. No structured image analysis pipeline (object detection, layout estimation, style analysis)
2. No depth/geometry estimation from input images
3. No object-level correspondence between image objects and generated 3D assets
4. No spatial relationship extraction from images
5. No "vibe"/style quantification system
6. No shelf/rack stacking capability for warehouse-type scenes

### 1.3 Our Approach: Image-Conditioned SAGE

**Architecture:**

```
Input Image(s)
    │
    ├─→ [VLM Scene Analysis] ──→ Structured Scene Description
    │     • Room type, dimensions, style
    │     • Object inventory with relative positions
    │     • Spatial relationships graph
    │     • Vibe/style descriptors
    │     • Shelf/rack configurations
    │
    ├─→ [Monocular Depth] ──→ Room geometry estimation
    │     • Depth Anything V2
    │     • Room dimensions inference
    │
    └─→ [Object Crops] ──→ Individual object references
          • SAM2 / GroundingDINO segmentation
          • CLIP embeddings for retrieval
          • Per-object style/material descriptors
              │
              ▼
    [SAGE Pipeline] (existing)
    ├─→ Room layout generation (informed by image analysis)
    ├─→ Object selection (guided by detected objects)
    ├─→ Object placement (constrained by spatial relationships)
    └─→ USD export + SimReady post-processing
```

### 1.4 Missing Capabilities Identified

1. **Shelf/Rack Stacking** — Current SAGE only places objects on flat surfaces. Warehouses need objects on multiple shelves at different heights.
2. **Style Transfer** — No mechanism to enforce style consistency from reference image.
3. **Spatial Relationship Preservation** — No structured way to encode "A is left of B, C is on top of D" from images.
4. **Clutter/Density Matching** — No way to match the visual density/clutter level of a reference image.

---

## 2. Implementation Architecture

### 2.1 New Module: `server/image_analysis/`

```
image_analysis/
├── __init__.py
├── scene_analyzer.py      # VLM-based structured scene analysis
├── depth_estimator.py     # Monocular depth estimation
├── object_detector.py     # Object detection + segmentation
├── style_extractor.py     # Style/vibe extraction
├── spatial_graph.py       # Spatial relationship graph builder
└── shelf_placer.py        # Multi-level shelf/rack object placement
```

### 2.2 New Client Script: `client/client_generation_from_image.py`

Modified task definition that:
1. Sends image to VLM for structured analysis
2. Generates enhanced room description from image analysis
3. Feeds enhanced description into existing SAGE pipeline
4. Adds image-specific placement constraints

### 2.3 USD SimReady Post-Processing: `server/simready/`

```
simready/
├── __init__.py
├── usd_processor.py       # Main USD post-processing pipeline
├── physics_prep.py        # Add collision meshes, mass properties
├── material_prep.py       # PBR material standardization
└── hierarchy_prep.py      # USD prim hierarchy standardization
```

---

## 3. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scene analysis | VLM (existing Qwen3-VL) | Already in pipeline, multimodal, understands spatial layout |
| Depth estimation | Depth Anything V2 | SOTA monocular depth, runs on single GPU |
| Object detection | VLM-based (no separate model) | Reduces complexity, VLM can enumerate objects |
| 3D asset generation | TRELLIS (existing) | Already integrated in SAGE server |
| Shelf placement | Custom constraint solver | No existing tool handles multi-level placement |
| USD format | Native USD via pxr library | Direct USD output, eliminate intermediate formats |
| SimReady compliance | Post-processing pass | Applies NVIDIA SimReady spec to all outputs |

---

## 4. SimReady USD Requirements (from NVIDIA docs)

### 4.1 Asset Structure
- Root prim with proper transform
- Geometry under Mesh prims
- Materials using UsdPreviewSurface or MDL
- Collision meshes (convex decomposition or trimesh)

### 4.2 Physics Properties
- Rigid body dynamics (mass, center of mass, inertia)
- Collision shapes (convex hull or mesh collider)
- Physics materials (friction, restitution)

### 4.3 Metadata
- Semantic labels (class, category)
- Asset info (name, version, source)
- Units specification (meters, kg, etc.)

---

## 5. Learnings & Recovery Notes

### 5.1 Environment Setup
- Branch: `upgrade/python-3.12-isaacsim-6.0` is the base
- Server uses `uv` with Python 3.12, client uses Python 3.13
- CUDA 12.6 toolkit installed for building pytorch3d, nvdiffrast, pointnet2_ops
- `~/sage` symlink needed for uv.lock absolute paths
- M2T2 and pointnet2_ops needed `pyproject.toml` with build-system declarations

### 5.2 SAGE Architecture
- MCP-based client-server architecture
- Client: VLM agent (Qwen3-VL-32B-Thinking) orchestrates via tool calls
- Server: FastMCP server exposes tools for room generation, object placement
- Key file: `server/layout_wo_robot.py` (single room), `server/layout_wo_robot_multiroom.py` (multi-room)
- Object pipeline: selection (`objects/object_selection_planner.py`) → placement (`objects/object_placement_planner.py`)
- Scene export: USD via Isaac Sim MCP extension
- Rendering: nvdiffrast-based room rendering (`server/room_render.py`)

### 5.3 Key Files Modified
- `M2T2/pyproject.toml` — Added build-system for uv compatibility
- `M2T2/pointnet2_ops/pyproject.toml` — Added build-system for uv compatibility
- `server/uv.lock` — Regenerated for correct paths

### 5.4 Recovery Instructions
If starting fresh:
1. Clone sage repo, checkout `image-based-gen` branch
2. `git submodule update --init --recursive`
3. Clone curobo: `git clone https://github.com/NVlabs/curobo.git ../curobo`
4. Symlink: `ln -sf $(pwd) ~/sage`
5. Install CUDA toolkit: `sudo apt install cuda-nvcc-12-6 cuda-cudart-dev-12-6 libcusparse-dev-12-6 libcublas-dev-12-6 libcusolver-dev-12-6 libcufft-dev-12-6 libcurand-dev-12-6`
6. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
7. Server: `cd server && uv sync`
8. Client: `cd client && uv sync`
