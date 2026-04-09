#!/usr/bin/env python3
"""
Run image3.png (playroom) pipeline with environment extraction and scene comparison.
Re-uses existing TRELLIS assets from layout_ec8b8760.
"""

import os
import sys
import json
import base64
from pathlib import Path

# Add server to path
SERVER_DIR = Path(__file__).parent
sys.path.insert(0, str(SERVER_DIR))

# Configuration
IMAGE_PATH = SERVER_DIR.parent / "test_images" / "image3.png"
EXISTING_LAYOUT_DIR = SERVER_DIR / "results" / "layout_ec8b8760"
OUTPUT_DIR = SERVER_DIR / "results" / "layout_image3_v2"

# The updated analysis prompt with environment extraction
ANALYSIS_PROMPT = """Analyze this image of an interior space and provide a detailed, structured description.

Return a JSON object with the following fields:

{
    "room_type": "type of room",
    "room_dimensions": {
        "width_m": estimated width in meters,
        "depth_m": estimated depth in meters,
        "height_m": estimated ceiling height in meters
    },
    "style": "overall design style",
    "environment": {
        "flooring": {
            "type": "concrete|tile|hardwood|carpet|linoleum|other",
            "color": "primary color",
            "material": "specific material description",
            "pattern": "solid|checkered|striped|wood_grain|other"
        },
        "walls": {
            "type": "drywall|concrete|brick|other",
            "color": "primary color",
            "material": "specific material",
            "finish": "matte|glossy|textured",
            "features": ["windows", "signage", "etc"]
        },
        "ceiling": {
            "type": "drop_ceiling|exposed|drywall|other",
            "color": "primary color",
            "features": ["lights", "ducts", "etc"]
        },
        "lighting": {
            "type": "natural|fluorescent|led|mixed",
            "brightness": "bright|moderate|dim",
            "color_temperature": "warm|neutral|cool",
            "sources": [{"type": "window|lamp|overhead", "description": "brief", "location": "where"}],
            "natural_light": true|false
        }
    },
    "vibe": {
        "clutter_level": "sparse|tidy|moderate|cluttered",
        "mood": "cozy|professional|casual|relaxed",
        "era": "contemporary|vintage|modern"
    },
    "objects": [
        {
            "name": "object type",
            "category": "furniture|decor|appliance|storage",
            "description": "brief description with color/material",
            "estimated_size": {"width_m": 0, "height_m": 0, "depth_m": 0},
            "material": "primary material",
            "style": "object style",
            "location": "where in room",
            "quantity": 1
        }
    ]
}

Focus on accurate environment details - flooring, walls, and lighting are critical.
List all visible objects. Skip any people in the scene.
"""


def encode_image(image_path: str) -> tuple:
    """Encode image to base64."""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return data, "image/png"


def main():
    print("=" * 60)
    print("SAGE Pipeline - Image 3 (Playroom) with Environment Extraction")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Analyze image using vlm.py
    print("\n[1/4] Analyzing image with environment extraction...")
    
    from vlm import call_vlm
    
    # Encode image
    img_data, media_type = encode_image(str(IMAGE_PATH))
    
    # Build message with image
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": img_data
                }
            },
            {
                "type": "text",
                "text": ANALYSIS_PROMPT
            }
        ]
    }]
    
    # Call VLM (uses key.py configuration)
    response = call_vlm(
        vlm_type="claude",
        model="",  # Will be overridden by MODEL_DICT
        max_tokens=8192,
        temperature=0.3,
        messages=messages
    )
    
    # Extract text from response
    response_text = response.content[0].text if response.content else ""
    
    # Parse JSON from response
    json_str = response_text
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        json_str = response_text[start:end].strip()
    elif "```" in response_text:
        start = response_text.find("```") + 3
        end = response_text.find("```", start)
        json_str = response_text[start:end].strip()
    
    try:
        analysis = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse error: {e}")
        print(f"  Response: {response_text[:500]}...")
        analysis = {"room_type": "playroom", "error": str(e)}
    
    # Save analysis
    analysis_path = OUTPUT_DIR / "image_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"  Room type: {analysis.get('room_type', 'unknown')}")
    print(f"  Style: {analysis.get('style', 'unknown')}")
    print(f"  Objects: {len(analysis.get('objects', []))}")
    
    # Print environment details
    env = analysis.get("environment", {})
    if env:
        if "flooring" in env:
            f = env["flooring"]
            print(f"  Flooring: {f.get('type', '?')} - {f.get('material', '')} ({f.get('color', '')})")
        if "walls" in env:
            w = env["walls"]
            print(f"  Walls: {w.get('type', '?')} - {w.get('color', '')} ({w.get('finish', '')})")
        if "lighting" in env:
            l = env["lighting"]
            print(f"  Lighting: {l.get('type', '?')} - {l.get('brightness', '')} ({l.get('color_temperature', '')})")
    
    print(f"\n  Analysis saved: {analysis_path}")
    
    # Step 2: Link existing TRELLIS assets
    print("\n[2/4] Linking existing TRELLIS assets...")
    
    gen_src = EXISTING_LAYOUT_DIR / "generation"
    gen_dst = OUTPUT_DIR / "generation"
    
    if gen_dst.exists():
        print(f"  Generation dir already exists")
    elif gen_dst.is_symlink():
        print(f"  Symlink exists")
    else:
        os.symlink(gen_src, gen_dst)
        print(f"  Linked: {gen_dst} -> {gen_src}")
    
    # Step 3: Copy layout and add environment
    print("\n[3/4] Loading existing layout with environment...")
    
    layout_src = EXISTING_LAYOUT_DIR / "layout_ec8b8760_messy.json"
    with open(layout_src) as f:
        layout = json.load(f)
    
    # Add environment data
    if env:
        layout["environment"] = env
    
    layout_dst = OUTPUT_DIR / "layout_image3_v2.json"
    with open(layout_dst, "w") as f:
        json.dump(layout, f, indent=2)
    
    print(f"  Layout with environment: {layout_dst}")
    print(f"  Objects: {len(layout['rooms'][0].get('objects', []))}")
    
    # Step 4: Build USD scene
    print("\n[4/4] Building USD scene...")
    
    from ply_to_usd import convert_layout_meshes, build_scene_usd
    
    usd_dir = OUTPUT_DIR / "final_usd"
    usd_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert meshes
    convert_layout_meshes(
        str(layout_dst),
        str(gen_dst),
        str(usd_dir)
    )
    
    # Build scene
    scene_path = usd_dir / "scene.usda"
    build_scene_usd(str(layout_dst), str(scene_path), str(usd_dir / "meshes"))
    
    print(f"\n  USD scene: {scene_path}")
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Output dir: {OUTPUT_DIR}")
    print("=" * 60)
    
    return str(scene_path)


if __name__ == "__main__":
    main()
