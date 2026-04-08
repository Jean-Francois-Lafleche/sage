#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
MCP Client for Image-Based Scene Generation

This client takes one or more reference images, analyzes them using a VLM to extract
structured scene descriptions, and then uses the SAGE MCP pipeline to generate a
matching 3D scene.

Key Features:
- Image analysis using VLM (scene_analyzer)
- Extracts: room type, dimensions, style, objects, spatial relationships
- Generates enhanced room descriptions from image analysis
- Feeds into existing SAGE pipeline for scene generation
- Supports warehouse/shelf scenarios with multi-level placement

Usage:
======
python client_generation_from_image.py --input_images <img1.png> [<img2.png> ...] --server_paths <server.py>
python client_generation_from_image.py --input_images <img1.png> --room_desc "Additional context" --server_paths <server.py>
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

# Import from the main client module
from client_generation_room_desc import MCPClientOAI, get_task_definition_text

# Import our image analysis module
try:
    from image_analysis.scene_analyzer import SceneAnalyzer, SceneAnalysis
    HAS_IMAGE_ANALYSIS = True
except ImportError:
    print("Warning: image_analysis module not found. Using basic image input only.")
    HAS_IMAGE_ANALYSIS = False


def get_image_based_task_definition(analysis: "SceneAnalysis", additional_context: str = "") -> str:
    """Generate task definition text from image analysis.
    
    This creates an enhanced prompt that guides the VLM agent to generate
    a scene matching the analyzed image(s).
    
    Args:
        analysis: SceneAnalysis object from the scene analyzer
        additional_context: Optional additional context from user
        
    Returns:
        Task definition string for the VLM agent
    """
    # Start with the analyzed room description
    room_desc = analysis.to_room_description()
    
    # Add specific placement guidance from spatial relationships
    placement_hints = []
    if analysis.spatial_graph:
        hints = analysis.spatial_graph.generate_placement_hints()
        placement_hints = hints[:20]  # Top 20 placement hints
    
    # Add shelf configurations for warehouse scenes
    shelf_guidance = ""
    if analysis.shelf_configs:
        shelf_parts = []
        for shelf in analysis.shelf_configs:
            shelf_parts.append(
                f"- {shelf.get('name', 'Shelf')}: {shelf.get('type', 'shelf')} with "
                f"{shelf.get('num_levels', 4)} levels, {shelf.get('width_m', 1.2):.1f}m wide"
            )
        if shelf_parts:
            shelf_guidance = "\n\nSHELF/RACK CONFIGURATIONS:\n" + "\n".join(shelf_parts)
            shelf_guidance += "\n[IMPORTANT] Place items at appropriate shelf levels, not just on the floor."

    # Construct the task definition
    task_text = f"""Task: Generate a 3D scene that matches the following reference image analysis.

=== REFERENCE IMAGE ANALYSIS ===

ROOM DESCRIPTION:
{room_desc}

DETECTED STYLE AND VIBE:
- Style: {analysis.style}
- Vibe: {json.dumps(analysis.vibe, indent=2) if analysis.vibe else "Not specified"}
{shelf_guidance}

=== PLACEMENT GUIDANCE FROM IMAGE ===
The following spatial relationships were detected in the reference image(s):
{chr(10).join(f"- {hint}" for hint in placement_hints) if placement_hints else "No specific placement hints detected."}

=== GENERATION INSTRUCTIONS ===

You will use tools to generate a complete, realistic scene that matches the reference image.
The process involves two main steps:
1. Generate the room layout (matching dimensions and style from analysis)
2. Place objects strategically to match the reference image's arrangement

STEP 1: GENERATE ROOM LAYOUT
Generate the scene layout first using the analyzed room dimensions ({analysis.room_dimensions[0]:.1f}m x {analysis.room_dimensions[1]:.1f}m x {analysis.room_dimensions[2]:.1f}m).
The style should be: {analysis.style}

STEP 2: PLACE OBJECTS
Place objects following the detected inventory and spatial relationships.

DETECTED OBJECTS TO PLACE (in priority order):
"""
    
    # Add object proposals from analysis
    proposals = analysis.to_object_proposals()
    for i, prop in enumerate(proposals[:25], 1):
        size = prop.get("estimated_size", {})
        size_str = f"{size.get('width', 0.5):.1f}x{size.get('height', 0.5):.1f}x{size.get('depth', 0.5):.1f}m"
        task_text += f"\n{i}. {prop['object_type']} ({prop.get('object_quantity', 1)}x): {prop.get('object_short_description', '')} [{size_str}]"
    
    task_text += """

[CRITICAL] Match the style, vibe, and spatial arrangement from the reference image.
[CRITICAL] For warehouse/storage scenes, use multi-level shelving and proper stacking.
[CRITICAL] Preserve the clutter level and mood detected in the analysis.

"""
    
    # Add the standard task definition sections
    task_text += get_task_definition_text(room_desc)
    
    # Add any additional context from user
    if additional_context:
        task_text += f"\n\nADDITIONAL USER CONTEXT:\n{additional_context}\n"
    
    return task_text


async def analyze_and_generate(
    image_paths: List[str],
    server_paths: List[str],
    additional_context: str = "",
    vlm_api_url: str = "http://localhost:8100/v1",
) -> None:
    """Analyze images and generate a matching scene.
    
    Args:
        image_paths: List of reference image paths
        server_paths: List of MCP server script paths
        additional_context: Optional additional context
        vlm_api_url: URL of the VLM API for image analysis
    """
    print("=" * 60)
    print("IMAGE-BASED SCENE GENERATION")
    print("=" * 60)
    
    # Step 1: Analyze images
    print(f"\n📷 Analyzing {len(image_paths)} image(s)...")
    
    if HAS_IMAGE_ANALYSIS:
        analyzer = SceneAnalyzer(api_url=vlm_api_url)
        analysis = analyzer.analyze_images(image_paths, additional_context)
        
        print(f"\n✅ Image Analysis Complete:")
        print(f"   Room Type: {analysis.room_type}")
        print(f"   Style: {analysis.style}")
        print(f"   Dimensions: {analysis.room_dimensions[0]:.1f}m x {analysis.room_dimensions[1]:.1f}m x {analysis.room_dimensions[2]:.1f}m")
        print(f"   Vibe: {analysis.vibe}")
        print(f"   Detected Objects: {len(analysis.objects)}")
        
        if analysis.shelf_configs:
            print(f"   Shelf Configurations: {len(analysis.shelf_configs)}")
        
        # Generate task definition from analysis
        task_definition = get_image_based_task_definition(analysis, additional_context)
    else:
        # Fallback: just use images as multimodal input without analysis
        print("⚠️ Image analysis module not available. Using basic multimodal input.")
        room_desc = additional_context or "Generate a scene matching the provided image(s)."
        task_definition = get_task_definition_text(room_desc)
        analysis = None
    
    # Step 2: Initialize MCP client
    print(f"\n🔌 Connecting to MCP servers...")
    client = MCPClientOAI()
    
    try:
        await client.connect_to_servers(server_paths)
        
        # Step 3: Run generation with the task definition and images
        print(f"\n🎨 Starting scene generation...")
        result, _ = await client.process_query(task_definition, image_paths)
        
        print(f"\n✅ Scene generation complete!")
        
        # Save analysis results
        if analysis:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            
            analysis_file = output_dir / "image_analysis.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis.to_dict(), f, indent=2)
            print(f"📄 Analysis saved to: {analysis_file}")
        
    finally:
        await client.cleanup()


async def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D scenes from reference images using SAGE"
    )
    parser.add_argument(
        "--input_images",
        nargs="+",
        required=True,
        help="Path(s) to reference image(s)"
    )
    parser.add_argument(
        "--server_paths",
        nargs="+",
        required=True,
        help="Path(s) to MCP server script(s)"
    )
    parser.add_argument(
        "--room_desc",
        type=str,
        default="",
        help="Additional room description/context"
    )
    parser.add_argument(
        "--vlm_api_url",
        type=str,
        default="http://localhost:8100/v1",
        help="URL of the VLM API for image analysis"
    )
    
    args = parser.parse_args()
    
    # Validate image paths
    for img_path in args.input_images:
        if not os.path.exists(img_path):
            print(f"Error: Image not found: {img_path}")
            sys.exit(1)
    
    # Validate server paths
    for server_path in args.server_paths:
        if not os.path.exists(server_path):
            print(f"Error: Server script not found: {server_path}")
            sys.exit(1)
    
    await analyze_and_generate(
        image_paths=args.input_images,
        server_paths=args.server_paths,
        additional_context=args.room_desc,
        vlm_api_url=args.vlm_api_url,
    )


if __name__ == "__main__":
    asyncio.run(main())
