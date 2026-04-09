# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM-based scene analyzer for image-conditioned scene generation.

This module analyzes input images using a Vision-Language Model (VLM) to extract
structured scene descriptions that can be used to guide the SAGE scene generation pipeline.
"""

from __future__ import annotations
import base64
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .spatial_graph import SpatialGraph, SpatialRelation, ObjectNode, RelationType
from .shelf_placer import ShelfPlacer, ShelfConfig, ShelfItem, ShelfLevel


@dataclass
class ObjectInfo:
    """Information about a detected object in the scene."""
    name: str
    category: str  # furniture, decor, appliance, storage, etc.
    description: str
    estimated_size: Tuple[float, float, float]  # (width, height, depth) in meters
    material: str
    style: str
    location_hint: str  # e.g., "against north wall", "center of room"
    quantity: int = 1
    is_container: bool = False  # Can support other objects
    shelf_levels: int = 0  # Number of shelf levels if applicable


@dataclass
class SceneAnalysis:
    """Complete analysis of a scene from one or more images.
    
    This dataclass contains all the information extracted from the VLM analysis
    that can be used to generate a matching 3D scene.
    """
    # Basic room info
    room_type: str
    room_dimensions: Tuple[float, float, float]  # (width, depth, height) in meters
    style: str  # e.g., "modern minimalist", "industrial", "rustic farmhouse"
    
    # Vibe descriptors
    vibe: Dict[str, str] = field(default_factory=dict)
    # e.g., {"clutter_level": "messy", "era": "contemporary", "mood": "cozy"}
    
    # Environment details (flooring, walls, ceiling, lighting)
    environment: Dict[str, Any] = field(default_factory=dict)
    # e.g., {"flooring": {...}, "walls": {...}, "ceiling": {...}, "lighting": {...}}
    
    # Detected objects
    objects: List[ObjectInfo] = field(default_factory=list)
    
    # Spatial relationships
    spatial_graph: Optional[SpatialGraph] = None
    
    # Shelf configurations (for warehouse/storage scenes)
    shelf_configs: List[Dict] = field(default_factory=list)
    
    # Raw VLM response for debugging
    raw_response: str = ""
    
    # Confidence score (0-1)
    confidence: float = 0.8
    
    # ------------------------------------------------------------------
    # Clutter / density helpers
    # ------------------------------------------------------------------

    _DENSITY_MAP = {
        "sparse": 0.6,
        "tidy": 0.8,
        "moderate": 1.0,
        "cluttered": 1.3,
        "chaotic": 1.6,
    }

    def get_density_multiplier(self) -> float:
        """Return a multiplier for object counts based on the detected clutter level.

        Maps the ``vibe["clutter_level"]`` value extracted by the VLM:
            sparse → 0.6, tidy → 0.8, moderate → 1.0, cluttered → 1.3, chaotic → 1.6

        Falls back to 1.0 when the clutter level is missing or unrecognised.
        """
        level = (self.vibe.get("clutter_level") or "").strip().lower()
        return self._DENSITY_MAP.get(level, 1.0)

    def get_min_spacing(self) -> float:
        """Return the minimum spacing (metres) between objects for this clutter level.

        Higher clutter → smaller minimum spacing:
            sparse → 0.5 m, tidy → 0.4 m, moderate → 0.3 m, cluttered → 0.15 m, chaotic → 0.08 m
        """
        _spacing = {
            "sparse": 0.5,
            "tidy": 0.4,
            "moderate": 0.3,
            "cluttered": 0.15,
            "chaotic": 0.08,
        }
        level = (self.vibe.get("clutter_level") or "").strip().lower()
        return _spacing.get(level, 0.3)

    def to_room_description(self) -> str:
        """Convert the analysis to a SAGE-compatible room description string."""
        parts = []
        
        # Room type and style
        parts.append(f"A {self.style} {self.room_type}")
        
        # Dimensions
        w, d, h = self.room_dimensions
        parts.append(f"approximately {w:.1f}m x {d:.1f}m with {h:.1f}m ceiling height")
        
        # Vibe
        vibe_parts = []
        if self.vibe.get("clutter_level"):
            vibe_parts.append(self.vibe["clutter_level"])
        if self.vibe.get("mood"):
            vibe_parts.append(self.vibe["mood"])
        if self.vibe.get("era"):
            vibe_parts.append(f"{self.vibe['era']} era")
        if vibe_parts:
            parts.append(f"The space is {', '.join(vibe_parts)}")
        
        # Key objects
        if self.objects:
            obj_descs = []
            for obj in self.objects[:10]:  # Top 10 objects
                if obj.quantity > 1:
                    obj_descs.append(f"{obj.quantity}x {obj.description}")
                else:
                    obj_descs.append(obj.description)
            parts.append(f"Key furnishings include: {', '.join(obj_descs)}")
        
        return ". ".join(parts) + "."

    def to_object_proposals(self) -> List[Dict]:
        """Convert detected objects to SAGE object proposal format."""
        proposals = []
        for obj in self.objects:
            proposal = {
                "object_type": obj.name.replace(" ", "_"),
                "object_short_description": obj.description[:60],
                "object_placement": obj.location_hint,
                "object_quantity": obj.quantity,
                "object_style": f"{obj.material} {obj.style}".strip(),
                "estimated_size": {
                    "width": obj.estimated_size[0],
                    "height": obj.estimated_size[1],
                    "depth": obj.estimated_size[2],
                }
            }
            proposals.append(proposal)
        return proposals

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "room_type": self.room_type,
            "room_dimensions": self.room_dimensions,
            "style": self.style,
            "vibe": self.vibe,
            "environment": self.environment,
            "objects": [
                {
                    "name": o.name,
                    "category": o.category,
                    "description": o.description,
                    "estimated_size": o.estimated_size,
                    "material": o.material,
                    "style": o.style,
                    "location_hint": o.location_hint,
                    "quantity": o.quantity,
                    "is_container": o.is_container,
                    "shelf_levels": o.shelf_levels,
                }
                for o in self.objects
            ],
            "spatial_graph": self.spatial_graph.to_dict() if self.spatial_graph else None,
            "shelf_configs": self.shelf_configs,
            "confidence": self.confidence,
        }


class SceneAnalyzer:
    """Analyzes images using a VLM to extract structured scene descriptions.
    
    This is the core component for image-conditioned scene generation. It uses
    a Vision-Language Model to understand the contents, layout, and style of
    input images, then produces structured outputs that guide the SAGE pipeline.
    
    Usage:
        analyzer = SceneAnalyzer()
        analysis = analyzer.analyze_images(["/path/to/image.png"])
        room_desc = analysis.to_room_description()
    """

    # VLM prompt for scene analysis
    ANALYSIS_PROMPT = """Analyze this image of an interior space and provide a detailed, structured description.

Return a JSON object with the following fields:

{
    "room_type": "type of room (e.g., living room, warehouse, office, bedroom)",
    "room_dimensions": {
        "width_m": estimated width in meters,
        "depth_m": estimated depth in meters,
        "height_m": estimated ceiling height in meters
    },
    "style": "overall design style (e.g., modern minimalist, industrial, rustic, contemporary)",
    "environment": {
        "flooring": {
            "type": "concrete|tile|hardwood|carpet|linoleum|epoxy|laminate|vinyl|stone|other",
            "color": "primary color (gray, beige, brown, etc.)",
            "material": "specific material description (polished concrete, ceramic tile, oak wood, etc.)",
            "pattern": "solid|checkered|striped|wood_grain|marble|speckled|none",
            "condition": "new|worn|damaged|clean|dirty"
        },
        "walls": {
            "type": "drywall|concrete|brick|metal|glass|paneling|other",
            "color": "primary color",
            "material": "specific material description",
            "finish": "matte|glossy|textured|rough|smooth",
            "features": ["windows", "signage", "shelving_mounted", "pipes_visible", "outlets", "etc."]
        },
        "ceiling": {
            "type": "drop_ceiling|exposed|drywall|metal|industrial|vaulted|other",
            "color": "primary color",
            "height_style": "standard|high|low|variable",
            "features": ["ducts", "pipes", "sprinklers", "lights", "skylights", "fans", "etc."]
        },
        "lighting": {
            "type": "fluorescent|led|natural|incandescent|mixed|industrial",
            "brightness": "bright|moderate|dim|variable",
            "color_temperature": "warm|neutral|cool|daylight",
            "sources": [
                {
                    "type": "overhead|window|lamp|track|pendant|recessed|strip",
                    "description": "brief description",
                    "location": "ceiling|wall|floor|window"
                }
            ],
            "shadows": "harsh|soft|mixed|none",
            "natural_light": true|false
        }
    },
    "vibe": {
        "clutter_level": "sparse|tidy|moderate|cluttered|chaotic",
        "mood": "cozy|professional|casual|formal|relaxed|energetic",
        "era": "vintage|retro|contemporary|futuristic|timeless",
        "lighting": "bright|dim|natural|artificial|mixed"
    },
    "objects": [
        {
            "name": "single-word object type (e.g., sofa, shelf, lamp)",
            "category": "furniture|decor|appliance|storage|lighting|textile",
            "description": "brief description with material/color/style (max 60 chars)",
            "estimated_size": {
                "width_m": width in meters,
                "height_m": height in meters,
                "depth_m": depth in meters
            },
            "material": "primary material (wood, metal, fabric, etc.)",
            "style": "object style (modern, rustic, industrial, etc.)",
            "location": "where in the room (against north wall, center, corner, etc.)",
            "quantity": number of this object visible,
            "is_container": true if object can hold/support other objects,
            "shelf_levels": number of shelf levels (0 if not a shelf/rack)
        }
    ],
    "spatial_relationships": [
        {
            "subject": "object name",
            "relation": "on_top_of|next_to|in_front_of|behind|left_of|right_of|inside|against_wall|on_shelf_level|stacked_on",
            "reference": "reference object or location",
            "level": optional shelf level number if applicable
        }
    ],
    "shelf_configurations": [
        {
            "name": "shelf/rack identifier",
            "type": "bookshelf|warehouse_rack|cabinet|display_shelf",
            "num_levels": number of shelf levels,
            "height_m": total height,
            "width_m": total width,
            "items_per_level": [
                {"level": 0, "items": ["item1", "item2"]},
                {"level": 1, "items": ["item3"]}
            ]
        }
    ]
}

Be thorough but realistic. Estimate dimensions based on typical object sizes and room proportions.
For warehouse/storage scenes, pay special attention to shelf configurations and stacked items.
Pay close attention to the ENVIRONMENT section - flooring, walls, ceiling, and lighting are critical for accurate scene reconstruction.
List at least 10-20 objects for a typical room, more for cluttered/complex scenes.
"""

    def __init__(
        self,
        api_url: str = "http://localhost:8100/v1",
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        api_key: str = "not-needed",
    ):
        """Initialize the scene analyzer.
        
        Args:
            api_url: URL of the OpenAI-compatible VLM API
            model: Model name to use
            api_key: API key (if required)
        """
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        
        # Try to import OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=api_url,
                api_key=api_key,
            )
        except ImportError:
            print("Warning: openai package not installed. Using requests fallback.")
            self.client = None

    def _encode_image(self, image_path: str) -> Tuple[str, str]:
        """Encode an image file to base64.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (base64_data, media_type)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Determine media type
        suffix = path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/png")
        
        # Read and encode
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        return data, media_type

    def _call_vlm(self, messages: List[Dict], max_tokens: int = 4096) -> str:
        """Call the VLM API.
        
        Args:
            messages: List of message dicts in OpenAI format
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content
        else:
            # Fallback to requests
            import requests
            resp = requests.post(
                f"{self.api_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def analyze_images(
        self,
        image_paths: List[str],
        additional_context: str = "",
        max_tokens: int = 8192,
    ) -> SceneAnalysis:
        """Analyze one or more images to extract scene information.
        
        Args:
            image_paths: List of paths to image files
            additional_context: Optional additional context to provide to the VLM
            max_tokens: Maximum tokens to generate
            
        Returns:
            SceneAnalysis object with extracted information
        """
        # Build message content
        content = []
        
        # Add images
        for path in image_paths:
            data, media_type = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{data}"
                }
            })
        
        # Add text prompt
        prompt = self.ANALYSIS_PROMPT
        if additional_context:
            prompt += f"\n\nAdditional context: {additional_context}"
        
        content.append({
            "type": "text",
            "text": prompt,
        })
        
        messages = [{"role": "user", "content": content}]
        
        # Call VLM with larger token limit
        response_text = self._call_vlm(messages, max_tokens=max_tokens)
        
        # Parse response
        return self._parse_response(response_text)

    def _recover_truncated_json(self, json_str: str) -> Optional[Dict]:
        """Attempt to recover a truncated JSON string.
        
        The VLM sometimes generates JSON that gets cut off at the token limit.
        This tries progressively more aggressive truncation + closing.
        """
        import re
        
        # Strategy 1: Find the last complete object in the "objects" array
        # and close everything after it
        for trim_point in range(len(json_str), max(0, len(json_str) - 2000), -1):
            candidate = json_str[:trim_point]
            # Try closing with various bracket combinations
            for closer in [
                '}]}]}',  # close object, objects array, shelf_configs, root
                '}]}',    # close object, array, root  
                '}]',     # close object, array
                ']}',     # close array, root
                '}',      # close root
                ']}'      # close array, root
            ]:
                try:
                    result = json.loads(candidate + closer)
                    if isinstance(result, dict) and 'room_type' in result:
                        print(f"  Recovered JSON at position {trim_point} with closer '{closer}'", file=sys.stderr)
                        return result
                except json.JSONDecodeError:
                    continue
        
        # Strategy 2: Extract just the top-level fields before "objects"
        try:
            # Find the objects array start
            obj_start = json_str.find('"objects"')
            if obj_start > 0:
                prefix = json_str[:obj_start].rstrip().rstrip(',')
                result = json.loads(prefix + '}')
                if isinstance(result, dict):
                    print(f"  Recovered partial JSON (no objects)", file=sys.stderr)
                    return result
        except json.JSONDecodeError:
            pass
        
        return None

    def _parse_response(self, response_text: str) -> SceneAnalysis:
        """Parse VLM response into SceneAnalysis object.
        
        Args:
            response_text: Raw text response from VLM
            
        Returns:
            SceneAnalysis object
        """
        # Extract JSON from response
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
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Try to salvage truncated JSON by closing open structures
            print(f"Warning: JSON parse error: {e}. Attempting truncated JSON recovery...", file=sys.stderr)
            data = self._recover_truncated_json(json_str)
            if data is None:
                print(f"Could not recover JSON. Response: {response_text[:300]}...", file=sys.stderr)
                return SceneAnalysis(
                    room_type="unknown",
                    room_dimensions=(4.0, 4.0, 2.7),
                    style="modern",
                    raw_response=response_text,
                    confidence=0.3,
                )
        
        # Extract room info
        dims = data.get("room_dimensions", {})
        room_dimensions = (
            dims.get("width_m", 4.0),
            dims.get("depth_m", 4.0),
            dims.get("height_m", 2.7),
        )
        
        # Extract objects
        objects = []
        for obj_data in data.get("objects", []):
            size = obj_data.get("estimated_size", {})
            obj = ObjectInfo(
                name=obj_data.get("name", "unknown"),
                category=obj_data.get("category", "furniture"),
                description=obj_data.get("description", "")[:60],
                estimated_size=(
                    size.get("width_m", 0.5),
                    size.get("height_m", 0.5),
                    size.get("depth_m", 0.5),
                ),
                material=obj_data.get("material", ""),
                style=obj_data.get("style", ""),
                location_hint=obj_data.get("location", "floor"),
                quantity=obj_data.get("quantity", 1),
                is_container=obj_data.get("is_container", False),
                shelf_levels=obj_data.get("shelf_levels", 0),
            )
            objects.append(obj)
        
        # Build spatial graph
        spatial_graph = SpatialGraph()
        for i, obj in enumerate(objects):
            node = ObjectNode(
                id=f"obj_{i:03d}",
                name=obj.name,
                category=obj.category,
                estimated_size=obj.estimated_size,
                material=obj.material,
                style=obj.style,
                is_container=obj.is_container,
                shelf_levels=obj.shelf_levels,
            )
            spatial_graph.add_object(node)
        
        # Add spatial relations
        for rel_data in data.get("spatial_relationships", []):
            try:
                relation = SpatialRelation(
                    subject=rel_data.get("subject", ""),
                    relation=RelationType(rel_data.get("relation", "next_to")),
                    reference=rel_data.get("reference", ""),
                    metadata={"level": str(rel_data.get("level", ""))} if rel_data.get("level") else {},
                )
                spatial_graph.add_relation(relation)
            except ValueError:
                pass  # Invalid relation type
        
        return SceneAnalysis(
            room_type=data.get("room_type", "room"),
            room_dimensions=room_dimensions,
            style=data.get("style", "modern"),
            vibe=data.get("vibe", {}),
            environment=data.get("environment", {}),
            objects=objects,
            spatial_graph=spatial_graph,
            shelf_configs=data.get("shelf_configurations", []),
            raw_response=response_text,
            confidence=0.8,
        )


def analyze_image(image_path: str, api_url: str = "http://localhost:8100/v1") -> SceneAnalysis:
    """Convenience function to analyze a single image.
    
    Args:
        image_path: Path to the image file
        api_url: URL of the VLM API
        
    Returns:
        SceneAnalysis object
    """
    analyzer = SceneAnalyzer(api_url=api_url)
    return analyzer.analyze_images([image_path])


if __name__ == "__main__":
    # Test with command line argument
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Analyzing: {image_path}")
        analysis = analyze_image(image_path)
        print(f"\nRoom type: {analysis.room_type}")
        print(f"Style: {analysis.style}")
        print(f"Dimensions: {analysis.room_dimensions}")
        print(f"Vibe: {analysis.vibe}")
        print(f"\nObjects ({len(analysis.objects)}):")
        for obj in analysis.objects[:10]:
            print(f"  - {obj.name}: {obj.description}")
        print(f"\nRoom description:\n{analysis.to_room_description()}")
    else:
        print("Usage: python scene_analyzer.py <image_path>")
