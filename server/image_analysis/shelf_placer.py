# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-level shelf and rack object placement library.

This module provides functionality that is MISSING from the base SAGE pipeline:
the ability to place objects on shelves/racks at specific vertical levels.
Essential for warehouse, storage, and shelving-heavy scenes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math


@dataclass
class ShelfItem:
    """An item placed on a shelf level.
    
    Attributes:
        name: Object name/type
        description: Detailed description for 3D generation
        width: Estimated width in meters
        height: Estimated height in meters
        depth: Estimated depth in meters
        x_offset: Horizontal offset from shelf left edge (meters)
        quantity: Number of this item
        style: Style descriptor
    """
    name: str
    description: str = ""
    width: float = 0.3
    height: float = 0.3
    depth: float = 0.3
    x_offset: float = 0.0
    quantity: int = 1
    style: str = ""


@dataclass
class ShelfLevel:
    """A single level/tier of a shelf or rack.
    
    Attributes:
        level_index: Zero-based level index (0 = bottom)
        height_from_ground: Height of this level's surface from ground (meters)
        usable_width: Width available for placing objects (meters)
        usable_depth: Depth available for placing objects (meters)
        level_height: Vertical clearance of this level (meters)
        items: Objects placed on this level
    """
    level_index: int
    height_from_ground: float
    usable_width: float = 1.0
    usable_depth: float = 0.5
    level_height: float = 0.4
    items: List[ShelfItem] = field(default_factory=list)

    @property
    def occupied_width(self) -> float:
        """Total width occupied by items on this level."""
        return sum(item.width * item.quantity for item in self.items)

    @property
    def remaining_width(self) -> float:
        """Remaining width available on this level."""
        return max(0, self.usable_width - self.occupied_width)

    def can_fit(self, item: ShelfItem) -> bool:
        """Check if an item can fit on this level."""
        total_item_width = item.width * item.quantity
        return (
            total_item_width <= self.remaining_width
            and item.height <= self.level_height
            and item.depth <= self.usable_depth
        )

    def place_item(self, item: ShelfItem) -> bool:
        """Place an item on this level. Returns True if successful."""
        if not self.can_fit(item):
            return False
        item.x_offset = self.occupied_width
        self.items.append(item)
        return True


@dataclass
class ShelfConfig:
    """Configuration for a shelf/rack unit.
    
    Attributes:
        name: Name identifier for this shelf
        num_levels: Number of shelf levels
        total_height: Total height of the shelf unit (meters)
        total_width: Total width of the shelf unit (meters)
        total_depth: Total depth of the shelf unit (meters)
        bottom_clearance: Gap between ground and first shelf (meters)
        shelf_thickness: Thickness of each shelf board (meters)
        style: Style descriptor (e.g., "industrial metal", "wooden bookcase")
        levels: Individual shelf level configurations
    """
    name: str = "shelf"
    num_levels: int = 4
    total_height: float = 2.0
    total_width: float = 1.2
    total_depth: float = 0.5
    bottom_clearance: float = 0.1
    shelf_thickness: float = 0.025
    style: str = "standard"
    levels: List[ShelfLevel] = field(default_factory=list)

    def __post_init__(self):
        if not self.levels:
            self._generate_levels()

    def _generate_levels(self) -> None:
        """Auto-generate evenly spaced shelf levels."""
        usable_height = self.total_height - self.bottom_clearance
        level_spacing = usable_height / self.num_levels
        level_clearance = level_spacing - self.shelf_thickness

        self.levels = []
        for i in range(self.num_levels):
            height = self.bottom_clearance + i * level_spacing
            self.levels.append(ShelfLevel(
                level_index=i,
                height_from_ground=height,
                usable_width=self.total_width - 0.02,  # Small margin
                usable_depth=self.total_depth - 0.02,
                level_height=level_clearance,
            ))


class ShelfPlacer:
    """Places objects on multi-level shelves and racks.
    
    This handles the complexity of placing objects at different vertical levels
    on shelving units — a capability missing from base SAGE which only supports
    floor-level and single-surface placement.
    
    Usage:
        config = ShelfConfig(num_levels=5, total_height=2.4, total_width=1.0)
        placer = ShelfPlacer(config)
        placer.place_on_level(0, ShelfItem(name="box", width=0.3, height=0.3))
        placements = placer.get_placement_commands()
    """

    def __init__(self, config: ShelfConfig) -> None:
        self.config = config

    def place_on_level(self, level: int, item: ShelfItem) -> bool:
        """Place an item on a specific shelf level.
        
        Args:
            level: Zero-based level index
            item: The item to place
            
        Returns:
            True if placement succeeded, False if level is full
        """
        if level < 0 or level >= len(self.config.levels):
            return False
        return self.config.levels[level].place_item(item)

    def auto_distribute(self, items: List[ShelfItem]) -> Dict[int, List[ShelfItem]]:
        """Automatically distribute items across shelf levels.
        
        Strategy:
        - Heavy/large items go on lower levels
        - Lighter/smaller items go higher
        - Fill levels evenly
        
        Args:
            items: List of items to distribute
            
        Returns:
            Dict mapping level index to list of placed items
        """
        # Sort items by size (largest first)
        sorted_items = sorted(
            items,
            key=lambda x: x.width * x.height * x.depth,
            reverse=True,
        )

        placements: Dict[int, List[ShelfItem]] = {
            i: [] for i in range(len(self.config.levels))
        }

        for item in sorted_items:
            placed = False
            # Try levels from bottom to top for large items, top to bottom for small
            item_volume = item.width * item.height * item.depth
            median_volume = 0.027  # ~30cm cube

            if item_volume >= median_volume:
                level_order = range(len(self.config.levels))
            else:
                level_order = range(len(self.config.levels) - 1, -1, -1)

            for level_idx in level_order:
                if self.config.levels[level_idx].can_fit(item):
                    self.config.levels[level_idx].place_item(item)
                    placements[level_idx].append(item)
                    placed = True
                    break

            if not placed:
                # Try any level
                for level_idx in range(len(self.config.levels)):
                    if self.config.levels[level_idx].can_fit(item):
                        self.config.levels[level_idx].place_item(item)
                        placements[level_idx].append(item)
                        placed = True
                        break

        return placements

    def get_placement_commands(self, shelf_object_id: str = "") -> List[Dict]:
        """Generate SAGE-compatible placement commands for all items.
        
        Each command specifies the object, its position relative to the shelf,
        and the vertical height for placement.
        
        Args:
            shelf_object_id: The SAGE object ID of the shelf unit
            
        Returns:
            List of placement command dictionaries
        """
        commands = []
        for level in self.config.levels:
            for item in level.items:
                cmd = {
                    "object_type": item.name,
                    "description": item.description or f"{item.style} {item.name}".strip(),
                    "quantity": item.quantity,
                    "place_location": shelf_object_id or self.config.name,
                    "placement_height": level.height_from_ground,
                    "x_offset": item.x_offset,
                    "constraints": (
                        f"Place on shelf level {level.level_index} "
                        f"(height {level.height_from_ground:.2f}m from ground), "
                        f"offset {item.x_offset:.2f}m from left edge"
                    ),
                }
                commands.append(cmd)
        return commands

    def get_placement_description(self) -> str:
        """Generate a human-readable description of all placements.
        
        Useful for feeding into the VLM agent as placement guidance.
        """
        lines = [f"Shelf '{self.config.name}' ({self.config.style}):"]
        lines.append(f"  Dimensions: {self.config.total_width:.1f}m W × {self.config.total_height:.1f}m H × {self.config.total_depth:.1f}m D")
        lines.append(f"  Levels: {self.config.num_levels}")
        lines.append("")

        for level in self.config.levels:
            if level.items:
                lines.append(f"  Level {level.level_index} (height {level.height_from_ground:.2f}m):")
                for item in level.items:
                    qty_str = f"{item.quantity}×" if item.quantity > 1 else ""
                    lines.append(f"    - {qty_str}{item.name}: {item.description}")
            else:
                lines.append(f"  Level {level.level_index} (height {level.height_from_ground:.2f}m): [empty]")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "config": {
                "name": self.config.name,
                "num_levels": self.config.num_levels,
                "total_height": self.config.total_height,
                "total_width": self.config.total_width,
                "total_depth": self.config.total_depth,
                "bottom_clearance": self.config.bottom_clearance,
                "shelf_thickness": self.config.shelf_thickness,
                "style": self.config.style,
            },
            "levels": [
                {
                    "level_index": lvl.level_index,
                    "height_from_ground": lvl.height_from_ground,
                    "items": [
                        {
                            "name": it.name,
                            "description": it.description,
                            "width": it.width,
                            "height": it.height,
                            "depth": it.depth,
                            "x_offset": it.x_offset,
                            "quantity": it.quantity,
                            "style": it.style,
                        }
                        for it in lvl.items
                    ],
                }
                for lvl in self.config.levels
            ],
        }


def create_warehouse_rack(
    name: str = "warehouse_rack",
    num_levels: int = 5,
    height: float = 3.0,
    width: float = 2.4,
    depth: float = 1.0,
) -> ShelfPlacer:
    """Factory function for creating a standard warehouse pallet rack.
    
    Args:
        name: Rack identifier
        num_levels: Number of levels (typically 3-6)
        height: Total height in meters
        width: Total width in meters (standard pallet rack ~2.4m)
        depth: Total depth in meters (standard ~1.0m for pallet)
        
    Returns:
        ShelfPlacer configured for a warehouse rack
    """
    config = ShelfConfig(
        name=name,
        num_levels=num_levels,
        total_height=height,
        total_width=width,
        total_depth=depth,
        bottom_clearance=0.15,
        shelf_thickness=0.05,
        style="industrial metal warehouse rack",
    )
    return ShelfPlacer(config)


def create_bookshelf(
    name: str = "bookshelf",
    num_levels: int = 5,
    height: float = 1.8,
    width: float = 0.8,
    depth: float = 0.3,
) -> ShelfPlacer:
    """Factory function for creating a standard bookshelf.
    
    Args:
        name: Shelf identifier
        num_levels: Number of shelf levels
        height: Total height in meters
        width: Total width in meters
        depth: Total depth in meters
        
    Returns:
        ShelfPlacer configured for a bookshelf
    """
    config = ShelfConfig(
        name=name,
        num_levels=num_levels,
        total_height=height,
        total_width=width,
        total_depth=depth,
        bottom_clearance=0.05,
        shelf_thickness=0.02,
        style="wooden bookshelf",
    )
    return ShelfPlacer(config)
