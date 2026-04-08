# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spatial relationship graph for scene objects."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple


class RelationType(str, Enum):
    """Types of spatial relationships between objects."""
    ON_TOP_OF = "on_top_of"
    NEXT_TO = "next_to"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    INSIDE = "inside"
    AGAINST_WALL = "against_wall"
    ON_SHELF_LEVEL = "on_shelf_level"
    STACKED_ON = "stacked_on"
    HANGING_FROM = "hanging_from"
    UNDER = "under"


@dataclass
class SpatialRelation:
    """A spatial relationship between two objects.
    
    Attributes:
        subject: The object being described (e.g., "lamp")
        relation: The spatial relationship type
        reference: The reference object (e.g., "desk") or location (e.g., "north_wall")
        confidence: VLM confidence in this relationship (0-1)
        metadata: Additional info (e.g., shelf_level=2, distance_estimate="close")
    """
    subject: str
    relation: RelationType
    reference: str
    confidence: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_placement_hint(self) -> str:
        """Convert to a natural language placement hint for the SAGE pipeline."""
        hints = {
            RelationType.ON_TOP_OF: f"place {self.subject} on top of {self.reference}",
            RelationType.NEXT_TO: f"place {self.subject} next to {self.reference}",
            RelationType.IN_FRONT_OF: f"place {self.subject} in front of {self.reference}",
            RelationType.BEHIND: f"place {self.subject} behind {self.reference}",
            RelationType.LEFT_OF: f"place {self.subject} to the left of {self.reference}",
            RelationType.RIGHT_OF: f"place {self.subject} to the right of {self.reference}",
            RelationType.INSIDE: f"place {self.subject} inside {self.reference}",
            RelationType.AGAINST_WALL: f"place {self.subject} against {self.reference}",
            RelationType.ON_SHELF_LEVEL: f"place {self.subject} on shelf level {self.metadata.get('level', '?')} of {self.reference}",
            RelationType.STACKED_ON: f"stack {self.subject} on {self.reference}",
            RelationType.HANGING_FROM: f"hang {self.subject} from {self.reference}",
            RelationType.UNDER: f"place {self.subject} under {self.reference}",
        }
        return hints.get(self.relation, f"{self.subject} {self.relation.value} {self.reference}")


@dataclass
class ObjectNode:
    """A node in the spatial graph representing a scene object.
    
    Attributes:
        id: Unique identifier for this object
        name: Human-readable name (e.g., "wooden bookshelf")
        category: Object category (e.g., "furniture", "decor", "appliance")
        estimated_size: Estimated dimensions as (width, height, depth) in meters
        material: Dominant material description
        style: Style descriptor (e.g., "modern", "rustic", "industrial")
        is_container: Whether this object can contain/support other objects
        shelf_levels: Number of shelf levels (if applicable)
    """
    id: str
    name: str
    category: str = "unknown"
    estimated_size: Optional[Tuple[float, float, float]] = None
    material: str = ""
    style: str = ""
    is_container: bool = False
    shelf_levels: int = 0


class SpatialGraph:
    """Graph representing spatial relationships between objects in a scene.
    
    This graph is built from VLM image analysis and used to guide
    object placement in the SAGE pipeline.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, ObjectNode] = {}
        self.relations: List[SpatialRelation] = []

    def add_object(self, node: ObjectNode) -> None:
        """Add an object node to the graph."""
        self.nodes[node.id] = node

    def add_relation(self, relation: SpatialRelation) -> None:
        """Add a spatial relation between objects."""
        self.relations.append(relation)

    def get_relations_for(self, object_id: str) -> List[SpatialRelation]:
        """Get all relations involving a specific object."""
        return [
            r for r in self.relations
            if r.subject == object_id or r.reference == object_id
        ]

    def get_placement_order(self) -> List[str]:
        """Determine optimal placement order (large/foundational objects first).
        
        Uses topological sort: objects that support others come first.
        """
        # Build dependency graph
        deps: Dict[str, set] = {nid: set() for nid in self.nodes}
        for rel in self.relations:
            if rel.relation in (
                RelationType.ON_TOP_OF,
                RelationType.INSIDE,
                RelationType.ON_SHELF_LEVEL,
                RelationType.STACKED_ON,
                RelationType.HANGING_FROM,
            ):
                if rel.subject in deps and rel.reference in self.nodes:
                    deps[rel.subject].add(rel.reference)

        # Topological sort (Kahn's algorithm)
        in_degree = {nid: 0 for nid in self.nodes}
        for nid, dep_set in deps.items():
            in_degree[nid] = len(dep_set)

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: List[str] = []

        while queue:
            # Sort by category priority: furniture > appliance > decor > other
            priority = {"furniture": 0, "appliance": 1, "storage": 2, "decor": 3}
            queue.sort(key=lambda x: priority.get(self.nodes[x].category, 4))
            node_id = queue.pop(0)
            order.append(node_id)
            for other_id, other_deps in deps.items():
                if node_id in other_deps:
                    other_deps.discard(node_id)
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        # Add any remaining nodes (cycles)
        for nid in self.nodes:
            if nid not in order:
                order.append(nid)

        return order

    def generate_placement_hints(self) -> List[str]:
        """Generate natural language placement hints for the SAGE VLM agent."""
        order = self.get_placement_order()
        hints = []
        for obj_id in order:
            obj_rels = [r for r in self.relations if r.subject == obj_id]
            if obj_rels:
                for rel in obj_rels:
                    hints.append(rel.to_placement_hint())
            else:
                node = self.nodes.get(obj_id)
                if node:
                    hints.append(f"place {node.name} in appropriate location")
        return hints

    def to_dict(self) -> Dict:
        """Serialize the graph to a dictionary."""
        return {
            "nodes": {
                nid: {
                    "id": n.id,
                    "name": n.name,
                    "category": n.category,
                    "estimated_size": n.estimated_size,
                    "material": n.material,
                    "style": n.style,
                    "is_container": n.is_container,
                    "shelf_levels": n.shelf_levels,
                }
                for nid, n in self.nodes.items()
            },
            "relations": [
                {
                    "subject": r.subject,
                    "relation": r.relation.value,
                    "reference": r.reference,
                    "confidence": r.confidence,
                    "metadata": r.metadata,
                }
                for r in self.relations
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SpatialGraph":
        """Deserialize a graph from a dictionary."""
        graph = cls()
        for nid, ndata in data.get("nodes", {}).items():
            size = ndata.get("estimated_size")
            if size and isinstance(size, (list, tuple)):
                size = tuple(size)
            graph.add_object(ObjectNode(
                id=ndata["id"],
                name=ndata["name"],
                category=ndata.get("category", "unknown"),
                estimated_size=size,
                material=ndata.get("material", ""),
                style=ndata.get("style", ""),
                is_container=ndata.get("is_container", False),
                shelf_levels=ndata.get("shelf_levels", 0),
            ))
        for rdata in data.get("relations", []):
            graph.add_relation(SpatialRelation(
                subject=rdata["subject"],
                relation=RelationType(rdata["relation"]),
                reference=rdata["reference"],
                confidence=rdata.get("confidence", 1.0),
                metadata=rdata.get("metadata", {}),
            ))
        return graph
