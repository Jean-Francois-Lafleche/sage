# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
VLM-based scene comparator for comparing rendered scenes against original reference images.

This module compares a rendered 3D scene against the original input image to identify
discrepancies such as:
- Wrong flooring/wall materials
- Missing or extra objects
- Incorrect object placement
- Wrong scale or proportions
- Lighting mismatches
- Style inconsistencies
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


@dataclass
class ComparisonIssue:
    """A single issue identified when comparing render to original."""
    category: str  # flooring, walls, ceiling, lighting, objects, layout, scale, style
    severity: str  # critical, major, minor, suggestion
    description: str
    location: str  # Where in the scene this issue occurs
    recommendation: str  # How to fix it


@dataclass 
class SceneComparison:
    """Results of comparing a rendered scene to the original reference image."""
    
    # Overall match score (0-10, 10 = perfect match)
    overall_score: float
    
    # Category-specific scores (0-10)
    scores: Dict[str, float] = field(default_factory=dict)
    # e.g., {"flooring": 8.0, "walls": 6.0, "objects": 7.5, "lighting": 5.0, ...}
    
    # List of identified issues
    issues: List[ComparisonIssue] = field(default_factory=list)
    
    # Summary of what matches well
    matches: List[str] = field(default_factory=list)
    
    # Summary text
    summary: str = ""
    
    # Raw VLM response for debugging
    raw_response: str = ""
    
    def get_critical_issues(self) -> List[ComparisonIssue]:
        """Return only critical severity issues."""
        return [i for i in self.issues if i.severity == "critical"]
    
    def get_issues_by_category(self, category: str) -> List[ComparisonIssue]:
        """Return issues for a specific category."""
        return [i for i in self.issues if i.category == category]
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "overall_score": self.overall_score,
            "scores": self.scores,
            "issues": [
                {
                    "category": i.category,
                    "severity": i.severity,
                    "description": i.description,
                    "location": i.location,
                    "recommendation": i.recommendation,
                }
                for i in self.issues
            ],
            "matches": self.matches,
            "summary": self.summary,
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown format for readable output."""
        lines = []
        lines.append(f"# Scene Comparison Report")
        lines.append(f"\n## Overall Score: {self.overall_score}/10\n")
        
        # Category scores
        lines.append("### Category Scores")
        for cat, score in sorted(self.scores.items()):
            emoji = "✅" if score >= 7 else "⚠️" if score >= 5 else "❌"
            lines.append(f"- {emoji} **{cat.title()}**: {score}/10")
        
        # What's working
        if self.matches:
            lines.append("\n### What's Working Well")
            for match in self.matches:
                lines.append(f"- ✓ {match}")
        
        # Issues by severity
        if self.issues:
            lines.append("\n### Issues Found")
            
            for severity in ["critical", "major", "minor", "suggestion"]:
                severity_issues = [i for i in self.issues if i.severity == severity]
                if severity_issues:
                    emoji = {"critical": "🔴", "major": "🟠", "minor": "🟡", "suggestion": "💡"}[severity]
                    lines.append(f"\n#### {emoji} {severity.title()} ({len(severity_issues)})")
                    for issue in severity_issues:
                        lines.append(f"- **[{issue.category}]** {issue.description}")
                        lines.append(f"  - Location: {issue.location}")
                        lines.append(f"  - Fix: {issue.recommendation}")
        
        # Summary
        if self.summary:
            lines.append(f"\n### Summary\n{self.summary}")
        
        return "\n".join(lines)


class SceneComparator:
    """Compares rendered scenes against original reference images using VLM.
    
    This is used after scene generation to validate that the output matches
    the input image and identify any discrepancies that need correction.
    
    Usage:
        comparator = SceneComparator()
        result = comparator.compare(
            original_image="/path/to/input.png",
            rendered_image="/path/to/render.png"
        )
        if result.overall_score < 7:
            print(result.to_markdown())
    """

    COMPARISON_PROMPT = """You are evaluating how well a 3D rendered scene matches an original reference photograph.

Compare the two images carefully and provide a detailed analysis.

IMAGE 1: Original reference photograph (the target we're trying to recreate)
IMAGE 2: 3D rendered scene (our attempt to recreate it)

Analyze the following aspects and return a JSON object:

{
    "overall_score": 0-10 (10 = perfect match),
    "scores": {
        "flooring": 0-10 (material, color, pattern match),
        "walls": 0-10 (material, color, features match),
        "ceiling": 0-10 (type, features match),
        "lighting": 0-10 (brightness, color temperature, sources match),
        "objects": 0-10 (correct objects present, right types),
        "object_placement": 0-10 (positions, orientations correct),
        "object_scale": 0-10 (sizes and proportions correct),
        "style": 0-10 (overall aesthetic match),
        "completeness": 0-10 (nothing major missing)
    },
    "issues": [
        {
            "category": "flooring|walls|ceiling|lighting|objects|object_placement|object_scale|style|completeness",
            "severity": "critical|major|minor|suggestion",
            "description": "what is wrong",
            "location": "where in the scene",
            "recommendation": "how to fix it"
        }
    ],
    "matches": [
        "list of things that match well between the images"
    ],
    "summary": "Brief overall assessment of the scene match quality and main areas needing improvement"
}

Severity guidelines:
- critical: Fundamentally wrong (wrong room type, completely wrong floor, major objects missing)
- major: Noticeable errors that significantly affect realism (wrong materials, missing important items)
- minor: Small discrepancies (slight color differences, minor position offsets)
- suggestion: Nice-to-have improvements (could enhance realism but not strictly necessary)

Be specific and actionable in your recommendations. Focus on what would make the biggest visual difference.
"""

    def __init__(
        self,
        api_url: str = "http://localhost:8100/v1",
        model: str = "Qwen/Qwen3-VL-8B-Instruct",
        api_key: str = "not-needed",
    ):
        """Initialize the scene comparator.
        
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
        """Encode an image file to base64."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        suffix = path.suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/png")
        
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        return data, media_type

    def _call_vlm(self, messages: List[Dict], max_tokens: int = 4096) -> str:
        """Call the VLM API."""
        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content
        else:
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

    def compare(
        self,
        original_image: str,
        rendered_image: str,
        additional_context: str = "",
        max_tokens: int = 4096,
    ) -> SceneComparison:
        """Compare a rendered scene against the original reference image.
        
        Args:
            original_image: Path to the original reference image
            rendered_image: Path to the rendered scene image
            additional_context: Optional additional context about the scene
            max_tokens: Maximum tokens to generate
            
        Returns:
            SceneComparison object with detailed analysis
        """
        # Build message content with both images
        content = []
        
        # Add original image (labeled as IMAGE 1)
        orig_data, orig_type = self._encode_image(original_image)
        content.append({
            "type": "text",
            "text": "IMAGE 1 - Original Reference Photograph:"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{orig_type};base64,{orig_data}"}
        })
        
        # Add rendered image (labeled as IMAGE 2)
        render_data, render_type = self._encode_image(rendered_image)
        content.append({
            "type": "text",
            "text": "IMAGE 2 - 3D Rendered Scene:"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{render_type};base64,{render_data}"}
        })
        
        # Add the prompt
        prompt = self.COMPARISON_PROMPT
        if additional_context:
            prompt += f"\n\nAdditional context: {additional_context}"
        
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        # Call VLM
        response_text = self._call_vlm(messages, max_tokens=max_tokens)
        
        # Parse response
        return self._parse_response(response_text)

    def _parse_response(self, response_text: str) -> SceneComparison:
        """Parse VLM response into SceneComparison object."""
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
            print(f"Warning: JSON parse error: {e}", file=sys.stderr)
            return SceneComparison(
                overall_score=0.0,
                summary=f"Failed to parse VLM response: {str(e)[:100]}",
                raw_response=response_text,
            )
        
        # Extract issues
        issues = []
        for issue_data in data.get("issues", []):
            issue = ComparisonIssue(
                category=issue_data.get("category", "unknown"),
                severity=issue_data.get("severity", "minor"),
                description=issue_data.get("description", ""),
                location=issue_data.get("location", ""),
                recommendation=issue_data.get("recommendation", ""),
            )
            issues.append(issue)
        
        return SceneComparison(
            overall_score=data.get("overall_score", 0.0),
            scores=data.get("scores", {}),
            issues=issues,
            matches=data.get("matches", []),
            summary=data.get("summary", ""),
            raw_response=response_text,
        )


def compare_scenes(
    original_image: str,
    rendered_image: str,
    api_url: str = "http://localhost:8100/v1"
) -> SceneComparison:
    """Convenience function to compare two scenes.
    
    Args:
        original_image: Path to the original reference image
        rendered_image: Path to the rendered scene image
        api_url: URL of the VLM API
        
    Returns:
        SceneComparison object
    """
    comparator = SceneComparator(api_url=api_url)
    return comparator.compare(original_image, rendered_image)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        original = sys.argv[1]
        rendered = sys.argv[2]
        print(f"Comparing:\n  Original: {original}\n  Rendered: {rendered}\n")
        
        result = compare_scenes(original, rendered)
        print(result.to_markdown())
    else:
        print("Usage: python scene_comparator.py <original_image> <rendered_image>")
