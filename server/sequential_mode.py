# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sequential execution mode for SAGE.

This module provides GPU memory management for running SAGE on a single GPU
by loading/unloading models sequentially rather than keeping everything in VRAM.

Pipeline stages and their GPU requirements:
1. VLM (Qwen3-VL via vLLM) — ~18GB for 8B model
2. CLIP + SBERT — ~2GB for material selection
3. TRELLIS — ~8-10GB for 3D object generation
4. FLUX — ~10GB for material/texture generation  
5. nvdiffrast — ~2GB for rendering

In sequential mode, only one heavy model is loaded at a time.

Usage:
    from sequential_mode import SequentialManager, sequential_enabled
    
    # Enable sequential mode
    manager = SequentialManager.instance()
    manager.enable()
    
    # Before a GPU-heavy stage
    manager.prepare_stage("trellis")  # Frees other models
    # ... do trellis work ...
    manager.release_stage("trellis")  # Optionally free early
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from enum import Enum
from typing import Optional, Dict, Callable, Any

import torch


class Stage(str, Enum):
    """GPU-heavy pipeline stages."""
    VLM = "vlm"
    CLIP = "clip"
    TRELLIS = "trellis"
    FLUX = "flux"
    RENDER = "render"
    IDLE = "idle"


# Global flag — checked by other modules
_sequential_enabled = os.environ.get("SAGE_SEQUENTIAL", "0") == "1"


def sequential_enabled() -> bool:
    """Check if sequential mode is enabled."""
    return _sequential_enabled


def enable_sequential():
    """Enable sequential mode globally."""
    global _sequential_enabled
    _sequential_enabled = True
    print("🔄 Sequential mode ENABLED — models will be loaded/unloaded as needed", file=sys.stderr)


def disable_sequential():
    """Disable sequential mode."""
    global _sequential_enabled
    _sequential_enabled = False
    print("⚡ Sequential mode DISABLED — all models stay resident", file=sys.stderr)


class SequentialManager:
    """Manages GPU memory by loading/unloading models for each pipeline stage.
    
    This is a singleton — use SequentialManager.instance() to get the shared instance.
    """
    
    _instance: Optional["SequentialManager"] = None
    _lock = threading.Lock()
    
    @classmethod
    def instance(cls) -> "SequentialManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.current_stage: Stage = Stage.IDLE
        self._stage_lock = threading.Lock()
        self._unloaders: Dict[str, Callable] = {}
        self._loaders: Dict[str, Callable] = {}
        self._loaded: Dict[str, bool] = {}
    
    def enable(self):
        """Enable sequential mode."""
        enable_sequential()
    
    def disable(self):
        """Disable sequential mode."""
        disable_sequential()
    
    def register_stage(
        self,
        stage_name: str,
        loader: Optional[Callable] = None,
        unloader: Optional[Callable] = None,
    ):
        """Register load/unload callbacks for a stage.
        
        Args:
            stage_name: Name of the stage
            loader: Function to call to load the model (returns the model object)
            unloader: Function to call to unload/free the model
        """
        if loader:
            self._loaders[stage_name] = loader
        if unloader:
            self._unloaders[stage_name] = unloader
        self._loaded[stage_name] = False
    
    def prepare_stage(self, stage_name: str) -> None:
        """Prepare for a GPU-heavy stage by freeing other models.
        
        In sequential mode, this will:
        1. Unload all other loaded stages
        2. Clear CUDA cache
        3. Load the requested stage's model
        
        In non-sequential mode, this is a no-op.
        
        Args:
            stage_name: Name of the stage to prepare for
        """
        if not sequential_enabled():
            return
        
        with self._stage_lock:
            if self.current_stage == stage_name:
                return  # Already in this stage
            
            print(f"🔄 Sequential: Preparing stage '{stage_name}'...", file=sys.stderr)
            
            # Unload other stages
            for name, is_loaded in self._loaded.items():
                if is_loaded and name != stage_name:
                    self._unload_stage(name)
            
            # Clear CUDA cache
            self._clear_gpu_memory()
            
            # Load the requested stage
            if stage_name in self._loaders and not self._loaded.get(stage_name, False):
                print(f"  Loading '{stage_name}' model...", file=sys.stderr)
                self._loaders[stage_name]()
                self._loaded[stage_name] = True
            
            self.current_stage = Stage(stage_name) if stage_name in Stage.__members__.values() else Stage.IDLE
            
            # Report memory
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024**3
                total = torch.cuda.get_device_properties(0).total_mem / 1024**3
                print(f"  GPU memory: {used:.1f}GB / {total:.1f}GB", file=sys.stderr)
    
    def release_stage(self, stage_name: str) -> None:
        """Release a stage's resources early.
        
        Useful when you know you're done with a stage and want to free
        memory before explicitly preparing the next one.
        
        Args:
            stage_name: Name of the stage to release
        """
        if not sequential_enabled():
            return
        
        with self._stage_lock:
            if self._loaded.get(stage_name, False):
                self._unload_stage(stage_name)
                self._clear_gpu_memory()
                self.current_stage = Stage.IDLE
    
    def _unload_stage(self, stage_name: str) -> None:
        """Unload a specific stage's model."""
        if stage_name in self._unloaders:
            print(f"  Unloading '{stage_name}' model...", file=sys.stderr)
            try:
                self._unloaders[stage_name]()
            except Exception as e:
                print(f"  Warning: Error unloading '{stage_name}': {e}", file=sys.stderr)
        self._loaded[stage_name] = False
    
    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory info in GB."""
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_mem / 1024**3,
            "free_gb": (torch.cuda.get_device_properties(0).total_mem - torch.cuda.memory_allocated()) / 1024**3,
        }


# Convenience functions for use in existing code

def prepare_for_clip():
    """Call before CLIP/material selection."""
    SequentialManager.instance().prepare_stage("clip")


def release_clip():
    """Call after CLIP/material selection is done."""
    SequentialManager.instance().release_stage("clip")


def prepare_for_trellis():
    """Call before 3D object generation."""
    SequentialManager.instance().prepare_stage("trellis")


def release_trellis():
    """Call after 3D generation is done."""
    SequentialManager.instance().release_stage("trellis")


def prepare_for_flux():
    """Call before material/texture generation."""
    SequentialManager.instance().prepare_stage("flux")


def release_flux():
    """Call after material generation is done."""
    SequentialManager.instance().release_stage("flux")


def prepare_for_render():
    """Call before rendering."""
    SequentialManager.instance().prepare_stage("render")


def release_render():
    """Call after rendering is done."""
    SequentialManager.instance().release_stage("render")
