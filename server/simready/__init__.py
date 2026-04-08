# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SimReady USD post-processing module.

Processes generated scene USD files to make them simulation-ready according to
NVIDIA SimReady guidelines: https://docs.omniverse.nvidia.com/simready/latest/overview.html
"""

from .usd_processor import SimReadyProcessor, process_scene_to_simready

__all__ = [
    "SimReadyProcessor",
    "process_scene_to_simready",
]
