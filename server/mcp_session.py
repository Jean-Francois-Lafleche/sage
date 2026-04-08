# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared MCP session ID for logging."""

import uuid
from datetime import datetime

mcp_init_id = str(datetime.now().strftime("%Y%m%d_%H%M%S")) + "_" + str(uuid.uuid4())[:8]


def get_mcp_init_id():
    global mcp_init_id
    return mcp_init_id
