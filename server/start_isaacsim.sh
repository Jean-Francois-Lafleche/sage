#!/bin/bash
# Start Isaac Sim with MCP extension for SAGE
cd /home/horde/.openclaw/workspace/sage/server
source .venv/bin/activate

export OMNI_KIT_ACCEPT_EULA=YES
export SLURM_JOB_ID=${SLURM_JOB_ID:-local}

# Remove local isaacsim from Python path to use pip-installed version
exec python -c "
import sys, os
# Remove current dir from path so pip-installed isaacsim takes precedence
sys.path = [p for p in sys.path if p not in ('', '.', os.getcwd())]
import isaacsim
isaacsim.main()
" -- isaacsim.exp.base.python \
  --ext-folder ./isaacsim \
  --enable isaac.sim.mcp_extension \
  --no-window \
  --/app/window/enabled=false
