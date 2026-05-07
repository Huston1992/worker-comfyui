#!/usr/bin/env bash

# Start SSH server if PUBLIC_KEY is set (enables remote access and dev-sync.sh)
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys
    chmod 700 ~/.ssh
    chmod 600 ~/.ssh/authorized_keys

    # Generate host keys if they don't exist (removed during image build for security)
    for key_type in rsa ecdsa ed25519; do
        key_file="/etc/ssh/ssh_host_${key_type}_key"
        if [ ! -f "$key_file" ]; then
            ssh-keygen -t "$key_type" -f "$key_file" -q -N ''
        fi
    done

    service ssh start && echo "worker-comfyui: SSH server started" || echo "worker-comfyui: SSH server could not be started" >&2
fi

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# ---------------------------------------------------------------------------
# Venv resolution
#
# Our build has TWO python environments:
#   - /opt/venv             — created by `uv venv` at Dockerfile top;
#                             holds runpod, requests, websocket-client
#   - /comfyui/.venv        — created by `comfy install --nvidia` and/or
#                             `uv pip install` from Impact-Pack RUN blocks;
#                             holds torch, ComfyUI deps, Impact-Pack deps
#
# The upstream start.sh assumed a single venv and called `python3`
# directly, expecting torch to be on PATH. In our build torch lives in
# /comfyui/.venv, so `python3` from /opt/venv raises `ModuleNotFoundError:
# torch`. Put /comfyui/.venv first on PATH when it exists, so every
# subsequent `python`/`python3` resolves to the env that actually has
# torch + Impact-Pack. Keep /opt/venv on PATH as fallback for runpod/etc.
# ---------------------------------------------------------------------------
if [ -x /comfyui/.venv/bin/python ]; then
    echo "worker-comfyui: Detected /comfyui/.venv — using it for runtime python"
    export PATH="/comfyui/.venv/bin:${PATH}"
    export VIRTUAL_ENV="/comfyui/.venv"
else
    echo "worker-comfyui: /comfyui/.venv not present — falling back to /opt/venv python"
fi
echo "worker-comfyui: python3 resolves to $(command -v python3)"

# ---------------------------------------------------------------------------
# GPU pre-flight check
# Verify that the GPU is accessible before starting ComfyUI. If PyTorch
# cannot initialize CUDA the worker will never be able to process jobs,
# so we fail fast with an actionable error message.
# ---------------------------------------------------------------------------
echo "worker-comfyui: Checking GPU availability..."
if ! GPU_CHECK=$(python3 -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    print(f'OK: {name}')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>&1); then
    echo "worker-comfyui: GPU is not available. PyTorch CUDA init failed:"
    echo "worker-comfyui: $GPU_CHECK"
    echo "worker-comfyui: This usually means the GPU on this machine is not properly initialized."
    echo "worker-comfyui: Please contact RunPod support and report this machine."
    exit 1
fi
echo "worker-comfyui: GPU available — $GPU_CHECK"

# ---------------------------------------------------------------------------
# Symlink /comfyui/models/facerestore_models -> /runpod-volume/...
#
# ReActor's downloader hardcodes /comfyui/models/facerestore_models/ as the
# destination for GFPGAN downloads. It does NOT consult ComfyUI's folder_paths
# or extra_model_paths.yaml — it just writes there directly.
#
# Without a symlink: every cold-start re-downloads ~660 MB of GFPGAN (v1.3 +
# v1.4) onto the container's local disk → handler.py POST to ComfyUI :8188
# times out after 30s while ComfyUI is busy downloading → first job fails.
# After container terminate, files are gone — next cold-start downloads again.
#
# Symlink to Volume: ReActor's "download" actually checks if file exists on
# Volume (where we pre-uploaded both v1.3 and v1.4) → finds them → skips
# download → handler responds in time. Files persist across all workers.
mkdir -p /comfyui/models
if [ ! -L /comfyui/models/facerestore_models ]; then
    rm -rf /comfyui/models/facerestore_models 2>/dev/null
    ln -sfn /runpod-volume/models/facerestore_models /comfyui/models/facerestore_models
    echo "worker-comfyui: facerestore_models symlinked to Volume"
fi
# Same trick for insightface — same hardcoded-path issue with inswapper_128.
if [ ! -L /comfyui/models/insightface ]; then
    rm -rf /comfyui/models/insightface 2>/dev/null
    ln -sfn /runpod-volume/models/insightface /comfyui/models/insightface
    echo "worker-comfyui: insightface symlinked to Volume"
fi

# RUNTIME VERIFY — print what is actually in /comfyui/custom_nodes at container
# boot. If Impact-Pack / Subpack are missing here but were present at build,
# something between build and runtime is wiping them (e.g. a Volume mount
# shadowing the path).
echo "worker-comfyui: [runtime verify] /comfyui/custom_nodes contents:"
ls -la /comfyui/custom_nodes/ || echo "worker-comfyui: [WARN] /comfyui/custom_nodes does not exist!"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

echo "worker-comfyui: Starting ComfyUI"

# Allow operators to tweak verbosity; default is DEBUG.
: "${COMFY_LOG_LEVEL:=DEBUG}"

# PID file used by the handler to detect if ComfyUI is still running
COMFY_PID_FILE="/tmp/comfyui.pid"

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout &
    echo $! > "$COMFY_PID_FILE"

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi