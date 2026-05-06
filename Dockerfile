# Build argument for base image selection
ARG BASE_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

# Stage 1: Base image with common dependencies
FROM ${BASE_IMAGE} AS base

# Build arguments for this stage with sensible defaults for standalone builds
ARG COMFYUI_VERSION=latest
ARG CUDA_VERSION_FOR_COMFY
ARG ENABLE_PYTORCH_UPGRADE=false
ARG PYTORCH_INDEX_URL

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1
# Speed up some cmake builds
ENV CMAKE_BUILD_PARALLEL_LEVEL=8

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    openssh-server \
    build-essential \
    cmake \
    pkg-config \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install uv (latest) using official installer and create isolated venv
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && ln -s /root/.local/bin/uvx /usr/local/bin/uvx \
    && uv venv /opt/venv

# Use the virtual environment for all subsequent commands
ENV PATH="/opt/venv/bin:${PATH}"

# Install comfy-cli + dependencies needed by it to install ComfyUI
RUN uv pip install comfy-cli pip setuptools wheel

# Install ComfyUI
RUN if [ -n "${CUDA_VERSION_FOR_COMFY}" ]; then \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --cuda-version "${CUDA_VERSION_FOR_COMFY}" --nvidia; \
    else \
      /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia; \
    fi

# Upgrade PyTorch if needed (for newer CUDA versions)
RUN if [ "$ENABLE_PYTORCH_UPGRADE" = "true" ]; then \
      uv pip install --force-reinstall torch torchvision torchaudio --index-url ${PYTORCH_INDEX_URL}; \
    fi

# Change working directory to ComfyUI
WORKDIR /comfyui

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Install Python runtime dependencies for the handler
RUN uv pip install runpod requests websocket-client

# Add application code and scripts
ADD src/start.sh src/network_volume.py handler.py test_input.json ./
RUN chmod +x /start.sh

# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Prevent pip from asking for confirmation during uninstall steps in custom nodes
ENV PIP_NO_INPUT=1

# Copy helper script to switch Manager network mode at container start
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Install ComfyUI-Impact-Pack + Subpack for character LoRA face/hand refinement.
#
# Why `uv pip install` (no VIRTUAL_ENV override):
#   - uv auto-detects /comfyui/.venv — the venv comfy-cli populated with
#     ComfyUI's own deps (einops, torch, etc). Our Impact-Pack deps land
#     alongside them → subpack's subcore.py imports resolve at runtime.
#   - uv caches wheels across Docker layers → keeps build under RunPod
#     timeout. Plain pip re-downloads ~2GB sam2+deps every time and the
#     build hits the infra timeout before reaching Docker RUN stages
#     (symptom: "Creating cache directory" then immediate fail).
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git && \
    cd ComfyUI-Impact-Pack && \
    git submodule update --init --recursive && \
    uv pip install -r requirements.txt

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git && \
    cd ComfyUI-Impact-Subpack && \
    uv pip install -r requirements.txt && \
    uv pip install ultralytics

# Pin numpy to Subpack-compatible version (needs Float64DType from numpy>=1.26.4)
RUN uv pip install 'numpy>=1.26.4,<2.0'

# Install ComfyUI-VideoHelperSuite for Wan 2.2 I2V video export (VHS_VideoCombine
# node saves frame batches as MP4). ComfyUI has native nodes_video.py but
# VideoHelperSuite is the de-facto standard that community video workflows
# target, and it handles codec options (H.264, frame rate, CRF) more robustly.
RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && \
    uv pip install -r requirements.txt

# Install ComfyUI-ReActor for face-swap on existing images (the 2026 community-
# grade swap recipe: InsightFace inswapper_128 ONNX + GFPGAN/CodeFormer face
# restoration + optional FaceDetailer harmonisation post-pass).
#
# Source: codeberg.org/Gourieff/comfyui-reactor-node (active fork — original
# github reactor-node was archived for compliance reasons; codeberg fork has
# active maintenance).
#
# Why the install is split into separate RUN layers (instead of one chain):
#   1) insightface has NO prebuilt wheel for Python 3.12 (our image's runtime).
#      pip falls back to source compile, which needs build-essential + cmake +
#      python3-dev + numpy already installed (Cython generates code referencing
#      numpy headers). build-essential/cmake are added in the apt step above.
#   2) Pre-installing numpy + cython explicitly before insightface means the
#      compile step sees them and doesn't fight pip's resolver mid-build.
#   3) Splitting into multiple RUN layers makes failures observable in build
#      logs — we know exactly which step broke.
#
# Models (inswapper_128.onnx, GFPGAN, buffalo_l) live on Network Volume, not
# in the image — keeps image lean.
RUN uv pip install --upgrade cython
RUN cd /comfyui/custom_nodes && \
    git clone https://codeberg.org/Gourieff/comfyui-reactor-node.git ComfyUI-ReActor
# uv prefers binary wheels by default — no --prefer-binary flag (it's a pip-only
# flag, uv exits 2 on it). Pin onnxruntime-gpu to a version with confirmed
# Python 3.12 + CUDA 12 wheel on PyPI (1.20.x line is stable for our combo).
RUN uv pip install "onnxruntime-gpu>=1.20,<1.23"
RUN uv pip install --no-build-isolation insightface
# segment-anything is required by ReActor's MaskHelper node. ReActor's own
# requirements.txt lists it as `segment_anything` (with underscore) — but the
# PyPI package name is `segment-anything` (with hyphen). Without this line the
# requirements.txt install silently no-ops on this dep, then `import
# segment_anything` in ReActor's nodes.py raises ModuleNotFoundError, and
# ComfyUI **skips the entire ReActor module** at startup — none of the ReActor
# nodes get registered. Symptom at job runtime:
#   "Node 'ReActorFaceSwap' not found. The custom node may not be installed."
RUN uv pip install segment-anything
# basicsr and facexlib are required by ReActor's r_facelib (FaceRestoreHelper).
# They're not listed in ReActor's requirements.txt because the project ships
# vendored copies as r_basicsr / r_facelib — but those vendored copies still
# import the real basicsr/facexlib packages internally for some utilities.
# Without these, `from r_facelib.utils.face_restoration_helper import
# FaceRestoreHelper` in nodes.py raises ModuleNotFoundError at ComfyUI startup
# and ALL ReActor nodes fail to register.
RUN uv pip install basicsr facexlib
RUN cd /comfyui/custom_nodes/ComfyUI-ReActor && \
    uv pip install -r requirements.txt
# Run ReActor's own install.py — it does runtime detection (CUDA version, torch
# version) and installs a matching onnxruntime variant + does is_installed()
# verification with strict version checks. Without running this, ComfyUI-Manager
# normally invokes it after clone — we clone via Dockerfile so we must do it
# ourselves. install.py is idempotent.
RUN cd /comfyui/custom_nodes/ComfyUI-ReActor && \
    /comfyui/.venv/bin/python install.py || echo "[WARN] install.py exited non-zero — check verify dump below"

# Diagnostic dump — always succeeds (final `true` ensures exit 0). Shows us
# exactly which packages got installed and which (if any) fail to import,
# with full traceback. Build proceeds regardless so we get a working image
# even if one of these has a soft failure we can fix later at runtime.
RUN echo '=== [verify] python version ===' && \
    /comfyui/.venv/bin/python --version && \
    echo '=== [verify] relevant installed packages ===' && \
    /comfyui/.venv/bin/python -m pip list 2>&1 | grep -iE 'insight|onnx|torch|numpy|cython|opencv|albumen' || true ; \
    for mod in onnxruntime insightface cv2 torch numpy segment_anything basicsr facexlib albumentations; do \
        echo "=== [verify] importing $mod ==="; \
        /comfyui/.venv/bin/python -c "import $mod; print('  [OK] $mod', getattr($mod, '__version__', '?'))" || echo "  [FAIL] $mod import failed (traceback above)"; \
    done; \
    echo "=== [verify] simulating ComfyUI loading ReActor's nodes.py ===" && \
    cd /comfyui/custom_nodes/ComfyUI-ReActor && \
    /comfyui/.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from nodes import NODE_CLASS_MAPPINGS as M; print('  [OK] ReActor loaded with', len(M), 'nodes:', list(M.keys())[:6])" \
        || echo "  [FAIL] ReActor nodes.py import failed (TRACEBACK ABOVE — this is what ComfyUI sees too)"; \
    true

# Mirror handler runtime deps into /comfyui/.venv.
#
# The upstream line `RUN uv pip install runpod requests websocket-client`
# earlier in this Dockerfile installs into /opt/venv because PATH makes
# /opt/venv/bin/python the active python at that point. But our start.sh
# prepends /comfyui/.venv/bin to PATH at container boot (because torch /
# ComfyUI / Impact-Pack deps live in /comfyui/.venv). So when /handler.py
# runs `import runpod`, python = /comfyui/.venv/bin/python — and the /opt
# /venv copy is invisible. Install the same deps into /comfyui/.venv so
# the handler works regardless of which venv is "active" at runtime.
RUN VIRTUAL_ENV=/comfyui/.venv uv pip install runpod requests websocket-client

# Sanity check using /comfyui/.venv/bin/python (same Python ComfyUI runtime uses
# because comfy-cli sets up .venv as the active env for /comfyui/main.py).
RUN /comfyui/.venv/bin/python -c "\
import sys; print('python exec:', sys.executable); \
import numpy; print('numpy:', numpy.__version__); \
import torch; print('torch:', torch.__version__); \
import einops; print('einops:', einops.__version__); \
import ultralytics; print('ultralytics:', ultralytics.__version__); \
from ultralytics.nn.tasks import DetectionModel, SegmentationModel; \
from ultralytics.utils import IterableSimpleNamespace; \
from ultralytics.utils.tal import TaskAlignedAssigner; \
import ultralytics.nn.modules, ultralytics.nn.modules.block, ultralytics.utils.loss; \
import dill._dill; \
from numpy.core.multiarray import scalar; \
from numpy.dtypes import Float64DType; \
print('[build verify] OK in', sys.executable)"

# Assert Impact-Pack + Subpack directories physically exist in the image at build
# completion. If anything after our git clones wipes them (e.g. another RUN, a
# COPY, or a VOLUME mount shadowing the path), this test fails the build loudly
# instead of letting us debug 'Node not found' at runtime.
RUN test -f /comfyui/custom_nodes/ComfyUI-Impact-Pack/__init__.py || (echo '[FATAL] Impact-Pack missing at end of build' && ls -la /comfyui/custom_nodes/ && exit 1)
RUN test -f /comfyui/custom_nodes/ComfyUI-Impact-Subpack/__init__.py || (echo '[FATAL] Impact-Subpack missing at end of build' && ls -la /comfyui/custom_nodes/ && exit 1)
RUN test -f /comfyui/custom_nodes/ComfyUI-VideoHelperSuite/__init__.py || (echo '[FATAL] VideoHelperSuite missing at end of build' && ls -la /comfyui/custom_nodes/ && exit 1)
RUN test -f /comfyui/custom_nodes/ComfyUI-ReActor/__init__.py || (echo '[FATAL] ReActor missing at end of build' && ls -la /comfyui/custom_nodes/ && exit 1)
RUN echo '[build verify] /comfyui/custom_nodes at image creation:' && ls -la /comfyui/custom_nodes/

# Set the default command to run when starting the container
CMD ["/start.sh"]

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
# Set default model type if none is provided
# Default 'none' — our endpoint uses Realism Illustrious from the Network Volume,
# not any baked-in model. Downloading Flux-fp8 (16 GB) or SDXL base as part of
# the image wastes build time (+5-10 min wget) and pushes image over 24 GB which
# hits RunPod's 30-min build timeout on the final registry push.
# All the `if [ "$MODEL_TYPE" = "X" ]` checks below will evaluate false → no wget,
# models/ dir stays empty, volume-mounted models handle everything at runtime.
ARG MODEL_TYPE=none

# Change working directory to ComfyUI
WORKDIR /comfyui

# Create necessary directories upfront
RUN mkdir -p models/checkpoints models/vae models/unet models/clip models/text_encoders models/diffusion_models models/model_patches

# Download checkpoints/vae/unet/clip models to include in image based on model type
RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
      wget -q -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
      wget -q -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
      wget -q -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "sd3" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
      wget -q -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
      wget -q -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "flux1-dev-fp8" ]; then \
      wget -q -O models/checkpoints/flux1-dev-fp8.safetensors https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors; \
    fi

RUN if [ "$MODEL_TYPE" = "z-image-turbo" ]; then \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/text_encoders/qwen_3_4b.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/diffusion_models/z_image_turbo_bf16.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/diffusion_models/z_image_turbo_bf16.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors && \
      wget -q --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/Z-Image-Turbo-Fun-Controlnet-Union.safetensors; \
    fi

# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models