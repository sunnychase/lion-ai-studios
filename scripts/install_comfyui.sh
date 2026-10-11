#!/bin/bash
# Installs ComfyUI at the commit Lion AI Studios was tested with, in its own venv, plus the studio's image node.
# Optional TensorFold engine: see tensorfold/README.md. Safe to re-run.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO/studio.env" ] && { set -a; . "$REPO/studio.env"; set +a; }
COMFY_DIR="${COMFY_DIR:-$HOME/ComfyUI}"; VENV="${COMFY_VENV:-$HOME/comfyui-env}"
COMMIT=b26625f2                      # ComfyUI v0.39.0 + 18 (MiniMax H3, Qwen-Image 2.1, MiniMax Music 3 nodes are built in)
[ -d "$COMFY_DIR/.git" ] || git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
git -C "$COMFY_DIR" fetch -q origin && git -C "$COMFY_DIR" checkout -q "$COMMIT"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q torch==2.10.0 torchvision torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
"$VENV/bin/pip" install -q -r "$COMFY_DIR/requirements.txt"
cp "$REPO/comfyui_nodes/lion_studio_image_gen.py" "$COMFY_DIR/custom_nodes/"
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(); print('ComfyUI env OK:', torch.__version__, torch.cuda.get_device_name(0))"
echo "ComfyUI ready at $COMFY_DIR (venv $VENV). Set COMFY_PY=$VENV/bin/python when running install_services.sh if you changed it."
