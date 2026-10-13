#!/bin/bash
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
# Installs the systemd --user units for the studio and ComfyUI (paths filled in from studio.env). Safe to re-run.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO/studio.env" ] || { echo "copy studio.env.example to studio.env first"; exit 1; }
set -a; . "$REPO/studio.env"; set +a
COMFY_PY="${COMFY_PY:-$HOME/comfyui-env/bin/python}"
mkdir -p ~/.config/systemd/user
sed "s#@REPO@#$REPO#g" "$REPO/systemd/lion-studio.service" > ~/.config/systemd/user/lion-studio.service
sed -e "s#@COMFY_DIR@#$COMFY_DIR#g" -e "s#@COMFY_PY@#$COMFY_PY#g" "$REPO/systemd/comfyui.service" > ~/.config/systemd/user/comfyui.service
systemctl --user daemon-reload
systemctl --user enable --now lion-studio.service
echo "Lion AI Studios: http://$STUDIO_HOST:$STUDIO_PORT  (ComfyUI starts on the first render)"
echo "Keep services running after logout: sudo loginctl enable-linger $USER"
