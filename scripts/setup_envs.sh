#!/bin/bash
# Created by Sunny Chase (https://github.com/sunnychase) in collaboration with Claude (Anthropic).
# Creates the studio's two helper Python environments under $STUDIO_HOME/envs (voice: ~6 GB with torch; midi: small)
# and the studio's folders. The web app itself needs only the system python3 (standard library). Safe to re-run.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$REPO/studio.env" ] && { set -a; . "$REPO/studio.env"; set +a; }
H="${STUDIO_HOME:-$HOME/lion-studio}"
mkdir -p "$H"/{library,state,uploads,voices,models/soundfonts,envs}
command -v ffmpeg >/dev/null || { echo "install ffmpeg first: sudo apt install ffmpeg"; exit 1; }

echo "== midi env"
python3 -m venv "$H/envs/midi"
"$H/envs/midi/bin/pip" install -q -r "$REPO/requirements/midi.txt"

echo "== voice env (PyTorch for CUDA 13, aarch64)"
python3 -m venv "$H/envs/tts"
"$H/envs/tts/bin/pip" install -q torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
"$H/envs/tts/bin/pip" install -q -r "$REPO/requirements/tts.txt"
"$H/envs/tts/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('voice env OK:', torch.__version__, torch.cuda.get_device_name(0))"
echo "done — models: python3 scripts/download_models.py --group voice --file MuseScore_General.sf2"
