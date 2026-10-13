#!/bin/bash
# The repository's checks (CI runs the same script): every Python file compiles, the page's inline scripts parse, and no
# tracked file carries a home-directory path, a private network address or something shaped like a Hugging Face token.
# Needs python3 and node. Run from anywhere: bash scripts/check.sh
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
fail=0
tmp=$(mktemp -d); trap 'rm -r "$tmp"' EXIT

echo "== python: py_compile"
n=0
while IFS= read -r f; do
  n=$((n + 1)); PYTHONPYCACHEPREFIX="$tmp/pyc" python3 -m py_compile "$f" || { echo "FAIL $f"; fail=1; }
done < <(git ls-files '*.py')
echo "$n files"

echo "== page: node --check on every inline <script> of app/index.html"
python3 - "$tmp" <<'PY' || fail=1
import pathlib, re, sys
page = pathlib.Path("app/index.html").read_text()
blocks = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", page, re.S)
if not blocks: sys.exit("no inline <script> blocks found")
for i, b in enumerate(blocks): pathlib.Path(sys.argv[1], f"block{i}.js").write_text(b)
print(f"{len(blocks)} blocks")
PY
for f in "$tmp"/block*.js; do node --check "$f" || { echo "FAIL $(basename "$f")"; fail=1; }; done

echo "== hygiene: home paths, private addresses, token-shaped strings"
hits=$(git grep -I -n -E '/home/[a-z]|192\.168\.[0-9]|\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b|\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}\b|hf_[A-Za-z0-9]{20,}' -- . | cut -c1-200)
if [ -n "$hits" ]; then echo "$hits"; echo "FAIL: use \$HOME, ~ or a studio.env setting instead; never commit addresses or tokens"; fail=1; else echo "clean"; fi

[ "$fail" = 0 ] && echo "ALL CHECKS PASSED" || echo "CHECKS FAILED"
exit "$fail"
