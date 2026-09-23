#!/usr/bin/env bash
# Package code + cached games as a private Kaggle dataset for notebooks/experiments.ipynb.
# Usage: bash scripts/package_kaggle_dataset.sh [create|version]
set -euo pipefail
MODE="${1:-}"
OWNER=$(python3 -c "import json;print(json.load(open('notebooks/kernel-metadata.json'))['id'].split('/')[0])")
DIST=dist/arcrl-code
[[ -d environment_files ]] || { echo "run scripts/download_games.py first"; exit 1; }
rm -rf "$DIST" && mkdir -p "$DIST"
zip -qr "$DIST/repo.zip" agent configs scripts src tests pyproject.toml requirements.txt Makefile \
    -x '*/__pycache__/*'
zip -qr "$DIST/environment_files.zip" environment_files
cat > "$DIST/dataset-metadata.json" <<JSON
{"title": "arcrl-code", "id": "$OWNER/arcrl-code", "licenses": [{"name": "other"}]}
JSON
echo "packaged $(du -sh "$DIST" | cut -f1) into $DIST (commit $(git rev-parse --short HEAD 2>/dev/null || echo no-git))"
KAGGLE="kaggle"
[[ -s .kaggle/access_token ]] && KAGGLE="env KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/kaggle"
case "$MODE" in
  create)  $KAGGLE datasets create -p "$DIST" ;;
  version) $KAGGLE datasets version -p "$DIST" -m "code $(git rev-parse --short HEAD 2>/dev/null || date +%F)" ;;
esac
