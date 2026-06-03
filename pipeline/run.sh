#!/usr/bin/env bash
# One command: every clip in the manifest -> events.jsonl (BUILD_SPEC Section 4).
#
#   bash pipeline/run.sh                       # Store 1 (default config)
#   MANIFEST=pipeline/config/store_2/clips_manifest.json \
#   ZONES=pipeline/config/store_2/zones.json \
#   LINES=pipeline/config/store_2/lines.json \
#   OUTPUT_PATH=data/events_store2.jsonl bash pipeline/run.sh    # any store
#
# Per-store config is selected by env var so `docker compose run -e MANIFEST=...`
# works without overriding the entrypoint. Any extra CLI args are forwarded too.
#
# Env:
#   MANIFEST/ZONES/LINES  per-store config (default = Store 1 top-level config)
#   OUTPUT_PATH           where events.jsonl is written (default data/events.jsonl)
#   WEIGHTS               YOLO weights (default yolo11m.pt; downloaded on first run)
set -euo pipefail

cd "$(dirname "$0")/.."

MANIFEST="${MANIFEST:-pipeline/config/store_1/clips_manifest.json}"
ZONES="${ZONES:-pipeline/config/store_1/zones.json}"
LINES="${LINES:-pipeline/config/store_1/lines.json}"
OUTPUT_PATH="${OUTPUT_PATH:-data/events.jsonl}"
WEIGHTS="${WEIGHTS:-yolo11m.pt}"
VID_STRIDE="${VID_STRIDE:-1}"
DEVICE="${DEVICE:-auto}"

echo "Running detection pipeline: manifest=${MANIFEST} -> ${OUTPUT_PATH} (weights=${WEIGHTS}, stride=${VID_STRIDE}, device=${DEVICE})"
python -m pipeline.run_pipeline \
  --manifest "${MANIFEST}" \
  --zones "${ZONES}" \
  --lines "${LINES}" \
  --output "${OUTPUT_PATH}" \
  --weights "${WEIGHTS}" \
  --stride "${VID_STRIDE}" \
  --device "${DEVICE}" \
  "$@"

echo "Done. Ingest into the API with:  python dashboard/replay.py --speed 20"
