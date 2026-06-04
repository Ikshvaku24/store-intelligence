# Store Intelligence

Turn raw CCTV footage from a retail store into **live, queryable analytics** — the
same kind of funnel and conversion numbers an online store gets, but for a physical
shop. You point the system at video clips, it detects and tracks people, figures out
where they went and whether they bought anything, and serves it all through a small
REST API with a live dashboard.

**Live demo:** <https://ikshvaku24.github.io/store-intelligence/> — a snapshot of the
real analytics output for both stores (funnel, visitors, zone dwell, demographics,
anomalies). The full live system runs locally via `docker compose up`.

The **North Star** is one number: **offline conversion rate** = visitors who bought ÷
unique visitors in a time window. Everything in the system either makes that number
*more accurate* (the detection side) or *more useful* (the API side).

---

## 1. What it does

From a handful of CCTV clips per store, the system answers questions like:

| Business question | Where it's answered |
|---|---|
| How many customers came in, and how many bought? | `/metrics` → `unique_visitors`, `conversion_rate` |
| Where in the store are we losing people? | `/funnel` → drop-off % per stage |
| Which zones get attention but no sales? | `/heatmap` dwell vs `/funnel` billing stage |
| Is a billing queue building up right now? | `/anomalies` → `BILLING_QUEUE_SPIKE` |
| Is conversion worse than usual today? | `/anomalies` → `CONVERSION_DROP` |
| Who shops here, and do they come in groups? | `/metrics` → `demographics` (gender/age), `groups` |
| Is a camera feed stale / down? | `/health` → `STALE_FEED` |

It also handles the messy realities of real footage: people entering in **groups**
(counted individually), **staff** (excluded from customer metrics), **re-entry** (same
person returning is not a new visitor), **partial occlusion**, **empty-store periods**
(returns clean zeros, never crashes), and **camera overlap** (the same person seen by
two cameras is de-duplicated, not double-counted).

---

## 2. How it's built (architecture)

Two **independent** processes joined only by an event schema, plus two operator tools:

```
  ┌─ Process 1: DETECTION PIPELINE (heavy, offline, GPU-friendly) ─┐
  │  video → YOLO person-detect → ByteTrack → geometry (tripwire,  │
  │  zones) → staff/dedup logic → schema-validated events.jsonl    │
  └───────────────────────────────┬───────────────────────────────┘
                                   │  events.jsonl  (the only contract)
  ┌────────────────────────────────▼──────────────────────────────┐
  │  Process 2: INTELLIGENCE API (lightweight, containerised, no   │
  │  ML deps) — ingest → metrics / funnel / heatmap / anomalies /  │
  │  health, all computed at query time, store-agnostic.          │
  └───────────────────────────────────────────────────────────────┘

  Calibration Studio  → draw zones/tripwires in the browser, extract frames
  Live Dashboard      → one metric updating in real time as events replay
```

**Why split them?** The API is the graded, must-always-work part — keeping it free of
torch/OpenCV means `docker compose up` builds in seconds and never breaks on a CV
dependency. The heavy detection pipeline lives in its own image. Full reasoning is in
[docs/DESIGN.md](docs/DESIGN.md) and [docs/CHOICES.md](docs/CHOICES.md).

### Repository layout

```
app/                FastAPI service (Process 2, graded): models, normalize (schema
                    adapter), db, ingestion, metrics, funnel, heatmap, anomalies,
                    pos, health, logging middleware
pipeline/           Detection pipeline (Process 1): detect, reid, geometry, sessions,
                    staff, vlm_staff, dedup, groups, emit, run_pipeline, calibrate
  config/           Per-store calibration: store_1/, store_2/, _TEMPLATE/ (copy to
                    add a store), stores.json (registry). Each store folder holds
                    clips_manifest.json, zones.json, lines.json
studio/             Calibration Studio (browser zone editor + frame extractor)
dashboard/          replay.py (event streamer) + index.html (live counter)
tests/              pytest suite (API + pipeline) with AI prompt blocks
docs/               DESIGN.md, CHOICES.md, PRESENTATION.pdf
docker/             Dockerfile.api (slim, the gate) + Dockerfile.pipeline (heavy)
scripts/            generate_sample_data.py, make_demo_pos.py (illustrative POS),
                    build_store_layout.py, build_demo.py, make_presentation.py
data/               (git-ignored) videos, frames, events, POS, layout
```

---

## 3. Prerequisites

- **Docker Desktop** (running) — for the API and the detection pipeline.
- **Python 3.11+** — for the dashboard replay, the Studio, and tests.
- **NVIDIA GPU** *(optional but recommended)* — the pipeline auto-uses it; CPU works too (slower, use a larger `VID_STRIDE`).
- **Gemini API key** *(optional)* — only for the behavioural staff confirmer. Without it, staff detection falls back to position + heuristics.

One-time Python setup (for replay / Studio / tests):

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements-api.txt        # API + dashboard + tests
pip install -r requirements-studio.txt     # only if you'll use the Studio's "Extract frames"
```

---

## 4. Quick start — just the API (the acceptance gate)

This needs **no videos** and proves the graded surface works:

```bash
git clone <your-repo> && cd store-intelligence
cp .env.example .env                         # (Windows: copy .env.example .env)
docker compose up -d --build                 # Postgres + API + dashboard
curl http://localhost:8000/health
curl http://localhost:8000/stores/STORE_BLR_002/metrics
```

- API: <http://localhost:8000> (interactive docs at `/docs`) · Dashboard: <http://localhost:8080>
- Tables are created automatically. `data/pos_transactions.csv` is auto-loaded if present.
- With no events yet, metrics return clean zeros (valid JSON) — that's correct.

To see numbers move, generate stand-in events and replay them:

```bash
python scripts/generate_sample_data.py       # writes data/sample_events.jsonl etc.
python dashboard/replay.py --events data/sample_events.jsonl --api http://localhost:8000 --speed 50
```

---

## 5. Full end-to-end — from your own videos to live analytics

Follow these in order. You only need to **(a) drop in videos, (b) extract frames,
(c) draw zones, (d) run** — everything else is a copy-paste command.

### Step 1 — Add a store (one folder per store)

Every store is a **self-contained folder** under `pipeline/config/`. Two come
ready (`store_1`, `store_2`); the registry that lists them is
`pipeline/config/stores.json`. To add your own store, e.g. `store_3`:

```bash
cp -r pipeline/config/_TEMPLATE pipeline/config/store_3
mkdir "data/cctv_footage/Store 3"        # drop your clips here
```

Then:

1. **Put the videos** in your footage folder (any names you like).
2. **Edit `pipeline/config/store_3/clips_manifest.json`** — one object per camera clip.
   `_TEMPLATE/clips_manifest.json` is a fully-commented sample; the fields are:

   | Field | Meaning |
   |---|---|
   | `clip_path` | path to the video (relative to repo root) |
   | `store_id` | any string — the API keys metrics off this |
   | `camera_id` | a **unique** id you choose; it's the key used in `zones.json`/`lines.json` |
   | `role` | `ENTRY` (counts in/out via a tripwire) · `FLOOR` / `BILLING` (zone & queue events) · `BACKROOM` (staff oracle) |
   | `start_ts` | real recording time, ISO-8601 UTC (read the burnt-in clock); only affects absolute time + POS correlation |
   | `fps` | `null` = read from the video at runtime (recommended), or a number |

3. **Register it** — add a `store_3` entry to `pipeline/config/stores.json` (copy the
   `store_2` block and change the paths + `store_id`). It now shows up in the Studio.

> Only `ENTRY`-role cameras emit ENTRY/EXIT/REENTRY (they need a tripwire). `FLOOR`/
> `BILLING` cameras emit zone and queue events (they need polygons). One physical
> person seen by two cameras is de-duplicated, not double-counted.

### Step 2 — Extract frames to draw on (Studio)

Start the Studio:

```bash
.venv\Scripts\python.exe -m uvicorn studio.app:app --port 8090
# open http://localhost:8090
```

Pick a store → click a camera → click **⤵ Extract frames**. This decodes that
camera's video into still frames (at the video's native resolution, so your drawings
line up with what the pipeline sees) and drops them in a dropdown.

> Requires `pip install -r requirements-studio.txt` (OpenCV). If you'd rather use the
> CLI: `python data/cctv_footage/frames.py` style scripts also work.

### Step 3 — Draw zones and tripwires (Studio)

Zones are polygons you draw **once** on a frame; the pipeline then reports any person
whose feet fall inside them. With a frame loaded:

**Floor / billing cameras — one polygon per shelf/counter area.** Click around the
area (4–8 points is plenty), then fill the form and **Save**:

| Field | What to put | Example |
|---|---|---|
| `zone_id` | SHORT_UPPERCASE id, unique within the store (appears in every event) | `MAKEUP_LIPS`, `LEFT_WALL_SHELF`, `BILLING_QUEUE` |
| `zone_name` | human label shown in dashboards | `Makeup - Lips`, `Billing Counter Queue` |
| `zone_type` | `SHELF` (wall products) · `DISPLAY` (island/table) · `BILLING` · `CONSULT` (makeover seat) | `SHELF` |
| **revenue** | tick for product zones (where buying happens) | ticked for shelves/displays |
| **queue** | tick ONLY for the billing queue lane (customer side) | ticked for `BILLING_QUEUE` |
| **staff** | tick ONLY for behind-the-counter / backroom areas | ticked for `BILLING_STAFF` |

For billing draw **two** zones: `BILLING_QUEUE` (the lane customers stand in — tick
queue + revenue) and `BILLING_STAFF` (behind the counter — tick staff). Anyone whose
feet enter a **staff** zone is flagged staff and excluded from customer metrics.

**Entry cameras — draw the tripwire.** The Studio auto-switches to tripwire mode.
Click **p1** then **p2** as a line straight **across the doorway** (just inside the
glass so feet stay visible), then a **3rd point on the store-floor (inside) side** —
that point tells the counter which direction is "in". Save. Inbound crossing = ENTRY,
outbound = EXIT, a return after an EXIT = REENTRY (not a new visitor).

Everything you draw is overlaid live (red = staff, blue = queue, amber = shelf, green =
display) so you can check alignment, and is saved straight into that store's
`zones.json` / `lines.json`.

### Step 4 — Build the detection image (once)

```bash
docker compose -f docker-compose.pipeline.yml build pipeline
```

### Step 5 — Run the pipeline per store

```bash
# Store 1 -> data/events_store1.jsonl   (each store = its own config folder)
docker compose -f docker-compose.pipeline.yml run --rm \
  -e MANIFEST=pipeline/config/store_1/clips_manifest.json \
  -e ZONES=pipeline/config/store_1/zones.json \
  -e LINES=pipeline/config/store_1/lines.json \
  -e OUTPUT_PATH=data/events_store1.jsonl \
  -e VID_STRIDE=3 -e REID_DEDUP_THRESHOLD=0.82 \
  -e REID_DEDUP_LOW=0.7 -e VLM_DEDUP_CONF=0.8 -e VLM_DEDUP_MAX_PAIRS=80 \
-e VLM_MIN_INTERVAL_S=0.08 -e VLM_MAX_CALLS=0 pipeline

# Store 2 (its own config) -> data/events_store2.jsonl
docker compose -f docker-compose.pipeline.yml run --rm \
  -e MANIFEST=pipeline/config/store_2/clips_manifest.json \
  -e ZONES=pipeline/config/store_2/zones.json \
  -e LINES=pipeline/config/store_2/lines.json \
  -e OUTPUT_PATH=data/events_store2.jsonl \
  -e VID_STRIDE=3 -e REID_DEDUP_THRESHOLD=0.82 \
  -e REID_DEDUP_LOW=0.7 -e VLM_DEDUP_CONF=0.8 -e VLM_DEDUP_MAX_PAIRS=80 \
-e VLM_MIN_INTERVAL_S=0.08 -e VLM_MAX_CALLS=0 pipeline
```

**Watch the log** — it should show, per clip, a progress line every 200 frames, then
after all clips:

```
[reid] embedding backend: mobilenet_v3_small (device cuda)   ← appearance dedup is alive
...
Dedup VLM tie-breaker: 6 borderline merge(s) confirmed of 23 pair(s) asked (band [0.74, 0.82))
Dedup: 142 tokens -> 11 people (threshold 0.82); relabeled ... events
Staff resolved: 2 staff visitor(s) (by source: {'vlm': 1, 'position': 1}); ...
Demographics: stamped 9 visitor(s) (...); Groups: 2 group(s) over 5 visitor(s) (...)
Wrote 318 events to data/events_store1.jsonl
```

The same per-person Gemini call powers staff, demographics, and the dedup tie-breaker;
all three need a `GEMINI_API_KEY` in the container (`VLM=on` in the log) and degrade
gracefully without one. **Delete `data/vlm_staff_cache.json` before a re-run** so the
new demographics prompt is used (visitor ids are minted fresh each run, so the cache is
cold across runs anyway — this just guarantees it).

> **On Windows PowerShell**, replace the trailing `\` line-continuations with a backtick `` ` ``.

### Step 6 — Load events into the API and look at the results

```bash
# Merge both stores for the dashboard (PowerShell):
Get-Content data\events_store1.jsonl, data\events_store2.jsonl | Set-Content -Encoding utf8 data\events.jsonl
#   (bash:  cat data/events_store1.jsonl data/events_store2.jsonl > data/events.jsonl)

# Conversion needs a POS feed whose store_id matches the footage store. If you have a
# real one, drop it in data/pos_transactions.csv (store_id, transaction_id, timestamp,
# basket_value_inr  — or the hackathon line-item format, auto-detected). If not, derive
# an illustrative one from the billing visits you just detected (clearly a stand-in):
python scripts/make_demo_pos.py --convert 1.0    # -> data/pos_transactions.csv

docker compose down -v          # reset the DB so only this run counts (counts inflate otherwise)
docker compose up -d --build    # rebuilds the API → bakes the new POS + loads it at startup

python dashboard/replay.py --events data/events.jsonl --api http://localhost:8000 --speed 20

curl http://localhost:8000/stores/STORE_BLR_002/metrics
curl http://localhost:8000/stores/STORE_MUM_1076/metrics
```

`/metrics` returns the North-Star block plus the new-schema extras the pipeline now
emits: `conversion_rate` (POS-correlated, **bounded ≤ 1.0** — a purchaser is folded
into the unique base), `abandonment_rate` (POS-reconciled — a buyer is not also an
abandoner), `demographics` (coarse `gender`/`age_bucket` from the staff VLM call),
`groups` (co-arrival group sizes), and `avg_dwell_ms_by_zone`. Demographics/groups are
populated by the pipeline run; they stay empty if you ingest events that predate these
fields or run with `--no-vlm`.

Open the **dashboard** at <http://localhost:8080> while the replay runs to watch the
visitor and conversion counters update live.

### Optional — check the detection with `visualize.py`

To *see* whether detection, your zones, and the tripwire line up with reality, render
annotated frames (person boxes + track ids + zone polygons + ENTER/EXIT labels):

```bash
docker compose -f docker-compose.pipeline.yml run --rm --entrypoint python pipeline \
  -m pipeline.visualize \
  --manifest pipeline/config/store_1/clips_manifest.json \
  --zones pipeline/config/store_1/zones.json \
  --lines pipeline/config/store_1/lines.json \
  --out data/cctv_footage/check_store1 --interval 5
```

Open the JPGs in `data/cctv_footage/check_store1/<camera>/`. If a zone or tripwire sits
in the wrong place, re-draw it in the Studio and re-run. To instead see the **final
staff/dedup decisions** (red = staff, green = customer, deduped id), first run the
pipeline with `-e DEBUG_DUMP=/work/data/debug_tracks.jsonl`, then add
`--overlay data/debug_tracks.jsonl` to the command above (no model needed).

### Optional — (re)generate `store_layout.json`

`store_layout.json` (the API uses it for open-hours and the canonical zone list) is
**not** produced by the pipeline. After drawing zones, regenerate it from the configs:

```bash
python scripts/build_store_layout.py        # writes data/store_layout.json
```

It reads `pipeline/config/stores.json` + each store's `zones.json` / `clips_manifest.json`
(no model needed) and preserves any open-hours you set. If the file is absent the API
still runs (treats the store as always-open, infers zone names).

---

## 6. Configuration & tuning

All knobs are environment variables (pass with `-e` on the pipeline `run`, or in `.env`):

| Variable | What it does | Default |
|---|---|---|
| `VID_STRIDE` | Process every Nth frame (higher = faster, coarser) | `1` (use `3`–`8`) |
| `DEVICE` | `auto` / `cpu` / `0` (GPU index) | `auto` |
| `WEIGHTS` | YOLO weights (`yolo11m.pt`, `yolo11n.pt` for speed) | `yolo11m.pt` |
| `REID_DEDUP_THRESHOLD` | Auto-merge visitor tokens with embedding cosine ≥ this | `0.82` |
| `REID_DEDUP_LOW` | Floor of the **VLM-vetted** borderline band `[low, threshold)`; pairs here merge only if the VLM confirms same-person | `0.74` |
| `VLM_DEDUP_CONF` / `VLM_DEDUP_MAX_PAIRS` | Min confidence to confirm a borderline merge / cap on borderline pairs sent to the VLM | `0.8` / `80` |
| `VLM_STAFF_CONF` | Min VLM confidence to mark someone staff | `0.75` |
| `VLM_MIN_INTERVAL_S` / `VLM_MAX_CALLS` | Gemini rate limit (set `0.4`/`0` on a paid tier) | `6.5` / `50` |
| `GEMINI_MODEL` | VLM model id | `gemini-2.5-flash` |
| `--no-vlm` (flag) | Skip ALL VLM calls (staff + demographics + dedup tie-breaker) | off |

The same per-person Gemini call drives three things — staff classification, coarse
**demographics** (`gender`/`age_bucket`), and the **dedup tie-breaker** — so on a paid
tier set `VLM_MAX_CALLS=0` (unlimited) or staff calls can exhaust the budget before
dedup. With no `GEMINI_API_KEY` the pipeline still runs: staff falls back to
position+heuristic, demographics stays empty, and dedup is embedding-only.

Per-store config selection is via `MANIFEST` / `ZONES` / `LINES` / `OUTPUT_PATH`.

---

## 7. The Calibration Studio (detail)

A separate browser tool (not part of the graded API). It lets you:

- **Extract frames** from any camera's clip (native resolution).
- **Draw zone polygons** and **entry tripwires** on a frame, with live overlay of what's
  already drawn (colour-coded: red = staff, blue = queue, amber = shelf, green = display).
- **Delete** zones, switch frames, switch cameras/stores.

It reads and writes the exact `pipeline/config/...` files the pipeline runs on, so what
you draw is what gets used. Run the pipeline itself from the terminal (Step 5).

```bash
.venv\Scripts\python.exe -m uvicorn studio.app:app --port 8090   # http://localhost:8090
```

---

## 8. API endpoints

| Method | Path | Returns |
|---|---|---|
| `POST` | `/events/ingest` | Batch ingest (≤500), per-event validation, **idempotent**, partial success |
| `GET` | `/stores/{id}/metrics` | unique visitors, conversion, dwell-by-zone, queue depth, abandonment, demographics, groups |
| `GET` | `/stores/{id}/funnel` | ENTRY → ZONE_VISIT → BILLING_QUEUE → PURCHASE (deduped on visitor) |
| `GET` | `/stores/{id}/heatmap` | per-zone visit_count, avg_dwell_ms, 0–100 score |
| `GET` | `/stores/{id}/anomalies` | queue spike, conversion drop, dead zone (INFO/WARN/CRITICAL + suggested_action) |
| `GET` | `/health` | DB connectivity + per-store last-event lag / `STALE_FEED` |

The API accepts **both** the original `StoreEvent` schema *and* the newer multi-shape
event schema (entry/zone/queue with demographics) — a normalization adapter
(`app/normalize.py`) folds them into one canonical row, so it works against held-out
events for any store. All endpoints exclude staff, return explicit zeros on zero
traffic, and are computed at query time.

---

## 9. Tests

```bash
pip install -r requirements-api.txt
pytest                              # ephemeral SQLite, no server needed
```

**69 tests, ~85% coverage** of the API + pure-Python pipeline core. Heavy CV modules
(detect/reid/calibrate) need torch/OpenCV and are exercised via the pipeline image
(omitted from coverage in `.coveragerc`). Each test file opens with a `# PROMPT:` /
`# CHANGES MADE:` block documenting the AI assistance (Part D).

---

## 10. Troubleshooting & edge cases

| Symptom | Cause & fix |
|---|---|
| **`events_store1/2.jsonl` not created** | You appended `python -m pipeline.run_pipeline …` after `pipeline`. The image entrypoint is `run.sh`, which selects config via **env vars** — use the Step 5 commands (`-e MANIFEST=… -e OUTPUT_PATH=…`), don't append a python command. |
| **Run looks "stuck" after the clips** | It's the post-run staff phase making throttled Gemini calls (CPU near 0% = waiting on the network). Watch for `[vlm] call N` ticks. On a paid tier add `-e VLM_MIN_INTERVAL_S=0.4 -e VLM_MAX_CALLS=0`, or `--no-vlm` to skip. |
| **No `Dedup: N -> M` line / too many unique visitors** | Appearance dedup needs embeddings. Confirm the log shows `[reid] embedding backend: mobilenet_v3_small`. If it says `backend: none`, torch/torchvision didn't load in the image (rebuild). Tune `REID_DEDUP_THRESHOLD` — lower (0.75) merges more, higher (0.90) merges less. With a key, the VLM also vets the `[REID_DEDUP_LOW, threshold)` band (`Dedup VLM tie-breaker: K of N` in the log). If unique drops *too* low (over-merge), raise `VLM_DEDUP_CONF` (0.85–0.9) or lower `VLM_DEDUP_MAX_PAIRS`. |
| **Too many / too few staff** | Raise `VLM_STAFF_CONF` (e.g. 0.85) to flag fewer; lower (0.65) to flag more. `VLM=off` in the log means the Gemini key didn't reach the container — check `.env`. |
| **`conversion_rate` is 0** | POS correlation matches on **store_id + time window**. The `store_id` in `pos_transactions.csv` must match the store you're querying, and timestamps must be on the same clock as the events. The hackathon-provided POS is for a store with no footage → 0 (correct, not a bug); run `python scripts/make_demo_pos.py` to derive an illustrative feed for the footage store, then `docker compose up -d --build` (the API bakes + loads POS at startup). |
| **`demographics` / `groups` are empty** | These come from the pipeline run: demographics needs the VLM (`VLM=on`; runs degrade to empty without a key), groups need ≥2 people co-arriving. Events ingested from before this feature won't have them — re-run the pipeline, clearing `data/vlm_staff_cache.json` first so the new demographics prompt is used. |
| **429 / rate limit from Gemini** | Free tier is ~10 req/min. Keep the default throttle, or upgrade and set `VLM_MIN_INTERVAL_S=0.4 VLM_MAX_CALLS=0`. |
| **Studio "Extract" returns 501** | `pip install -r requirements-studio.txt` (OpenCV missing in the studio env). |
| **Zones don't line up / no zone events** | Draw on frames extracted by the Studio (native resolution). Frames from another tool at a different resolution will be offset. |
| **`docker compose up` fails** | Make sure Docker Desktop is running and ports 8000/8080/5432 are free. Re-run `docker compose down -v` then `up -d --build`. |
| **Counts inflate across replays** | The DB accumulates. Always `docker compose down -v` before replaying a fresh run. |
| **No GPU** | `-e DEVICE=cpu -e VID_STRIDE=10 -e WEIGHTS=yolo11n.pt` (slower but works). |

---

## 11. Honest limitations

- Cross-camera identity is **best-effort** (blurred faces, similar clothing) — it's
  flagged with a confidence, never used to gate the headline visitor count.
- The funnel is an **aggregate** unique-count funnel, not a per-individual trace.
- Queue depth in a tight billing area is reported as a flagged trend, not an exact head-count.
- Conversion needs a POS feed whose `store_id` matches the footage. The provided feed is
  for a non-footage store, so `scripts/make_demo_pos.py` derives an **illustrative**
  stand-in from detected billing visits — it demonstrates the real correlation
  mechanism, it is not ground-truth sales.
- **Demographics** (`gender`/`age_bucket`) are a **coarse, best-effort** VLM guess on
  blurred footage — the model returns null when unsure and we never force a guess.
- **Groups** are inferred from co-arrival + shared-camera overlap (size 2–4); a
  floor-wide cluster is dropped as noise, so the count is conservative.
- The **VLM dedup tie-breaker** only vets the borderline embedding band and can still
  mis-merge two similarly-dressed people on this footage; it is capped, flagged, and
  never overrides the provably-different (same-camera-overlap) block.

See [docs/DESIGN.md](docs/DESIGN.md) §7 & §10 and [docs/CHOICES.md](docs/CHOICES.md)
Decisions 4–5 for the full reasoning and trade-offs.
