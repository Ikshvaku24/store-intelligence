# DESIGN.md — Store Intelligence

## 1. Dataset reality (two stores)

> **Round-2 update:** the dataset now spans **two stores**; §9 documents the
> multi-store onboarding, the new event schema, and the fixes that followed. This
> section describes the primary store (`STORE_BLR_002`) the design was built on.

`STORE_BLR_002` is several **camera angles of one store**, not separate stores: every
clip's wall-clock timestamps fall in the same short window and the cameras are
functional complements of one floor plan (a door, two product walls, a billing
counter). Camera roles:

| Role | Camera | Responsibility |
|---|---|---|
| ENTRY | cam_3 | **Sole emitter** of ENTRY / EXIT / REENTRY |
| FLOOR_SKINCARE | cam_1 | `ZONE_*` for skincare wall + fragrance table |
| FLOOR_MAKEUP | cam_2 | `ZONE_*` for makeup wall; vanity = consult zone |
| BILLING | cam_5 | Queue join/abandon; behind-counter = staff zone |

(The Round-2 reorganisation dropped Store 1's backroom camera; **Store 2** has two
entry angles + a floor + a billing camera, recorded at different times.) Making
**entry counting single-camera** removes cross-camera double-counting *by
construction* instead of by solving a hard re-identification problem. The system is
store-agnostic — the API keys off `store_id` (normalising equivalent ids) and the
pipeline processes whatever each store's manifest lists, so a new store is just a new
config folder, no code change.

## 2. The two-process split

The single most important architectural decision is that the system is **two
independent processes joined only by the event schema**:

* **Process 1 — detection pipeline** (`pipeline/`): heavy, offline, GPU-friendly.
  Video → YOLO person detection → ByteTrack → geometry/session logic →
  schema-validated `events.jsonl`.
* **Process 2 — intelligence API** (`app/`): lightweight, graded, **no torch**.
  Ingests events → query-time metrics/funnel/heatmap/anomalies.

Why it is non-negotiable: the graded API image builds in seconds with no ML
dependencies, so `docker compose up` (the acceptance gate) is robust; the API is
**validated against `sample_events.jsonl` without running any video**; and the
same `events.jsonl` powers both batch scoring and the live dashboard via
accelerated replay. The deviation from the suggested single-image layout (two
Dockerfiles, `pipeline/config/`) is deliberate and is what protects the gate.

## 3. Identity model — confidence as a first-class signal

Identity is tiered by reliability, and the tier is **carried in the event** via
two additive metadata keys (`id_source`, `id_confidence`) so the API can surface,
never hide, weak signals:

* **within-camera** (solid): a ByteTrack track id maps 1:1 to a `visitor_id`.
* **re-entry** (medium): at the ENTRY camera only, a returning person is matched
  against a recently-exited gallery using a CNN appearance embedding (torchvision
  **MobileNetV3-small**) **and** an HSV histogram (two signals reject similar-clothing
  collisions). Same `visitor_id`, emit `REENTRY`.
* **cross-camera** (low, flagged): best-effort embedding match within a ±5s
  synchronized window. It is deliberately reported at low confidence and **never
  gates entry counts** — because entry counting is single-camera, a cross-camera
  mismatch can only mis-attach a zone visit, which degrades gracefully to
  aggregate stage counts.

`visitor_id` is resolved as a **stable physical-person token within the clip
window** (see CHOICES.md): a re-entry reuses it, so unique-visitor counts and the
funnel dedup on `visitor_id` and a person who leaves and returns is counted once.

## 4. Geometry-first zones and staff

Calibration replaces training data. Zone polygons and the entry tripwire are
hand-drawn **on the distorted frame** (via the browser Calibration Studio), so no
lens calibration is needed and the most reliable signal available (position) is
deterministic and explainable. Staff detection is a **cascade**: position oracle
(backroom / behind-counter polygons — near-perfect where it applies) → a **Gemini
VLM behavioural confirmer** for the ambiguous floor cases position can't see (a
consultant applying makeup, operating the POS), gated at high confidence and biased
to "customer" → a persistence/serve-many heuristic fallback. The VLM is used because
on this footage position alone misses floor staff; see §8 and CHOICES.md for the
override. After the run, appearance dedup (MobileNetV3 embeddings) collapses the
per-camera/fragmented tokens of one person. The API filters `is_staff=true` out of
every customer metric.

## 5. Conversion without identity

Conversion — the North Star — is computed by **POS time-window correlation**:
a visitor present in the billing zone within a 5-minute window *before* a POS
transaction is counted as converted. There is no `customer_id`, so correlation is
by store + time only. This is the most robust metric on a tiny dataset because it
needs no cross-camera identity. (One refinement: the POS fetch window is extended
past the event window's end by the correlation window, since a purchase completes
*after* the customer's last detected billing event.)

## 6. Production concerns

Structured JSON logs (one line per request: `trace_id`, `store_id`, `endpoint`,
`latency_ms`, `event_count`, `status_code`); idempotent ingest via dialect-aware
`ON CONFLICT DO NOTHING` on `event_id`; graceful degradation (DB errors → HTTP 503
with a structured body, a catch-all handler so no stack trace ever reaches a
client); and an accurate `/health` that reports real DB connectivity and per-store
last-event lag with a `stale_feed` flag. Storage is Postgres (native
conflict-ignore + a credible scaling story); the same SQLAlchemy Core code runs on
SQLite for the test suite, which is how 88% coverage is achieved without a server.

## 7. Honest limits

* **Cross-camera Re-ID** is unreliable here (blurred faces, black-on-black,
  barrel distortion); it is flagged, not asserted, and never gates counts.
* **The funnel is aggregate**, not a per-individual trace — entry-anchored unique
  counts per stage — because reliable cross-camera linking is not achievable.
* **Conversion-drop** has no real baseline on a 90-second clip; it returns
  `baseline: "insufficient_history"` rather than fabricating a trend.
* **Dead-zone** uses a 5-min window for the clips (production: 30 min) and only
  fires during open hours from `store_layout.json`.
* **Queue depth** is noisy in the tight billing space; it is reported as a flagged
  trend, not an exact headcount.
* **Tight-group entry** can under-count under mutual occlusion at the blurred
  door; we lower the confidence floor near the line and flag, never fabricate.

## 8. AI-Assisted Decisions

Three decisions where an LLM was consulted and explicitly agreed with or overridden:

* **RT-DETR vs YOLOv8m/YOLO11m on crowded billing/group frames.** The AI's first
  instinct was "use the lightest model for speed." We **overrode the framing**:
  the pipeline is offline, so speed is not the gate — occlusion recall on touching
  bodies is. We kept YOLO11m as the default and recorded an RT-DETR A/B for the
  crowded billing clip (NMS-free set prediction separates adjacent bodies); the
  detector choice is pluggable via the `--weights` flag so the A/B is reproducible.

* **VLM-for-staff, overridden in favour of position rules.** We evaluated using a
  vision-language model to classify staff from behind-counter crops. The footage
  justified rejecting it: **there is no uniform**, so appearance carries little
  signal, while the backroom camera is a near-perfect positional oracle. We kept a
  VLM only as an optional confirmer on genuinely ambiguous crops, capped at one
  cached call per `visitor_id`, with the geometry cascade as the always-available
  fallback. Geometry beat the VLM on this data; a reliable badge/uniform would flip
  that verdict.

* **Reconciling the ambiguous `visitor_id` semantics.** The schema comment
  ("unique per visit session") conflicts with the REENTRY rule ("same id after a
  prior EXIT"). The LLM initially proposed minting a fresh id per session. We
  **overrode it** to a stable physical-person token within the clip window,
  because the brief's REENTRY definition and the unique-visitor metric both require
  that a returning person is *not* re-counted. This is documented loudly in
  CHOICES.md because it is a near-certain follow-up question.

## 9. Adapting to the new multi-shape dataset (Round-2 update)

A revised dataset arrived mid-build: a real `sample_events.jsonl`, a real
`pos_transactions.csv`, a **second store** with a different camera topology, and
**store-layout planograms**. The new event stream is **three differently-shaped
events with inconsistent field names** — entry/exit (`id_token`, `store_code`,
`event_timestamp`, plus demographics `gender_pred`/`age_pred`/`age_bucket` and
groups `group_id`/`group_size`), zone_entered/exited (`track_id`, `store_id`,
`event_time`, `zone_type`/`is_revenue_zone`/`zone_hotspot`), and
queue_completed/abandoned (`queue_join_ts`/`served_ts`/`exit_ts`, `wait_seconds`,
`queue_position_at_join`, `abandoned`). The old `StoreEvent` rejected **100%** of it.

* **A normalization adapter at ingest, not a rewrite.** `app/normalize.py` folds
  each wire shape onto the existing canonical row, so the metrics/funnel/heatmap
  layer is untouched. event_type is mapped to the canonical vocabulary;
  per-camera `track_id`s are namespaced `T{n}@{camera}` (the source does **not**
  unify identity across cameras — entry `id_token` vs zone/queue `track_id` — which
  *vindicates* our aggregate-funnel stance from §3/§5); `store_code` and `store_id`
  are folded to one canonical id (`store_1076`↔`ST1076`), applied symmetrically on
  read; an `event_id` is synthesized deterministically for idempotency; and the
  rich extras land in `metadata`. Already-canonical events pass through untouched.

* **Demographics & groups are a pass-through win.** Because the graders *supply*
  gender/age/group on their events, the high-value move was to **aggregate** them
  (gender & age-bucket splits, group count / size) rather than build an age-gender
  model. These are attributed over the unique-visitor base, never per-camera token,
  so identity non-unification can't inflate them.

* **Dwell without `dwell_ms`.** The new zone events carry no duration, so dwell is
  derived by **pairing each `zone_exited` with the matching prior `zone_entered`**
  on `(visitor_id, zone_id)` — a superset that still honours an explicit `dwell_ms`
  when present.

* **Per-store config, zero new code.** The pipeline already takes
  `--manifest/--zones/--lines`, so each store is just its own config dir
  (`pipeline/config/store_2/`). Store 2 has two entry cameras recorded at different
  times (independent windows → no double-count) and **no backroom camera**, so its
  staff signal is billing-counter position + the VLM behavioural confirmer. fps is
  now read from the video at runtime (manifest value optional), removing the
  hand-measured-fps error class. Zones are drawn from the **layout planograms**,
  carrying `zone_type`/`is_revenue_zone` into events to match the graders' schema.

## 10. Closing the loop on the North Star (conversion, demographics, groups)

The detection pipeline and the API meet at the `StoreEvent` schema, but a few
query-time metrics were only as good as the fields the pipeline emitted. Three
additions close that gap:

* **Conversion is now bounded and demonstrable.** Conversion correlates billing-zone
  presence to POS transactions by time window (no identity). Because the provided POS
  feed is for a store with no footage, `scripts/make_demo_pos.py` generates an
  *illustrative* feed for the footage store from the billing presences we detected, so
  the mechanism is shown end-to-end. The denominator now includes POS-confirmed buyers
  (a purchaser is a visitor), so the rate can never exceed 100% even when entry-basis
  undercounts the floor; and a visitor who purchased is no longer also counted as a
  queue abandoner.

* **Demographics ride the staff VLM call.** Staff resolution already issues one Gemini
  vision call per person. That call now also returns a coarse `gender`/`age_bucket`
  (best-effort — faces are blurred, so `null`/`U` is allowed and never forced), mapped
  through the dedup relabel onto the canonical visitor and stamped into event metadata.
  The API's `demographics` block reads those keys over the unique-visitor base, so it
  can't double-count a person across their per-camera tracks.

* **Groups from co-arrival.** `pipeline/groups.py` forms a group only from non-staff
  visitors first seen within seconds of each other **and** overlapping in dwell on a
  shared camera (size 2–4; floor-wide clusters dropped as noise). It runs purely on
  event timestamps — no extra model — and degrades to *no* groups rather than a fake
  giant one.

All three are approximate on this footage and surfaced as such; the design choice is
to populate and **flag** them rather than leave the blocks empty or print an
unjustified number. Each is additive — the schema, the six endpoints, and the
acceptance gate are unchanged.
