# CHOICES.md — Three decisions, with the reasoning

Each decision lists the options considered, what the AI suggested, and what we
chose and why.

---

## Decision 1 — Detection model & the staff-classifier question

**Options considered:** YOLOv8n/s/m, YOLO11m, YOLOv9, RT-DETR, MediaPipe pose;
and separately, a VLM vs geometry for staff classification.

**What the AI suggested:** the lightest model that hits real-time (YOLOv8n/s),
and a VLM to read "is this person behind the counter a cashier?" from crops.

**What we chose and why.** We use **YOLO11m (COCO `person`) + ByteTrack** as the
default and treat RT-DETR as an **A/B candidate for the crowded billing and group
frames**. The AI optimised for the wrong constraint: the detection pipeline is
**offline**, so inference speed is not the gate — what matters is **recall under
occlusion** (touching bodies at the makeup wall and door). A heavier model is
affordable, and ByteTrack's low-score second association is exactly the mechanism
that rescues motion-blurred, low-confidence boxes through short occlusions, so we
run a deliberately low confidence floor (~0.20) and an enlarged track buffer. The
detector is pluggable (`--weights`), which makes the RT-DETR A/B a one-flag change
rather than a rewrite — RT-DETR's NMS-free set prediction is worth measuring
precisely where boxes overlap.

On staff, the decision **evolved — and that evolution is the answer.** We first
**overrode** the AI's VLM suggestion for a position-first cascade: where a backroom
or behind-counter polygon applies, position is near-perfect and needs no model.
Running it on the real footage then exposed the gap the AI had flagged: the backroom
camera is often empty and the consultant works the **floor** (applying makeup,
operating the POS) where position is silent. So we **adopted** a Gemini VLM
behavioural confirmer as a genuine cascade stage — position → VLM → heuristic — with
the exact prompt in `pipeline/vlm_staff.py`. Honest evaluation: the VLM's first
version **over-flagged**, calling too many ambiguous shoppers "staff", so we made it
conservative — a high confidence gate (`VLM_STAFF_CONF`, default 0.75) and a prompt
that **defaults to CUSTOMER**, demands clear repeated role evidence, and uses the
uniform cue the brief mentions. Net: geometry is the trustworthy backbone; the VLM
earns its place only on the floor cases geometry cannot see, and only when confident.
We also moved appearance embeddings from OSNet/torchreid (which silently fails on
torch 2.x) to **torchvision MobileNetV3**, which is what makes the visitor dedup
actually fire.

---

## Decision 2 — Event schema: flat `StoreEvent`, `event_id` as PK, identity made visible

**Options considered:** a normalized multi-table schema (visitors, sessions,
zone-visits) vs a single flat event row; whether to add fields beyond the brief's
schema; and how to represent unreliable identity.

**What the AI suggested:** a normalized relational model with a `visitors` table
and foreign keys, and dropping low-confidence detections to "keep the data clean."

**What we chose and why.** A **flat `StoreEvent`** with `event_id` (UUID) as the
**primary key and idempotency key**, stored once with `ON CONFLICT DO NOTHING`.
A normalized model would have coupled the graded API to identity decisions made in
the pipeline — exactly the fragile dependency the two-process split exists to
avoid. Flat events keep ingestion idempotent and the API store-agnostic, and
query-time aggregation over a tiny dataset is simple and portable.

We **rejected "drop low-confidence detections"** outright: confidence is a
first-class signal that is **flagged, never suppressed** (`confidence` is in every
event and is *never* used to filter). We then **added two optional metadata keys**
the brief's schema does not mandate — `id_source`
(`within_camera`/`reentry_match`/`cross_camera_match`) and `id_confidence` — so the
API can down-weight or flag unreliable cross-camera identity **without breaking the
required schema** (extra optional metadata keys are permitted; all required keys
remain present and correctly typed). We also resolved the ambiguous `visitor_id`
semantics here: it is a **stable physical-person token within the clip window**, so
a re-entry reuses it and unique-visitor counts dedup correctly. Surfacing
uncertainty instead of hiding it is what makes the funnel's "what's approximate
about this?" answer honest.

---

## Decision 3 — Single-camera authoritative entry counting + an aggregate funnel

**Options considered:** (a) fleet-wide cross-camera Re-ID that builds one global
identity per person and traces them stage-by-stage through the store, vs
(b) single-camera authoritative entry counting with an entry-anchored **aggregate**
funnel.

**What the AI suggested:** the rigorous option — global cross-camera Re-ID linking
every appearance into one track, producing a true per-individual funnel.

**What we chose and why.** We chose **(b)**. The "rigorous" option is the wrong bet
on *this* footage: blurred (anonymised) faces make face-Re-ID impossible,
black-on-black clothing collapses appearance embeddings, and barrel distortion plus
per-camera lighting wreck cross-camera matching. A global-Re-ID funnel built on
that foundation would be confidently wrong, and worse, a false cross-camera match
could **corrupt the headline visitor count**. Instead, the **ENTRY camera is the
sole authority** for ENTRY/EXIT, which makes cross-camera double-counting
impossible by construction. The funnel is then **entry-anchored and aggregate**:
unique non-staff entries → unique visitors with a zone visit → unique visitors in a
billing queue → POS purchases, each deduped on `visitor_id`. High-confidence
cross-camera links attach zone visits to entry sessions when available; otherwise
stages fall back to aggregate unique counts. This is accurate on the clips and
**degrades gracefully** exactly where the data is weak. The trade-off is explicit:
we trade a per-individual trace (which we cannot produce reliably) for an aggregate
funnel (which we can), and we say so. A full fix needs per-camera homography and a
controlled identity model — a "more data / more time" item, and the honest answer
to "at 40 live stores, what breaks first?" (the GPU-bound detection layer
saturates first; the query-time API and Postgres scale comfortably).

## Decision 4 — Absorbing the new schema with an adapter, not a rewrite

**Context.** Round 2 shipped a new authoritative dataset whose events come in
**three shapes with inconsistent field names** (entry uses `id_token`/`store_code`/
`event_timestamp`; zone uses `track_id`/`store_id`/`event_time`; queue uses
`queue_*_ts`), plus demographics and a second store. The existing graded API
**rejected 100%** of the provided `sample_events.jsonl` — so the 40-point API block
was failing on the new data.

**Options considered:** (a) rewrite the schema, models, DB, and every metric to the
new shapes; (b) a thin **normalization adapter at the ingest boundary** that maps
each wire shape onto the canonical row everything downstream already speaks.

**What the AI suggested:** initially, change the `StoreEvent` model to the new
fields directly (the obvious "make it match" move).

**What we chose and why.** We chose **(b)**. Rewriting the model would have
re-touched the DB schema, all six endpoints, and the whole test-suite for a wire
format that is *itself internally inconsistent* — high blast radius for the most
valuable, most controllable points. The adapter (`app/normalize.py`) localizes all
the messiness in one tested place: event_type folding, `store_code`↔`store_id`
canonicalization (applied symmetrically on read), per-camera `track_id` namespacing,
deterministic `event_id` synthesis for idempotency, and demographics/queue extras
carried into `metadata`. Downstream code is untouched, **all prior tests stay
green**, and our own pipeline's canonical events still ingest unchanged. The new
schema also **confirmed** an earlier call (Decision 3): it does *not* unify identity
across cameras (`id_token` vs `track_id`), exactly the assumption our aggregate
funnel was built on. The trade-off: the adapter must track any future field
additions — but that is one small module, not the graded core. A genuinely
malformed event still fails per-event validation, so partial-success is preserved.
