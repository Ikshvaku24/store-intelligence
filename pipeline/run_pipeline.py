"""Pipeline orchestrator (BUILD_SPEC Section 7): clips -> events.jsonl.

For each clip in the manifest, independently: decode + track, sample appearance
features for identity, run tripwire + zone geometry, apply the staff cascade and
session/identity logic, build schema-validated events, then write them all to a
single events.jsonl sorted by timestamp.

Heavy CV deps (detect/reid) are imported lazily, so this module imports cleanly
in the light venv; only `run()` pulls them in.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any, Optional

from pipeline import geometry
from pipeline.emit import EventWriter, build_event, frame_timestamp
from pipeline.sessions import SessionManager
from pipeline.staff import StaffEvidence, resolve_staff

REID_INTERVAL = 10           # sample appearance features every N frames
DWELL_CADENCE_S = 30         # ZONE_DWELL emitted per 30s of continued presence
QUEUE_DWELL_MIN_S = 3        # min presence in queue lane before it counts to depth
MIN_TRACK_FRAMES = 5         # drop floor events from tracks seen < this many frames (flicker)
_KEEP_SHORT = {"ENTRY", "EXIT", "REENTRY"}  # tripwire-gated; never dropped by track length


def _parse_ts(s: str) -> dt.datetime:
    s = s.replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)


def run(
    manifest_path: str,
    zones_path: str,
    lines_path: str,
    output_path: str,
    weights: str = "yolo11m.pt",
    vid_stride: int = 1,
    device: Optional[str] = None,
    use_vlm: Optional[bool] = None,
    min_track_frames: int = MIN_TRACK_FRAMES,
    dedup_threshold: float = 0.0,
    debug_dump: Optional[str] = None,
) -> int:
    manifest = _load(manifest_path)
    zones_cfg = _load(zones_path)
    lines_cfg = _load(lines_path)

    from pipeline.reid import FeatureExtractor  # lazy (torch/cv2)

    mgr = SessionManager()
    writer = EventWriter()
    evidence = StaffEvidence()
    extractor = FeatureExtractor(device=device)
    print(f"Re-ID device: {extractor.device}")

    track_rows: Optional[list] = [] if debug_dump else None

    for clip in manifest:
        _process_clip(
            clip,
            zones_cfg.get(clip["camera_id"], {}),
            lines_cfg.get(clip["camera_id"]),
            mgr,
            writer,
            evidence,
            extractor,
            weights,
            vid_stride,
            device,
            track_rows,
        )

    # --- resolve staff once (original tokens): position -> VLM -> heuristic ---
    # "Once a person is staff, everyone else is a customer": we only add to staff.
    print(f"All clips processed. Resolving staff over {len(evidence.visitors)} visitor "
          f"tokens (VLM={'on' if use_vlm else 'off'}; VLM calls are rate-limited, so this "
          f"step can pause between calls)...", flush=True)
    vlm = _make_vlm(use_vlm)
    # Bias against false-positive staff: only a *confident* VLM verdict promotes to
    # staff (most people in a store are customers). Tunable via VLM_STAFF_CONF.
    vlm_conf = float(os.environ.get("VLM_STAFF_CONF", "0.75"))
    staff_ids, decisions = resolve_staff(evidence, mgr.staff_ids(), vlm, vlm_conf_threshold=vlm_conf)
    print(f"Staff resolved ({len(staff_ids)} staff). Deduplicating visitor tokens...", flush=True)

    # --- best-effort global dedup: collapse track fragments + cross-camera dups ---
    # cam1 & cam2 are the same room and ByteTrack fragments tracks, so one person
    # yields many tokens. Merge by mean embedding; never merge two tokens that
    # overlap in time on the SAME camera (different people). Approximate -> flagged.
    merge_map: dict[str, str] = {}
    if dedup_threshold and dedup_threshold > 0:
        embs = evidence.mean_embeddings()
        if embs:
            from pipeline.dedup import build_merge_map  # lazy

            merge_map = build_merge_map(embs, _same_camera_overlap(evidence), dedup_threshold)
    relabeled = writer.relabel_visitors(merge_map)
    staff_ids = {merge_map.get(s, s) for s in staff_ids}

    flipped = writer.finalize_staff(staff_ids)
    by_src: dict[str, int] = {}
    for d in decisions:
        if d["decision"] == "staff":
            by_src[d["source"]] = by_src.get(d["source"], 0) + 1

    # Drop flicker tracks: sum frames per *canonical* visitor (after merge).
    canon_frames: dict[str, int] = {}
    for vid, v in evidence.visitors.items():
        c = merge_map.get(vid, vid)
        canon_frames[c] = canon_frames.get(c, 0) + v.frames
    short = {c for c, fr in canon_frames.items() if fr < min_track_frames}
    removed = writer.drop_short_tracks(short, _KEEP_SHORT)

    if merge_map:
        people = len(set(merge_map.values()))
        print(f"Dedup: {len(merge_map)} tokens -> {people} people "
              f"(threshold {dedup_threshold}); relabeled {relabeled} events")
    print(
        f"Staff resolved: {len(staff_ids)} staff visitor(s) "
        f"(by source: {by_src or 'none'}); re-stamped {flipped} events; "
        f"VLM={'on' if vlm is not None else 'off'}"
    )
    print(f"Dropped {removed} event(s) from {len(short)} short/flicker track(s) "
          f"(< {min_track_frames} frames)")

    if debug_dump and track_rows:
        _write_debug_dump(debug_dump, track_rows, merge_map, staff_ids)
        print(f"Wrote {len(track_rows)} detection rows -> {debug_dump} (staff overlay)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    n = writer.write(output_path)
    print(f"Wrote {n} events to {output_path}")
    return n


def _same_camera_overlap(evidence) -> set:
    """Token pairs that share a camera AND overlap in time there -> different
    people -> must never be merged by dedup."""
    vids = list(evidence.visitors)
    blocked = set()
    for i in range(len(vids)):
        a = evidence.visitors[vids[i]]
        for j in range(i + 1, len(vids)):
            b = evidence.visitors[vids[j]]
            if (a.cameras & b.cameras) and a.first_ts <= b.last_ts and b.first_ts <= a.last_ts:
                blocked.add((vids[i], vids[j]))
    return blocked


def _write_debug_dump(path: str, rows: list, merge_map: dict, staff_ids: set) -> None:
    """Per-detection rows (bbox + final visitor_id + is_staff) for the staff overlay."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            vid = merge_map.get(r["visitor_id"], r["visitor_id"])
            out = {**r, "visitor_id": vid, "is_staff": vid in staff_ids}
            fh.write(json.dumps(out) + "\n")


def _make_vlm(use_vlm: Optional[bool]):
    """Build the Gemini staff confirmer when a key is present (or forced on).

    Returns None to skip (no key, forced off, or SDK unavailable) -- the resolver
    then falls back to position + heuristic, so the pipeline always runs."""
    if use_vlm is False:
        return None
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if use_vlm is None and not key:
        return None
    try:
        from pipeline.vlm_staff import GeminiStaffClassifier  # lazy

        vlm = GeminiStaffClassifier()
        return vlm if vlm.available else None
    except Exception:  # noqa: BLE001
        return None


def _process_clip(
    clip: dict[str, Any],
    zones: dict[str, Any],
    line: Optional[dict[str, Any]],
    mgr: SessionManager,
    writer: EventWriter,
    evidence: StaffEvidence,
    extractor,
    weights: str,
    vid_stride: int = 1,
    device: Optional[str] = None,
    track_rows: Optional[list] = None,
) -> None:
    from pipeline import detect  # lazy (ultralytics)
    from pipeline.reid import crop_bbox, encode_jpeg

    store_id = clip["store_id"]
    camera_id = clip["camera_id"]
    role = clip["role"]
    start_ts = _parse_ts(clip["start_ts"])
    fps = _clip_fps(clip)

    zone_polys = {name: cfg["polygon"] for name, cfg in zones.items()}

    tripwire = None
    if role == "ENTRY" and line:
        inside_sign = geometry.inside_sign_from_point(line["p1"], line["p2"], line["inside_point"])
        tripwire = geometry.TripwireCounter(
            line["p1"], line["p2"], inside_sign, line.get("min_frames_each_side", 3)
        )

    features: dict[int, dict[str, Any]] = {}     # track_id -> {emb, hist}
    track_zone: dict[int, dict[str, dict]] = {}  # track_id -> {zone -> {enter_ts,last_dwell}}

    import time as _time
    print(f"[{camera_id}] {role}: processing {clip['clip_path']} (fps={fps:.2f}, stride={vid_stride})...",
          flush=True)
    _t0 = _time.time()
    _processed = 0

    for frame_index, tracks, frame in detect.track_clip(
        clip["clip_path"], weights=weights, vid_stride=vid_stride, device=device
    ):
        ts = frame_timestamp(start_ts, frame_index, fps)
        _processed += 1
        if _processed % 200 == 0:
            rate = _processed / max(1e-3, _time.time() - _t0)
            print(f"[{camera_id}]   ...{_processed} frames processed "
                  f"(frame_index {frame_index}, {rate:.1f} fps, {len(features)} tracks)", flush=True)

        # --- resolve each track to a visitor + zone membership for this frame ---
        resolved = []
        frame_crops: dict[int, tuple[float, bytes]] = {}
        for tr in tracks:
            fp = tr.foot_point
            if frame is not None and frame_index % REID_INTERVAL == 0:
                crop = crop_bbox(frame, tr.bbox)
                if crop is not None:
                    emb = extractor.embed(crop)
                    hist = extractor.hsv_histogram(crop)
                    f = features.setdefault(tr.track_id, {})
                    if emb:
                        f["emb"] = emb
                    if hist:
                        f["hist"] = hist
                    cj = encode_jpeg(crop)
                    if cj:
                        x1, y1, x2, y2 = tr.bbox
                        frame_crops[tr.track_id] = (abs((x2 - x1) * (y2 - y1)), cj)
            f = features.get(tr.track_id, {})
            a = mgr.assign(camera_id, tr.track_id, role, f.get("emb"), f.get("hist"), ts)
            in_zones = list(geometry.zones_for_point(zone_polys, fp))

            staff_now = role == "BACKROOM" or any(zones[z].get("is_staff_zone") for z in in_zones)
            if staff_now:
                mgr.mark_staff(a.visitor_id)
            resolved.append((tr, a, in_zones))

        # --- staff evidence: proximity ("serving" signal), crops, presence span ---
        near_map: dict[str, set] = {}
        for i, (tr_i, a_i, _zi) in enumerate(resolved):
            fxi, fyi = tr_i.foot_point
            wi = max(tr_i.bbox[2] - tr_i.bbox[0], 1.0)
            near = near_map.setdefault(a_i.visitor_id, set())
            for j, (tr_j, a_j, _zj) in enumerate(resolved):
                if i == j or a_i.visitor_id == a_j.visitor_id:
                    continue
                fxj, fyj = tr_j.foot_point
                if abs(fxi - fxj) <= 1.5 * wi and abs(fyi - fyj) <= 1.5 * wi:
                    near.add(a_j.visitor_id)
        for tr, a, in_zones in resolved:
            area, cj = frame_crops.get(tr.track_id, (None, None))
            emb = features.get(tr.track_id, {}).get("emb")
            evidence.observe(
                a.visitor_id, ts, camera_id, in_zones,
                near_map.get(a.visitor_id, ()), area, cj, emb,
            )
            if track_rows is not None:
                x1, y1, x2, y2 = tr.bbox
                track_rows.append({
                    "camera": camera_id, "frame_index": frame_index,
                    "t": round(frame_index / fps, 2),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "visitor_id": a.visitor_id, "conf": round(tr.confidence, 3),
                })

        # queue depth = distinct non-staff foot-points dwelling in a queue zone.
        queue_depth = sum(
            1
            for tr, a, zs in resolved
            if not mgr.is_staff(a.visitor_id) and any(zones[z].get("is_queue_zone") for z in zs)
        )

        for tr, a, in_zones in resolved:
            vid = a.visitor_id
            staff = mgr.is_staff(vid)
            conf = tr.confidence

            # --- ENTRY / EXIT / REENTRY (entry camera only) ---
            if tripwire is not None:
                crossing = tripwire.update(tr.track_id, tr.foot_point)
                if crossing == "ENTRY":
                    if a.is_reentry:
                        _emit(writer, mgr, store_id, camera_id, vid, "REENTRY", ts, conf, staff,
                              meta=_idmeta(a))
                    elif not a.suppress_entry:
                        _emit(writer, mgr, store_id, camera_id, vid, "ENTRY", ts, conf, staff,
                              meta=_idmeta(a))
                elif crossing == "EXIT":
                    _emit(writer, mgr, store_id, camera_id, vid, "EXIT", ts, conf, staff,
                          meta=_idmeta(a))
                    f = features.get(tr.track_id, {})
                    mgr.on_exit(vid, f.get("emb"), f.get("hist"), ts)

            # --- zone enter / dwell / exit (floor + billing cameras) ---
            prev = track_zone.setdefault(tr.track_id, {})
            for z in in_zones:
                zinfo = zones[z]
                sku = zinfo.get("sku_zone") or z
                if z not in prev:
                    prev[z] = {"enter_ts": ts, "last_dwell": ts}
                    if zinfo.get("is_queue_zone"):
                        _emit(writer, mgr, store_id, camera_id, vid, "BILLING_QUEUE_JOIN", ts, conf,
                              staff, zone_id=z, meta={"queue_depth": queue_depth, **_zone_meta(zinfo), **_idmeta(a)})
                    else:
                        _emit(writer, mgr, store_id, camera_id, vid, "ZONE_ENTER", ts, conf, staff,
                              zone_id=z, meta={"sku_zone": sku, **_zone_meta(zinfo), **_idmeta(a)})
                else:
                    if (ts - prev[z]["last_dwell"]).total_seconds() >= DWELL_CADENCE_S:
                        prev[z]["last_dwell"] = ts
                        dwell_ms = int((ts - prev[z]["enter_ts"]).total_seconds() * 1000)
                        _emit(writer, mgr, store_id, camera_id, vid, "ZONE_DWELL", ts, conf, staff,
                              zone_id=z, dwell_ms=dwell_ms, meta={"sku_zone": sku, **_zone_meta(zinfo)})

            for z in list(prev):
                if z not in in_zones:
                    info = prev.pop(z)
                    dwell_ms = int((ts - info["enter_ts"]).total_seconds() * 1000)
                    if zones[z].get("is_queue_zone"):
                        # Best-effort abandon; the API downgrades any that correlate
                        # to a POS purchase within the window (BUILD_SPEC 10.7).
                        _emit(writer, mgr, store_id, camera_id, vid, "BILLING_QUEUE_ABANDON", ts,
                              conf, staff, zone_id=z, dwell_ms=dwell_ms, meta=_zone_meta(zones[z]))
                    else:
                        _emit(writer, mgr, store_id, camera_id, vid, "ZONE_EXIT", ts, conf, staff,
                              zone_id=z, dwell_ms=dwell_ms,
                              meta={"sku_zone": zones[z].get("sku_zone") or z, **_zone_meta(zones[z])})

    print(f"[{camera_id}] done: {_processed} frames in {_time.time() - _t0:.1f}s, "
          f"{len(features)} tracks.", flush=True)


def _idmeta(a) -> dict[str, Any]:
    return {"id_source": a.id_source, "id_confidence": round(a.id_confidence, 3)}


def _zone_meta(zinfo: dict) -> dict[str, Any]:
    """New-schema zone attributes (zone_type/zone_name/is_revenue_zone) for metadata,
    so pipeline events carry the same zone semantics the graders' schema uses."""
    out: dict[str, Any] = {}
    for key in ("zone_type", "zone_name", "is_revenue_zone"):
        if zinfo.get(key) is not None:
            out[key] = zinfo[key]
    return out


def _emit(writer, mgr, store_id, camera_id, vid, etype, ts, conf, staff, zone_id=None, dwell_ms=0, meta=None):
    meta = dict(meta or {})
    meta["session_seq"] = mgr.next_seq(vid)
    ev = build_event(
        store_id=store_id,
        camera_id=camera_id,
        visitor_id=vid,
        event_type=etype,
        timestamp=ts,
        confidence=conf,
        zone_id=zone_id,
        dwell_ms=dwell_ms,
        is_staff=staff,
        metadata=meta,
    )
    writer.add(ev)


def _clip_fps(clip: dict) -> float:
    """Frame rate for a clip: the manifest value if set, else read from the video.

    Per-store onboarding (e.g. Store 2) can leave ``fps`` null/0 in the manifest and
    have the true native rate read from the container at runtime -- removing the
    hand-measured-fps error class that previously stretched every timestamp (ADR-010).
    """
    fps = clip.get("fps")
    if fps:
        return float(fps)
    import cv2  # lazy (pipeline container only)

    cap = cv2.VideoCapture(clip["clip_path"])
    val = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not val or val <= 0:
        raise SystemExit(
            f"Could not read fps from {clip['clip_path']}; set 'fps' in the manifest."
        )
    return float(val)


def _load(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the detection pipeline -> events.jsonl")
    ap.add_argument("--manifest", default="pipeline/config/store_1/clips_manifest.json")
    ap.add_argument("--zones", default="pipeline/config/store_1/zones.json")
    ap.add_argument("--lines", default="pipeline/config/store_1/lines.json")
    ap.add_argument("--output", default=os.environ.get("OUTPUT_PATH", "data/events.jsonl"))
    ap.add_argument("--weights", default=os.environ.get("WEIGHTS", "yolo11m.pt"))
    ap.add_argument("--stride", type=int, default=int(os.environ.get("VID_STRIDE", "1")),
                    help="process every Nth frame (CPU speed-up; timestamps stay correct)")
    ap.add_argument("--device", default=os.environ.get("DEVICE", "auto"),
                    help="auto (GPU if available) | cpu | 0 | cuda:0")
    ap.add_argument("--no-vlm", action="store_true",
                    help="disable the Gemini behavioural staff confirmer")
    ap.add_argument("--min-track-frames", type=int,
                    default=int(os.environ.get("MIN_TRACK_FRAMES", str(MIN_TRACK_FRAMES))),
                    help="drop floor events from tracks seen fewer than N frames (flicker)")
    ap.add_argument("--dedup-threshold", type=float,
                    default=float(os.environ.get("REID_DEDUP_THRESHOLD", "0.82")),
                    help="merge visitor tokens with mean-embedding cosine >= this (0 disables)")
    ap.add_argument("--no-dedup", action="store_true", help="disable embedding visitor dedup")
    ap.add_argument("--debug-dump", default=os.environ.get("DEBUG_DUMP"),
                    help="write per-detection rows here for the staff overlay (visualize --overlay)")
    args = ap.parse_args()
    run(args.manifest, args.zones, args.lines, args.output, args.weights, args.stride,
        args.device, use_vlm=False if args.no_vlm else None,
        min_track_frames=args.min_track_frames,
        dedup_threshold=0.0 if args.no_dedup else args.dedup_threshold,
        debug_dump=args.debug_dump)


if __name__ == "__main__":
    main()
