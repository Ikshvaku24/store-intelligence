"""Best-effort global visitor dedup (BUILD_SPEC Section 9.7, extended).

One physical person yields many ``visitor_id`` tokens here, for two reasons:
  * cam1 (skincare) and cam2 (makeup) are the SAME room shot from two angles, so a
    shopper is seen on both with a separate per-camera token;
  * ByteTrack fragments a track on occlusion / leaving frame, minting a fresh token.

This collapses tokens that are very likely the same person using their mean OSNet
embedding. It is **conservative and flagged** -- appearance Re-ID is weak on this
footage (blurred faces, black-on-black clothing) -- and it never merges two tokens
that overlap in time on the SAME camera, because those are provably different
people. Pure-Python so it is unit-testable without torch.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def build_merge_map(
    embeddings: dict[str, list[float]],
    blocked_pairs: Optional[Iterable[tuple[str, str]]] = None,
    threshold: float = 0.82,
) -> dict[str, str]:
    """Return ``{visitor_id -> canonical_visitor_id}`` collapsing duplicates.

    Greedy highest-similarity-first union, skipping any merge that would place a
    ``blocked_pairs`` couple in the same group. Canonical id = lexicographically
    smallest token in each group (deterministic). Tokens never merged map to
    themselves.
    """
    ids = list(embeddings)
    blocked = set()
    for a, b in (blocked_pairs or ()):
        blocked.add((a, b))
        blocked.add((b, a))

    parent = {i: i for i in ids}
    members: dict[str, set[str]] = {i: {i} for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def blocked_between(g1: set[str], g2: set[str]) -> bool:
        # smaller group on the outside keeps this cheap
        small, large = (g1, g2) if len(g1) <= len(g2) else (g2, g1)
        return any((x, y) in blocked for x in small for y in large)

    candidates = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            s = cosine(embeddings[ids[i]], embeddings[ids[j]])
            if s >= threshold:
                candidates.append((s, ids[i], ids[j]))
    candidates.sort(key=lambda c: c[0], reverse=True)

    for _s, a, b in candidates:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if blocked_between(members[ra], members[rb]):
            continue
        parent[rb] = ra
        members[ra] |= members[rb]
        members[rb] = set()

    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)

    mapping: dict[str, str] = {}
    for root, ms in groups.items():
        canon = min(ms)
        for m in ms:
            mapping[m] = canon
    return mapping
