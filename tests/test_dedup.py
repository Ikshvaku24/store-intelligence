# PROMPT: "Write pytest tests for build_merge_map(embeddings, blocked_pairs,
#   threshold): tokens with cosine >= threshold merge to one canonical id; tokens
#   below threshold stay separate; a blocked pair (two tracks overlapping on the
#   same camera = different people) is never merged even with high similarity;
#   canonical id is the lexicographically smallest in each group."
# CHANGES MADE: Added the transitivity-with-block case (a~c~b but a,b blocked must
#   not all collapse) and an empty-input guard, which the first draft missed.
from __future__ import annotations

from pipeline.dedup import build_merge_map, cosine


def test_cosine_basics():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([], [1]) == 0.0


def test_merges_similar_tokens():
    embs = {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.0, 1.0]}
    m = build_merge_map(embs, threshold=0.9)
    assert m["a"] == m["b"]      # near-identical -> merged
    assert m["c"] != m["a"]      # orthogonal -> separate
    assert m["a"] == "a"         # canonical = smallest id


def test_blocked_pair_never_merges():
    embs = {"a": [1.0, 0.0], "b": [1.0, 0.0]}      # identical
    m = build_merge_map(embs, blocked_pairs=[("a", "b")], threshold=0.9)
    assert m["a"] != m["b"]      # same camera + overlapping time => different people


def test_block_prevents_transitive_collapse():
    # a~c and c~b by similarity, but a,b are blocked -> they must not share a group.
    embs = {"a": [1.0, 0.0, 0.0], "c": [1.0, 0.0, 0.0], "b": [1.0, 0.0, 0.0]}
    m = build_merge_map(embs, blocked_pairs=[("a", "b")], threshold=0.9)
    assert m["a"] != m["b"]


def test_empty_and_single():
    assert build_merge_map({}) == {}
    assert build_merge_map({"x": [1.0, 2.0]}) == {"x": "x"}


# --- VLM borderline-band tie-breaker (gated confirmer) ---
import math

# cosine ~0.78 -> inside the borderline band [0.74, 0.82); 0.85 -> auto band.
_A = [1.0, 0.0]
_B = [0.78, math.sqrt(1 - 0.78 ** 2)]
_C = [0.85, math.sqrt(1 - 0.85 ** 2)]


def test_borderline_needs_confirm():
    embs = {"a": _A, "b": _B}
    # no confirmer -> borderline pair stays separate (embedding-only behaviour)
    assert build_merge_map(embs, threshold=0.82, low_threshold=0.74) == {"a": "a", "b": "b"}
    # confirm False -> still separate; confirm True -> merged
    assert build_merge_map(embs, threshold=0.82, low_threshold=0.74,
                           confirm=lambda a, b: False)["b"] == "b"
    assert build_merge_map(embs, threshold=0.82, low_threshold=0.74,
                           confirm=lambda a, b: True)["b"] == "a"


def test_auto_band_ignores_confirmer():
    # >= threshold merges regardless of the confirmer's answer
    embs = {"a": _A, "c": _C}
    m = build_merge_map(embs, threshold=0.82, low_threshold=0.74, confirm=lambda a, b: False)
    assert m["c"] == "a"


def test_block_beats_confirm_in_borderline():
    embs = {"a": _A, "b": _B}
    m = build_merge_map(embs, blocked_pairs=[("a", "b")], threshold=0.82,
                        low_threshold=0.74, confirm=lambda a, b: True)
    assert m["a"] != m["b"]   # provably-different (same-camera overlap) is never merged
