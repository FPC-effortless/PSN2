"""Tests for AttractorLibrary — insert/query/prune, cache fix (#15)."""
import torch
import pytest
from psn2.attractor import AttractorLibrary, COS_INSERT_THRESHOLD
from psn2.vsa import normalize


def test_insert_and_query():
    al = AttractorLibrary(dim=64, max_size=100)
    v = normalize(torch.randn(64))
    inserted = al.add(v)
    assert inserted
    assert len(al) == 1
    results = al.query(v, k=1)
    assert len(results) == 1
    assert results[0][0] > 0.99


def test_no_duplicate_insert():
    al = AttractorLibrary(dim=64, max_size=100)
    v = normalize(torch.randn(64))
    al.add(v)
    inserted_again = al.add(v)
    assert not inserted_again
    assert len(al) == 1


def test_prune_on_capacity():
    al = AttractorLibrary(dim=64, max_size=5)
    for i in range(6):
        al.add(normalize(torch.randn(64)), utility=float(i))
    # Should have pruned to keep_fraction=0.9 of max_size=5 -> keep 4
    assert len(al) <= 5


def test_cache_invalidated_on_insert():
    """Fix #15: cache should be invalidated after insert."""
    al = AttractorLibrary(dim=64, max_size=100)
    v1 = normalize(torch.randn(64))
    al.add(v1)
    t1 = al.as_tensor()
    v2 = normalize(torch.randn(64))
    al.add(v2)
    t2 = al.as_tensor()
    assert t2.shape[0] == 2, "Cache should reflect new entry"
    assert t1.shape[0] == 1


def test_state_dict_roundtrip():
    al = AttractorLibrary(dim=64, max_size=100)
    for _ in range(3):
        al.add(normalize(torch.randn(64)))
    state = al.state_dict()
    al2 = AttractorLibrary(dim=64, max_size=100)
    al2.load_state_dict(state)
    assert len(al2) == 3
