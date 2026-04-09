import torch
import pytest
from psn2.vsa import normalize
from psn2.ers import ExperienceReplaySubstrate, ERSTuple

def test_ers_tuple_initialization():
    dim = 512
    inp = normalize(torch.randn(dim))
    trace = normalize(torch.randn(dim))
    out = normalize(torch.randn(dim))
    
    tup = ERSTuple(input_vsa=inp, trace_vsa=trace, output_vsa=out, source="test", utility_score=0.5)
    assert tup.decay_state == 1.0

def test_ers_write_and_retrieve():
    ers = ExperienceReplaySubstrate(dim=512, max_working=10)
    inp = normalize(torch.randn(512))
    
    # Write to Working tier
    tup = ERSTuple(input_vsa=inp, trace_vsa=normalize(torch.randn(512)), output_vsa=normalize(torch.randn(512)), source="test", utility_score=0.5)
    ers.write("Working", tup)
    
    assert len(ers.working) == 1
    
    # Retrieve with highly similar query
    query = inp.clone()
    results = ers.retrieve(query, k=1)
    
    assert len(results) == 1
    assert results[0].source == "test"
    
    # Retrieve with orthogonal query should return empty (unless by chance cos > tau_ers, highly unlikely in 512d)
    ortho_query = normalize(torch.randn(512))
    results_ortho = ers.retrieve(ortho_query, k=1)
    assert len(results_ortho) == 0

def test_ers_promotion():
    ers = ExperienceReplaySubstrate(dim=512, max_working=10, max_episodic=10)
    inp = normalize(torch.randn(512))
    tup = ERSTuple(input_vsa=inp, trace_vsa=normalize(torch.randn(512)), output_vsa=normalize(torch.randn(512)), source="test", utility_score=0.8) # > 0.7 for promotion
    
    ers.write("Working", tup)
    assert len(ers.working) == 1
    
    ers.attempt_promotions()
    
    # Needs to promote from Working to Episodic
    assert len(ers.working) == 0
    assert len(ers.episodic) == 1
    
    ers.attempt_promotions()
    
    # Episodic to Semantic
    assert len(ers.episodic) == 0
    assert len(ers.semantic) == 1

def test_ers_memory_bounds():
    ers = ExperienceReplaySubstrate(dim=512, max_working=2)
    
    for i in range(5):
        tup = ERSTuple(input_vsa=normalize(torch.randn(512)), trace_vsa=normalize(torch.randn(512)), output_vsa=normalize(torch.randn(512)), source=f"test_{i}", utility_score=0.1 * i)
        ers.write("Working", tup)
        
    # Working tier uses FIFO eviction (PRD Section 11.2)
    assert len(ers.working) == 2
    # Should keep the last 2 written (test_3 and test_4)
    sources = [t.source for t in ers.working]
    assert "test_3" in sources
    assert "test_4" in sources
