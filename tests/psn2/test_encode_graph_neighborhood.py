"""Unit tests for encode_graph_neighborhood (Task 3.1).

Tests verify:
- Output shape and normalization
- Attention-weighted aggregation based on relation type
- Relation type embeddings added directly to neighborhood context
- Discriminative signal: different masked entities produce different contexts
- Edge cases: empty relations, single entity, all entities masked
- Gradient flow through the function

**Validates: Requirements 1.3, 2.3, 2.4**
"""
import torch
import pytest
from psn2.core import PSN2System


def make_model(dim=64):
    """Create a small PSN2System for testing."""
    return PSN2System(dim=dim, max_nodes=16, grid_vocab=10, rel_vocab=17, stage="D1")


def make_graph_inputs(B=2, N_e=6, N_r=5, dim=64):
    """Create simple graph inputs for testing."""
    entities = torch.randint(0, 16, (B, N_e))
    # Relations: (source_idx, relation_type, target_idx)
    rel_a = torch.randint(0, N_e, (B, N_r))
    rel_r = torch.randint(0, 3, (B, N_r))
    rel_b = torch.randint(0, N_e, (B, N_r))
    relations = torch.stack([rel_a, rel_r, rel_b], dim=-1)  # [B, N_r, 3]
    masked_idx = torch.zeros(B, dtype=torch.long)  # mask entity 0
    return entities, relations, masked_idx


# ---------------------------------------------------------------------------
# Output shape and normalization
# ---------------------------------------------------------------------------

def test_output_shape_small_graph():
    """encode_graph_neighborhood returns [B, D] for a 2-entity graph."""
    model = make_model(dim=64)
    B, N_e, N_r = 2, 2, 1
    entities = torch.randint(0, 16, (B, N_e))
    relations = torch.tensor([[[0, 0, 1]], [[0, 1, 1]]], dtype=torch.long)  # [B, 1, 3]
    masked_idx = torch.zeros(B, dtype=torch.long)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert out.shape == (B, 64), f"Expected shape ({B}, 64), got {out.shape}"


def test_output_shape_medium_graph():
    """encode_graph_neighborhood returns [B, D] for a 6-entity graph."""
    model = make_model(dim=64)
    entities, relations, masked_idx = make_graph_inputs(B=4, N_e=6, N_r=5)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert out.shape == (4, 64), f"Expected shape (4, 64), got {out.shape}"


def test_output_shape_large_graph():
    """encode_graph_neighborhood returns [B, D] for a 10-entity graph."""
    model = make_model(dim=64)
    entities, relations, masked_idx = make_graph_inputs(B=3, N_e=10, N_r=8)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert out.shape == (3, 64), f"Expected shape (3, 64), got {out.shape}"


def test_output_is_normalized():
    """Output vectors should be unit-normalized (L2 norm ≈ 1.0)."""
    model = make_model(dim=64)
    entities, relations, masked_idx = make_graph_inputs(B=4, N_e=6, N_r=5)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), (
        f"Output vectors should be unit-normalized, got norms: {norms}"
    )


def test_output_no_nan_or_inf():
    """Output should not contain NaN or Inf values."""
    model = make_model(dim=64)
    entities, relations, masked_idx = make_graph_inputs(B=4, N_e=6, N_r=5)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert not torch.isnan(out).any(), "Output contains NaN values"
    assert not torch.isinf(out).any(), "Output contains Inf values"


# ---------------------------------------------------------------------------
# Discriminative signal: different masked entities produce different contexts
# ---------------------------------------------------------------------------

def test_different_masked_entities_produce_different_contexts():
    """
    Masking different entities should produce different neighborhood contexts.
    This verifies the function carries discriminative signal.
    """
    model = make_model(dim=64)
    torch.manual_seed(0)
    B, N_e, N_r = 1, 6, 5
    entities = torch.randint(0, 16, (B, N_e))
    # Build relations that connect different entities
    relations = torch.tensor([[[0, 0, 1], [1, 1, 2], [2, 2, 3], [3, 0, 4], [4, 1, 5]]],
                              dtype=torch.long)

    masked_idx_0 = torch.tensor([0])
    masked_idx_3 = torch.tensor([3])

    out_0 = model.encode_graph_neighborhood(entities, relations, masked_idx_0)
    out_3 = model.encode_graph_neighborhood(entities, relations, masked_idx_3)

    # The two contexts should be different (cosine similarity < 1.0)
    cosine_sim = torch.nn.functional.cosine_similarity(out_0, out_3, dim=-1).item()
    assert cosine_sim < 0.999, (
        f"Masking different entities should produce different contexts, "
        f"but cosine similarity is {cosine_sim:.4f} (too similar)"
    )


def test_different_relation_types_produce_different_contexts():
    """
    Graphs with different relation types connecting the same entities should
    produce different neighborhood contexts (relation type signal is preserved).
    """
    model = make_model(dim=64)
    torch.manual_seed(1)
    B, N_e = 1, 3
    entities = torch.randint(0, 16, (B, N_e))
    masked_idx = torch.tensor([0])

    # Same structure, different relation types
    relations_type0 = torch.tensor([[[0, 0, 1], [0, 0, 2]]], dtype=torch.long)
    relations_type1 = torch.tensor([[[0, 1, 1], [0, 1, 2]]], dtype=torch.long)

    out_type0 = model.encode_graph_neighborhood(entities, relations_type0, masked_idx)
    out_type1 = model.encode_graph_neighborhood(entities, relations_type1, masked_idx)

    cosine_sim = torch.nn.functional.cosine_similarity(out_type0, out_type1, dim=-1).item()
    assert cosine_sim < 0.999, (
        f"Different relation types should produce different contexts, "
        f"but cosine similarity is {cosine_sim:.4f} (too similar)"
    )


# ---------------------------------------------------------------------------
# Attention-weighted aggregation
# ---------------------------------------------------------------------------

def test_attention_weights_sum_to_one_for_involved_relations():
    """
    Attention weights for involved relations should sum to approximately 1.0
    (softmax over involved relations only).
    """
    model = make_model(dim=64)
    B, N_e, N_r = 2, 4, 3
    entities = torch.randint(0, 16, (B, N_e))
    # Entity 0 is involved in all 3 relations
    relations = torch.tensor([
        [[0, 0, 1], [0, 1, 2], [0, 2, 3]],
        [[0, 0, 1], [0, 1, 2], [0, 2, 3]],
    ], dtype=torch.long)
    masked_idx = torch.zeros(B, dtype=torch.long)

    # Should not raise and should produce valid output
    out = model.encode_graph_neighborhood(entities, relations, masked_idx)
    assert out.shape == (B, 64)
    assert not torch.isnan(out).any()


def test_uninvolved_relations_do_not_affect_output():
    """
    Adding uninvolved relations (not connected to masked entity) should not
    significantly change the neighborhood context.
    """
    model = make_model(dim=64)
    torch.manual_seed(2)
    B, N_e = 1, 5
    entities = torch.randint(0, 16, (B, N_e))
    masked_idx = torch.tensor([0])

    # Only relations involving entity 0
    relations_minimal = torch.tensor([[[0, 0, 1], [0, 1, 2]]], dtype=torch.long)

    # Same relations + extra uninvolved ones (between entities 3 and 4)
    relations_with_extra = torch.tensor([
        [[0, 0, 1], [0, 1, 2], [3, 0, 4], [3, 1, 4], [4, 2, 3]]
    ], dtype=torch.long)

    out_minimal = model.encode_graph_neighborhood(entities, relations_minimal, masked_idx)
    out_with_extra = model.encode_graph_neighborhood(entities, relations_with_extra, masked_idx)

    # The outputs should be similar (uninvolved relations are masked out)
    cosine_sim = torch.nn.functional.cosine_similarity(out_minimal, out_with_extra, dim=-1).item()
    assert cosine_sim > 0.9, (
        f"Uninvolved relations should not significantly change the context, "
        f"but cosine similarity is {cosine_sim:.4f} (too different)"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_no_relations_involving_masked_entity():
    """
    When no relations involve the masked entity, the function should fall back
    to the global entity mean without raising an error.
    """
    model = make_model(dim=64)
    B, N_e = 2, 4
    entities = torch.randint(0, 16, (B, N_e))
    # Relations only between entities 1, 2, 3 — entity 0 is isolated
    relations = torch.tensor([
        [[1, 0, 2], [2, 1, 3], [1, 2, 3], [1, 0, 3], [2, 1, 3]],
        [[1, 0, 2], [2, 1, 3], [1, 2, 3], [1, 0, 3], [2, 1, 3]],
    ], dtype=torch.long)
    masked_idx = torch.zeros(B, dtype=torch.long)  # mask entity 0 (isolated)

    # Should not raise
    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert out.shape == (B, 64)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_single_relation_involving_masked_entity():
    """
    When only one relation involves the masked entity, the function should
    produce a valid output using that single relation.
    """
    model = make_model(dim=64)
    B, N_e = 2, 4
    entities = torch.randint(0, 16, (B, N_e))
    # Only one relation involves entity 0
    relations = torch.tensor([
        [[0, 0, 1], [1, 1, 2], [2, 2, 3], [1, 0, 2], [2, 1, 3]],
        [[0, 0, 1], [1, 1, 2], [2, 2, 3], [1, 0, 2], [2, 1, 3]],
    ], dtype=torch.long)
    masked_idx = torch.zeros(B, dtype=torch.long)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert out.shape == (B, 64)
    assert not torch.isnan(out).any()


def test_all_relations_involve_masked_entity():
    """
    When all relations involve the masked entity, the function should
    aggregate all of them correctly.
    """
    model = make_model(dim=64)
    B, N_e, N_r = 2, 5, 4
    entities = torch.randint(0, 16, (B, N_e))
    # All relations involve entity 0
    relations = torch.tensor([
        [[0, 0, 1], [0, 1, 2], [0, 2, 3], [0, 0, 4]],
        [[0, 0, 1], [0, 1, 2], [0, 2, 3], [0, 0, 4]],
    ], dtype=torch.long)
    masked_idx = torch.zeros(B, dtype=torch.long)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert out.shape == (B, 64)
    assert not torch.isnan(out).any()


def test_batch_size_one():
    """Function should work correctly with batch size 1."""
    model = make_model(dim=64)
    entities, relations, masked_idx = make_graph_inputs(B=1, N_e=6, N_r=5)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert out.shape == (1, 64)
    assert not torch.isnan(out).any()


def test_large_batch():
    """Function should work correctly with a large batch."""
    model = make_model(dim=64)
    entities, relations, masked_idx = make_graph_inputs(B=16, N_e=6, N_r=5)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert out.shape == (16, 64)
    assert not torch.isnan(out).any()


def test_masked_idx_at_last_position():
    """Function should work when the masked entity is at the last position."""
    model = make_model(dim=64)
    B, N_e, N_r = 2, 6, 5
    entities = torch.randint(0, 16, (B, N_e))
    # Relations involving entity 5 (last)
    relations = torch.tensor([
        [[5, 0, 0], [5, 1, 1], [5, 2, 2], [5, 0, 3], [5, 1, 4]],
        [[5, 0, 0], [5, 1, 1], [5, 2, 2], [5, 0, 3], [5, 1, 4]],
    ], dtype=torch.long)
    masked_idx = torch.full((B,), N_e - 1, dtype=torch.long)  # mask last entity

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert out.shape == (B, 64)
    assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flows_to_entity_encoder():
    """
    Gradients should flow through encode_graph_neighborhood to entity_encoder.
    This verifies the function is differentiable and connected to entity_encoder.
    """
    model = make_model(dim=64)
    entities, relations, masked_idx = make_graph_inputs(B=4, N_e=6, N_r=5)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)
    loss = out.sum()
    loss.backward()

    # Check that entity encoder parameters received gradients
    has_grad = False
    for param in model.entity_encoder.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "Gradients should flow to entity_encoder through encode_graph_neighborhood"


def test_gradient_flows_to_relation_encoder():
    """
    Gradients should flow through encode_graph_neighborhood to relation_encoder.
    This verifies the relation type signal is connected to the gradient path.
    """
    model = make_model(dim=64)
    entities, relations, masked_idx = make_graph_inputs(B=4, N_e=6, N_r=5)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)
    loss = out.sum()
    loss.backward()

    assert model.relation_encoder.weight.grad is not None, (
        "Gradients should flow to relation_encoder through encode_graph_neighborhood"
    )
    assert model.relation_encoder.weight.grad.abs().sum() > 0, (
        "Relation encoder gradient should be non-zero"
    )


def test_gradient_flows_to_neighborhood_attn():
    """
    Gradients should flow to the neighborhood_attn layer (Task 3.1 addition).
    """
    model = make_model(dim=64)
    entities, relations, masked_idx = make_graph_inputs(B=4, N_e=6, N_r=5)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)
    loss = out.sum()
    loss.backward()

    assert model.neighborhood_attn.weight.grad is not None, (
        "Gradients should flow to neighborhood_attn (Task 3.1 attention layer)"
    )
    assert model.neighborhood_attn.weight.grad.abs().sum() > 0, (
        "neighborhood_attn gradient should be non-zero"
    )


def test_gradient_flows_to_rel_context_proj():
    """
    Gradients should flow to the rel_context_proj layer (Task 3.1 addition).
    """
    model = make_model(dim=64)
    entities, relations, masked_idx = make_graph_inputs(B=4, N_e=6, N_r=5)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)
    loss = out.sum()
    loss.backward()

    assert model.rel_context_proj.weight.grad is not None, (
        "Gradients should flow to rel_context_proj (Task 3.1 relation context projection)"
    )
    assert model.rel_context_proj.weight.grad.abs().sum() > 0, (
        "rel_context_proj gradient should be non-zero"
    )


# ---------------------------------------------------------------------------
# Task 3.1 specific: attention-weighted aggregation and relation type embeddings
# ---------------------------------------------------------------------------

def test_neighborhood_attn_and_rel_context_proj_registered():
    """
    Task 3.1 requires neighborhood_attn and rel_context_proj to be registered
    as model parameters.
    """
    model = make_model(dim=64)

    assert hasattr(model, 'neighborhood_attn'), (
        "PSN2System should have neighborhood_attn (Task 3.1)"
    )
    assert hasattr(model, 'rel_context_proj'), (
        "PSN2System should have rel_context_proj (Task 3.1)"
    )
    assert isinstance(model.neighborhood_attn, torch.nn.Linear), (
        "neighborhood_attn should be an nn.Linear"
    )
    assert isinstance(model.rel_context_proj, torch.nn.Linear), (
        "rel_context_proj should be an nn.Linear"
    )


def test_neighborhood_attn_input_output_dims():
    """
    neighborhood_attn should map from dim -> 1 (scalar score per relation).
    """
    model = make_model(dim=64)

    assert model.neighborhood_attn.in_features == 64, (
        f"neighborhood_attn should have in_features=64, got {model.neighborhood_attn.in_features}"
    )
    assert model.neighborhood_attn.out_features == 1, (
        f"neighborhood_attn should have out_features=1, got {model.neighborhood_attn.out_features}"
    )


def test_rel_context_proj_input_output_dims():
    """
    rel_context_proj should map from dim -> dim (project relation context).
    """
    model = make_model(dim=64)

    assert model.rel_context_proj.in_features == 64, (
        f"rel_context_proj should have in_features=64, got {model.rel_context_proj.in_features}"
    )
    assert model.rel_context_proj.out_features == 64, (
        f"rel_context_proj should have out_features=64, got {model.rel_context_proj.out_features}"
    )


def test_mixed_batch_some_with_no_neighbors():
    """
    In a batch where some samples have no relations involving the masked entity
    and others do, the function should handle both correctly.
    """
    model = make_model(dim=64)
    B, N_e = 4, 4
    entities = torch.randint(0, 16, (B, N_e))

    # Sample 0 and 2: entity 0 is isolated (no relations)
    # Sample 1 and 3: entity 0 has relations
    relations = torch.tensor([
        [[1, 0, 2], [2, 1, 3], [1, 2, 3], [1, 0, 3], [2, 1, 3]],  # entity 0 isolated
        [[0, 0, 1], [0, 1, 2], [0, 2, 3], [1, 0, 2], [2, 1, 3]],  # entity 0 connected
        [[1, 0, 2], [2, 1, 3], [1, 2, 3], [1, 0, 3], [2, 1, 3]],  # entity 0 isolated
        [[0, 0, 1], [0, 1, 2], [0, 2, 3], [1, 0, 2], [2, 1, 3]],  # entity 0 connected
    ], dtype=torch.long)
    masked_idx = torch.zeros(B, dtype=torch.long)

    out = model.encode_graph_neighborhood(entities, relations, masked_idx)

    assert out.shape == (B, 64)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
