"""T21: USL Codec — bidirectional understand() and generate() pipeline."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from psn2.vsa import normalize, bundle
from psn2.usl.lal import LexicalAttractorLibrary, WordShape
from psn2.usl.linguistic_bonds import LinguisticBondTypes
from psn2.usl.syntax_encoder import SyntaxEncoder
from psn2.usl.syntax_decoder import SyntaxDecoder
from psn2.usl.pragmatics import PragmaticsRenderer


class USLCodec(nn.Module):
    """
    Bidirectional understand() and generate() pipeline.

    understand(tokens): token_ids → LAL retrieve → syntax_encoder → bond proposals → shape
    generate(shape):    shape → syntax_decoder → pragmatics_renderer → token_ids
    """

    def __init__(self, dim: int, vocab_size: int, lal: LexicalAttractorLibrary):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.lal = lal
        self.ling_bonds = LinguisticBondTypes(dim)
        self.syntax_encoder = SyntaxEncoder(dim)
        self.syntax_decoder = SyntaxDecoder(dim, vocab_size)
        self.pragmatics = PragmaticsRenderer(dim, vocab_size)
        # Fallback embedding for unknown tokens
        self.unk_embedding = nn.Embedding(vocab_size, dim)

    def _token_to_vec(self, token_id: int, device) -> torch.Tensor:
        """Retrieve semantic vector for a token id."""
        token_str = str(token_id)
        ws = self.lal.retrieve_by_token(token_str)
        if ws is not None:
            return ws.semantic_v.to(device)
        return self.unk_embedding(torch.tensor(token_id % self.vocab_size, device=device))

    def understand(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        token_ids: [seq_len] int tensor
        Returns (shape_vec [dim], bond_proposals list)
        """
        device = token_ids.device
        # Guard: empty token sequence returns zero vector and no proposals
        if token_ids.numel() == 0:
            return torch.zeros(self.dim, device=device), []
        word_vecs = torch.stack([self._token_to_vec(int(t.item()), device) for t in token_ids], dim=0)
        # Syntax encoder produces bond proposals
        bond_proposals = self.syntax_encoder(word_vecs)
        # Bundle word vectors into a shape (Phase B equivalent)
        shape_vec = normalize(bundle(list(word_vecs)))
        return shape_vec, bond_proposals

    def generate(self, shape_vec: torch.Tensor, n_tokens: int = 8,
                 verifier_fn=None) -> Tuple[torch.Tensor, bool]:
        """
        shape_vec: [dim]
        Returns (token_logits [n_tokens, vocab_size], passed_verifier bool)
        """
        token_logits, _ = self.syntax_decoder(shape_vec, n_tokens)
        token_logits = self.pragmatics(shape_vec, token_logits)
        passed = True
        if verifier_fn is not None:
            passed = verifier_fn(shape_vec, token_logits)
        return token_logits, passed

    def roundtrip_fidelity(self, token_ids: torch.Tensor) -> float:
        """
        Encode → decode → re-encode; measure cosine similarity of shapes.
        Target: >= 0.85
        """
        shape1, _ = self.understand(token_ids)
        token_logits, _ = self.generate(shape1, n_tokens=len(token_ids))
        # Re-encode using argmax tokens
        recon_ids = token_logits.argmax(dim=-1)
        shape2, _ = self.understand(recon_ids)
        return float(F.cosine_similarity(shape1.unsqueeze(0), shape2.unsqueeze(0)).item())
