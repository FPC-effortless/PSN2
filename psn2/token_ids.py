"""Stable hashed diagnostic identities; not a lossless language tokenizer.

Version2 changes old process-dependent IDs: old checkpoints require the old
encoding and cannot be evaluated interchangeably with this mapping.
"""
import hashlib

TOKEN_ID_VERSION = "blake2b-psn2-id-v2"

def stable_token_id(text: str, vocab: int) -> int:
    if type(vocab) is not int or vocab < 1:
        raise ValueError("vocab must be positive")
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=b"psn2-id-v2").digest()
    return int.from_bytes(digest, "big") % vocab
