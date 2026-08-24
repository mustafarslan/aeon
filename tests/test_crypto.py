"""
V4 Stage 4 task 6 Phase B: crypto.py's pure encoding functions
(encrypt_metadata/decrypt_metadata/is_encrypted_metadata/
max_plaintext_bytes) -- no Postgres needed, unlike Keystore itself (see
test_control_plane.py for the DB-backed round-trip tests).
"""
import os

import pytest

from aeon_py.crypto import (
    decrypt_metadata,
    encrypt_metadata,
    is_encrypted_metadata,
    max_plaintext_bytes,
)


def _dek() -> bytes:
    return os.urandom(32)


class TestEncryptDecryptMetadata:
    def test_round_trips(self):
        dek = _dek()
        stored = encrypt_metadata(dek, "hello world")
        assert decrypt_metadata(dek, stored) == "hello world"

    def test_empty_string_round_trips(self):
        dek = _dek()
        stored = encrypt_metadata(dek, "")
        assert decrypt_metadata(dek, stored) == ""

    def test_unicode_round_trips(self):
        dek = _dek()
        text = "café éè 中文"
        stored = encrypt_metadata(dek, text)
        assert decrypt_metadata(dek, stored) == text

    def test_different_deks_produce_undecryptable_ciphertext(self):
        # This IS the crypto-erase mechanism at the smallest possible
        # scale: a different key cannot decrypt content encrypted under
        # another key.
        stored = encrypt_metadata(_dek(), "secret")
        with pytest.raises(Exception):
            decrypt_metadata(_dek(), stored)

    def test_is_encrypted_metadata_true_for_encrypted_value(self):
        stored = encrypt_metadata(_dek(), "hello")
        assert is_encrypted_metadata(stored) is True

    def test_is_encrypted_metadata_false_for_legacy_plaintext(self):
        # A node minted before a keystore was ever configured for this
        # deployment -- must be distinguishable WITHOUT attempting
        # decrypt (crypto.py's module docstring: mixed deployments).
        assert is_encrypted_metadata("just some plain text") is False
        assert is_encrypted_metadata("") is False

    def test_decrypt_raises_on_unmarked_value(self):
        with pytest.raises(ValueError, match="not marked as encrypted"):
            decrypt_metadata(_dek(), "plain text, not encrypted")

    def test_ciphertext_varies_across_calls_same_plaintext(self):
        # Nonce discipline: encrypting the same plaintext twice must NOT
        # produce identical ciphertext (a fixed/reused nonce would be a
        # real cryptographic vulnerability, not just a style nit).
        dek = _dek()
        a = encrypt_metadata(dek, "same text")
        b = encrypt_metadata(dek, "same text")
        assert a != b


class TestMaxPlaintextBytes:
    def test_larger_metadata_size_yields_larger_budget(self):
        assert max_plaintext_bytes(512) > max_plaintext_bytes(256)

    def test_512_byte_field_exceeds_default_256_effective_budget(self):
        # v4-plan.md Stage 4 task 6 decision record's central numeric
        # claim: metadata_size=512 yields MORE usable plaintext than the
        # pre-existing ~250-char convention (dreamer.py), not less --
        # confirmed here, not just asserted in the plan doc.
        assert max_plaintext_bytes(512) > 250

    def test_never_negative(self):
        assert max_plaintext_bytes(0) == 0
        assert max_plaintext_bytes(1) == 0

    def test_budget_is_actually_usable_round_trip(self):
        # Proves the number ISN'T just arithmetic -- a plaintext exactly
        # at the computed budget must actually fit and round-trip through
        # a real encrypt call within a metadata_size-length field.
        metadata_size = 512
        budget = max_plaintext_bytes(metadata_size)
        plaintext = "x" * budget
        stored = encrypt_metadata(_dek(), plaintext)
        assert len(stored.encode("utf-8")) <= metadata_size - 1

    def test_one_byte_over_budget_would_not_fit(self):
        metadata_size = 512
        budget = max_plaintext_bytes(metadata_size)
        plaintext = "x" * (budget + 1)
        stored = encrypt_metadata(_dek(), plaintext)
        assert len(stored.encode("utf-8")) > metadata_size - 1


class TestGetCryptoEraseKek:
    """get_crypto_erase_kek() (dependencies.py) -- Postgres-independent,
    belongs here rather than a DB-gated file. Not previously covered by any
    test (confirmed by grep before adding these), despite COMPLIANCE.md §5.2
    now stating its exactly-32-byte requirement as a security guarantee in
    writing. DEFAULT_CRYPTO_ERASE_KEK_HEX is read once at import time, so
    these monkeypatch the already-imported module attribute directly rather
    than the environment variable -- get_crypto_erase_kek() itself carries no
    @lru_cache (only the downstream get_keystore() does), so each call here
    re-evaluates fresh against the patched value."""

    def test_none_when_unset(self, monkeypatch):
        from aeon_py import dependencies

        monkeypatch.setattr(dependencies, "DEFAULT_CRYPTO_ERASE_KEK_HEX", None)
        assert dependencies.get_crypto_erase_kek() is None

    def test_none_on_malformed_hex(self, monkeypatch):
        from aeon_py import dependencies

        monkeypatch.setattr(dependencies, "DEFAULT_CRYPTO_ERASE_KEK_HEX", "not-hex-zz")
        assert dependencies.get_crypto_erase_kek() is None

    def test_none_on_16_byte_key(self, monkeypatch):
        # AESGCM would happily accept this (AES-128) -- but crypto.py's
        # DEK_SIZE_BYTES is hardcoded 32 (AES-256), so a 16-byte KEK would
        # silently wrap 256-bit keys at 128-bit strength. Rejected, not
        # merely accepted-with-a-weaker-guarantee.
        from aeon_py import dependencies

        monkeypatch.setattr(dependencies, "DEFAULT_CRYPTO_ERASE_KEK_HEX", os.urandom(16).hex())
        assert dependencies.get_crypto_erase_kek() is None

    def test_none_on_24_byte_key(self, monkeypatch):
        from aeon_py import dependencies

        monkeypatch.setattr(dependencies, "DEFAULT_CRYPTO_ERASE_KEK_HEX", os.urandom(24).hex())
        assert dependencies.get_crypto_erase_kek() is None

    def test_returns_bytes_for_valid_32_byte_key(self, monkeypatch):
        from aeon_py import dependencies

        raw = os.urandom(32)
        monkeypatch.setattr(dependencies, "DEFAULT_CRYPTO_ERASE_KEK_HEX", raw.hex())
        assert dependencies.get_crypto_erase_kek() == raw
