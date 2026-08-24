"""
Aeon crypto-erase keystore (v4-plan.md Stage 4 task 6 Phase B) -- per-
subject-per-scope encryption for the shared Atlas store's metadata field.

Scope, per the task 6 decision record: BLOB/METADATA payloads only, not
centroid vectors. Vectors stay plaintext in the mmap file -- the
vector-encryption design (ATLAS_VERSION 3, stream-cipher-in-place,
decrypt-at-open into anonymous memory, the resulting RAM ceiling) was
considered and NOT pursued; embedding inversion means a destroyed
subject's vector remains partially informative about their original text
even after their key is destroyed (task 8's adopter-facing guarantee
documentation must state this).

Two independent pieces:

1. Keystore (below): one random 256-bit DEK per (subject_id, scope) pair,
   generated on first use, wrapped (AES-256-GCM) under a single
   deployment-wide KEK read from AEON_CRYPTO_ERASE_KEK_HEX (dependencies.py
   -- 503 when unset, same fail-closed pattern as
   AEON_AUDIT_LOG_EXPORT_KEY_HEX). DELETING a subject_scope_keys row is
   the actual crypto-erase primitive -- see control_plane/schema.py's
   column comment for why this can't be HKDF-derived from the KEK alone
   (a derived key can only be revoked by destroying the KEK itself, which
   would destroy every other subject's key too).

2. encrypt_metadata()/decrypt_metadata()/max_plaintext_bytes() (below):
   AES-256-GCM (not a bare stream cipher -- unlike the retired vector
   design, this field has no NodeHeader-style fixed-layout constraint, so
   there's no reason to give up the authentication tag). base64(nonce ||
   ciphertext || tag) is what actually gets written through
   Atlas.insert()'s metadata parameter and read back via
   get_node_metadata() -- required because get_node_metadata() returns
   std::string(meta), which stops at the first NUL byte, and raw
   ciphertext is likely to contain one at any real length.

Read/write sites (v4-plan.md's task 6 Phase B enumeration, checked by
grep, not assumed): promotion.py's promote_fragment() is the only WRITE
site (encrypts result.redacted_text before dest_atlas.atlas.insert());
server.py's GET /admin/knowledge is the only shared-store READ site
today. promotion.py's OWN read of source_atlas.atlas.get_node_metadata()
(the PRIVATE store, before promotion) is correctly untouched -- crypto-
erase doesn't cover the private store (no per-owner model there).
"""

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING, Optional, Union

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# sqlalchemy/control_plane.schema are NOT imported at module level -- this
# keeps encrypt_metadata()/decrypt_metadata()/is_encrypted_metadata()/
# max_plaintext_bytes() (pure functions, no Postgres needed) importable
# with zero DB extras installed, same "must keep working with zero DB
# dependencies installed" convention promotion.py's own TYPE_CHECKING
# comment states explicitly. Only Keystore (the class that actually talks
# to Postgres) imports them, lazily, inside its own methods.
if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

DEK_SIZE_BYTES = 32  # AES-256
NONCE_SIZE_BYTES = 12  # AESGCM standard nonce size
GCM_TAG_SIZE_BYTES = 16  # appended to ciphertext by AESGCM.encrypt()


class Keystore:
    """Postgres-backed store of per-(subject_id, scope) DEKs, each wrapped
    under a single deployment-wide KEK. Sync psycopg3, in-process --
    mirrors GovernanceDB/AdminDB/ErasureDB's shape and optionality
    conventions (accepts a DATABASE_URL or a shared Engine).
    """

    def __init__(self, database_url_or_engine: Union[str, "Engine"], kek: bytes):
        if isinstance(database_url_or_engine, str):
            from sqlalchemy import create_engine
            self._engine: "Engine" = create_engine(database_url_or_engine, pool_pre_ping=True)
            self._owns_engine = True
        else:
            self._engine = database_url_or_engine
            self._owns_engine = False
        self._aesgcm = AESGCM(kek)

    def _wrap(self, dek: bytes) -> str:
        nonce = os.urandom(NONCE_SIZE_BYTES)
        ciphertext = self._aesgcm.encrypt(nonce, dek, None)
        return base64.b64encode(nonce + ciphertext).decode("ascii")

    def _unwrap(self, wrapped_dek: str) -> bytes:
        raw = base64.b64decode(wrapped_dek)
        nonce, ciphertext = raw[:NONCE_SIZE_BYTES], raw[NONCE_SIZE_BYTES:]
        return self._aesgcm.decrypt(nonce, ciphertext, None)

    def get_or_create_dek(self, subject_id: str, scope: int) -> bytes:
        """Returns the DEK for (subject_id, scope), generating and storing
        a new one (wrapped) if none exists yet. Safe under concurrent
        first-use: ON CONFLICT DO NOTHING means a losing concurrent insert
        re-reads the WINNING row rather than trusting its own generated
        (and now orphaned) DEK.
        """
        from sqlalchemy import select
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from .control_plane.schema import subject_scope_keys

        with self._engine.begin() as conn:
            row = conn.execute(
                select(subject_scope_keys.c.wrapped_dek).where(
                    subject_scope_keys.c.subject_id == subject_id,
                    subject_scope_keys.c.scope == scope,
                )
            ).mappings().first()
            if row is not None:
                return self._unwrap(row["wrapped_dek"])

            dek = os.urandom(DEK_SIZE_BYTES)
            conn.execute(
                pg_insert(subject_scope_keys)
                .values(subject_id=subject_id, scope=scope, wrapped_dek=self._wrap(dek))
                .on_conflict_do_nothing(
                    index_elements=["subject_id", "scope"]
                )
            )
            row = conn.execute(
                select(subject_scope_keys.c.wrapped_dek).where(
                    subject_scope_keys.c.subject_id == subject_id,
                    subject_scope_keys.c.scope == scope,
                )
            ).mappings().first()
            return self._unwrap(row["wrapped_dek"])

    def get_dek(self, subject_id: str, scope: int) -> Optional[bytes]:
        """Read-only lookup -- returns None if no key exists for
        (subject_id, scope) rather than creating one. Use this on READ
        paths (e.g. the knowledge browser decrypting a listed node): a
        missing key after erasure is the whole point of the feature, not
        an error to paper over by minting a fresh, useless replacement."""
        from sqlalchemy import select

        from .control_plane.schema import subject_scope_keys

        with self._engine.begin() as conn:
            row = conn.execute(
                select(subject_scope_keys.c.wrapped_dek).where(
                    subject_scope_keys.c.subject_id == subject_id,
                    subject_scope_keys.c.scope == scope,
                )
            ).mappings().first()
            if row is None:
                return None
            return self._unwrap(row["wrapped_dek"])

    def destroy_key(self, subject_id: str, scope: int) -> bool:
        """Deletes the (subject_id, scope) DEK row -- the actual crypto-
        erase primitive (v4-plan.md task 6's gate: "demonstrates actual
        key destruction end-to-end, not just a tombstone flag"). Returns
        whether a row existed to delete. Same DELETE-vs-WAL/PITR/backups
        caveat as every other destructive Postgres operation this plan
        documents -- the KEK-wrapping is what makes that survive (a
        recovered wrapped row is unusable without the KEK).
        """
        from sqlalchemy import delete

        from .control_plane.schema import subject_scope_keys

        with self._engine.begin() as conn:
            result = conn.execute(
                delete(subject_scope_keys).where(
                    subject_scope_keys.c.subject_id == subject_id,
                    subject_scope_keys.c.scope == scope,
                )
            )
            return result.rowcount > 0

    def dispose(self) -> None:
        if self._owns_engine:
            self._engine.dispose()


# Prefix marking an encrypted metadata payload, distinguishing it from
# plaintext at read time WITHOUT needing a schema field (NodeHeader has no
# room -- see the task 6 decision record) or a DEK to attempt decryption
# with. Load-bearing for mixed deployments: a shared store that enables
# crypto-erase after nodes already exist has some nodes minted before a
# keystore was configured (permanently plaintext -- not retroactively
# encrypted, out of scope for this increment) alongside newly-promoted
# encrypted ones. is_encrypted_metadata() lets a reader tell them apart
# before ever touching the keystore.
_ENC_MARKER = "AEONENC1:"


def encrypt_metadata(dek: bytes, plaintext: str) -> str:
    """Encrypts plaintext for storage in the shared Atlas store's metadata
    field. Returns _ENC_MARKER + base64(nonce || ciphertext || tag).

    Does NOT length-check its own output -- the caller is responsible for
    checking `plaintext`'s encoded UTF-8 length against
    max_plaintext_bytes(atlas.metadata_size) BEFORE calling this (see that
    function's doc comment for why: Atlas.insert() truncates silently
    rather than raising on overflow, which corrupts ciphertext).
    """
    nonce = os.urandom(NONCE_SIZE_BYTES)
    ciphertext = AESGCM(dek).encrypt(nonce, plaintext.encode("utf-8"), None)
    return _ENC_MARKER + base64.b64encode(nonce + ciphertext).decode("ascii")


def is_encrypted_metadata(stored: str) -> bool:
    """True if `stored` (a value returned by Atlas.get_node_metadata())
    was written by encrypt_metadata() -- checked BEFORE attempting
    decrypt_metadata(), so a legacy-plaintext node (minted before a
    keystore was configured for this deployment) is returned as-is rather
    than raising a decrypt error."""
    return stored.startswith(_ENC_MARKER)


def decrypt_metadata(dek: bytes, stored: str) -> str:
    """Inverse of encrypt_metadata(). Raises ValueError if `stored` isn't
    marked as encrypted -- callers must check is_encrypted_metadata()
    first (see its doc comment)."""
    if not is_encrypted_metadata(stored):
        raise ValueError(
            "decrypt_metadata: stored value is not marked as encrypted "
            "(missing _ENC_MARKER) -- check is_encrypted_metadata() first"
        )
    raw = base64.b64decode(stored[len(_ENC_MARKER):])
    nonce, ciphertext = raw[:NONCE_SIZE_BYTES], raw[NONCE_SIZE_BYTES:]
    return AESGCM(dek).decrypt(nonce, ciphertext, None).decode("utf-8")


def max_plaintext_bytes(metadata_size: int) -> int:
    """Given an Atlas store's metadata_size (bytes), returns the maximum
    UTF-8-encoded plaintext length that fits after the marker prefix,
    nonce, GCM tag, and base64 overhead, reserving one byte for
    get_node_metadata()'s implicit C-string null terminator
    (Atlas::get_node_metadata() returns std::string(meta), which stops at
    the first NUL).

    Callers MUST length-check encrypt_metadata()'s intended plaintext
    against this BEFORE calling Atlas.insert() -- insert() truncates
    silently at metadata_size - 1 rather than raising, which for
    plaintext is lossy-but-tolerable (a shortened sentence) but for
    ciphertext is silent corruption (a cut nonce/tag/base64 group,
    discovered only as a decrypt failure far from the write that caused
    it).
    """
    usable_bytes = metadata_size - 1 - len(_ENC_MARKER)
    # base64 emits 4 output chars per 3 input bytes; only whole groups
    # decode safely.
    raw_budget = (usable_bytes // 4) * 3
    plaintext_budget = raw_budget - NONCE_SIZE_BYTES - GCM_TAG_SIZE_BYTES
    return max(0, plaintext_budget)
