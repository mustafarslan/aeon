"""add subject_scope_keys (task 6 phase B)

Revision ID: d6315ab6e406
Revises: d0e24ce99c88
Create Date: 2026-08-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd6315ab6e406'
down_revision: Union[str, Sequence[str], None] = 'd0e24ce99c88'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    v4-plan.md Stage 4 task 6 Phase B: one row per (subject_id, scope)
    pair, holding a wrapped (AES-256-GCM, under an env-configured KEK)
    per-subject-per-scope DEK for the shared store's encrypted metadata
    field. Rows are meant to be DELETED as the actual crypto-erase
    primitive, not soft-flagged -- see control_plane/schema.py's column
    comment.
    """
    op.create_table(
        'subject_scope_keys',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('subject_id', sa.String(length=256), nullable=False),
        sa.Column('scope', sa.Numeric(precision=20, scale=0), nullable=False),
        sa.Column('wrapped_dek', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('subject_id', 'scope', name='uq_subject_scope_keys_subject_scope'),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('subject_scope_keys')
