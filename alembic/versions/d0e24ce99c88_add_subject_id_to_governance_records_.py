"""add subject_id to governance_records (task 6 phase A)

Revision ID: d0e24ce99c88
Revises: 57beb688a09c
Create Date: 2026-08-23 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd0e24ce99c88'
down_revision: Union[str, Sequence[str], None] = '57beb688a09c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    v4-plan.md Stage 4 task 6 Phase A (crypto-erase design spike):
    promote_fragment() now requires a subject_id identifying whichever
    private-store owner a promoted fragment's content derives from --
    the (subject_id, dest_scope) pair a future crypto-erase DEK lookup
    resolves through NodeHeader.governance_record_id. NOT NULL with no
    server_default: this repo carries no pre-existing governance_records
    rows to backfill (pre-GA) -- see control_plane/schema.py's column
    comment.
    """
    op.add_column(
        'governance_records',
        sa.Column('subject_id', sa.String(length=256), nullable=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('governance_records', 'subject_id')
