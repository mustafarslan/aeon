"""add supersession governance actions (Stage 5 task 2)

Revision ID: 8f3a1c9e2b7d
Revises: d6315ab6e406
Create Date: 2026-08-23 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8f3a1c9e2b7d'
down_revision: Union[str, Sequence[str], None] = 'd6315ab6e406'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # NOT auto-detected (CHECK constraint diffs never are -- same gap the
    # 'add erasure workflow' migration hit): widen
    # ck_governance_records_action to admit supersession.py's two new
    # action values (governance.py's GOVERNANCE_RECORD_ACTIONS).
    op.drop_constraint('ck_governance_records_action', 'governance_records', type_='check')
    op.create_check_constraint(
        'ck_governance_records_action',
        'governance_records',
        "action IN ('promotion', 'promotion_rejected', 'promotion_unscoped_anomaly', "
        "'erasure', 'supersession', 'supersession_revoked')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('ck_governance_records_action', 'governance_records', type_='check')
    op.create_check_constraint(
        'ck_governance_records_action',
        'governance_records',
        "action IN ('promotion', 'promotion_rejected', 'promotion_unscoped_anomaly', 'erasure')",
    )
