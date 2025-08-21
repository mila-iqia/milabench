"""Convert SEGINT to valid runs

Revision ID: 1f7d08d59388
Revises: 149eb1393c1e
Create Date: 2025-05-27 11:31:41.244610

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import select, update



# revision identifiers, used by Alembic.
revision: str = '1f7d08d59388'
down_revision: Union[str, None] = '149eb1393c1e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    from milabench.metrics.sqlalchemy import Pack, Metric

    session = sa.orm.Session(op.get_bind())

    # Update packs with return_code -15 to early_stop
    early_stop_packs = select(Metric.pack_id).where(
        Metric.name == 'return_code',
        Metric.value == -15
    ).distinct()
    
    session.execute(
        update(Pack)
        .where(Pack._id.in_(early_stop_packs))
        .values(status='early_stop')
    )

    # Update packs with non-zero return codes (except -15) to error
    error_packs = select(Metric.pack_id).where(
        Metric.name == 'return_code',
        Metric.value != -15,
        Metric.value != 0
    ).distinct()
    session.execute(
        update(Pack)
        .where(Pack._id.in_(error_packs))
        .values(status='error')
    )

    # Update packs with return_code 0 to done
    done_packs = select(Metric.pack_id).where(
        Metric.name == 'return_code',
        Metric.value == 0
    ).distinct()
    session.execute(
        update(Pack)
        .where(Pack._id.in_(done_packs))
        .values(status='done')
    )


def downgrade() -> None:
    """Downgrade schema."""
    pass
