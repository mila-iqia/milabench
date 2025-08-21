"""Add a status metric

Revision ID: 79f64511ee0e
Revises: 1f7d08d59388
Create Date: 2025-05-27 11:46:26.442911

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import select, insert


# revision identifiers, used by Alembic.
revision: str = '79f64511ee0e'
down_revision: Union[str, None] = '1f7d08d59388'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    from milabench.metrics.sqlalchemy import Pack, Metric

    session = sa.orm.Session(op.get_bind())

    # Get early stop packs
    early_stop_packs = select(Metric).where(
        Metric.name == 'return_code',
        Metric.value == -15
    ).distinct()

    # Create new status metrics for early stop packs
    session.execute(
        insert(Metric).from_select(
            ['exec_id', 'pack_id', 'name', 'value', 'order', 'gpu_id', 'job_id'],
            select(
                early_stop_packs.c.exec_id,
                early_stop_packs.c.pack_id,
                sa.literal('status'),
                sa.literal(0),
                early_stop_packs.c.order,
                early_stop_packs.c.gpu_id,
                early_stop_packs.c.job_id
            )
        )
    )

    # Update packs with non-zero return codes (except -15) to error
    error_packs = select(Metric).where(
        Metric.name == 'return_code',
        Metric.value != -15,
        Metric.value != 0
    ).distinct()

    session.execute(
        insert(Metric).from_select(
            ['exec_id', 'pack_id', 'name', 'value', 'order', 'gpu_id', 'job_id'],
            select(
                error_packs.c.exec_id,
                error_packs.c.pack_id,
                sa.literal('status'),
                sa.literal(1),
                error_packs.c.order,
                error_packs.c.gpu_id,
                error_packs.c.job_id
            )
        )
    )

    # Update packs with return_code 0 to done
    done_packs = select(Metric).where(
        Metric.name == 'return_code',
        Metric.value == 0
    ).distinct()

    session.execute(
        insert(Metric).from_select(
            ['exec_id', 'pack_id', 'name', 'value', 'order', 'gpu_id', 'job_id'],
            select(
                done_packs.c.exec_id,
                done_packs.c.pack_id,
                sa.literal('status'),
                sa.literal(0),
                done_packs.c.order,
                done_packs.c.gpu_id,
                done_packs.c.job_id
            )
        )
    )

def downgrade() -> None:
    """Downgrade schema."""
    pass
