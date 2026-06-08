"""Add last_accessed to report_cache

Revision ID: a1b2c3d4e5f6
Revises: f2c8a3d19e47
Create Date: 2026-06-08 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'f2c8a3d19e47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'report_cache',
        sa.Column('last_accessed', sa.DateTime(), nullable=True),
    )
    op.execute("UPDATE report_cache SET last_accessed = created_at WHERE last_accessed IS NULL")
    op.create_index('idx_report_cache_last_accessed', 'report_cache', ['last_accessed'])


def downgrade() -> None:
    op.drop_index('idx_report_cache_last_accessed', table_name='report_cache')
    op.drop_column('report_cache', 'last_accessed')
