"""Add report_cache table

Revision ID: f2c8a3d19e47
Revises: b3a1f7e20d41
Create Date: 2026-06-05 13:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f2c8a3d19e47'
down_revision: Union[str, None] = 'b3a1f7e20d41'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'report_cache',
        sa.Column('_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('exec_id', sa.Integer(), sa.ForeignKey('execs._id'), nullable=False),
        sa.Column('profile', sa.String(length=256), nullable=False),
        sa.Column('bench', sa.String(length=256), nullable=False),
        sa.Column('fail', sa.Integer(), nullable=True),
        sa.Column('n', sa.Float(), nullable=True),
        sa.Column('ngpu', sa.Float(), nullable=True),
        sa.Column('perf', sa.Float(), nullable=True),
        sa.Column('sem', sa.Float(), nullable=True),
        sa.Column('std', sa.Float(), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('log_score', sa.Float(), nullable=True),
        sa.Column('weight', sa.Float(), nullable=True),
        sa.Column('enabled', sa.Float(), nullable=True),
        sa.Column('order', sa.Float(), nullable=True),
        sa.Column('weight_total', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('_id'),
        sa.UniqueConstraint('exec_id', 'profile', 'bench', name='uq_report_cache_row'),
    )
    op.create_index('idx_report_cache_exec_profile', 'report_cache', ['exec_id', 'profile'])
    op.create_index('idx_report_cache_created', 'report_cache', ['created_at'])


def downgrade() -> None:
    op.drop_index('idx_report_cache_created', table_name='report_cache')
    op.drop_index('idx_report_cache_exec_profile', table_name='report_cache')
    op.drop_table('report_cache')
