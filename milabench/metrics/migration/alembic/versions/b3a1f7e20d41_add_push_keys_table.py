"""Add push_keys table

Revision ID: b3a1f7e20d41
Revises: e9ab9d168607
Create Date: 2026-05-27 09:30:00.000000

"""
from typing import Sequence, Union
import os

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b3a1f7e20d41'
down_revision: Union[str, None] = 'e9ab9d168607'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('push_keys',
        sa.Column('_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(length=256), nullable=False),
        sa.Column('key', sa.String(length=64), nullable=False),
        sa.Column('created_time', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('_id'),
        sa.UniqueConstraint('name'),
        sa.UniqueConstraint('key'),
    )
    op.create_index('push_keys_key', 'push_keys', ['key'], unique=False)
    op.create_index('push_keys_name', 'push_keys', ['name'], unique=False)

    app_user = os.getenv("POSTGRES_USER")
    if app_user:
        op.execute(sa.text(f'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE push_keys TO "{app_user}"'))
        op.execute(sa.text(f'GRANT USAGE, SELECT ON SEQUENCE push_keys__id_seq TO "{app_user}"'))


def downgrade() -> None:
    op.drop_index('push_keys_name', table_name='push_keys')
    op.drop_index('push_keys_key', table_name='push_keys')
    op.drop_table('push_keys')
