"""Declarative base shared by all ORM models."""

from datetime import datetime

from sqlalchemy import DateTime, MetaData
from sqlalchemy.orm import DeclarativeBase

# Deterministic constraint names. Without them Postgres invents names, and Alembic cannot reliably
# drop or alter those constraints in later migrations.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Base class for every table."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    # Any `Mapped[datetime]` becomes TIMESTAMP WITH TIME ZONE, so a naive timestamp column can't be declared by mistake.
    type_annotation_map = {datetime: DateTime(timezone=True)}
