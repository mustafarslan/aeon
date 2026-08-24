import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# shell/aeon_py isn't installed as a package when running alembic
# directly from a fresh checkout before `pip install -e .` -- add it to
# sys.path the same way tests/test_phase8.py does, so `import aeon_py...`
# below works regardless of install state.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shell"))

from aeon_py.control_plane.schema import metadata as aeon_control_plane_metadata

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# AEON_CONTROL_PLANE_DATABASE_URL (not alembic.ini's sqlalchemy.url, which
# stays a placeholder) is the single source of truth for which database
# migrations run against -- same env var control_plane/dependencies.py
# reads, so `alembic upgrade head` and the running app can never point at
# different databases by accident.
_db_url = os.environ.get("AEON_CONTROL_PLANE_DATABASE_URL")
if _db_url:
    config.set_main_option("sqlalchemy.url", _db_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = aeon_control_plane_metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
