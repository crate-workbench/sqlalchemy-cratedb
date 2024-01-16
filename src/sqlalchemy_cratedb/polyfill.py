import itertools

import sqlalchemy as sa
from sqlalchemy.event import listen


def polyfill_autoincrement_timestamp():
    """
    Configure SQLAlchemy model columns with an alternative to `autoincrement=True`.
    Use the current timestamp instead.

    This is used by CrateDB's LangChain adapter.

    TODO: Maybe enable through a dialect parameter `crate_polyfill_autoincrement` or such.
    """
    import sqlalchemy.sql.schema as schema
    from sqlalchemy import func

    init_dist = schema.Column.__init__

    def __init__(self, *args, **kwargs):
        if "autoincrement" in kwargs:
            del kwargs["autoincrement"]
            if "default" not in kwargs:
                kwargs["default"] = func.now()
        init_dist(self, *args, **kwargs)

    schema.Column.__init__ = __init__  # type: ignore[method-assign]


def check_uniqueness_factory(sa_entity, *attribute_names):
    """
    Run a manual column value uniqueness check on a table, and raise an IntegrityError if applicable.

    CrateDB does not support the UNIQUE constraint on columns. This attempts to emulate it.

    This is used by CrateDB's LangChain adapter.

    TODO: Maybe enable through a dialect parameter `crate_polyfill_unique` or such.
    """

    # Synthesize a canonical "name" for the constraint,
    # composed of all column names involved.
    constraint_name: str = "-".join(attribute_names)

    def check_uniqueness(mapper, connection, target):
        from sqlalchemy.exc import IntegrityError

        if isinstance(target, sa_entity):
            # TODO: How to use `session.query(SqlExperiment)` here?
            stmt = mapper.selectable.select()
            for attribute_name in attribute_names:
                stmt = stmt.filter(getattr(sa_entity, attribute_name) == getattr(target, attribute_name))
            stmt = stmt.compile(bind=connection.engine)
            results = connection.execute(stmt)
            if results.rowcount > 0:
                raise IntegrityError(
                    statement=stmt,
                    params=[],
                    orig=Exception(
                        f"DuplicateKeyException in table '{target.__tablename__}' " f"on constraint '{constraint_name}'"
                    ),
                )

    return check_uniqueness


def polyfill_refresh_after_dml_session(session: sa.orm.Session):
    """
    Run `REFRESH TABLE <tablename>` after each INSERT, UPDATE, and DELETE operation.

    CrateDB is eventually consistent, i.e. write operations are not flushed to
    disk immediately, so readers may see stale data. In a traditional OLTP-like
    application, this is not applicable.

    This SQLAlchemy extension makes sure that data is synchronized after each
    operation manipulating data.

    > `after_{insert,update,delete}` events only apply to the session flush operation
    > and do not apply to the ORM DML operations described at ORM-Enabled INSERT,
    > UPDATE, and DELETE statements. To intercept ORM DML events, use
    > `SessionEvents.do_orm_execute().`
    > -- https://docs.sqlalchemy.org/en/20/orm/events.html#sqlalchemy.orm.MapperEvents.after_insert

    > Intercept statement executions that occur on behalf of an ORM Session object.
    > -- https://docs.sqlalchemy.org/en/20/orm/events.html#sqlalchemy.orm.SessionEvents.do_orm_execute

    > Execute after flush has completed, but before commit has been called.
    > -- https://docs.sqlalchemy.org/en/20/orm/events.html#sqlalchemy.orm.SessionEvents.after_flush

    This is used by CrateDB's LangChain adapter.

    TODO: Maybe enable through a dialect parameter `crate_dml_refresh` or such.
    """  # noqa: E501
    listen(session, "after_flush", do_flush)


def polyfill_refresh_after_dml_engine(engine: sa.engine.Engine):
    def receive_after_execute(
        conn: sa.engine.Connection, clauseelement, multiparams, params, execution_options, result
    ):
        """
        Run a `REFRESH TABLE ...` command after each DML operation (INSERT, UPDATE, DELETE).

        This is used by CrateDB's Singer/Meltano and `rdflib-sqlalchemy` adapters.
        """

        if isinstance(clauseelement, (sa.sql.Insert, sa.sql.Update, sa.sql.Delete)):
            if not isinstance(clauseelement.table, sa.sql.Join):
                full_table_name = f'"{clauseelement.table.name}"'
                if clauseelement.table.schema is not None:
                    full_table_name = f'"{clauseelement.table.schema}".' + full_table_name
                conn.execute(sa.text(f'REFRESH TABLE {full_table_name};'))

    sa.event.listen(engine, "after_execute", receive_after_execute)


def do_flush(session, flush_context):
    """
    SQLAlchemy event handler for the 'after_flush' event,
    invoking `REFRESH TABLE` on each table which has been modified.
    """
    dirty_entities = itertools.chain(session.new, session.dirty, session.deleted)
    dirty_classes = {entity.__class__ for entity in dirty_entities}
    for class_ in dirty_classes:
        refresh_table(session, class_)


def refresh_table(connection, target):
    """
    Invoke a `REFRESH TABLE` statement.
    """
    sql = f"REFRESH TABLE {target.__tablename__}"
    connection.execute(sa.text(sql))
