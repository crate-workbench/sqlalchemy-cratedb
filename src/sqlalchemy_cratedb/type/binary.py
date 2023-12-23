import base64
import sqlalchemy as sa


class LargeBinary(sa.String):

    """A type for large binary byte data.

    The :class:`.LargeBinary` type corresponds to a large and/or unlengthed
    binary type for the target platform, such as BLOB on MySQL and BYTEA for
    PostgreSQL.  It also handles the necessary conversions for the DBAPI.

    """

    __visit_name__ = "large_binary"

    def bind_processor(self, dialect):
        if dialect.dbapi is None:
            return None

        # TODO: DBAPIBinary = dialect.dbapi.Binary

        def process(value):
            if value is not None:
                # TODO: return DBAPIBinary(value)
                return base64.b64encode(value).decode()
            else:
                return None

        return process

    # Python 3 has native bytes() type
    # both sqlite3 and pg8000 seem to return it,
    # psycopg2 as of 2.5 returns 'memoryview'
    def result_processor(self, dialect, coltype):
        if dialect.returns_native_bytes:
            return None

        def process(value):
            if value is not None:
                return base64.b64decode(value)
            return value

        return process
