# Changelog


## Unreleased
- Added utility support wrapping CrateDB's special table options

## 2024/06/25 0.38.0
- Added/reactivated documentation as `sqlalchemy-cratedb`
- Added `CrateIdentifierPreparer`, in order to quote reserved words
  like `object` properly, for example when used as column names.
- Fixed `CrateDialect.get_pk_constraint` to return `list` instead of `set` type
- Added re-usable patches and polyfills from application adapters.
  New utilities: `patch_autoincrement_timestamp`, `refresh_after_dml`,
  `check_uniqueness_factory`
- Added `table_kwargs` context manager to enable pandas/Dask to support
  CrateDB dialect table options.
- Fixed SQL rendering of special DDL table options in `CrateDDLCompiler`.
  Before, configuring `crate_"translog.durability"` was not possible.
- Unlocked supporting timezone-aware `DateTime` fields
- Added support for marshalling Python `datetime.date` values on `sa.DateTime` fields

## 2024/06/13 0.37.0
- Added support for CrateDB's [FLOAT_VECTOR] data type and its accompanying
  [KNN_MATCH] function, for HNSW matches. For SQLAlchemy column definitions,
  you can use it like `FloatVector(dimensions=1536)`.
- Fixed `get_table_names()` reflection method to respect the
  `schema` query argument in SQLAlchemy connection URLs.

[FLOAT_VECTOR]: https://cratedb.com/docs/crate/reference/en/latest/general/ddl/data-types.html#float-vector
[KNN_MATCH]: https://cratedb.com/docs/crate/reference/en/latest/general/builtins/scalar-functions.html#scalar-knn-match

## 2024/06/11 0.36.1
- Dependencies: Use `crate==1.0.0dev0`

## 2024/06/11 0.36.0
- Dependencies: Use `dask[dataframe]`
- Maintenance release after splitting packages `crate-python` vs.
  `sqlalchemy-cratedb`

## 2023/09/29 0.34.0

- Fix handling URL parameters `timeout` and `pool_size`
- Improve DDL compiler to ignore foreign key and uniqueness constraints.
- Ignore SQL's `FOR UPDATE` clause. Thanks, @surister.


## 2023/07/17 0.33.0

- Rename leftover occurrences of `Object`. The new symbol to represent
  CrateDB's `OBJECT` column type is now `ObjectType`.

- DQL: Use CrateDB's native `ILIKE` operator instead of using SA's
  generic implementation `lower() LIKE lower()`. Thanks, @hlcianfagna.


## 2023/07/06 0.32.0

- DDL: Allow turning off column store using `crate_columnstore=False`.
  Thanks, @fetzerms.

- DDL: Allow setting `server_default` on columns to enable
  server-generated defaults. Thanks, @JanLikar.

- Allow handling datetime values tagged with time zone info when inserting or updating.

- Fix SQL statement caching for CrateDB's `OBJECT` type. Thanks, @faymarie.

- Refactor `OBJECT` type to use SQLAlchemy's JSON type infrastructure.

- Added `insert_bulk` fast-path `INSERT` method for pandas, in
  order to support efficient batch inserts using CrateDB's "bulk operations" endpoint.

- Add documentation and software tests for usage with Dask


## 2023/04/18 0.31.1

- Core: Re-enable support for `INSERT/UPDATE...RETURNING` in
  SQLAlchemy 2.0 by adding the new `insert_returning` and `update_returning` flags
  in the CrateDB dialect.


## 2023/03/30 0.31.0

- Core: Support `INSERT...VALUES` with multiple value sets by enabling
  `supports_multivalues_insert` on the CrateDB dialect, it is used by pandas'
  `method="multi"` option

- Core: Enable the `insertmanyvalues` feature, which lets you control
  the batch size of `INSERT` operations using the `insertmanyvalues_page_size`
  engine-, connection-, and statement-options.

- ORM: Remove support for the legacy `session.bulk_save_objects` API
  on SQLAlchemy 2.0, in favor of the new `insertmanyvalues` feature. Performance
  optimizations from `bulk_save()` have been made inherently part of `add_all()`.
  Note: The legacy mode will still work on SQLAlchemy 1.x, while SQLAlchemy 2.x users
  MUST switch to the new method now.


## 2023/03/02 0.30.1

- Fixed SQLAlchemy 2.0 incompatibility with `CrateDialect.{has_schema,has_table}`


## 2023/02/16 0.30.0

- Added deprecation warning about dropping support for SQLAlchemy 1.3 soon, it
  is effectively EOL.

- Added support for SQLAlchemy 2.0. See also [What's New in SQLAlchemy 2.0]
  and [SQLAlchemy 2.0 migration guide].

- Updated to geojson 3.0.0.

[SQLAlchemy 2.0 migration guide]: https://docs.sqlalchemy.org/en/20/changelog/migration_20.html
[What's New in SQLAlchemy 2.0]: https://docs.sqlalchemy.org/en/20/changelog/whatsnew_20.html


## 2022/12/08 0.29.0

- Added support for `crate_index` and `nullable` attributes in
  ORM column definitions.

- Added support for converting `TIMESTAMP` columns to timezone-aware
  `datetime` objects, using the new `time_zone` keyword argument.


## 2022/12/02 0.28.0

- Added a generic data type converter to the `Cursor` object, for converting
  fetched data from CrateDB data types to Python data types.

- Fixed generating appropriate syntax for OFFSET/LIMIT clauses. It was possible
  that SQL statement clauses like `LIMIT -1` could have been generated. Both
  PostgreSQL and CrateDB only accept `LIMIT ALL` instead.

- Added support for ORM computed columns


## 2022/10/10 0.27.2
- Improved `CrateDialect.get_pk_constraint` to be compatible
  with breaking changes in CrateDB >=5.1.0.


## 2022/06/02 0.27.0

- Added support for Python 3.9 and 3.10.

- Dropped support for Python 3.4, 3.5 and 3.6.

- Dropped support for SQLAlchemy 1.1 and 1.2.

- Dropped support for CrateDB < 2.0.0.

- Added support for enabling SSL using SQLAlchemy DB URI with parameter
  `?ssl=true`.

- Added support for SQLAlchemy 1.4

### Notes

For learning about the transition to SQLAlchemy 1.4, we recommend the
corresponding documentation [What’s New in SQLAlchemy 1.4?].

### Breaking changes

#### Textual column expressions
SQLAlchemy 1.4 became stricter on some details. It requires to wrap [CrateDB
system columns] like `_score` in a [SQLAlchemy literal_column] type.
Before, it was possible to use a query like this:

    session.query(Character.name, '_score')

It must now be written like:

    session.query(Character.name, sa.literal_column('_score'))

Otherwise, SQLAlchemy will complain like:

    sqlalchemy.exc.ArgumentError: Textual column expression '_score' should be
    explicitly declared with text('_score'), or use column('_score') for more
    specificity


[CrateDB system columns]: https://cratedb.com/docs/crate/reference/en/4.8/general/ddl/system-columns.html
[SQLAlchemy literal_column]: https://docs.sqlalchemy.org/en/14/core/sqlelement.html#sqlalchemy.sql.expression.literal_column
[What’s New in SQLAlchemy 1.4?]: https://docs.sqlalchemy.org/en/14/changelog/migration_14.html


## 2020/09/28 0.26.0
- Propagate connect parameter `pool_size` to urllib3 as `maxsize` parameter
  in order to make the connection pool size configurable.

## 2020/08/05 0.25.0
- Added support for the `RETURNING` clause. This requires CrateDB 4.2 or greater.
  In case you use any server side generated columns in your primary key constraint
  with earlier CrateDB versions, you can turn off this feature by passing
  `implicit_returning=False` in the `create_engine()` call.
- Added support for `geo_point` and `geo_json` types

## 2020/05/27 0.24.0
- Upgraded SQLAlchemy support to 1.3.
- Added official Python 3.8 support.
- Made it so that the dialect is now aware of the return type of the
  `date_trunc` function.
- Added driver attribute, as SQLAlchemy relies on interfaces having that string for identification.

## 2019/08/01 0.23.1
- Extended the type mapping for the upcoming type name changes in CrateDB 4.0.
- Added support for Python 3.7 and made that version the recommended one.

## 2018/05/02 0.22.0
- BREAKING: Dropped support for Python 2.7 and 3.3
  If you are using this package with Python 2.7 or 3.3 already, you will not be
  able to install newer versions of this package.
- Add support for SQLAlchemy 1.2
- Updated `get_table_names()` method to only return tables but not views.
  This enables compatibility with CrateDB 3.0 and newer.

## 2018/03/14 0.21.3
- Fixed an issue that caused `metadata.create_all(bind=engine)` to fail
  creating tables that contain an `ObjectArray` column.

## 2018/01/03 0.21.1
- Fixed an issue that prevented the usage of SQLAlchemy types `NUMERIC` and
  `DECIMAL` as column types.

## 2017/12/07 0.21.0
- Prepared primary key retrieval for CrateDB 2.3.0. Preserved
  backwards-compatibility for lower versions.

## 2017/02/02 0.18.0
- Added support for `Insert` from select
- Support `get_columns` and `get_pk_constraint`

## 2016/12/19 0.17.0
- BREAKING: Dropped support for SQLAlchemy < 1.0.0
- Fix: The dialect didn't work properly with alpha and beta
  versions of sqlalchemy due to a wrong version check
  (e.g.: sandman2 depends on 1.1.0b3)
- Added support for native Arrays
- Fix: `sa.inspect(engine).get_table_names` failed due
  to an attribute error

## 2016/11/21 0.16.5
- Added compatibility for SQLAlchemy version 1.1

## 2016/10/18 0.16.4
- Fix: Updates in nested `OBJECT` columns have been ignored

## 2016/06/23 0.16.1
- Fix: `Date` column type is now correctly created as `TIMESTAMP` column
  when creating the table

## 2016/06/09 0.16.0
- Added support for serialization of Decimals

## 2016/05/17 0.15.0
- Added support for client certificates
- Dropped support for Python 2.6

## 2016/02/05 0.14.0
- Added support for serialization of date and datetime objects

## 2015/10/12 0.13.5
- Fix: use proper `CLUSTERED` clause syntax in `CREATE TABLE` statement

## 2015/06/29 0.13.3
- Fix: Allow `ObjectArray`s to be set to `None`

## 2015/05/29 0.13.1
- Fixed compatibility issues with SQLAlchemy 1.0.x
- Map SQLAlchemy's `TEXT` column type to CrateDB's `STRING` type

## 2015/03/10 0.13.0
- Add support for table creation using the SQLAlchemy ORM functionality
- Fix: Match predicate now properly handles term literal

## 2015/02/13 0.12.5
- Changed update statement generation to be compatible with CrateDB 0.47.X

## 2015/02/04 0.12.4
- Add missing functionality in `CrateDialect`, including:
  default schema name, server version info,
  check if table/schema exists, list all tables/schemas

## 2014/10/20 0.12.2
- Add `match` predicate to support fulltext search

## 2014/07/14 0.10.3
- Fix: Columns that have an onupdate definition are now correctly updated

## 2014/05/16 0.10.0
- Implemented `ANY` operator on object array containment checks

## 2014/05/13 0.9.5
- Bugfix: Updates of complex types will only be rewritten if the dialect
  is set to `crate`

## 2014/05/07 0.9.1
- use new crate doc theme

## 2014/03/11 0.4.0
- Fix a bug where setting an empty list on a multivalued field results in
  returning `None` after refreshing the session

## 2014/01/27 0.3.0
- Add the `ObjectArray` type
- Rename `Craty` type to `Object`. `Craty` can still be imported to
  maintain backward compatibility.

## 2014/01/15 0.2.0
- Compatibility adjustments for SQLAlchemy >= 0.9.x

## 2013/12/06 0.1.9
- Support native booleans

## 2013/11/25 0.1.7
- Raise an exception if timezone aware datetime objects are stored

## 2013/11/08 0.1.3
- Fix datetime parsing that didn't work with crate >= 0.18.4 due
  to the fixed datetime mapping

## 2013/11/08 0.1.2
- Document `count()` and `group_by()` support

## 2013/10/09 0.0.9
- `DateTime` and `Date` can now be nullable

## 2013/10/04 0.0.8
- Fix an ORM error with the `Craty` type and where the `update`
  statement wasn't correctly generated

## 2013/10/02 0.0.7
- Support the `Date` and `DateTime` types

## 2013/10/01 0.0.6
- Initial release of SQLAlchemy dialect including complex types

## 2013/09/05 0.0.2
- Initial release of Python DBAPI driver
