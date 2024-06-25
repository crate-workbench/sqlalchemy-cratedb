# -*- coding: utf-8; -*-
#
# Licensed to CRATE Technology GmbH ("Crate") under one or more contributor
# license agreements.  See the NOTICE file distributed with this work for
# additional information regarding copyright ownership.  Crate licenses
# this file to you under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may
# obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations
# under the License.
#
# However, if you have executed another commercial license agreement
# with Crate these terms will supersede the license and you may use the
# software solely pursuant to the terms of the relevant commercial agreement.

from __future__ import absolute_import

import os
import sys
import unittest
import doctest
from pprint import pprint
import logging

from crate.client import connect
from sqlalchemy_cratedb import SA_VERSION, SA_2_0, SA_1_4
from tests.settings import crate_host

log = logging.getLogger()
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
log.addHandler(ch)


def cprint(s):
    if isinstance(s, bytes):
        s = s.decode('utf-8')
    print(s)


def docs_path(*parts):
    return os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), *parts
        )
    )


def provision_database():

    drop_tables()

    with connect(crate_host) as conn:
        cursor = conn.cursor()

        with open(docs_path('tests/assets/locations.sql')) as s:
            stmt = s.read()
            cursor.execute(stmt)
            stmt = ("SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name = 'locations'")
            cursor.execute(stmt)
            assert cursor.fetchall()[0][0] == 1

        # `/assets` is located within the Docker container used for running CrateDB.
        # docker compose -f tests/docker-compose.yml up
        data_path = "/assets/locations.jsonl"

        # load testing data into crate
        cursor.execute("COPY locations FROM ?", (data_path,))
        # refresh location table so imported data is visible immediately
        cursor.execute("REFRESH TABLE locations")
        # create blob table
        cursor.execute("CREATE BLOB TABLE myfiles CLUSTERED INTO 1 SHARDS " +
                       "WITH (number_of_replicas=0)")

        # create users
        cursor.execute("CREATE USER me WITH (password = 'my_secret_pw')")
        cursor.execute("CREATE USER trusted_me")

        cursor.close()

    ddl_statements = [
        """
        CREATE TABLE characters (
            id STRING PRIMARY KEY,
            name STRING,
            quote STRING,
            details OBJECT,
            more_details ARRAY(OBJECT),
            INDEX name_ft USING FULLTEXT(name) WITH (analyzer = 'english'),
            INDEX quote_ft USING FULLTEXT(quote) WITH (analyzer = 'english')
            )""",
        """
        CREATE VIEW characters_view
            AS SELECT * FROM characters
        """,
        """
        CREATE TABLE cities (
            name STRING PRIMARY KEY,
            coordinate GEO_POINT,
            area GEO_SHAPE
        )""",
        """
        CREATE TABLE search (
            name STRING PRIMARY KEY,
            text STRING,
            embedding FLOAT_VECTOR(3)
        )""",
    ]
    _execute_statements(ddl_statements)


def drop_tables():
    """
    Drop all tables, views, and users created by the test suite.
    """
    ddl_statements = [
        "DROP TABLE IF EXISTS archived_tasks",
        "DROP TABLE IF EXISTS characters",
        "DROP TABLE IF EXISTS cities",
        "DROP TABLE IF EXISTS foobar",
        "DROP TABLE IF EXISTS locations",
        "DROP BLOB TABLE IF EXISTS myfiles",
        "DROP TABLE IF EXISTS search",
        'DROP TABLE IF EXISTS "test-testdrive"',
        "DROP TABLE IF EXISTS todos",
        'DROP TABLE IF EXISTS "user"',
        "DROP VIEW IF EXISTS characters_view",
        "DROP USER IF EXISTS me",
        "DROP USER IF EXISTS trusted_me",
    ]
    _execute_statements(ddl_statements)


def _execute_statements(statements, on_error="raise"):
    with connect(crate_host) as conn:
        cursor = conn.cursor()
        for stmt in statements:
            _execute_statement(cursor, stmt, on_error=on_error)
        cursor.close()


def _execute_statement(cursor, stmt, on_error="raise"):
    if on_error not in ["ignore", "raise"]:
        raise ValueError(f"Invalid value for `on_error` argument: {on_error}")
    try:
        cursor.execute(stmt)
    except Exception:  # pragma: no cover
        # FIXME: Why does this croak on statements like ``DROP TABLE cities``?
        # Note: When needing to debug the test environment, you may want to
        #       enable this logger statement.
        # log.exception("Executing SQL statement failed")
        if on_error == "ignore":
            pass
        elif on_error == "raise":
            raise


def setUp(test):

    provision_database()

    test.globs['crate_host'] = crate_host
    test.globs['pprint'] = pprint
    test.globs['print'] = cprint


def tearDown(test):
    pass


def create_test_suite():
    suite = unittest.TestSuite()
    flags = (doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)

    sqlalchemy_integration_tests = [
        'docs/getting-started.rst',
        'docs/crud.rst',
        'docs/working-with-types.rst',
        'docs/advanced-querying.rst',
    ]

    # Don't run DataFrame integration tests on SQLAlchemy 1.4 and earlier, or Python 3.7.
    skip_dataframe = SA_VERSION < SA_2_0 or sys.version_info < (3, 8)
    if not skip_dataframe:
        sqlalchemy_integration_tests += [
            'docs/dataframe.rst',
        ]

    # Don't run reflection integration tests on SQLAlchemy 1.3 and earlier and Python 3.10 and 3.11.
    skip_reflection = SA_VERSION < SA_1_4 and (3, 10) <= sys.version_info < (3, 12)
    if not skip_reflection:
        sqlalchemy_integration_tests += [
            'docs/inspection-reflection.rst',
        ]

    s = doctest.DocFileSuite(
        *sqlalchemy_integration_tests,
        module_relative=False,
        setUp=setUp,
        tearDown=tearDown,
        optionflags=flags,
        encoding='utf-8'
    )
    suite.addTest(s)

    return suite


def load_tests(loader, tests, pattern):
    """
    Provide test suite to test discovery.

    https://docs.python.org/3/library/unittest.html#load-tests-protocol
    """
    return create_test_suite()
