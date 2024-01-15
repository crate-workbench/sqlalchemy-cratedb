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

from datetime import datetime, tzinfo, timedelta
import datetime as dt
from unittest import TestCase, skipIf
from unittest.mock import patch, MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from sqlalchemy_cratedb import SA_VERSION, SA_1_4

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

from crate.client.cursor import Cursor


fake_cursor = MagicMock(name='fake_cursor')
FakeCursor = MagicMock(name='FakeCursor', spec=Cursor)
FakeCursor.return_value = fake_cursor


class CST(tzinfo):
    """
    Timezone object for CST
    """

    def utcoffset(self, date_time):
        return timedelta(seconds=-3600)

    def dst(self, date_time):
        return timedelta(seconds=-7200)


@skipIf(SA_VERSION < SA_1_4, "SQLAlchemy 1.3 suddenly has problems with these test cases")
@patch('crate.client.connection.Cursor', FakeCursor)
class SqlAlchemyDateAndDateTimeTest(TestCase):

    def setUp(self):
        self.engine = sa.create_engine('crate://')
        Base = declarative_base()

        class Character(Base):
            __tablename__ = 'characters'
            name = sa.Column(sa.String, primary_key=True)
            date = sa.Column(sa.Date)
            timestamp = sa.Column(sa.DateTime)

        fake_cursor.description = (
            ('characters_name', None, None, None, None, None, None),
            ('characters_date', None, None, None, None, None, None)
        )
        self.session = Session(bind=self.engine)
        self.Character = Character

    def test_date_can_handle_datetime(self):
        """ date type should also be able to handle iso datetime strings.

        this verifies that the fallback in the Date result_processor works.
        """
        fake_cursor.fetchall.return_value = [
            ('Trillian', '2013-07-16T00:00:00.000Z')
        ]
        self.session.query(self.Character).first()

    def test_date_can_handle_tz_aware_datetime(self):
        character = self.Character()
        character.name = "Athur"
        character.timestamp = datetime(2009, 5, 13, 19, 19, 30, tzinfo=CST())
        self.session.add(character)


Base = declarative_base()


class FooBar(Base):
    __tablename__ = "foobar"
    name = sa.Column(sa.String, primary_key=True)
    date = sa.Column(sa.Date)
    datetime = sa.Column(sa.DateTime)


@pytest.fixture
def session(cratedb_service):
    engine = cratedb_service.database.engine
    session = sessionmaker(bind=engine)()

    Base.metadata.drop_all(engine, checkfirst=True)
    Base.metadata.create_all(engine, checkfirst=True)
    return session


@pytest.mark.skipif(SA_VERSION < SA_1_4, reason="Test case not supported on SQLAlchemy 1.3")
def test_datetime_notz(session):
    """
    An integration test for `sa.Date` and `sa.DateTime`, not using timezones.
    """

    # Insert record.
    foo_item = FooBar(
        name="foo",
        date=dt.date(2009, 5, 13),
        datetime=dt.datetime(2009, 5, 13, 19, 19, 30, 123456),
    )
    session.add(foo_item)
    session.commit()
    session.execute(sa.text("REFRESH TABLE foobar"))

    # Query record.
    result = session.execute(sa.select(FooBar.name, FooBar.date, FooBar.datetime)).mappings().first()

    # Compare outcome.
    assert result["date"].year == 2009
    assert result["datetime"].year == 2009
    assert result["datetime"].tzname() is None
    assert result["datetime"].timetz() == dt.time(19, 19, 30, 123000)
    assert result["datetime"].tzinfo is None


@pytest.mark.skipif(SA_VERSION < SA_1_4, reason="Test case not supported on SQLAlchemy 1.3")
def test_datetime_tz(session):
    """
    An integration test for `sa.Date` and `sa.DateTime`, now using timezones.
    """

    # Insert record.
    foo_item = FooBar(
        name="foo",
        date=dt.date(2009, 5, 13),
        datetime=dt.datetime(2009, 5, 13, 19, 19, 30, 123456, tzinfo=CST()),
    )
    session.add(foo_item)
    session.commit()
    session.execute(sa.text("REFRESH TABLE foobar"))

    # Query record.
    result = session.execute(sa.select(FooBar.name, FooBar.date, FooBar.datetime)).mappings().first()

    # Compare outcome.
    assert result["date"].year == 2009
    assert result["datetime"].year == 2009
    assert result["datetime"].tzname() is None
    assert result["datetime"].timetz() == dt.time(19, 19, 30, 123000)
    assert result["datetime"].tzinfo is None
