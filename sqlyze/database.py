import os

DIRNAME = os.path.dirname(__file__)
ENVPATH = os.path.join(DIRNAME, ".env")
import pandas as pd

from typing import Dict, List, Union, Optional, Any
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pydantic.main import BaseModel

from sqlalchemy import create_engine, text
from sqlalchemy.sql.elements import TextClause
from sqlalchemy.sql.selectable import TextualSelect
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.engine.base import Engine, Connection
from sqlalchemy.ext.asyncio.engine import AsyncEngine, AsyncConnection
from config import settings


class DataBase:
    def __init__(self, settings=settings):
        self._async_engine = create_async_engine(
            settings.ASYNC_DATABASE_URL, pool_pre_ping=True
        )
        self._engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

    @property
    def async_engine(self) -> AsyncEngine:
        if not self._async_engine:
            self._async_engine = create_async_engine(settings.ASYNC_DATABASE_URL)
        return self._async_engine

    @property
    def engine(self) -> Engine:
        if not self._async_engine:
            self.async_engine

        if not self._engine:
            self._engine = create_engine(settings.DATABASE_URL)
        return self._engine

    @property
    def con(self) -> Any:
        con = self._engine.begin()
        return con

    def _build_text_sql(
        self, sql: Union[str, TextClause, TextualSelect]
    ) -> Union[TextClause, TextualSelect]:
        if isinstance(sql, TextClause) or isinstance(sql, TextualSelect):
            return sql
        text_sql = text(sql)
        return text_sql

    def vaildate(self, data: DataFrame, model: type[BaseModel]):
        df = pd.DataFrame(
            [model(**record).dict() for record in data.to_dict(orient="records")]
        )
        return df

    def read_sql(
        self, sql: Union[str, TextClause, TextualSelect], /, **kwargs
    ) -> DataFrame:
        text_sql = self._build_text_sql(sql)
        with self.engine.begin() as conn:
            data = conn.execute(text_sql.bindparams(**kwargs))
            dataframe = DataFrame(data=data.fetchall(), columns=data.keys())
        return dataframe

    async def async_read_sql(self, sql: TextClause, /, **kwargs) -> DataFrame:
        async with self.async_engine.begin() as conn:
            data = await conn.execute(sql.bindparams(**kwargs))
            dataframe = DataFrame(data=data.fetchall(), columns=data.keys())
        return dataframe


db = DataBase(settings)
