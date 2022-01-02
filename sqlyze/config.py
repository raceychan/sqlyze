# import pika
import ssl

# from pika.connection import ConnectionParameters
from typing import Any, Dict, List, Optional, Union, Set
from pydantic import BaseSettings, PostgresDsn, EmailStr, HttpUrl, AnyUrl, validator
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from utils import build_url

DATABASE = "postgresql"
# DATABASE = "mysql"


class MysqlDsn(AnyUrl):
    allowed_schemes = {"mysql"}


class Settings(BaseSettings):
    class Config:
        #        env_file = ".env"
        env_file_encoding = "utf-8"
        allow_reuse = True

    class DefaultValues:
        null = 0

    EVENTS: Set[str] = {"instantiate", "sql_update", "data_update"}

    MYSQL_DB: str
    MYSQL_DRIVER: str
    MYSQL_HOST: str
    MYSQL_PORT: str = "3306"
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_SCHEMA: Optional[str] = None

    POSTGRESQL_DB: str
    POSTGRESQL_DRIVER: str
    POSTGRESQL_HOST: str
    POSTGRESQL_PORT: str
    POSTGRESQL_USER: str
    POSTGRESQL_PASSWORD: str
    POSTGRESQL_SCHEMA: str

    DB_PARAMS_SCHEMA: Set[str] = {
        "db",
        "driver",
        "user",
        "password",
        "host",
        "port",
        "schema",
    }

    DATABASE_URL: Optional[Union[PostgresDsn, MysqlDsn]] = None

    @validator("DATABASE_URL")
    def assemble_db_url(
        cls, v: Optional[str], values: Dict[str, Any], config, field, **kwargs
    ) -> Any:
        """TODO: receive **kwargs to update db_params"""
        if v and isinstance(v, str):
            return v
        db_type = kwargs.get("db_type", DATABASE)
        db_params_schema: Dict[str, Any] = values.get("DB_PARAMS_SCHEMA", "")
        db_params = {
            schema: values.get(f"{db_type.upper()}_{schema.upper()}", "")
            for schema in db_params_schema
        }
        return build_url(db_params=db_params, db_params_schema=db_params_schema)

    ASYNC_DATABASE_URL: Optional[Union[PostgresDsn, MysqlDsn]] = None

    @validator("ASYNC_DATABASE_URL")
    def assemble_aysnc_url(
        cls, v: Optional[str], values: Dict[str, Any], config, field, **kwargs
    ) -> Any:
        """TODO: re-write to be like: return self.assemble_db_url(db_driver = 'aiomysql')"""
        if isinstance(v, str):
            return v
        db_type = kwargs.get("db_type", DATABASE)

        db_params_schema: Dict[str, Any] = values.get("DB_PARAMS_SCHEMA", "")
        db_params = {
            schema: values.get(f"{db_type.upper()}_{schema.upper()}", "")
            for schema in db_params_schema
        }
        if db_type == "mysql":
            db_params[f"driver"] = "aiomysql"
        elif db_type == "postgresql":
            db_params[f"driver"] = "asyncpg"
        return build_url(db_params=db_params, db_params_schema=db_params_schema)

    RABBITMQ_DRIVER: str
    RABBITMQ_HOST: str
    RABBITMQ_PORT: str
    RABBITMQ_USER: str
    RABBITMQ_PASSWORD: str


#    RMQ_CON_PARAM: Optional[ConnectionParameters] = None

# @validator("RMQ_CON_PARAM", pre=True)
# def assemble_mq_con(
# cls, v: Optional[str], values: Dict[str, Any], config, field
# ) -> ConnectionParameters:
# if isinstance(v, ConnectionParameters):
# return v
# rabbitmq_user = values.get("RABBITMQ_USER")
# rabbitmq_password = values.get("RABBITMQ_PASSWORD")
# rabbitmq_host = values.get("RABBITMQ_HOST")
# rabbitmq_port = values.get("RABBITMQ_PORT")

# credentials = pika.PlainCredentials(rabbitmq_user, rabbitmq_password)
# context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
# ssl_options = pika.SSLOptions(context)
# connection_params = pika.ConnectionParameters(
# host=rabbitmq_host,
# port=rabbitmq_port,
# credentials=credentials,
# ssl_options=ssl_options,
# )
# return connection_params


settings = Settings(_env_file=".env")
defaults = Settings.DefaultValues()
