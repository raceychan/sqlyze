"""
api:
sales_sql = Sql('select id, name from table where id =:id and name =: name')
@sales_sql.executor()
# use the func sig here and put it in text.bindparams
def sales_data(sql:TextClause, id:int, name:str):
    df = db.read_sql(sql)
    df.covert_dtypes()
"""
import sys
import inspect
import pandas as pd
from pathlib import Path
from functools import wraps
from inspect import Parameter
from sqlalchemy.engine.row import LegacyRow
from sqlalchemy.sql.elements import TextClause
from pydantic import parse_obj_as, ValidationError, BaseModel, create_model
from pandas.core.series import Series
from pandas.core.frame import DataFrame
from typing import Callable, TypeVar, Type, List, Union, Any  # , Dict
from database import db

T = TypeVar("T")


class Pawn:
    def __init__(self, sql):
        pass

    def get_typed_series(self: "Pawn", typed_dict: dict):
        dtypes = {column: Series(dtype=dtype) for column, dtype in typed_dict.items()}
        return dtypes

    def get_template_from_series(self, types_dict: dict):
        series = self.get_typed_series(types_dict)
        dtemplate = DataFrame(series)
        return dtemplate

    def get_pydantic_types(self, basemodel: BaseModel):
        annotations = basemodel.__annotations__.items()
        dtypes = {column: dtype for column, dtype in annotations}
        return dtypes


class Sql(TextClause):
    def __init__(
        self,
        text: str,
        func_exe: Callable = None,
        doc: Any = None,
        sql_obj: Type = None,
        *,
        path: Union[str, Path] = None,
    ) -> None:

        super(Sql, self).__init__(text=text)
        if func_exe and doc is None:
            self.func_exe = func_exe
            self.sql_obj = sql_obj
            doc = self.func_exe.__doc__
            wraps(self.func_exe)(self)
            self.func_sig = inspect.signature(self.func_exe)
            self.func_params = self.func_sig.parameters
            self.func_return = self.func_sig.return_annotation
            self.validate_signature()
        self.__doc__ = doc  # type: ignore
        self.path = path

    def __get__(self, obj: T, obj_type: Type[T] = None) -> T:
        if obj is None:
            return self
        if self.func_exe is None:
            raise AttributeError("Unreachable attribute")
        return self.__call__

    def query(self, func_exe: Callable) -> Type:
        return type(self)(
            text=self.text, func_exe=func_exe, doc=self.__doc__, sql_obj=self
        )

    def validate_signature(self) -> None:
        params_keys = set(self._bindparams.keys())  # type: ignore
        signature_keys = set(self.func_params.keys())
        if not params_keys.issubset(signature_keys):
            missed_params = params_keys - signature_keys
            raise ValueError(f"Missing {missed_params} for sql params in func params")

        if "return_schema" in self.func_params:
            self.default_return_schema = self.func_params["return_schema"].default
        else:  # TODO: rewrite the following logic into a function
            return_schema_param = Parameter(
                name="return_schema",
                annotation=Union[BaseModel, type],
                kind=Parameter.KEYWORD_ONLY,
            )
            func_params = list(self.func_params.values())
            func_params.append(return_schema_param)
            self.func_exe.__signature__ = self.func_sig.replace(parameters=func_params)
            self.default_return_schema = None

    def append_params(self, func, param):
        parameter = Parameter(
            name="return_schema",
            annotation=Union[BaseModel, type],
            kind=Parameter.KEYWORD_ONLY,
        )
        func_params = list(self.func_params.values())
        func_params.append(parameter)
        self.func_exe.__signature__ = self.func_sig.replace(parameters=func_params)
        self.default_return_schema = None

    def validate_parameters(self, func: Callable, *args, **kwargs) -> None:
        params = inspect.signature(func).parameters
        values = list(params.values())
        for index, arg in enumerate(args):
            param = values[index]
            if param.annotation is inspect._empty:  # type: ignore
                continue
            try:
                parse_obj_as(param.annotation, arg)
            except ValidationError:
                raise TypeError(
                    f"Argument {param.name} must be of type {param.annotation}"
                )

        for k, v in kwargs.items():
            if params[k].annotation is inspect._empty:  # type: ignore
                continue
            try:
                parse_obj_as(params[k].annotation, v)
            except ValidationError:
                raise TypeError(f"Argument {k} must be {params[k].annotation}")

    def validate_result(
        self, return_schema: Type[T], func_result: Union[List[LegacyRow], LegacyRow]
    ) -> Union[List[Type[T]], T, Any]:
        if isinstance(func_result, LegacyRow):
            schema = return_schema
        elif isinstance(func_result, list):
            schema = List[return_schema]
        else:
            schema = List[return_schema]
        try:
            return parse_obj_as(schema, func_result)
        except RuntimeError as re:
            if isinstance(func_result, schema):  # type: ignore
                return func_result
        finally:
            return func_result

    def validate_return(
        self, func_result: Union[List[T], T], func_return: Type[T], return_handler=None
    ) -> Union[List[T], T]:
        try:
            return parse_obj_as(func_return, func_result)
        except ValidationError as ve:
            raise TypeError(f"Invalid function return for type {func_return}")

        except RuntimeError as re:  # return type not yet supported by pydantic, handled specifically
            if isinstance(func_result, func_return):
                return func_result
            if issubclass(func_return, DataFrame):  # pre-defined return handler
                return self.pydantic_to_dataframe(func_result)
            elif return_handler is not None:  # customized return handler
                return return_handler(func_result)
            else:
                raise TypeError(f"type {func_return} not parsable by pydantic")

    def pydantic_to_dataframe(
        self, pydantic_models: Union[List[BaseModel], BaseModel]
    ) -> DataFrame:
        list_of_dicts = [obj.__dict__ for obj in pydantic_models]
        df = pd.DataFrame(list_of_dicts).convert_dtypes()
        return df

    def __call__(self, func: Callable = None, *func_args, **func_kwargs) -> Any:
        if func:
            func_args = (func,) + func_args
        func = self.func_exe
        return_schema = func_kwargs.get("return_schema") or self.default_return_schema

        try:
            func_kwargs.pop("return_schema")
        except KeyError:
            if return_schema is None:
                raise KeyError("return_schema not defined")

        self.validate_parameters(func, *func_args, **func_kwargs)
        self.sql_obj._bindparams = self.sql_obj.bindparams(
            **func_kwargs
        )._bindparams  # type: ignore
        func_result = func(*func_args, **func_kwargs)

        if return_schema:
            func_result = self.validate_result(return_schema, func_result)

        if self.func_return is not inspect._empty:  # type: ignore
            return_handler = func_kwargs.get("return_handler")
            func_result = self.validate_return(
                func_result, self.func_return, return_handler
            )

        return func_result


housing_sql = Sql(
    """
select *
from housing
where total_rooms <:total_rooms
and median_income >:median_income
"""
)


class Housing(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: int

    class HousingData:
        housing_sql: Sql[Housing] = housing_sql  # type: ignore

    # NOTE: type hint allows: def query(df: DataFrame([], dtypes = []))
    # above should delear the default schema for housing sql

    # NOTE:
    dtypes = {
        "a": pd.Series(dtype="int"),
        "b": pd.Series(dtype="str"),
        "c": pd.Series(dtype="float"),
    }

    def test(self, df: DataFrame):
        list(inspect.signature(test).parameters.values())[0].annotation

    def __init__(self):
        pass

    @housing_sql.query
    def housing_data(self, total_rooms: int, median_income: str) -> pd.DataFrame:
        with db.engine.connect() as conn:
            result = conn.execute(housing_sql).fetchall()
        return result


# HousingData().housing_data(
# total_rooms=1000, median_income=15, return_schema=Housing
# )  # type: ignore
"""
db=DataBase.build_db(env=prod)

class Housing:

    houses = Sql(path='houses.sql', db = db).
    houses.schema({
        'longitude': float,
        'latitude': float,
        'housing_median_age': int
    })

    houses.schema = Housing

    @houses.fetch
    def get_housing_data(self):
        with db.engine.connect() as conn:
            result = conn.execute(houses).fetchall()
        return result
"""


if __name__ == "__main__":
    print("1")

