import math
import datetime
import builtins
from typing import Any, Union


def log(x: float, base: float = 10) -> float:
    return math.log(x, base)


def log1p(x: float) -> float:
    return math.log1p(x)


def exp(x: float) -> float:
    return math.exp(x)


def sqrt(x: float) -> float:
    return math.sqrt(x)


def clip(x: float, min_val: float, max_val: float) -> float:
    return builtins.max(min_val, builtins.min(x, max_val))


def round(x: float, ndigits: int = 0) -> float:
    return builtins.round(x, ndigits)


def floor(x: float) -> float:
    return math.floor(x)


def ceil(x: float) -> float:
    return math.ceil(x)


def abs(x: float) -> float:
    return builtins.abs(x)


def len(x: Any) -> int:
    return builtins.len(x)


def startswith(x: str, prefix: str) -> bool:
    return x.startswith(prefix)


def endswith(x: str, suffix: str) -> bool:
    return x.endswith(suffix)


def lower(x: str) -> str:
    return x.lower()


def upper(x: str) -> str:
    return x.upper()


def replace_values(x: str, old: str, new: str) -> str:
    return x.replace(old, new)


def year(x: Union[datetime.datetime, str]) -> int:
    if isinstance(x, str):
        x = datetime.datetime.fromisoformat(x)
    return x.year


def month(x: Union[datetime.datetime, str]) -> int:
    if isinstance(x, str):
        x = datetime.datetime.fromisoformat(x)
    return x.month


def day(x: Union[datetime.datetime, str]) -> int:
    if isinstance(x, str):
        x = datetime.datetime.fromisoformat(x)
    return x.day


def weekday(x: Union[datetime.datetime, str]) -> int:
    if isinstance(x, str):
        x = datetime.datetime.fromisoformat(x)
    return x.weekday()


def hour(x: Union[datetime.datetime, str]) -> int:
    if isinstance(x, str):
        x = datetime.datetime.fromisoformat(x)
    return x.hour


def days_between(x1: Union[datetime.datetime, str], x2: Union[datetime.datetime, str]) -> int:
    if isinstance(x1, str):
        x1 = datetime.datetime.fromisoformat(x1)
    if isinstance(x2, str):
        x2 = datetime.datetime.fromisoformat(x2)
    return builtins.abs((x2 - x1).days)


def is_null(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def strip(x: str) -> str:
    return x.strip()


def not_null(x: Any) -> bool:
    return not is_null(x)


def min(x1: float, x2: float) -> float:
    return builtins.min(x1, x2)


def max(x1: float, x2: float) -> float:
    return builtins.max(x1, x2)


def format_timestamp(x: Union[datetime.datetime, str], format: str) -> str:
    if isinstance(x, str):
        x = datetime.datetime.fromisoformat(x)
    return x.strftime(format)
