"""JSON helpers with optional orjson acceleration."""

from __future__ import annotations

import json
from typing import Any

try:
    import orjson  # type: ignore
except ImportError:  # pragma: no cover - exercised in environments without orjson
    orjson = None


def loads(data: str | bytes) -> Any:
    if orjson is not None:
        return orjson.loads(data)
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    return json.loads(data)


def dumps(data: Any, *, pretty: bool = False, sort_keys: bool = False) -> str:
    if orjson is not None:
        option = 0
        if pretty:
            option |= orjson.OPT_INDENT_2
        if sort_keys:
            option |= orjson.OPT_SORT_KEYS
        return orjson.dumps(data, option=option).decode("utf-8")
    indent = 2 if pretty else None
    separators = None if pretty else (",", ":")
    return json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=sort_keys, separators=separators)

