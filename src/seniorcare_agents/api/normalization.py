import json
from typing import Any


def unwrap_record_list(value: Any, *, allow_singleton: bool = False) -> list[dict[str, Any]]:
    """Normalize MCP list results, including JSON text and simulation envelopes."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    # Some Streamable HTTP MCP adapter versions collapse a one-item structured
    # list into the contained object. List-returning API boundaries opt into
    # restoring that record to a one-item collection.
    if allow_singleton and isinstance(value, dict):
        return [value]
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]
