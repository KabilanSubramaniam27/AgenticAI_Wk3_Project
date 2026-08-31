import json
from typing import Any


def unwrap_record_list(value: Any) -> list[dict[str, Any]]:
    """Normalize MCP list results, including JSON text and simulation envelopes."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if not isinstance(value, list):
        return []
    return [record for record in value if isinstance(record, dict)]
