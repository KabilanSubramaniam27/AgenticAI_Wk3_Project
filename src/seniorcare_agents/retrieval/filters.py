from typing import Any


def matches_filters(
    row: dict[str, Any], categories: list[str], geography: dict | None, filters: dict | None
) -> bool:
    if categories and row.get("category") not in categories:
        return False
    if int(row.get("source_trust_tier") or 4) > 3:
        return False
    for key, value in (filters or {}).items():
        current = row.get(key)
        if isinstance(current, list):
            if value not in current:
                return False
        elif current != value:
            return False
    if not geography:
        return True
    state = geography.get("state")
    county = geography.get("county")
    city = geography.get("city")
    if state and row.get("state") not in {None, state}:
        return False
    if not county and not city:
        return True
    local = {
        str(row.get("county") or ""),
        str(row.get("city") or ""),
        *map(str, row.get("service_area") or []),
    }
    broad = {"Virginia", "Richmond Metro", "Central Virginia", "US", "United States"}
    return bool(local & broad or (county and county in local) or (city and city in local))
