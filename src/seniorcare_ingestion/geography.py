ALIASES = {
    "va": "Virginia",
    "virginia": "Virginia",
    "richmond": "Richmond City",
    "richmond city": "Richmond City",
    "henrico": "Henrico County",
    "henrico county": "Henrico County",
    "chesterfield": "Chesterfield County",
    "chesterfield county": "Chesterfield County",
    "hanover": "Hanover County",
    "hanover county": "Hanover County",
}


def normalize_place(value: str | None) -> str | None:
    return ALIASES.get(value.strip().lower(), value.strip()) if value else None
