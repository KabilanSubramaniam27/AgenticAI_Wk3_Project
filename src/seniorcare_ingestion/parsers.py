import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pymupdf
import trafilatura
from bs4 import BeautifulSoup


def parse_html(data: bytes) -> tuple[str, str]:
    html = data.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled"
    text = trafilatura.extract(html, include_links=True, include_tables=True, output_format="txt")
    return title, text or soup.get_text("\n", strip=True)


def parse_pdf(path: Path) -> list[tuple[str, str, int]]:
    document = pymupdf.open(path)
    title = document.metadata.get("title") or path.stem
    return [
        (title, document.load_page(number).get_text("text"), number + 1)
        for number in range(document.page_count)
    ]


def parse_structured(path: Path) -> list[dict]:
    if path.suffix == ".zip":
        rows: list[dict[str, Any]] = []
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".csv"):
                    with archive.open(name) as stream:
                        rows.extend(
                            csv.DictReader(
                                io.TextIOWrapper(stream, encoding="utf-8-sig", errors="replace")
                            )
                        )
        return rows
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    for key in ("data", "results", "items"):
        if isinstance(value.get(key), list):
            return value[key]
    return [value]
