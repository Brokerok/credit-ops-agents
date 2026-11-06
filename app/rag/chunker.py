"""Policy documents are markdown with a small front matter block; each `##` section becomes one chunk.

Section-level chunks keep rule text intact, so a citation always points at a human-readable section id
(e.g. "CP-3.2") instead of an arbitrary 512-token window.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FRONT = re.compile(r"^---\n(.*?)\n---\n", re.S)
_SECTION = re.compile(r"^## +(.+?)\n", re.M)


@dataclass(frozen=True)
class Chunk:
    doc: str  # file stem, e.g. "credit_policy"
    section: str  # section id + title, e.g. "CP-3.2 Maximum LTV"
    roles: tuple[str, ...]  # who may read it
    content: str


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    m = _FRONT.match(text)
    if not m:
        return {}, text
    meta: dict[str, str] = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        meta[k.strip()] = v.strip()
    return meta, text[m.end() :]


def chunk_document(path: Path) -> list[Chunk]:
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    roles = tuple(r.strip() for r in meta.get("roles", "").strip("[]").split(",") if r.strip())
    parts = _SECTION.split(body)
    chunks: list[Chunk] = []
    # parts = [preamble, title1, body1, title2, body2, ...]
    for i in range(1, len(parts) - 1, 2):
        title, text = parts[i].strip(), parts[i + 1].strip()
        if text:
            chunks.append(Chunk(path.stem, title, roles, f"{title}\n{text}"))
    return chunks


def load_policy_dir(policy_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(policy_dir.glob("*.md")):
        chunks.extend(chunk_document(p))
    return chunks
