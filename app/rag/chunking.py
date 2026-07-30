"""Belge chunking modülü.

Markdown kaynak belgelerini (## / ### başlıklı bölümler) semantik olarak anlamlı
parçalara böler. Önce başlık bazlı böler, sonra çok uzun bölümleri paragraf sınırlarını
koruyarak overlap ile alt-chunk'lara ayırır. Her chunk; kaynak dosya, bölüm başlığı ve
chunk id taşır — RAG evaluation'da "doğru chunk'ı buldu mu" ölçümü ve kaynak atfı bu
metadata'ya dayanır (bkz. p1_proje_plani.md İlke 1 ve İlke 4).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    source_file: str
    section_title: str
    text: str

    def __repr__(self) -> str:  # pragma: no cover - sadece okunabilirlik için
        preview = self.text[:60].replace("\n", " ")
        return f"Chunk({self.chunk_id}, {self.section_title!r}, {preview!r}...)"


def _split_into_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Markdown metnini ## / ### başlıklarına göre (başlık, içerik) çiftlerine böler."""
    lines = markdown_text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "Giriş"
    current_lines: list[str] = []

    for line in lines:
        header_match = re.match(r"^(#{1,3})\s+(.*)", line)
        if header_match:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = header_match.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    return [
        (title, "\n".join(lines).strip())
        for title, lines in sections
        if "\n".join(lines).strip()
    ]


def _split_long_section(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Bir bölüm max_chars'tan uzunsa, paragraf sınırlarını koruyarak overlap ile böler."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
                tail = current[-overlap_chars:] if overlap_chars else ""
                current = f"{tail}\n\n{para}".strip()
            else:
                # tek paragraf bile max_chars'tan uzun; zorla böl
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i : i + max_chars])
                current = ""

    if current:
        chunks.append(current)

    return chunks


def chunk_markdown_file(
    file_path: str,
    markdown_text: str,
    max_chars: int = 600,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """Bir markdown kaynak belgesini Chunk listesine dönüştürür."""
    sections = _split_into_sections(markdown_text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for title, section_text in sections:
        sub_texts = _split_long_section(section_text, max_chars, overlap_chars)
        for sub_text in sub_texts:
            chunk_index += 1
            chunks.append(
                Chunk(
                    chunk_id=f"{file_path}::chunk{chunk_index}",
                    source_file=file_path,
                    section_title=title,
                    text=sub_text,
                )
            )

    return chunks
