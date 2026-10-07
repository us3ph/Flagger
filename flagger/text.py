"""Download-free preprocessing and conservative sentence segmentation."""

from dataclasses import dataclass
import re
import unicodedata


def normalize_text(text: str) -> str:
    """Keep tense, negation, dates and freshness words; do not remove stopwords."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"’": "'", "“": '"', "”": '"'}))
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass(frozen=True)
class Claim:
    text: str
    start: int
    end: int


ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "u.s", "u.k"}
LINE_PREFIX = re.compile(r"^\s*(?:(?:[-*+•]|\d+[.)])\s+|>\s*)?")


def split_claims(answer: str) -> list[Claim]:
    """Return sentence spans in original Python Unicode character offsets.

    Newlines and semicolons separate units. Decimal/version dots, titles and
    initials are preserved. Headings and fenced code are skipped. This is a
    sentence-level baseline, not a semantic atomic-fact extractor.
    """
    claims: list[Claim] = []
    offset = 0
    in_code = False

    def emit(start: int, end: int) -> None:
        while start < end and answer[start].isspace():
            start += 1
        while end > start and answer[end - 1].isspace():
            end -= 1
        text = answer[start:end]
        if any(ch.isalnum() for ch in text):
            claims.append(Claim(text, start, end))

    for line_with_end in answer.splitlines(keepends=True):
        line = line_with_end.rstrip("\r\n")
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_code = not in_code
            offset += len(line_with_end)
            continue
        if in_code or not stripped or re.match(r"^#{1,6}\s", stripped):
            offset += len(line_with_end)
            continue
        start = offset + LINE_PREFIX.match(line).end()
        end = offset + len(line)
        cursor = start
        while cursor < end:
            char = answer[cursor]
            if char not in ".!?;":
                cursor += 1
                continue
            if char == ".":
                # Inside prices, versions, dates, URLs and abbreviations.
                if cursor + 1 < end and not answer[cursor + 1].isspace() and answer[cursor + 1] not in '\"\')]}':
                    cursor += 1
                    continue
                token_match = re.search(r"([\w.]+)$", answer[start:cursor])
                token = token_match.group(1).lower() if token_match else ""
                if token in ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                    cursor += 1
                    continue
            boundary = cursor + 1
            while boundary < end and answer[boundary] in '.!?\"\')]}':
                boundary += 1
            if boundary == end or answer[boundary].isspace() or char == ";":
                emit(start, boundary)
                start = boundary
                cursor = boundary
            else:
                cursor += 1
        emit(start, end)
        offset += len(line_with_end)
    return claims
