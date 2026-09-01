"""Parse race results out of whatever the runner pastes in.

Deliberately not a per-site scraper. Timing companies each render their own
React table, but every one of them survives select-all-copy into plain text, so
that is the interface we target: one parser, every site, nothing to break when
somebody ships a redesign.

The parser is expected to be imperfect. Everything it produces is shown back in
an editable table -- the goal is to save typing, not to be trusted blindly.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime

from taper.athlete import RaceResult, Surface

# --- distances -------------------------------------------------------------

MARATHON_M = 42195.0
HALF_MARATHON_M = 21097.5

# Checked before the numeric patterns, so these must be names that a number
# never legitimately precedes.
_NAMED_DISTANCES: dict[str, float] = {
    "full marathon": MARATHON_M,
    "half marathon": HALF_MARATHON_M,
    "half-marathon": HALF_MARATHON_M,
    "marathon": MARATHON_M,
}

# Checked only after the numeric patterns have had their turn, because a number
# in front of these changes what they mean: "mile" is 1609m but "10 mile" is not.
_BARE_DISTANCES: dict[str, float] = {
    "half": HALF_MARATHON_M,
    "mile": 1609.344,
}

# Bare decimals runners use as shorthand for the two road classics.
_SHORTHAND = {"26.2": MARATHON_M, "13.1": HALF_MARATHON_M}

_RE_KM = re.compile(r"\b(\d+(?:\.\d+)?)\s*k(?:m)?\b", re.I)
_RE_MILES = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:mi|mile|miles)\b", re.I)
_RE_METERS = re.compile(r"\b(\d{3,5})\s*m(?:eters?)?\b", re.I)


def parse_distance(text: str) -> float | None:
    """Best-effort distance in metres from a fragment of results text.

    Order matters. Marathon names are matched first because no number precedes
    them, then the runner shorthands, then numeric units, and only then the bare
    names -- otherwise "10 mile classic" matches the word "mile" and comes back
    as 1609m instead of ten of them.
    """
    low = text.lower()

    for name in sorted(_NAMED_DISTANCES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", low):
            return _NAMED_DISTANCES[name]

    for token, metres in _SHORTHAND.items():
        if re.search(rf"(?<!\d){re.escape(token)}(?!\d)", low):
            return metres

    if m := _RE_KM.search(low):
        return float(m.group(1)) * 1000.0
    if m := _RE_MILES.search(low):
        return float(m.group(1)) * 1609.344
    if m := _RE_METERS.search(low):
        metres = float(m.group(1))
        if 400 <= metres <= 50000:
            return metres

    for name in sorted(_BARE_DISTANCES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(name)}\b", low):
            return _BARE_DISTANCES[name]
    return None


# --- times -----------------------------------------------------------------

_RE_HMS = re.compile(r"\b(\d{1,2}):([0-5]\d):([0-5]\d)(?:\.(\d+))?\b")
_RE_MS = re.compile(r"\b(\d{1,3}):([0-5]\d)(?:\.(\d+))?\b")


def _all_times(text: str) -> list[float]:
    """Every clock-looking token in the text, as seconds."""
    found: list[tuple[int, int, float]] = []  # (start, end, seconds)

    for m in _RE_HMS.finditer(text):
        h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
        frac = float(f"0.{m.group(4)}") if m.group(4) else 0.0
        found.append((m.start(), m.end(), h * 3600 + mi * 60 + s + frac))

    for m in _RE_MS.finditer(text):
        # Skip anything already consumed as part of an h:mm:ss match.
        if any(start <= m.start() < end for start, end, _ in found):
            continue
        mi, s = int(m.group(1)), int(m.group(2))
        frac = float(f"0.{m.group(3)}") if m.group(3) else 0.0
        found.append((m.start(), m.end(), mi * 60 + s + frac))

    return [seconds for _, _, seconds in sorted(found)]


def parse_finish_time(text: str) -> float | None:
    """Pick the finish time out of a results row.

    Results rows routinely carry several clock values: gun time, chip time, and
    a pace column. Pace is always much smaller than the finish time, so the
    largest value wins -- except when two candidates are close, which means gun
    and chip rather than time and pace, and there the smaller (chip) is the one
    the runner would claim.
    """
    times = [t for t in _all_times(text) if t > 0]
    if not times:
        return None
    times.sort()
    longest = times[-1]
    close = [t for t in times if t >= longest * 0.9]
    return min(close)


# --- dates -----------------------------------------------------------------

_DATE_FORMATS = (
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y",
    "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
    "%d %B %Y", "%d %b %Y",
)

_RE_DATE_CANDIDATES = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b",
    re.I,
)


def parse_date(text: str) -> date | None:
    for match in _RE_DATE_CANDIDATES.finditer(text):
        candidate = match.group(0).replace(".", "").strip()
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
    # A bare year is better than nothing for trajectory work; assume mid-year.
    if m := re.search(r"\b(19[5-9]\d|20[0-4]\d)\b", text):
        return date(int(m.group(1)), 7, 1)
    return None


# --- placings --------------------------------------------------------------

_RE_PLACE_OF = re.compile(r"\b(\d{1,6})\s*(?:/|of)\s*(\d{1,6})\b", re.I)
_RE_ORDINAL = re.compile(r"\b(\d{1,6})(?:st|nd|rd|th)\b", re.I)


def parse_placing(text: str) -> tuple[int | None, int | None]:
    if m := _RE_PLACE_OF.search(text):
        place, field_size = int(m.group(1)), int(m.group(2))
        if place <= field_size:
            return place, field_size
    if m := _RE_ORDINAL.search(text):
        return int(m.group(1)), None
    return None, None


# --- whole-blob parsing ----------------------------------------------------

def _looks_like_header(line: str) -> bool:
    low = line.lower()
    header_words = ("place", "name", "time", "pace", "bib", "div", "age", "gun", "chip")
    hits = sum(1 for w in header_words if w in low)
    return hits >= 3 and not _all_times(line)


def parse_pasted_results(blob: str, default_surface: Surface = Surface.ROAD) -> list[RaceResult]:
    """Parse a pasted block of results text into races, one per line.

    A line counts as a race only if it yields both a distance and a time. Lines
    that give a time but no distance are common in single-race result tables
    (the distance sits in the page heading, not the row), so we carry the most
    recent distance seen forward as context.
    """
    races: list[RaceResult] = []
    context_distance: float | None = None
    context_date: date | None = None

    for raw in blob.splitlines():
        line = raw.strip()
        if not line or _looks_like_header(line):
            continue

        distance = parse_distance(line)
        time_s = parse_finish_time(line)
        when = parse_date(line)

        # A heading line: names a distance and/or date but has no result on it.
        if time_s is None:
            if distance is not None:
                context_distance = distance
            if when is not None:
                context_date = when
            continue

        distance = distance or context_distance
        if distance is None:
            continue

        place, field_size = parse_placing(line)
        races.append(RaceResult(
            distance_m=distance,
            finish_time_s=time_s,
            race_date=when or context_date,
            name=_guess_race_name(line),
            place_overall=place,
            field_size=field_size,
            surface=default_surface,
            source="paste",
        ))

    return races


_RE_NAME_NOISE = re.compile(
    r"\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?"      # times
    r"|\d+(?:\.\d+)?\s*(?:k|km|mi|miles?|m)\b"  # distances
    r"|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\b\d+\s*(?:/|of)\s*\d+\b",
    re.I,
)


def _guess_race_name(line: str) -> str:
    """Strip the machine-readable bits and keep whatever prose is left."""
    text = _RE_NAME_NOISE.sub(" ", line)
    text = re.sub(r"[\t|,;]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" -–—")
    words = [w for w in text.split() if not w.isdigit()]
    return " ".join(words)[:120].strip()


def parse_csv_results(blob: str, default_surface: Surface = Surface.ROAD) -> list[RaceResult]:
    """Parse a CSV export, matching columns by fuzzy header name."""
    reader = csv.DictReader(io.StringIO(blob))
    if not reader.fieldnames:
        return []

    def column(row: dict[str, str], *names: str) -> str:
        """First non-empty cell whose header matches, trying names in the order
        given -- so a 'Distance' column beats a 'Race' column for the distance,
        even though the race name often contains one too."""
        for name in names:
            for key, value in row.items():
                if key and name in key.strip().lower() and value and value.strip():
                    return value.strip()
        return ""

    races: list[RaceResult] = []
    for row in reader:
        joined = " ".join(v for v in row.values() if v)
        distance = (parse_distance(column(row, "distance", "event", "race"))
                    or parse_distance(joined))
        time_s = (parse_finish_time(column(row, "chip", "finish", "time", "result", "gun"))
                  or parse_finish_time(joined))
        if distance is None or time_s is None:
            continue
        place, field_size = parse_placing(column(row, "place", "position", "overall") or joined)
        races.append(RaceResult(
            distance_m=distance,
            finish_time_s=time_s,
            race_date=parse_date(column(row, "date", "when") or joined),
            name=column(row, "name", "event", "race", "title"),
            place_overall=place,
            field_size=field_size,
            surface=default_surface,
            source="csv",
        ))
    return races


def parse_any(blob: str, default_surface: Surface = Surface.ROAD) -> list[RaceResult]:
    """Dispatch on shape: CSV if it parses as CSV with a plausible header."""
    head = blob.strip().splitlines()[:1]
    if head and head[0].count(",") >= 2:
        if races := parse_csv_results(blob, default_surface):
            return races
    return parse_pasted_results(blob, default_surface)
