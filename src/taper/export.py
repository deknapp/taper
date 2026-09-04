"""Plain-text export of a runner's record history.

A backup you can read. The point of writing it as text rather than as JSON or a
database dump is that it stays legible in twenty years, in any editor, with no
version of this program available to open it -- which is the timescale a running
career actually spans.

It is deliberately a *report*, not a serialisation: it carries what was run and
what it means, including the efforts the terrain screen threw out and why, so a
future reader is not left wondering where a missing personal best went.
"""
from __future__ import annotations

import textwrap
from datetime import date

from taper.athlete import AthleteProfile
from taper.records import STANDARD_DISTANCES, detect_records, progression, rejected_efforts
from taper.units import format_duration

LINE = "-" * 72


def _heading(title: str) -> list[str]:
    return ["", title.upper(), LINE]


def _wrapped(text: str, indent: str = " " * 17) -> list[str]:
    """A sentence folded to the report width, so it stays readable in a terminal."""
    return textwrap.wrap(text, width=len(LINE), initial_indent=indent,
                         subsequent_indent=indent) or [indent.rstrip()]


def _signed(seconds: float) -> str:
    """A time difference, always carrying its sign."""
    sign = "-" if seconds > 0 else "+"
    return f"{sign}{format_duration(abs(seconds))}"


def records_report(profile: AthleteProfile, today: date | None = None) -> str:
    """The whole record history as text, ready to write to a file."""
    today = today or date.today()
    days = profile.training_days
    races = profile.races

    records = detect_records(days, races)
    lines = [
        f"taper -- record history for {profile.name or 'an unnamed runner'}",
        f"exported {today.isoformat()}",
        LINE,
    ]

    lines += _heading("personal records")
    if records:
        for record in records:
            source = "race" if record.effort.source == "race" else "training"
            when = record.set_on.isoformat() if record.set_on else "undated"
            lines.append(
                f"  {record.label:<15}{record.effort.formatted_time:>9}   {when}   "
                f"{source:<9}{record.effort.name[:20]}".rstrip())
            if record.improvement_s:
                lines.append(f"  {'':<15}{_signed(record.improvement_s):>9}   "
                             f"on the previous best")
    else:
        lines.append("  Nothing yet. A record needs a run whose whole distance lands on")
        lines.append("  a standard one, on terrain flat enough to be comparable.")

    lines += _heading("progression")
    any_chain = False
    for label, _ in STANDARD_DISTANCES:
        chain = progression(days, races, label)
        if len(chain) < 2:
            continue
        any_chain = True
        lines.append(f"  {label}")
        previous = None
        for effort in chain:
            when = effort.day.isoformat() if effort.day else "undated"
            delta = f"   {_signed(previous - effort.time_s)}" if previous else ""
            lines.append(f"    {when}   {effort.formatted_time:>9}{delta}")
            previous = effort.time_s
        lines.append("")
    if not any_chain:
        lines.append("  No distance has been improved on yet.")

    lines += _heading("races")
    dated = sorted((r for r in races), key=lambda r: (r.race_date or date.min))
    if dated:
        for race in dated:
            when = race.race_date.isoformat() if race.race_date else "undated  "
            place = ""
            if race.place_overall:
                place = (f"  {race.place_overall}"
                         f"{'/' + str(race.field_size) if race.field_size else ''}")
            lines.append(
                f"  {when}  {race.distance_m / 1000:>7.2f} km  "
                f"{format_duration(race.finish_time_s):>9}  "
                f"{race.name[:24]:<24}{place}".rstrip())
    else:
        lines.append("  No races on record.")

    rejected = rejected_efforts(days, races)
    if rejected:
        lines += _heading("efforts the terrain screen excluded")
        lines.append("  Fast, real, and not comparable to a flat road time. Kept here so")
        lines.append("  a missing personal best is never a mystery.")
        lines.append("")
        for effort in rejected[:25]:
            when = effort.day.isoformat() if effort.day else "undated"
            lines.append(f"  {effort.label:<15}{effort.formatted_time:>9}   {when}")
            lines += _wrapped(effort.reason)
        if len(rejected) > 25:
            lines.append(f"  ... and {len(rejected) - 25} more.")

    lines += _heading("how to read this")
    lines.append("  Records are drawn from races and from training days whose whole")
    lines.append("  distance lands on a standard one. A day holding more than one")
    lines.append("  activity is never counted: its distances and times were added")
    lines.append("  together, so it describes no single continuous run.")
    lines.append("")
    lines.append("  An effort must be on road or track, climb no more than 12 m/km, and")
    lines.append("  drop no more than 4 m/km net, or the clock is measuring the hill")
    lines.append("  rather than the runner.")
    lines.append("")
    # No line in a backup should carry invisible trailing space: it survives
    # every copy, and shows up as noise in any diff of two exports.
    return "\n".join(line.rstrip() for line in lines) + "\n"


def suggested_filename(profile: AthleteProfile, today: date | None = None) -> str:
    today = today or date.today()
    stem = "".join(c if c.isalnum() else "-" for c in (profile.name or "runner").lower())
    stem = "-".join(part for part in stem.split("-") if part) or "runner"
    return f"{stem}-records-{today.isoformat()}.txt"
