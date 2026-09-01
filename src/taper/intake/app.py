"""Local web app for building an athlete profile.

Runs on localhost only. Nothing leaves the machine, and the saved profile is a
JSON file in the working directory -- there is no account, no server, and no
third-party API involved anywhere in the intake path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from taper.athlete import AthleteProfile, Surface
from taper.insights import (
    best_per_year, current_fitness, formatted_equivalents, race_insights, readiness_flags,
)
from taper.intake.parsers import parse_any
from taper.profile_io import from_dict, save, to_dict
from taper.units import format_duration

FORM_HTML = Path(__file__).parent / "form.html"
DEFAULT_PROFILE_DIR = Path("profiles")

app = FastAPI(title="taper intake", docs_url=None, redoc_url=None)


class PasteRequest(BaseModel):
    text: str
    surface: str = "road"


class ProfileRequest(BaseModel):
    profile: dict[str, Any]


class SaveRequest(BaseModel):
    profile: dict[str, Any]
    filename: str | None = None


def _load_profile(payload: dict[str, Any]) -> AthleteProfile:
    try:
        return from_dict(payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=f"Could not read profile: {exc}") from exc


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return FORM_HTML.read_text()


@app.post("/api/parse-races")
def parse_races(req: PasteRequest) -> dict[str, Any]:
    """Parse pasted results text or CSV into race rows for the editable table."""
    try:
        surface = Surface(req.surface)
    except ValueError:
        surface = Surface.ROAD

    races = parse_any(req.text, default_surface=surface)
    rows = [{
        "distance_m": r.distance_m,
        "finish_time_s": r.finish_time_s,
        "finish_time": format_duration(r.finish_time_s),
        "race_date": r.race_date.isoformat() if r.race_date else None,
        "name": r.name,
        "place_overall": r.place_overall,
        "field_size": r.field_size,
        "surface": r.surface.value,
        "source": r.source,
    } for r in races]

    return {
        "races": rows,
        "count": len(rows),
        "message": _paste_feedback(len(rows), req.text),
    }


def _paste_feedback(count: int, text: str) -> str:
    if count:
        return (f"Found {count} race{'s' if count != 1 else ''}. "
                f"Check the table -- fix anything it read wrong.")
    if not text.strip():
        return "Nothing pasted yet."
    return ("No races found. Every row needs both a distance and a finish time. "
            "If the distance is only in the page heading, paste that line too.")


@app.post("/api/insights")
def insights(req: ProfileRequest) -> dict[str, Any]:
    """Everything derivable from the profile as it currently stands."""
    profile = _load_profile(req.profile)
    fitness = current_fitness(profile)

    return {
        "races": [{
            "name": i.race.name,
            "date": i.race.race_date.isoformat() if i.race.race_date else None,
            "distance_m": i.race.distance_m,
            "finish_time": format_duration(i.race.finish_time_s),
            "vdot": round(i.vdot, 1),
        } for i in race_insights(profile)],
        "trajectory": [{"year": y, "vdot": round(v, 1)} for y, v in best_per_year(profile)],
        "fitness": None if fitness is None else {
            "vdot": round(fitness.vdot, 1),
            "confidence": fitness.confidence,
            "note": fitness.note,
            "equivalents": [{"label": l, "time": t}
                            for l, t in formatted_equivalents(fitness.vdot)],
        },
        "flags": [{"severity": f.severity, "message": f.message}
                  for f in readiness_flags(profile)],
    }


@app.post("/api/save")
def save_profile(req: SaveRequest) -> dict[str, Any]:
    profile = _load_profile(req.profile)
    stem = (req.filename or profile.name or "runner").strip()
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in stem.lower()).strip("-")
    path = save(profile, DEFAULT_PROFILE_DIR / f"{safe or 'runner'}.json")
    return {"path": str(path.resolve()), "profile": to_dict(profile)}


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
