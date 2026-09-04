"""Local web app for keeping the training log.

The intake form captures who a runner is, once. This is the other half: what
they did today, and how their body felt about it. It writes straight into
taper.db rather than to a JSON profile, because this data accumulates daily and
is the thing the injury model will eventually be fitted against.

Runs on localhost, stores everything in one SQLite file, and talks to nothing.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from taper.athlete import (
    AthleteProfile, Injury, InjuryEpisode, RaceResult, Surface, Symptom, Tissue,
    TrainingDay, Wellness,
)
from taper.db import DEFAULT_DB_PATH, Database
from taper.export import records_report, suggested_filename
from taper.history import summarise
from taper.insights import current_fitness
from taper.layoffs import find_layoffs
from taper.records import detect_records, progression, rejected_efforts
from taper.units import format_duration

PAGE_HTML = Path(__file__).parent / "page.html"

app = FastAPI(title="taper log", docs_url=None, redoc_url=None)

# Set by serve(); every request opens its own connection against it, because
# SQLite objects are bound to the thread that created them.
_DB_PATH: Path = DEFAULT_DB_PATH


def db() -> Database:
    return Database(_DB_PATH)


def _enum(kind, value: str | None, default):
    """Read an enum from the form without letting a typo become a 500."""
    if not value:
        return default
    try:
        return kind(value)
    except ValueError:
        return default


def _date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def _required_date(value: str | None, field: str) -> date:
    parsed = _date(value)
    if parsed is None:
        raise HTTPException(status_code=400, detail=f"{field} needs a valid date.")
    return parsed


def _resolve(database: Database, athlete_id: int | None) -> int:
    target = athlete_id or database.default_athlete_id()
    if target is None:
        raise HTTPException(
            status_code=404,
            detail="No athlete yet. Create one before logging against it.")
    return target


# -- request bodies --------------------------------------------------------

class AthleteRequest(BaseModel):
    name: str = ""


class DayRequest(BaseModel):
    athlete_id: int | None = None
    day: str
    distance_km: float = 0.0
    duration_s: float | None = None
    avg_hr: int | None = None
    rpe: float | None = None
    elevation_gain_m: float | None = None
    elevation_loss_m: float | None = None
    surface: str = "road"
    name: str = ""
    kind: str = "easy"
    notes: str = ""


class DeleteDayRequest(BaseModel):
    athlete_id: int | None = None
    day: str


class RaceRow(BaseModel):
    distance_m: float
    finish_time_s: float
    race_date: str | None = None
    name: str = ""
    place_overall: int | None = None
    field_size: int | None = None
    surface: str = "road"
    elevation_gain_m: float | None = None


class RacesRequest(BaseModel):
    athlete_id: int | None = None
    races: list[RaceRow] = Field(default_factory=list)


class SymptomRequest(BaseModel):
    athlete_id: int | None = None
    day: str
    body_part: str
    severity: float
    tissue: str = "other"
    affected_running: bool = False
    notes: str = ""


class DeleteSymptomRequest(BaseModel):
    athlete_id: int | None = None
    day: str
    body_part: str


class WellnessRequest(BaseModel):
    athlete_id: int | None = None
    day: str
    sleep_hours: float | None = None
    sleep_quality: int | None = None
    soreness: int | None = None
    stress: int | None = None
    motivation: int | None = None
    resting_hr: int | None = None
    body_mass_kg: float | None = None
    notes: str = ""


class EpisodeRow(BaseModel):
    body_part: str
    tissue: str = "other"
    onset_date: str
    resolved_date: str | None = None
    peak_severity: float | None = None
    days_lost: int = 0
    notes: str = ""


class EpisodesRequest(BaseModel):
    athlete_id: int | None = None
    episodes: list[EpisodeRow] = Field(default_factory=list)


# -- pages -----------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return PAGE_HTML.read_text()


@app.get("/api/state")
def state(athlete: int | None = None, days: int = Query(60, ge=1, le=3650)) -> dict[str, Any]:
    """Everything the page needs in one round trip."""
    with db() as database:
        athletes = database.list_athletes()
        if not athletes:
            return {"athletes": [], "athlete_id": None, "profile": None}

        athlete_id = _resolve(database, athlete)
        profile = database.load_profile(athlete_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"No athlete {athlete_id}.")

        window_start = date.today() - timedelta(days=days - 1)
        recent = [d for d in profile.training_days if d.day >= window_start]
        return {
            "athletes": athletes,
            "athlete_id": athlete_id,
            "profile": {"name": profile.name},
            "training_days": [_day_json(d) for d in recent],
            "races": [_race_json(r) for r in profile.races],
            "symptoms": [_symptom_json(s) for s in profile.symptoms
                         if s.day >= window_start],
            "wellness": [_wellness_json(w) for w in profile.wellness
                         if w.day >= window_start],
            "episodes": [_episode_json(e) for e in profile.episodes],
            "records": _records_json(profile),
            "rejected": [_effort_json(e) for e in rejected_efforts(
                profile.training_days, profile.races)[:12]],
            "layoffs": [_layoff_json(l) for l in find_layoffs(profile.training_days)],
            "summary": _summary_json(profile),
        }


def _day_json(d: TrainingDay) -> dict[str, Any]:
    return {
        "day": d.day.isoformat(), "distance_km": d.distance_km,
        "duration_s": d.duration_s, "avg_hr": d.avg_hr, "rpe": d.rpe,
        "elevation_gain_m": d.elevation_gain_m, "elevation_loss_m": d.elevation_loss_m,
        "surface": d.surface.value, "name": d.name, "kind": d.kind,
        "sessions": d.sessions, "source": d.source, "notes": d.notes,
        "duration": format_duration(d.duration_s) if d.duration_s else "",
    }


def _race_json(r: RaceResult) -> dict[str, Any]:
    return {
        "distance_m": r.distance_m, "finish_time_s": r.finish_time_s,
        "finish_time": format_duration(r.finish_time_s),
        "race_date": r.race_date.isoformat() if r.race_date else None,
        "name": r.name, "place_overall": r.place_overall, "field_size": r.field_size,
        "surface": r.surface.value, "elevation_gain_m": r.elevation_gain_m,
        "source": r.source,
    }


def _symptom_json(s: Symptom) -> dict[str, Any]:
    return {"day": s.day.isoformat(), "body_part": s.body_part, "severity": s.severity,
            "tissue": s.tissue.value, "affected_running": s.affected_running,
            "notes": s.notes, "is_flare": s.is_flare}


def _wellness_json(w: Wellness) -> dict[str, Any]:
    return {"day": w.day.isoformat(), "sleep_hours": w.sleep_hours,
            "sleep_quality": w.sleep_quality, "soreness": w.soreness,
            "stress": w.stress, "motivation": w.motivation,
            "resting_hr": w.resting_hr, "body_mass_kg": w.body_mass_kg,
            "notes": w.notes}


def _episode_json(e: InjuryEpisode) -> dict[str, Any]:
    return {"body_part": e.body_part, "tissue": e.tissue.value,
            "onset_date": e.onset_date.isoformat(),
            "resolved_date": e.resolved_date.isoformat() if e.resolved_date else None,
            "peak_severity": e.peak_severity, "days_lost": e.days_lost,
            "notes": e.notes, "is_open": e.is_open()}


def _effort_json(e) -> dict[str, Any]:
    return {"label": e.label, "time": e.formatted_time,
            "day": e.day.isoformat() if e.day else None, "name": e.name,
            "source": e.source, "reason": e.reason, "eligible": e.eligible}


def _layoff_json(l) -> dict[str, Any]:
    return {"start": l.start.isoformat(), "end": l.end.isoformat(), "kind": l.kind,
            "days": l.days, "confidence": l.confidence, "reason": l.reason,
            "ongoing": l.ongoing, "baseline_weekly_km": round(l.baseline_weekly_km, 1)}


def _records_json(profile: AthleteProfile) -> list[dict[str, Any]]:
    out = []
    for record in detect_records(profile.training_days, profile.races):
        chain = progression(profile.training_days, profile.races, record.label)
        out.append({
            "label": record.label,
            "time": record.effort.formatted_time,
            "set_on": record.set_on.isoformat() if record.set_on else None,
            "name": record.effort.name,
            "source": record.effort.source,
            "vdot": round(record.effort.vdot, 1),
            "improvement_s": record.improvement_s,
            "improvement": (format_duration(record.improvement_s)
                            if record.improvement_s else None),
            "chain": [{"day": e.day.isoformat() if e.day else None,
                       "time": e.formatted_time} for e in chain],
        })
    return out


def _summary_json(profile: AthleteProfile) -> dict[str, Any]:
    report = summarise(profile)
    fitness = current_fitness(profile)
    coverage = report.training
    return {
        "warnings": report.warnings,
        "days_logged": coverage.days_logged,
        "real_days": coverage.real_days,
        "total_km": round(coverage.total_km, 1),
        "coverage": round(coverage.coverage_fraction, 3),
        "first_day": coverage.first_day.isoformat() if coverage.first_day else None,
        "last_day": coverage.last_day.isoformat() if coverage.last_day else None,
        "methods": coverage.methods,
        "sources": coverage.sources,
        "races": report.races,
        "calibrated": report.calibration.fitted,
        "calibration_note": report.calibration.note,
        "vdot": round(fitness.vdot, 1) if fitness else None,
        "symptoms": len(profile.symptoms),
        "flares": sum(1 for s in profile.symptoms if s.is_flare),
        "episodes": len(profile.episodes),
        "open_episodes": sum(1 for e in profile.episodes if e.is_open()),
    }


# -- writes ----------------------------------------------------------------

@app.post("/api/athlete")
def create_athlete(req: AthleteRequest) -> dict[str, Any]:
    with db() as database:
        athlete_id = database.save_profile(AthleteProfile(name=req.name.strip()))
        return {"athlete_id": athlete_id}


@app.post("/api/training-day")
def save_day(req: DayRequest) -> dict[str, Any]:
    day = _required_date(req.day, "The day")
    with db() as database:
        athlete_id = _resolve(database, req.athlete_id)
        entry = TrainingDay(
            day=day, distance_km=max(0.0, req.distance_km), duration_s=req.duration_s,
            avg_hr=req.avg_hr, rpe=req.rpe, elevation_gain_m=req.elevation_gain_m,
            elevation_loss_m=req.elevation_loss_m,
            surface=_enum(Surface, req.surface, Surface.ROAD), name=req.name.strip(),
            kind=req.kind or "easy", source="manual", notes=req.notes.strip())
        database.upsert_training_days(athlete_id, [entry])
        return {"saved": entry.day.isoformat(), "athlete_id": athlete_id}


@app.post("/api/training-day/delete")
def delete_day(req: DeleteDayRequest) -> dict[str, Any]:
    day = _required_date(req.day, "The day")
    with db() as database:
        athlete_id = _resolve(database, req.athlete_id)
        removed = database.delete_training_days(athlete_id, since=day, until=day)
        return {"deleted": removed}


@app.post("/api/races")
def save_races(req: RacesRequest) -> dict[str, Any]:
    """Replace the race list. The table owns it as a set, as intake does."""
    with db() as database:
        athlete_id = _resolve(database, req.athlete_id)
        profile = database.load_profile(athlete_id)
        profile.races = [
            RaceResult(
                distance_m=r.distance_m, finish_time_s=r.finish_time_s,
                race_date=_date(r.race_date), name=r.name.strip(),
                place_overall=r.place_overall, field_size=r.field_size,
                surface=_enum(Surface, r.surface, Surface.ROAD),
                elevation_gain_m=r.elevation_gain_m, source="manual")
            for r in req.races if r.distance_m > 0 and r.finish_time_s > 0]
        database.save_profile(profile, athlete_id=athlete_id)
        return {"races": len(profile.races)}


@app.post("/api/symptom")
def save_symptom(req: SymptomRequest) -> dict[str, Any]:
    day = _required_date(req.day, "The day")
    part = req.body_part.strip()
    if not part:
        raise HTTPException(status_code=400, detail="A symptom needs a body part.")
    with db() as database:
        athlete_id = _resolve(database, req.athlete_id)
        database.upsert_symptoms(athlete_id, [Symptom(
            day=day, body_part=part, severity=max(0.0, min(10.0, req.severity)),
            tissue=_enum(Tissue, req.tissue, Tissue.OTHER),
            affected_running=req.affected_running, notes=req.notes.strip())])
        return {"saved": part}


@app.post("/api/symptom/delete")
def delete_symptom(req: DeleteSymptomRequest) -> dict[str, Any]:
    day = _required_date(req.day, "The day")
    with db() as database:
        athlete_id = _resolve(database, req.athlete_id)
        return {"deleted": database.delete_symptom(athlete_id, day, req.body_part)}


@app.post("/api/wellness")
def save_wellness(req: WellnessRequest) -> dict[str, Any]:
    day = _required_date(req.day, "The day")
    with db() as database:
        athlete_id = _resolve(database, req.athlete_id)
        database.upsert_wellness(athlete_id, [Wellness(
            day=day, sleep_hours=req.sleep_hours, sleep_quality=req.sleep_quality,
            soreness=req.soreness, stress=req.stress, motivation=req.motivation,
            resting_hr=req.resting_hr, body_mass_kg=req.body_mass_kg,
            notes=req.notes.strip())])
        return {"saved": day.isoformat()}


@app.post("/api/episodes")
def save_episodes(req: EpisodesRequest) -> dict[str, Any]:
    with db() as database:
        athlete_id = _resolve(database, req.athlete_id)
        episodes = []
        for row in req.episodes:
            onset = _date(row.onset_date)
            if onset is None or not row.body_part.strip():
                continue
            episodes.append(InjuryEpisode(
                body_part=row.body_part.strip(),
                tissue=_enum(Tissue, row.tissue, Tissue.OTHER), onset_date=onset,
                resolved_date=_date(row.resolved_date),
                peak_severity=row.peak_severity, days_lost=row.days_lost,
                notes=row.notes.strip()))
        return {"episodes": database.replace_episodes(athlete_id, episodes)}


# -- export ----------------------------------------------------------------

@app.get("/api/export/records.txt", response_class=PlainTextResponse)
def export_records(athlete: int | None = None) -> PlainTextResponse:
    """The record history as a text file, for keeping somewhere safe."""
    with db() as database:
        athlete_id = _resolve(database, athlete)
        profile = database.load_profile(athlete_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"No athlete {athlete_id}.")
        return PlainTextResponse(
            records_report(profile),
            headers={"Content-Disposition":
                     f'attachment; filename="{suggested_filename(profile)}"'})


def serve(host: str = "127.0.0.1", port: int = 8001,
          db_path: str | Path = DEFAULT_DB_PATH) -> None:
    import uvicorn

    global _DB_PATH
    _DB_PATH = Path(db_path)
    Database(_DB_PATH).close()   # create and migrate before the first request
    uvicorn.run(app, host=host, port=port, log_level="warning")
