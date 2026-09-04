"""Local SQLite storage.

One file, no server, no account. The schema draws a hard line between *real*
history -- races that were actually run, training that actually happened,
injuries that actually occurred -- and anything the simulator produced. Real
rows carry a `source` saying where they came from; simulated days live in
separate tables entirely and can never be mistaken for evidence.

That separation is the point. A simulator that quietly folds its own output back
into its inputs stops being a model of anything.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

from taper.athlete import (
    AthleteProfile, GoalRace, Injury, LifeLoad, Occupation, Physiology, RaceResult,
    Sex, Surface, Tissue, TrainingBackground, TrainingDay,
)

SCHEMA_VERSION = 3
DEFAULT_DB_PATH = Path("taper.db")

# Sources that represent things that really happened, as opposed to anything the
# simulator invented. Used to keep the two apart in every query that feeds the
# engine.
REAL_SOURCES = ("manual", "paste", "csv", "import", "estimated")

_SCHEMA = """
CREATE TABLE athlete (
    id                  INTEGER PRIMARY KEY,
    name                TEXT    NOT NULL DEFAULT '',
    birth_date          TEXT,
    sex                 TEXT    NOT NULL DEFAULT 'unspecified',
    height_cm           REAL,
    body_mass_kg        REAL,
    resting_hr          INTEGER,
    max_hr              INTEGER,
    vo2max_tested       REAL,
    current_weekly_km   REAL,
    peak_weekly_km_ever REAL,
    longest_recent_run_km REAL,
    runs_per_week       REAL,
    years_running       REAL,
    strength_days_per_week REAL NOT NULL DEFAULT 0,
    cross_training_hours_per_week REAL NOT NULL DEFAULT 0,
    primary_surface     TEXT    NOT NULL DEFAULT 'road',
    hilliness_m_per_km  REAL,
    sleep_hours         REAL,
    occupation          TEXT    NOT NULL DEFAULT 'sedentary',
    life_stress         INTEGER NOT NULL DEFAULT 3,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL
);

CREATE TABLE race (
    id              INTEGER PRIMARY KEY,
    athlete_id      INTEGER NOT NULL REFERENCES athlete(id) ON DELETE CASCADE,
    distance_m      REAL    NOT NULL,
    finish_time_s   REAL    NOT NULL,
    race_date       TEXT,
    name            TEXT    NOT NULL DEFAULT '',
    place_overall   INTEGER,
    field_size      INTEGER,
    surface         TEXT    NOT NULL DEFAULT 'road',
    elevation_gain_m REAL,
    source          TEXT    NOT NULL DEFAULT 'manual',
    created_at      TEXT    NOT NULL
);
CREATE INDEX race_athlete_date ON race(athlete_id, race_date);

CREATE TABLE injury (
    id          INTEGER PRIMARY KEY,
    athlete_id  INTEGER NOT NULL REFERENCES athlete(id) ON DELETE CASCADE,
    tissue      TEXT    NOT NULL,
    body_part   TEXT    NOT NULL DEFAULT '',
    start_date  TEXT,
    weeks_out   REAL,
    recurrences INTEGER NOT NULL DEFAULT 0,
    notes       TEXT    NOT NULL DEFAULT '',
    source      TEXT    NOT NULL DEFAULT 'manual'
);
CREATE INDEX injury_athlete ON injury(athlete_id);

-- Real training that actually happened. The daily impulse series the engine
-- runs on. One row per day, rest days included.
CREATE TABLE training_day (
    id              INTEGER PRIMARY KEY,
    athlete_id      INTEGER NOT NULL REFERENCES athlete(id) ON DELETE CASCADE,
    day             TEXT    NOT NULL,
    distance_km     REAL    NOT NULL DEFAULT 0,
    duration_s      REAL,
    avg_hr          INTEGER,
    rpe             REAL,
    elevation_gain_m REAL,
    elevation_loss_m REAL,
    surface         TEXT    NOT NULL DEFAULT 'road',
    name            TEXT    NOT NULL DEFAULT '',
    kind            TEXT    NOT NULL DEFAULT 'easy',
    sessions        INTEGER NOT NULL DEFAULT 1,
    source          TEXT    NOT NULL DEFAULT 'manual',
    notes           TEXT    NOT NULL DEFAULT '',
    UNIQUE(athlete_id, day)
);
CREATE INDEX training_day_athlete_day ON training_day(athlete_id, day);

CREATE TABLE goal (
    id              INTEGER PRIMARY KEY,
    athlete_id      INTEGER NOT NULL REFERENCES athlete(id) ON DELETE CASCADE,
    distance_m      REAL    NOT NULL,
    race_date       TEXT    NOT NULL,
    name            TEXT    NOT NULL DEFAULT '',
    target_time_s   REAL,
    elevation_gain_m REAL,
    surface         TEXT    NOT NULL DEFAULT 'road',
    is_active       INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX goal_athlete ON goal(athlete_id, is_active);

-- Simulated output, kept strictly apart from the real tables above.
CREATE TABLE sim_run (
    id          INTEGER PRIMARY KEY,
    athlete_id  INTEGER NOT NULL REFERENCES athlete(id) ON DELETE CASCADE,
    label       TEXT    NOT NULL DEFAULT '',
    created_at  TEXT    NOT NULL,
    start_date  TEXT    NOT NULL,
    end_date    TEXT    NOT NULL,
    params_json TEXT    NOT NULL DEFAULT '{}'
);

CREATE TABLE sim_day (
    id          INTEGER PRIMARY KEY,
    sim_run_id  INTEGER NOT NULL REFERENCES sim_run(id) ON DELETE CASCADE,
    day         TEXT    NOT NULL,
    load        REAL    NOT NULL DEFAULT 0,
    fitness     REAL    NOT NULL DEFAULT 0,
    fatigue     REAL    NOT NULL DEFAULT 0,
    performance REAL,
    UNIQUE(sim_run_id, day)
);
CREATE INDEX sim_day_run ON sim_day(sim_run_id, day);

CREATE TABLE setting (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


class Database:
    """A taper database file.

    Deliberately thin: the domain model lives in the dataclasses, and this maps
    them on and off disk. Callers get whole `AthleteProfile` objects, not rows.
    """

    def __init__(self, path: str | Path = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, detect_types=0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._migrate()

    # -- schema ------------------------------------------------------------

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            with self.conn:
                self.conn.executescript(_SCHEMA)
                self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            return
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"{self.path} uses schema version {version}; this build understands "
                f"{SCHEMA_VERSION}. Upgrade taper to open it.")

        # v1 -> v2. Elevation loss and the activity name were on TrainingDay but
        # had no columns, so both were dropped on write. Elevation loss is not
        # cosmetic: the record screen needs it to reject net-downhill efforts,
        # and without it that screen silently passed everything.
        if version < 2:
            with self.conn:
                self.conn.execute(
                    "ALTER TABLE training_day ADD COLUMN elevation_loss_m REAL")
                self.conn.execute(
                    "ALTER TABLE training_day ADD COLUMN name TEXT NOT NULL DEFAULT ''")
                self.conn.execute("PRAGMA user_version = 2")

        # v2 -> v3. How many activities were merged into a day. Without it a
        # day cannot say whether its time describes one run or several added up.
        if version < 3:
            with self.conn:
                self.conn.execute(
                    "ALTER TABLE training_day ADD COLUMN sessions INTEGER NOT NULL "
                    "DEFAULT 1")
                self.conn.execute("PRAGMA user_version = 3")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self.conn:
            yield self.conn

    # -- athletes ----------------------------------------------------------

    def list_athletes(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, name, updated_at FROM athlete "
            "ORDER BY updated_at DESC, id DESC").fetchall()
        return [dict(r) for r in rows]

    def default_athlete_id(self) -> int | None:
        row = self.conn.execute(
            "SELECT id FROM athlete ORDER BY updated_at DESC, id DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    def save_profile(self, profile: AthleteProfile, athlete_id: int | None = None) -> int:
        """Insert or update an athlete and replace their child rows.

        Races, injuries and the goal are replaced wholesale because the intake
        form owns them as a set. Training days are *not* touched here -- they
        are accumulated real history and are only ever added to.
        """
        now = datetime.now().isoformat(timespec="seconds")
        p, t, l = profile.physiology, profile.training, profile.life
        fields = {
            "name": profile.name, "birth_date": _iso(profile.birth_date),
            "sex": profile.sex.value, "height_cm": p.height_cm,
            "body_mass_kg": p.body_mass_kg, "resting_hr": p.resting_hr,
            "max_hr": p.max_hr, "vo2max_tested": p.vo2max_tested,
            "current_weekly_km": t.current_weekly_km,
            "peak_weekly_km_ever": t.peak_weekly_km_ever,
            "longest_recent_run_km": t.longest_recent_run_km,
            "runs_per_week": t.runs_per_week, "years_running": t.years_running,
            "strength_days_per_week": t.strength_days_per_week,
            "cross_training_hours_per_week": t.cross_training_hours_per_week,
            "primary_surface": t.primary_surface.value,
            "hilliness_m_per_km": t.hilliness_m_per_km,
            "sleep_hours": l.sleep_hours, "occupation": l.occupation.value,
            "life_stress": l.life_stress, "updated_at": now,
        }

        with self._tx() as conn:
            if athlete_id is None:
                cols = ", ".join([*fields, "created_at"])
                marks = ", ".join(["?"] * (len(fields) + 1))
                cur = conn.execute(f"INSERT INTO athlete ({cols}) VALUES ({marks})",
                                   [*fields.values(), now])
                athlete_id = int(cur.lastrowid)
            else:
                sets = ", ".join(f"{k} = ?" for k in fields)
                conn.execute(f"UPDATE athlete SET {sets} WHERE id = ?",
                             [*fields.values(), athlete_id])
                conn.execute("DELETE FROM race WHERE athlete_id = ?", (athlete_id,))
                conn.execute("DELETE FROM injury WHERE athlete_id = ?", (athlete_id,))
                conn.execute("DELETE FROM goal WHERE athlete_id = ?", (athlete_id,))

            for r in profile.races:
                conn.execute(
                    "INSERT INTO race (athlete_id, distance_m, finish_time_s, race_date, "
                    "name, place_overall, field_size, surface, elevation_gain_m, source, "
                    "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (athlete_id, r.distance_m, r.finish_time_s, _iso(r.race_date), r.name,
                     r.place_overall, r.field_size, r.surface.value, r.elevation_gain_m,
                     r.source, now))

            for inj in profile.injuries:
                conn.execute(
                    "INSERT INTO injury (athlete_id, tissue, body_part, start_date, "
                    "weeks_out, recurrences, notes) VALUES (?,?,?,?,?,?,?)",
                    (athlete_id, inj.tissue.value, inj.body_part, _iso(inj.start_date),
                     inj.weeks_out, inj.recurrences, inj.notes))

            if profile.goal:
                g = profile.goal
                conn.execute(
                    "INSERT INTO goal (athlete_id, distance_m, race_date, name, "
                    "target_time_s, elevation_gain_m, surface) VALUES (?,?,?,?,?,?,?)",
                    (athlete_id, g.distance_m, _iso(g.race_date), g.name, g.target_time_s,
                     g.elevation_gain_m, g.surface.value))

        return athlete_id

    def load_profile(self, athlete_id: int) -> AthleteProfile | None:
        row = self.conn.execute(
            "SELECT * FROM athlete WHERE id = ?", (athlete_id,)).fetchone()
        if row is None:
            return None

        profile = AthleteProfile(
            name=row["name"], birth_date=_parse_date(row["birth_date"]),
            sex=Sex(row["sex"]),
            physiology=Physiology(
                resting_hr=row["resting_hr"], max_hr=row["max_hr"],
                vo2max_tested=row["vo2max_tested"], height_cm=row["height_cm"],
                body_mass_kg=row["body_mass_kg"]),
            training=TrainingBackground(
                current_weekly_km=row["current_weekly_km"],
                peak_weekly_km_ever=row["peak_weekly_km_ever"],
                longest_recent_run_km=row["longest_recent_run_km"],
                runs_per_week=row["runs_per_week"], years_running=row["years_running"],
                strength_days_per_week=row["strength_days_per_week"],
                cross_training_hours_per_week=row["cross_training_hours_per_week"],
                primary_surface=Surface(row["primary_surface"]),
                hilliness_m_per_km=row["hilliness_m_per_km"]),
            life=LifeLoad(sleep_hours=row["sleep_hours"],
                          occupation=Occupation(row["occupation"]),
                          life_stress=row["life_stress"]),
        )

        profile.races = [
            RaceResult(
                distance_m=r["distance_m"], finish_time_s=r["finish_time_s"],
                race_date=_parse_date(r["race_date"]), name=r["name"],
                place_overall=r["place_overall"], field_size=r["field_size"],
                surface=Surface(r["surface"]), elevation_gain_m=r["elevation_gain_m"],
                source=r["source"])
            for r in self.conn.execute(
                "SELECT * FROM race WHERE athlete_id = ? ORDER BY race_date", (athlete_id,))]

        profile.injuries = [
            Injury(tissue=Tissue(i["tissue"]), body_part=i["body_part"],
                   start_date=_parse_date(i["start_date"]), weeks_out=i["weeks_out"],
                   recurrences=i["recurrences"], notes=i["notes"])
            for i in self.conn.execute(
                "SELECT * FROM injury WHERE athlete_id = ? ORDER BY start_date",
                (athlete_id,))]

        profile.training_days = self.training_days(athlete_id)

        goal_row = self.conn.execute(
            "SELECT * FROM goal WHERE athlete_id = ? AND is_active = 1 "
            "ORDER BY race_date LIMIT 1", (athlete_id,)).fetchone()
        if goal_row:
            profile.goal = GoalRace(
                distance_m=goal_row["distance_m"],
                race_date=_parse_date(goal_row["race_date"]),
                name=goal_row["name"], target_time_s=goal_row["target_time_s"],
                elevation_gain_m=goal_row["elevation_gain_m"],
                surface=Surface(goal_row["surface"]))
        return profile

    def delete_athlete(self, athlete_id: int) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM athlete WHERE id = ?", (athlete_id,))

    # -- training days: accumulated real history ---------------------------

    def training_days(self, athlete_id: int, since: date | None = None,
                      until: date | None = None) -> list[TrainingDay]:
        sql = "SELECT * FROM training_day WHERE athlete_id = ?"
        args: list[Any] = [athlete_id]
        if since:
            sql += " AND day >= ?"
            args.append(since.isoformat())
        if until:
            sql += " AND day <= ?"
            args.append(until.isoformat())
        sql += " ORDER BY day"
        return [
            TrainingDay(
                day=_parse_date(r["day"]), distance_km=r["distance_km"],
                duration_s=r["duration_s"], avg_hr=r["avg_hr"], rpe=r["rpe"],
                elevation_gain_m=r["elevation_gain_m"],
                elevation_loss_m=r["elevation_loss_m"], surface=Surface(r["surface"]),
                name=r["name"], kind=r["kind"], sessions=r["sessions"],
                source=r["source"], notes=r["notes"])
            for r in self.conn.execute(sql, args)]

    def upsert_training_days(self, athlete_id: int, days: Iterable[TrainingDay]) -> int:
        """Add or replace real training days. Returns how many rows were written.

        Upsert rather than insert: re-importing an overlapping range should
        correct the overlap, not duplicate it.
        """
        rows = [
            (athlete_id, _iso(d.day), d.distance_km, d.duration_s, d.avg_hr, d.rpe,
             d.elevation_gain_m, d.elevation_loss_m, d.surface.value, d.name, d.kind,
             d.sessions, d.source, d.notes)
            for d in days]
        if not rows:
            return 0
        with self._tx() as conn:
            conn.executemany(
                "INSERT INTO training_day (athlete_id, day, distance_km, duration_s, "
                "avg_hr, rpe, elevation_gain_m, elevation_loss_m, surface, name, kind, "
                "sessions, source, notes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(athlete_id, day) DO UPDATE SET "
                "distance_km=excluded.distance_km, duration_s=excluded.duration_s, "
                "avg_hr=excluded.avg_hr, rpe=excluded.rpe, "
                "elevation_gain_m=excluded.elevation_gain_m, "
                "elevation_loss_m=excluded.elevation_loss_m, surface=excluded.surface, "
                "name=excluded.name, kind=excluded.kind, sessions=excluded.sessions, "
                "source=excluded.source, notes=excluded.notes",
                rows)
        return len(rows)

    def delete_training_days(self, athlete_id: int, since: date | None = None,
                             until: date | None = None) -> int:
        sql = "DELETE FROM training_day WHERE athlete_id = ?"
        args: list[Any] = [athlete_id]
        if since:
            sql += " AND day >= ?"
            args.append(since.isoformat())
        if until:
            sql += " AND day <= ?"
            args.append(until.isoformat())
        with self._tx() as conn:
            return conn.execute(sql, args).rowcount

    # -- settings ----------------------------------------------------------

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM setting WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO setting (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    def all_settings(self) -> dict[str, str]:
        return {r["key"]: r["value"]
                for r in self.conn.execute("SELECT key, value FROM setting")}
