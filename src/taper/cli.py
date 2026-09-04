"""Command line entry point."""
from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from taper import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="taper", description="A running simulator grounded in exercise-physiology research.")
    parser.add_argument("--version", action="version", version=f"taper {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    intake = sub.add_parser("intake", help="open the athlete intake form in a browser")
    intake.add_argument("--port", type=int, default=8000)
    intake.add_argument("--host", default="127.0.0.1")
    intake.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window automatically")

    log = sub.add_parser(
        "log", help="open the training log in a browser: days, races, symptoms, records")
    log.add_argument("--port", type=int, default=8001)
    log.add_argument("--host", default="127.0.0.1")
    log.add_argument("--db", default="taper.db", help="database file (default: taper.db)")
    log.add_argument("--no-browser", action="store_true",
                     help="do not open a browser window automatically")

    export = sub.add_parser(
        "export", help="write the record history to a text file")
    export.add_argument("-o", "--output", default=None,
                        help="file to write (default: a dated name in this directory)")
    export.add_argument("--db", default="taper.db", help="database file (default: taper.db)")
    export.add_argument("--athlete", type=int, default=None,
                        help="athlete id (default: the most recently updated)")

    show = sub.add_parser("show", help="print a summary of a saved profile")
    show.add_argument("path")

    imp = sub.add_parser(
        "import-strava",
        help="import a Strava bulk export's activities.csv into the training log")
    imp.add_argument("path", help="path to activities.csv from your Strava archive")
    imp.add_argument("--db", default="taper.db", help="database file (default: taper.db)")
    imp.add_argument("--athlete", type=int, default=None,
                     help="athlete id (default: the most recently updated)")
    imp.add_argument("--dry-run", action="store_true",
                     help="report what would be imported without writing")

    args = parser.parse_args(argv)

    if args.command == "intake":
        from taper.intake.app import serve

        url = f"http://{args.host}:{args.port}/"
        print(f"taper intake -> {url}")
        print("Everything stays on this machine. Ctrl-C to stop.")
        if not args.no_browser:
            threading.Timer(0.7, webbrowser.open, args=(url,)).start()
        serve(host=args.host, port=args.port)
        return 0

    if args.command == "log":
        from taper.logapp.app import serve

        url = f"http://{args.host}:{args.port}/"
        print(f"taper log -> {url}")
        print(f"Writing to {args.db}. Everything stays on this machine. Ctrl-C to stop.")
        if not args.no_browser:
            threading.Timer(0.7, webbrowser.open, args=(url,)).start()
        serve(host=args.host, port=args.port, db_path=args.db)
        return 0

    if args.command == "export":
        return _export(args.output, args.db, args.athlete)

    if args.command == "show":
        return _show(args.path)

    if args.command == "import-strava":
        return _import_strava(args.path, args.db, args.athlete, args.dry_run)

    return 1


def _export(output: str | None, db_path: str, athlete_id: int | None) -> int:
    from pathlib import Path

    from taper.db import Database
    from taper.export import records_report, suggested_filename

    if not Path(db_path).exists():
        print(f"No database at {db_path}. Run 'taper log' or 'taper import-strava' first.")
        return 1

    db = Database(db_path)
    try:
        target = athlete_id or db.default_athlete_id()
        if target is None:
            print("No athlete in the database yet.")
            return 1
        profile = db.load_profile(target)
    finally:
        db.close()

    path = Path(output) if output else Path(suggested_filename(profile))
    path.write_text(records_report(profile))
    print(f"Wrote {path} ({path.stat().st_size} bytes).")
    return 0


def _import_strava(path: str, db_path: str, athlete_id: int | None, dry_run: bool) -> int:
    from pathlib import Path

    from taper.db import Database
    from taper.importers.strava import parse_activities_csv
    from taper.layoffs import find_layoffs
    from taper.records import detect_records, rejected_efforts

    source = Path(path)
    if not source.exists():
        print(f"No such file: {source}")
        return 1

    result = parse_activities_csv(source.read_text(errors="replace"))
    if not result.days:
        for warning in result.warnings:
            print(f"  {warning}")
        return 1

    print(f"Read {result.rows_seen} rows from {source.name}")
    print(f"  {result.runs_used} runs, {result.cross_used} cross-training sessions")
    print(f"  {result.first_day} to {result.last_day}")
    print(f"  {len(result.days)} day rows, including {result.rest_days_filled} rest days")
    if result.skipped_types:
        skipped = ", ".join(f"{name} x{n}" for name, n in
                            sorted(result.skipped_types.items(), key=lambda kv: -kv[1])[:6])
        print(f"  skipped: {skipped}")
    for warning in result.warnings:
        print(f"  note: {warning}")

    layoffs = find_layoffs(result.days)
    if layoffs:
        print(f"\n{len(layoffs)} possible layoff{'s' if len(layoffs) != 1 else ''} "
              f"found in the log. These are candidates, not conclusions -- confirm or "
              f"dismiss each one before the injury model uses it:")
        for layoff in layoffs:
            print(f"  [{layoff.confidence:>8}] {layoff.start} to {layoff.end} "
                  f"({layoff.days}d, {layoff.kind})")
            print(f"             {layoff.reason}")

    records = detect_records(result.days, [])
    if records:
        print("\nPersonal records in the log, on terrain that makes them comparable:")
        for record in records:
            print(f"  {record.label:<16}{record.effort.formatted_time:>9}  "
                  f"{record.set_on}  {record.effort.name[:32]}")
    rejected = rejected_efforts(result.days, [])
    if rejected:
        print(f"\n{len(rejected)} fast effort{'s' if len(rejected) != 1 else ''} "
              f"excluded by the terrain screen (use --dry-run output to review):")
        for effort in rejected[:5]:
            print(f"  {effort.label:<8}{effort.formatted_time:>9}  {effort.reason[:66]}")

    if dry_run:
        print("\nDry run -- nothing written.")
        return 0

    db = Database(db_path)
    try:
        target = athlete_id or db.default_athlete_id()
        if target is None:
            print("\nNo athlete in the database yet. Run 'taper intake' first to "
                  "create one, then import again.")
            return 1
        written = db.upsert_training_days(target, result.days)
        print(f"\nWrote {written} training days to {db_path} for athlete {target}.")
    finally:
        db.close()
    return 0


def _show(path: str) -> int:
    from taper.insights import current_fitness, formatted_equivalents, readiness_flags
    from taper.profile_io import load
    from taper.units import format_duration, km_to_miles

    profile = load(path)
    print(f"{profile.name or 'Unnamed runner'}")

    weekly = profile.training.current_weekly_km
    if weekly:
        print(f"  {km_to_miles(weekly):.1f} mi/week ({weekly:.1f} km)")
    print(f"  {len(profile.races)} races, {len(profile.injuries)} injuries on record")

    fitness = current_fitness(profile)
    if fitness is None:
        print("\n  No race results, so no fitness estimate yet.")
        return 0

    print(f"\n  VDOT {fitness.vdot:.1f} [{fitness.confidence}]")
    print(f"  {fitness.note}")
    print("\n  Equivalent performances:")
    for label, time in formatted_equivalents(fitness.vdot):
        print(f"    {label:<16}{time}")

    if profile.goal:
        from taper.physiology import predict_time

        predicted = predict_time(profile.goal.distance_m, fitness.vdot)
        line = f"\n  Goal: {profile.goal.name or 'race'} on {profile.goal.race_date}"
        print(line)
        print(f"    Predicted at current fitness: {format_duration(predicted)}")
        if profile.goal.target_time_s:
            gap = profile.goal.target_time_s - predicted
            verb = "ahead of" if gap > 0 else "behind"
            print(f"    Target {format_duration(profile.goal.target_time_s)} "
                  f"-- {format_duration(abs(gap))} {verb} target pace")

    flags = readiness_flags(profile)
    if flags:
        print("\n  Readiness:")
        for flag in flags:
            print(f"    [{flag.severity}] {flag.message}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
