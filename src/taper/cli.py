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

    show = sub.add_parser("show", help="print a summary of a saved profile")
    show.add_argument("path")

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

    if args.command == "show":
        return _show(args.path)

    return 1


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
