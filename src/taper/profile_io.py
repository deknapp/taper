"""Save and load athlete profiles as JSON.

Plain JSON on purpose: a profile is a document a runner might want to read, diff
in git, or hand-edit when the form gets something wrong.
"""
from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

from taper.athlete import (
    AthleteProfile, GoalRace, Injury, LifeLoad, Occupation, Physiology,
    RaceResult, Sex, Surface, TrainingBackground, Tissue,
)

SCHEMA_VERSION = 1

T = TypeVar("T")


def _encode(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (Sex, Surface, Tissue, Occupation)):
        return value.value
    if is_dataclass(value):
        return {f.name: _encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    return value


def to_dict(profile: AthleteProfile) -> dict[str, Any]:
    payload = _encode(profile)
    payload["schema_version"] = SCHEMA_VERSION
    return payload


def _decode_value(annotation: Any, value: Any) -> Any:
    """Rebuild one field from JSON, following the dataclass's own annotations."""
    if value is None:
        return None

    # Unwrap `X | None`, taking the non-None arm.
    origin = get_origin(annotation)
    if origin is not None:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if origin is list:
            inner = args[0] if args else Any
            return [_decode_value(inner, v) for v in value]
        if len(args) == 1:
            return _decode_value(args[0], value)

    if annotation is date or annotation == "date":
        return date.fromisoformat(value) if isinstance(value, str) else value
    if isinstance(annotation, type):
        if issubclass(annotation, (Sex, Surface, Tissue, Occupation)):
            return annotation(value)
        if is_dataclass(annotation):
            return _decode_dataclass(annotation, value)
    return value


def _decode_dataclass(cls: type[T], payload: dict[str, Any]) -> T:
    from typing import get_type_hints

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in payload:
            continue
        kwargs[f.name] = _decode_value(hints.get(f.name, Any), payload[f.name])
    return cls(**kwargs)


def from_dict(payload: dict[str, Any]) -> AthleteProfile:
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"Profile uses schema version {version}, but this build only understands "
            f"{SCHEMA_VERSION}. Upgrade taper to read it."
        )
    return _decode_dataclass(AthleteProfile, payload)


def save(profile: AthleteProfile, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(profile), indent=2, sort_keys=False) + "\n")
    return path


def load(path: str | Path) -> AthleteProfile:
    return from_dict(json.loads(Path(path).read_text()))
