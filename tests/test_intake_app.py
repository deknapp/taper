from datetime import date, timedelta

from fastapi.testclient import TestClient

from taper.intake.app import app

client = TestClient(app)


def test_form_is_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "runner intake" in res.text


def test_parse_endpoint_reads_a_results_table():
    blob = (
        "Springfield Half Marathon - April 15, 2024\n"
        "Place Name      Bib Age Gun     Chip    Pace\n"
        "12    Jane Doe  841 34  1:34:40 1:34:22 7:12\n"
    )
    res = client.post("/api/parse-races", json={"text": blob})
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 1
    race = data["races"][0]
    assert race["distance_m"] == 21097.5
    assert race["finish_time"] == "1:34:22"
    assert race["race_date"] == "2024-04-15"


def test_parse_endpoint_explains_an_empty_result():
    res = client.post("/api/parse-races", json={"text": "12  Jane Doe  1:34:22"})
    data = res.json()
    assert data["count"] == 0
    assert "distance" in data["message"]


# 3:10:49 for the marathon is the VDOT 50 row of Daniels' table.
VDOT_50_MARATHON_S = 3 * 3600 + 10 * 60 + 49


def _profile_with_marathon(days_ago: int) -> dict:
    return {
        "name": "Jane",
        "training": {"current_weekly_km": 64.4, "strength_days_per_week": 2},
        "races": [{"distance_m": 42195.0, "finish_time_s": VDOT_50_MARATHON_S,
                   "race_date": (date.today() - timedelta(days=days_ago)).isoformat(),
                   "name": "Boston"}],
    }


def test_insights_endpoint_derives_vdot_and_equivalents():
    res = client.post("/api/insights", json={"profile": _profile_with_marathon(60)})
    assert res.status_code == 200
    data = res.json()
    assert data["fitness"]["vdot"] == 50.0
    assert data["fitness"]["confidence"] == "measured"
    labels = [e["label"] for e in data["fitness"]["equivalents"]]
    assert labels == ["1 mile", "5K", "10K", "Half marathon", "Marathon"]
    assert data["races"][0]["vdot"] == 50.0


def test_insights_ages_a_stale_result_and_says_so():
    """A race older than a year is decayed forward, and labelled as such rather
    than quietly presented as current fitness."""
    data = client.post("/api/insights",
                       json={"profile": _profile_with_marathon(365 * 3)}).json()
    assert data["fitness"]["confidence"] == "aged"
    assert data["fitness"]["vdot"] < 50.0
    assert "optimistic" in data["fitness"]["note"]
    # The race's own VDOT is untouched -- only the present-day estimate decays.
    assert data["races"][0]["vdot"] == 50.0


def test_insights_endpoint_with_no_races():
    res = client.post("/api/insights", json={"profile": {"name": "Nobody"}})
    data = res.json()
    assert data["fitness"] is None
    assert any("weekly mileage" in f["message"] for f in data["flags"])


def test_insights_rejects_a_malformed_profile():
    res = client.post("/api/insights", json={"profile": {"sex": "not-a-sex"}})
    assert res.status_code == 400


def test_save_writes_json(tmp_path, monkeypatch):
    import taper.intake.app as app_module

    monkeypatch.setattr(app_module, "DEFAULT_PROFILE_DIR", tmp_path)
    res = client.post("/api/save", json={"profile": {"name": "Jane Doe"}})
    assert res.status_code == 200
    written = tmp_path / "jane-doe.json"
    assert written.exists()
    assert res.json()["path"] == str(written.resolve())
