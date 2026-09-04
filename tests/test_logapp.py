"""Tests for the training-log web app.

The app is thin on purpose, so these check the seam it owns: that what the form
sends is what the database ends up holding, that a bad field is a 400 rather
than a 500, and that the export route hands back a real file.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

import taper.logapp.app as logapp
from taper.athlete import AthleteProfile
from taper.db import Database

TODAY = date.today().isoformat()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    path = tmp_path / "log.db"
    Database(path).close()
    monkeypatch.setattr(logapp, "_DB_PATH", path)
    return TestClient(logapp.app)


@pytest.fixture()
def runner(client):
    """A client with one athlete already created."""
    client.post("/api/athlete", json={"name": "Test Runner"})
    return client


def state(client, **params):
    response = client.get("/api/state", params=params)
    assert response.status_code == 200
    return response.json()


# -- the page --------------------------------------------------------------

def test_the_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "training log" in response.text


# -- getting started -------------------------------------------------------

def test_an_empty_database_reports_no_athletes(client):
    assert state(client)["athletes"] == []


def test_an_athlete_can_be_created(client):
    response = client.post("/api/athlete", json={"name": "Test Runner"})
    assert response.status_code == 200
    assert state(client)["athletes"][0]["name"] == "Test Runner"


def test_logging_against_nobody_is_refused_rather_than_crashing(client):
    response = client.post("/api/training-day", json={"day": TODAY, "distance_km": 10})
    assert response.status_code == 404
    assert "athlete" in response.json()["detail"].lower()


def test_asking_for_an_athlete_that_does_not_exist_is_a_404(runner):
    assert runner.get("/api/state", params={"athlete": 999}).status_code == 404


# -- training days ---------------------------------------------------------

def test_a_logged_day_comes_back_in_the_state(runner):
    runner.post("/api/training-day", json={
        "day": TODAY, "distance_km": 16.2, "duration_s": 4980, "avg_hr": 152,
        "rpe": 6.5, "elevation_gain_m": 240, "elevation_loss_m": 310,
        "surface": "trail", "name": "Foothills", "kind": "long", "notes": "warm"})
    day = state(runner)["training_days"][0]

    assert day["distance_km"] == pytest.approx(16.2)
    assert day["avg_hr"] == 152
    assert day["elevation_loss_m"] == pytest.approx(310)
    assert day["surface"] == "trail"
    assert day["name"] == "Foothills"
    assert day["duration"] == "1:23:00"


def test_a_manually_logged_day_is_sourced_as_manual(runner):
    runner.post("/api/training-day", json={"day": TODAY, "distance_km": 10})
    assert state(runner)["training_days"][0]["source"] == "manual"


def test_logging_the_same_day_twice_corrects_it(runner):
    runner.post("/api/training-day", json={"day": TODAY, "distance_km": 10})
    runner.post("/api/training-day", json={"day": TODAY, "distance_km": 12})
    days = state(runner)["training_days"]
    assert len(days) == 1
    assert days[0]["distance_km"] == pytest.approx(12)


def test_a_rest_day_is_stored_as_rest(runner):
    runner.post("/api/training-day", json={"day": TODAY, "distance_km": 0, "kind": "off"})
    day = state(runner)["training_days"][0]
    assert day["kind"] == "off"
    assert day["distance_km"] == 0


def test_a_day_can_be_deleted(runner):
    runner.post("/api/training-day", json={"day": TODAY, "distance_km": 10})
    response = runner.post("/api/training-day/delete", json={"day": TODAY})
    assert response.json()["deleted"] == 1
    assert state(runner)["training_days"] == []


def test_an_unreadable_date_is_a_400_not_a_500(runner):
    response = runner.post("/api/training-day",
                           json={"day": "the day before yesterday", "distance_km": 10})
    assert response.status_code == 400
    assert "date" in response.json()["detail"]


def test_an_unknown_surface_falls_back_rather_than_failing(runner):
    runner.post("/api/training-day",
                json={"day": TODAY, "distance_km": 10, "surface": "lava"})
    assert state(runner)["training_days"][0]["surface"] == "road"


def test_a_negative_distance_is_clamped(runner):
    runner.post("/api/training-day", json={"day": TODAY, "distance_km": -5})
    assert state(runner)["training_days"][0]["distance_km"] == 0


def test_the_window_limits_which_days_come_back(runner):
    for offset in (1, 100):
        runner.post("/api/training-day", json={
            "day": (date.today() - timedelta(days=offset)).isoformat(),
            "distance_km": 10})
    assert len(state(runner, days=30)["training_days"]) == 1
    assert len(state(runner, days=365)["training_days"]) == 2


# -- races -----------------------------------------------------------------

def test_races_round_trip_through_the_form(runner):
    runner.post("/api/races", json={"races": [
        {"distance_m": 21097.5, "finish_time_s": 5400, "race_date": "2024-04-14",
         "name": "Spring Half", "place_overall": 42, "field_size": 1200,
         "surface": "road", "elevation_gain_m": 110}]})
    race = state(runner)["races"][0]
    assert race["name"] == "Spring Half"
    assert race["finish_time"] == "1:30:00"
    assert race["place_overall"] == 42


def test_saving_races_replaces_the_list_rather_than_appending(runner):
    runner.post("/api/races", json={"races": [
        {"distance_m": 5000, "finish_time_s": 1200}]})
    runner.post("/api/races", json={"races": [
        {"distance_m": 10000, "finish_time_s": 2400}]})
    races = state(runner)["races"]
    assert len(races) == 1
    assert races[0]["distance_m"] == 10000


def test_a_race_with_no_time_is_dropped_rather_than_stored_as_zero(runner):
    runner.post("/api/races", json={"races": [
        {"distance_m": 5000, "finish_time_s": 0},
        {"distance_m": 10000, "finish_time_s": 2400}]})
    assert len(state(runner)["races"]) == 1


def test_saving_races_does_not_disturb_the_training_log(runner):
    runner.post("/api/training-day", json={"day": TODAY, "distance_km": 10})
    runner.post("/api/races", json={"races": [
        {"distance_m": 5000, "finish_time_s": 1200}]})
    assert len(state(runner)["training_days"]) == 1


def test_a_race_result_unlocks_a_fitness_estimate(runner):
    assert state(runner)["summary"]["vdot"] is None
    runner.post("/api/races", json={"races": [
        {"distance_m": 10000, "finish_time_s": 2400, "race_date": "2024-05-01"}]})
    assert state(runner)["summary"]["vdot"] > 0


# -- symptoms --------------------------------------------------------------

def test_a_symptom_round_trips(runner):
    runner.post("/api/symptom", json={
        "day": TODAY, "body_part": "left achilles", "severity": 5,
        "tissue": "tendon", "affected_running": True, "notes": "tight"})
    symptom = state(runner)["symptoms"][0]
    assert symptom["body_part"] == "left achilles"
    assert symptom["severity"] == 5
    assert symptom["is_flare"] is True


def test_a_mild_symptom_is_not_a_flare(runner):
    runner.post("/api/symptom", json={
        "day": TODAY, "body_part": "knee", "severity": 1})
    assert state(runner)["symptoms"][0]["is_flare"] is False


def test_a_symptom_needs_a_body_part(runner):
    response = runner.post("/api/symptom",
                           json={"day": TODAY, "body_part": "   ", "severity": 3})
    assert response.status_code == 400


def test_severity_is_clamped_to_the_scale(runner):
    runner.post("/api/symptom",
                json={"day": TODAY, "body_part": "knee", "severity": 99})
    assert state(runner)["symptoms"][0]["severity"] == 10


def test_re_rating_the_same_part_on_the_same_day_corrects_it(runner):
    for severity in (5, 2):
        runner.post("/api/symptom",
                    json={"day": TODAY, "body_part": "knee", "severity": severity})
    symptoms = state(runner)["symptoms"]
    assert len(symptoms) == 1
    assert symptoms[0]["severity"] == 2


def test_a_symptom_can_be_taken_back(runner):
    runner.post("/api/symptom", json={"day": TODAY, "body_part": "knee", "severity": 5})
    response = runner.post("/api/symptom/delete",
                           json={"day": TODAY, "body_part": "knee"})
    assert response.json()["deleted"] == 1
    assert state(runner)["symptoms"] == []


# -- wellness --------------------------------------------------------------

def test_a_check_in_round_trips(runner):
    runner.post("/api/wellness", json={
        "day": TODAY, "sleep_hours": 7.5, "sleep_quality": 4, "soreness": 2,
        "stress": 3, "motivation": 5, "resting_hr": 47, "body_mass_kg": 71.4})
    entry = state(runner)["wellness"][0]
    assert entry["sleep_hours"] == pytest.approx(7.5)
    assert entry["resting_hr"] == 47


def test_a_mostly_empty_check_in_is_allowed(runner):
    runner.post("/api/wellness", json={"day": TODAY, "sleep_hours": 6})
    entry = state(runner)["wellness"][0]
    assert entry["sleep_hours"] == pytest.approx(6)
    assert entry["soreness"] is None


def test_one_check_in_per_day(runner):
    runner.post("/api/wellness", json={"day": TODAY, "sleep_hours": 6})
    runner.post("/api/wellness", json={"day": TODAY, "sleep_hours": 8})
    entries = state(runner)["wellness"]
    assert len(entries) == 1
    assert entries[0]["sleep_hours"] == pytest.approx(8)


# -- episodes --------------------------------------------------------------

def test_episodes_round_trip(runner):
    runner.post("/api/episodes", json={"episodes": [
        {"body_part": "left achilles", "tissue": "tendon",
         "onset_date": "2024-03-01", "resolved_date": "2024-04-15",
         "peak_severity": 7, "days_lost": 32, "notes": "hill block"}]})
    episode = state(runner)["episodes"][0]
    assert episode["body_part"] == "left achilles"
    assert episode["days_lost"] == 32
    assert episode["is_open"] is False


def test_an_unresolved_episode_reads_as_open(runner):
    runner.post("/api/episodes", json={"episodes": [
        {"body_part": "shin", "onset_date": "2024-03-01"}]})
    assert state(runner)["episodes"][0]["is_open"] is True


def test_an_episode_without_a_body_part_is_dropped(runner):
    runner.post("/api/episodes", json={"episodes": [
        {"body_part": "  ", "onset_date": "2024-03-01"},
        {"body_part": "shin", "onset_date": "2024-03-01"}]})
    assert len(state(runner)["episodes"]) == 1


def test_an_episode_without_an_onset_is_dropped(runner):
    runner.post("/api/episodes", json={"episodes": [
        {"body_part": "shin", "onset_date": "whenever"}]})
    assert state(runner)["episodes"] == []


def test_saving_episodes_replaces_the_list(runner):
    runner.post("/api/episodes", json={"episodes": [
        {"body_part": "shin", "onset_date": "2024-03-01"}]})
    runner.post("/api/episodes", json={"episodes": []})
    assert state(runner)["episodes"] == []


# -- records and the export ------------------------------------------------

def test_records_appear_once_there_is_something_to_rank(runner):
    runner.post("/api/races", json={"races": [
        {"distance_m": 5000, "finish_time_s": 1200, "race_date": "2024-05-01",
         "name": "Spring 5K", "elevation_gain_m": 10}]})
    records = state(runner)["records"]
    assert records[0]["label"] == "5K"
    assert records[0]["time"] == "20:00"
    assert records[0]["vdot"] > 0


def test_a_downhill_effort_is_reported_as_rejected_with_a_reason(runner):
    runner.post("/api/training-day", json={
        "day": TODAY, "distance_km": 5.0, "duration_s": 1150,
        "elevation_gain_m": 20, "elevation_loss_m": 400})
    rejected = state(runner)["rejected"]
    assert rejected and "Net downhill" in rejected[0]["reason"]


def test_the_export_is_served_as_a_named_attachment(runner):
    runner.post("/api/races", json={"races": [
        {"distance_m": 5000, "finish_time_s": 1200, "race_date": "2024-05-01"}]})
    response = runner.get("/api/export/records.txt")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert "test-runner-records-" in response.headers["content-disposition"]


def test_the_export_contains_the_record_history(runner):
    runner.post("/api/races", json={"races": [
        {"distance_m": 5000, "finish_time_s": 1200, "race_date": "2024-05-01",
         "name": "Spring 5K", "elevation_gain_m": 10}]})
    text = runner.get("/api/export/records.txt").text
    assert "PERSONAL RECORDS" in text
    assert "20:00" in text
    assert "Test Runner" in text


def test_the_export_works_on_an_empty_log(runner):
    response = runner.get("/api/export/records.txt")
    assert response.status_code == 200
    assert "Nothing yet" in response.text


# -- the rest of the state payload ----------------------------------------

def test_the_summary_counts_what_was_logged(runner):
    runner.post("/api/training-day", json={"day": TODAY, "distance_km": 10})
    runner.post("/api/symptom", json={"day": TODAY, "body_part": "knee", "severity": 6})
    summary = state(runner)["summary"]
    assert summary["days_logged"] == 1
    assert summary["total_km"] == pytest.approx(10)
    assert summary["flares"] == 1


def test_two_athletes_keep_separate_logs(client):
    first = client.post("/api/athlete", json={"name": "A"}).json()["athlete_id"]
    second = client.post("/api/athlete", json={"name": "B"}).json()["athlete_id"]
    client.post("/api/training-day",
                json={"athlete_id": first, "day": TODAY, "distance_km": 10})

    assert len(state(client, athlete=first)["training_days"]) == 1
    assert state(client, athlete=second)["training_days"] == []
