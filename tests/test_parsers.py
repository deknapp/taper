from datetime import date

from taper.intake.parsers import (
    parse_any, parse_date, parse_distance, parse_finish_time,
    parse_pasted_results, parse_placing,
)


def test_distance_named_and_numeric():
    assert parse_distance("Boston Marathon") == 42195.0
    assert parse_distance("half marathon") == 21097.5
    # 'half marathon' must win over the bare 'half' entry.
    assert parse_distance("Rock n Roll Half Marathon") == 21097.5
    assert parse_distance("Turkey Trot 5K") == 5000.0
    assert parse_distance("10 mile classic") == 16093.44
    assert parse_distance("1500m final") == 1500.0
    assert parse_distance("26.2") == 42195.0
    assert parse_distance("a race with no distance") is None


def test_finish_time_prefers_result_over_pace():
    # Place, name, finish, pace -- pace must not win.
    assert parse_finish_time("12  Jane Doe  1:34:22  7:12") == 5662.0


def test_finish_time_prefers_chip_over_gun():
    # Gun 3:25:10, chip 3:24:55: close together, so the smaller is the chip.
    assert parse_finish_time("Jane Doe  3:25:10  3:24:55") == 12295.0


def test_finish_time_mmss():
    assert parse_finish_time("Turkey Trot 5K  19:42") == 1182.0


def test_dates():
    assert parse_date("2024-04-15") == date(2024, 4, 15)
    assert parse_date("Apr 15, 2024") == date(2024, 4, 15)
    assert parse_date("4/15/2024") == date(2024, 4, 15)
    assert parse_date("ran this back in 2019") == date(2019, 7, 1)


def test_placing():
    assert parse_placing("finished 42/1300") == (42, 1300)
    assert parse_placing("came 7th overall") == (7, None)
    assert parse_placing("no placing here") == (None, None)


def test_paste_from_a_results_table():
    blob = """
    Place  Name           Bib   Age  Gun      Chip     Pace
    12     Jane Doe       841   34   1:34:40  1:34:22  7:12
    88     Ann Roe        112   29   1:47:03  1:46:58  8:10
    """
    # No distance on the rows, and none in context, so nothing is claimed.
    assert parse_pasted_results(blob) == []

    with_heading = "Springfield Half Marathon - April 15, 2024\n" + blob
    races = parse_pasted_results(with_heading)
    assert len(races) == 2
    assert races[0].distance_m == 21097.5
    assert races[0].finish_time_s == 5662.0
    assert races[0].race_date == date(2024, 4, 15)


def test_paste_one_race_per_line():
    blob = """
    Boston Marathon        2024-04-15   3:12:44
    Turkey Trot 5K         2023-11-23   19:42
    Local Half Marathon    2023-05-07   1:29:03
    """
    races = parse_pasted_results(blob)
    assert [r.distance_m for r in races] == [42195.0, 5000.0, 21097.5]
    assert [r.finish_time_s for r in races] == [11564.0, 1182.0, 5343.0]
    assert races[0].name == "Boston Marathon"


def test_csv_roundtrip():
    blob = (
        "Race,Date,Distance,Chip Time,Overall Place\n"
        "Boston Marathon,2024-04-15,Marathon,3:12:44,1204/25000\n"
        "Turkey Trot,2023-11-23,5K,19:42,7/850\n"
    )
    races = parse_any(blob)
    assert len(races) == 2
    assert races[0].distance_m == 42195.0
    assert races[0].place_overall == 1204
    assert races[0].field_size == 25000
    assert races[1].finish_time_s == 1182.0
