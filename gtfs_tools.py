"""Look up Centro bus schedule data from a GTFS folder.

GTFS is just a folder of CSV-style .txt files. We load the ones we need
with pandas and answer simple questions from them.

To switch to a newer schedule, replace the .txt files in the data/ folder
(or change DATA_DIR). Nothing else needs to change.
"""
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TZ = ZoneInfo("America/New_York")
LOOKAHEAD_DAYS = 14  # how far ahead we search for the next day with service

_cache = {}


def _load():
    """Read the files once and keep them in memory."""
    if not _cache:
        read = lambda name: pd.read_csv(
            os.path.join(DATA_DIR, name), dtype=str, keep_default_na=False
        )
        _cache["stops"] = read("stops.txt")
        _cache["routes"] = read("routes.txt")
        _cache["trips"] = read("trips.txt")
        _cache["stop_times"] = read("stop_times.txt")
        _cache["calendar"] = read("calendar.txt")
        _cache["calendar_dates"] = read("calendar_dates.txt")
        _cache["feed_info"] = read("feed_info.txt")
    return _cache


def feed_coverage():
    """Which dates does this schedule file cover?"""
    info = _load()["feed_info"].iloc[0]
    return {
        "start": info["feed_start_date"],  # YYYYMMDD
        "end": info["feed_end_date"],
        "version": info["feed_version"],
    }


def _to_seconds(hms):
    """'6:08:09' -> seconds. GTFS allows 25:10:00 for after-midnight trips."""
    h, m, s = (int(p) for p in hms.split(":"))
    return h * 3600 + m * 60 + s


def _fmt12(hms):
    """'14:05:00' -> '2:05 PM'. Also handles after-midnight times like '25:10:00'."""
    h, m, _ = (int(p) for p in hms.strip().split(":"))
    h24 = h % 24
    suffix = "AM" if h24 < 12 else "PM"
    return f"{h24 % 12 or 12}:{m:02d} {suffix}"


def _now(date=None, time=None):
    """Use the real current time unless a test date/time is given."""
    now = datetime.now(TZ)
    if date:
        d = datetime.strptime(date, "%Y-%m-%d")
        now = now.replace(year=d.year, month=d.month, day=d.day)
    if time:
        t = datetime.strptime(time, "%H:%M")
        now = now.replace(hour=t.hour, minute=t.minute, second=0)
    return now


def _label(day):
    """'Monday 2026-11-23'"""
    return day.strftime("%A %Y-%m-%d")


def _in_feed(day):
    cov = feed_coverage()
    return cov["start"] <= day.strftime("%Y%m%d") <= cov["end"]


def _active_service_ids(day):
    """Which service_ids run on this calendar date?"""
    data = _load()
    ymd = day.strftime("%Y%m%d")
    weekday = day.strftime("%A").lower()
    cal = data["calendar"]
    running = set(
        cal[
            (cal["start_date"] <= ymd)
            & (cal["end_date"] >= ymd)
            & (cal[weekday] == "1")
        ]["service_id"]
    )
    exc = data["calendar_dates"]
    exc = exc[exc["date"] == ymd]
    running |= set(exc[exc["exception_type"] == "1"]["service_id"])  # added
    running -= set(exc[exc["exception_type"] == "2"]["service_id"])  # removed
    return running


def _stop_table(stop_id):
    """Every scheduled stop at this stop, with route, trip, and service info."""
    data = _load()
    st = data["stop_times"]
    st = st[st["stop_id"] == str(stop_id)]
    df = st.merge(data["trips"], on="trip_id").merge(data["routes"], on="route_id")
    df = df.copy()
    df["secs"] = df["departure_time"].map(_to_seconds)
    return df


def _departures_on(df, day, after_secs=0, limit=5, latest=False):
    """Departures on one date. latest=True returns the LAST ones of the day."""
    rows = df[df["service_id"].isin(_active_service_ids(day)) & (df["secs"] >= after_secs)]
    rows = rows.sort_values("secs")
    rows = rows.tail(limit) if latest else rows.head(limit)
    return [
        {
            "route": r.route_short_name,
            "trip_label": r.trip_headsign,  # Centro's own text, e.g. "40 Syracuse University to Hub"
            "time": _fmt12(r.departure_time)
            + (" (after midnight)" if r.secs >= 86400 else ""),
        }
        for r in rows.itertuples()
    ]


def find_stops(name, limit=15):
    """Search stops by part of their name, e.g. 'Marshall' or 'Destiny'."""
    stops = _load()["stops"]
    hits = stops[stops["stop_name"].str.contains(name, case=False, na=False, regex=False)]
    return [
        {"stop_id": r.stop_id, "stop_name": r.stop_name}
        for r in hits.head(limit).itertuples()
    ]


def routes_at_stop(stop_id, date=None):
    """Routes that serve this stop, and whether each one is actually running soon.

    A route can appear in the schedule file but have no service for weeks
    (for example a State Fair shuttle, or a route that skips Thanksgiving week).
    """
    now = _now(date)
    df = _stop_table(stop_id)
    days = [now + timedelta(days=i) for i in range(LOOKAHEAD_DAYS)]
    active = {d: _active_service_ids(d) for d in days if _in_feed(d)}
    out = []
    for (short, long_), g in df.groupby(["route_short_name", "route_long_name"]):
        sids = set(g["service_id"])
        running = [d for d, ids in active.items() if sids & ids]
        out.append(
            {
                "route": short,
                "name": long_,
                f"days_with_service_next_{LOOKAHEAD_DAYS}": len(running),
                "next_service_date": _label(running[0]) if running else None,
            }
        )
    return out


def next_departures(stop_id, limit=5, date=None, time=None, latest=False):
    """Scheduled departures at a stop (scheduled, NOT live).

    date  "YYYY-MM-DD" (default today)    time "HH:MM" 24-hour (default now)
    latest=True gives the last departures of that day instead of the next ones.
    If nothing is left on that date, also reports the next day with service.
    """
    now = _now(date, time)
    cov = feed_coverage()
    if not _in_feed(now):
        return {
            "error": "The schedule file does not cover this date.",
            "date_asked": now.strftime("%Y-%m-%d"),
            "schedule_runs_from": cov["start"],
            "schedule_runs_to": cov["end"],
        }

    df = _stop_table(stop_id)
    after = 0 if latest else now.hour * 3600 + now.minute * 60
    deps = _departures_on(df, now, after, limit, latest)
    result = {
        "stop_id": str(stop_id),
        "date": _label(now),
        "as_of": now.strftime("%I:%M %p").lstrip("0"),
        "source": "scheduled times from Centro schedule file (not live)",
        "departures": deps,
    }
    if latest:
        result.pop("as_of")  # the current clock time is irrelevant for "last bus" questions
    if not deps:
        result["note"] = "No scheduled departures left at this stop on this date."
        result["next_service"] = None
        for i in range(1, LOOKAHEAD_DAYS + 1):
            day = now + timedelta(days=i)
            if not _in_feed(day):
                break
            nxt = _departures_on(df, day, 0, limit)
            if nxt:
                result["next_service"] = {"date": _label(day), "departures": nxt}
                break
        if result["next_service"] is None:
            result["next_service_note"] = (
                f"No service at this stop in the next {LOOKAHEAD_DAYS} days."
            )
    return result


if __name__ == "__main__":
    # Quick self-test: py gtfs_tools.py
    print("Coverage:", feed_coverage())
    stop = find_stops("Marshall")[0]
    print("Stop:", stop)
    print("Routes:", routes_at_stop(stop["stop_id"]))
    print(next_departures(stop["stop_id"]))  # today
