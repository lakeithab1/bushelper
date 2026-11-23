"""The assistant: an OpenAI model that can call our bus lookup functions.

ask()        -> grounded assistant (uses tools + notes.md)
ask_plain()  -> plain chatbot with no tools (the baseline to compare against)
"""
import json
import os

import gtfs_tools

# Change this to a model your OpenAI account can use (check OpenAI's model list).
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

NOTES_PATH = os.path.join(os.path.dirname(__file__), "notes.md")


def _notes():
    with open(NOTES_PATH, encoding="utf-8") as f:
        return f.read()


def system_prompt():
    cov = gtfs_tools.feed_coverage()
    today = gtfs_tools._now().strftime("%A %Y-%m-%d")
    return f"""You are a bus helper for new Syracuse University students who do not have a car and want to ride Centro buses.

Rules:
1. For routes, stops, and departure times, ALWAYS use the tools. Never answer from memory and never invent routes, stops, or times.
2. For fares, student ID rules, and disruptions, use ONLY the notes below. If the notes do not cover it, say you don't know and suggest checking centro.org or calling Centro.
3. The schedule data is scheduled times only, NOT live arrivals. Say so when giving times. The schedule file runs {cov['start']} to {cov['end']}.
4. If a tool returns an error or no results, say that plainly. Do not guess.
5. Say where each answer came from (schedule file or notes).
6. Each departure has a "trip_label" copied from Centro's schedule (for example "40 Syracuse University to Hub"). Quote it exactly. Never rephrase it, and never guess the bus's direction or final destination from it. If the label is unclear, say so.
7. When a stop search returns several stops, check all of them that are relevant. Say which stop each answer is about.
8. Today is {today}. For questions about another day, pass that date to next_departures (YYYY-MM-DD). For the first bus of a day use time "00:00". For the last bus of a day set latest to true.
9. routes_at_stop shows whether each route actually runs soon. If a route has 0 days of service in the next 14 days, say it is in the schedule but has no service in the next two weeks, and do not recommend it.
10. If next_departures returns no departures but has next_service, tell the user when service resumes. A time marked "after midnight" belongs to the previous day's schedule; say so.
11. Keep answers short and clear.

NOTES (verified information from Centro and SU):
{_notes()}
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_stops",
            "description": "Search bus stops by part of the stop name, e.g. 'Marshall' or 'Destiny'. Returns stop ids.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "routes_at_stop",
            "description": "List the bus routes that serve a stop id.",
            "parameters": {
                "type": "object",
                "properties": {"stop_id": {"type": "string"}},
                "required": ["stop_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "next_departures",
            "description": "Scheduled departures at a stop id (scheduled times, not live). Defaults to the next departures today. Use date/time for other days, and latest for the last buses of a day.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stop_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "date": {"type": "string", "description": "YYYY-MM-DD, default today"},
                    "time": {"type": "string", "description": "HH:MM in 24-hour time, default now"},
                    "latest": {"type": "boolean", "description": "true = the last departures of the day"},
                },
                "required": ["stop_id"],
            },
        },
    },
]


def run_tool(name, args, test_date=None, test_time=None):
    """Run one of our lookup functions. Returns something JSON-friendly."""
    if name == "find_stops":
        return gtfs_tools.find_stops(args["name"])
    if name == "routes_at_stop":
        return gtfs_tools.routes_at_stop(args["stop_id"], date=test_date)
    if name == "next_departures":
        return gtfs_tools.next_departures(
            args["stop_id"],
            limit=args.get("limit", 5),
            date=test_date or args.get("date"),
            time=test_time or args.get("time"),
            latest=args.get("latest", False),
        )
    return {"error": f"unknown tool {name}"}


def _client():
    from openai import OpenAI  # imported here so tests without the package still work

    return OpenAI()  # reads OPENAI_API_KEY from the environment


def ask(history, test_date=None, test_time=None, max_steps=6):
    """history = [{'role': 'user'|'assistant', 'content': '...'}, ...]

    Returns (answer_text, tool_log). tool_log lists every lookup the model made,
    which is useful to show sources and for your report.
    """
    client = _client()
    messages = [{"role": "system", "content": system_prompt()}] + history
    tool_log = []
    for _ in range(max_steps):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content, tool_log
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = run_tool(call.function.name, args, test_date, test_time)
            tool_log.append({"tool": call.function.name, "args": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )
    return "Sorry, I couldn't finish that lookup. Please try rephrasing.", tool_log


def ask_plain(question):
    """Baseline: the same model with NO tools and NO notes."""
    client = _client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content
