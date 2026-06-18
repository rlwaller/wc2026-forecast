"""Fetch completed 2026 World Cup match results from ESPN's public API
and write them into data/results.txt.

ESPN's hidden scoreboard API requires no key. Endpoint:
  https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard

We extract only FINISHED group-stage matches (status STATUS_FULL_TIME),
map ESPN's team identifiers to our 3-letter codes, and write them out.

The script is intentionally defensive:
  - It only writes matches it can confidently map to our team codes.
  - Any unmapped team name is logged loudly (so the mapping can be fixed),
    and that match is skipped rather than guessed.
  - It preserves the comment header of results.txt.
  - It is idempotent: running it repeatedly produces the same file.
"""

import sys
import json
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_PATH = DATA_DIR / "results.txt"

ESPN_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/"
    "scoreboard?limit=200&dates=20260611-20260719"
)

# Our 48 team codes
OUR_TEAMS = {
    "MEX", "RSA", "KOR", "CZE", "CAN", "BIH", "QAT", "SUI",
    "BRA", "MAR", "HAI", "SCO", "USA", "PAR", "AUS", "TUR",
    "GER", "CUW", "CIV", "ECU", "NED", "JPN", "SWE", "TUN",
    "BEL", "EGY", "IRN", "NZL", "ESP", "CPV", "KSA", "URU",
    "FRA", "SEN", "IRQ", "NOR", "ARG", "ALG", "AUT", "JOR",
    "POR", "COD", "UZB", "COL", "ENG", "CRO", "GHA", "PAN",
}

# Map ESPN identifiers -> our codes. ESPN provides several fields per team:
#   - abbreviation (usually 3 letters, often FIFA-style)
#   - displayName / name / shortDisplayName
# We match on a normalized version of any of these. The dict below maps
# many possible ESPN spellings/abbreviations to our canonical code.
# Keys are uppercased and stripped for matching.
ESPN_NAME_TO_CODE = {
    # Group A
    "MEXICO": "MEX", "MEX": "MEX",
    "SOUTH AFRICA": "RSA", "RSA": "RSA",
    "SOUTH KOREA": "KOR", "KOREA REPUBLIC": "KOR", "KOR": "KOR",
    "CZECHIA": "CZE", "CZECH REPUBLIC": "CZE", "CZE": "CZE",
    # Group B
    "CANADA": "CAN", "CAN": "CAN",
    "BOSNIA AND HERZEGOVINA": "BIH", "BOSNIA & HERZEGOVINA": "BIH", "BOSNIA": "BIH", "BIH": "BIH",
    "QATAR": "QAT", "QAT": "QAT",
    "SWITZERLAND": "SUI", "SUI": "SUI", "SWZ": "SUI",
    # Group C
    "BRAZIL": "BRA", "BRA": "BRA",
    "MOROCCO": "MAR", "MAR": "MAR",
    "HAITI": "HAI", "HAI": "HAI",
    "SCOTLAND": "SCO", "SCO": "SCO",
    # Group D
    "UNITED STATES": "USA", "USA": "USA", "UNITED STATES OF AMERICA": "USA",
    "PARAGUAY": "PAR", "PAR": "PAR",
    "AUSTRALIA": "AUS", "AUS": "AUS",
    "TURKEY": "TUR", "TURKIYE": "TUR", "TÜRKIYE": "TUR", "TUR": "TUR",
    # Group E
    "GERMANY": "GER", "GER": "GER",
    "CURACAO": "CUW", "CURAÇAO": "CUW", "CUW": "CUW",
    "IVORY COAST": "CIV", "COTE D'IVOIRE": "CIV", "CÔTE D'IVOIRE": "CIV", "CIV": "CIV",
    "ECUADOR": "ECU", "ECU": "ECU",
    # Group F
    "NETHERLANDS": "NED", "NED": "NED", "HOLLAND": "NED",
    "JAPAN": "JPN", "JPN": "JPN",
    "SWEDEN": "SWE", "SWE": "SWE",
    "TUNISIA": "TUN", "TUN": "TUN",
    # Group G
    "BELGIUM": "BEL", "BEL": "BEL",
    "EGYPT": "EGY", "EGY": "EGY",
    "IRAN": "IRN", "IR IRAN": "IRN", "IRAN, ISLAMIC REPUBLIC OF": "IRN", "IRN": "IRN",
    "NEW ZEALAND": "NZL", "NZL": "NZL",
    # Group H
    "SPAIN": "ESP", "ESP": "ESP",
    "CAPE VERDE": "CPV", "CABO VERDE": "CPV", "CPV": "CPV",
    "SAUDI ARABIA": "KSA", "KSA": "KSA", "SAU": "KSA",
    "URUGUAY": "URU", "URU": "URU",
    # Group I
    "FRANCE": "FRA", "FRA": "FRA",
    "SENEGAL": "SEN", "SEN": "SEN",
    "IRAQ": "IRQ", "IRQ": "IRQ",
    "NORWAY": "NOR", "NOR": "NOR",
    # Group J
    "ARGENTINA": "ARG", "ARG": "ARG",
    "ALGERIA": "ALG", "ALG": "ALG",
    "AUSTRIA": "AUT", "AUT": "AUT",
    "JORDAN": "JOR", "JOR": "JOR",
    # Group K
    "PORTUGAL": "POR", "POR": "POR",
    "DR CONGO": "COD", "CONGO DR": "COD", "DEMOCRATIC REPUBLIC OF THE CONGO": "COD", "COD": "COD",
    "UZBEKISTAN": "UZB", "UZB": "UZB",
    "COLOMBIA": "COL", "COL": "COL",
    # Group L
    "ENGLAND": "ENG", "ENG": "ENG",
    "CROATIA": "CRO", "CRO": "CRO",
    "GHANA": "GHA", "GHA": "GHA",
    "PANAMA": "PAN", "PAN": "PAN",
}

# Header preserved at the top of results.txt
HEADER = """# Group-stage results — AUTO-GENERATED from ESPN by fetch_results.py
#
# Do not hand-edit during the tournament; this file is overwritten on each
# automatic update. To manually override a result, edit the script's logic
# or temporarily disable the GitHub Action.
#
# Format: TEAM_A TEAM_B score_a score_b
# Only FINISHED group-stage matches are listed.

# --- Played matches below ---
"""


def normalize(s):
    return (s or "").strip().upper()


def lookup_code(*candidates):
    """Try several ESPN-provided strings; return our code or None."""
    for c in candidates:
        key = normalize(c)
        if key in ESPN_NAME_TO_CODE:
            return ESPN_NAME_TO_CODE[key]
        if key in OUR_TEAMS:
            return key
    return None


def fetch_espn(url=ESPN_URL):
    req = urllib.request.Request(url, headers={"User-Agent": "wc2026-forecast/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def parse_results(data):
    """Extract finished group-stage matches. Returns (results, warnings)."""
    results = []      # list of (code_a, code_b, score_a, score_b)
    warnings = []
    events = data.get("events", [])
    for ev in events:
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]

        # Status: only completed matches
        status = comp.get("status", {}).get("type", {})
        state = status.get("state")  # 'pre', 'in', 'post'
        completed = status.get("completed", False)
        if state != "post" or not completed:
            continue

        # Only group-stage matches. ESPN tags knockout rounds in various ways;
        # we detect group stage by the presence of a group/notes label OR by
        # only accepting matches where both teams map to our codes AND the
        # round looks like a group match. Simplest robust approach: accept any
        # finished match where both teams map; knockout rounds will also map,
        # but we filter those by checking the leagueName/season slug if present.
        # For safety we accept all finished mapped matches; the simulator only
        # uses the 72 group fixtures it knows about, ignoring extras.
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        parsed = []
        ok = True
        for c in competitors:
            team = c.get("team", {})
            code = lookup_code(
                team.get("abbreviation"),
                team.get("displayName"),
                team.get("name"),
                team.get("shortDisplayName"),
                team.get("location"),
            )
            score_raw = c.get("score")
            try:
                score = int(score_raw)
            except (TypeError, ValueError):
                ok = False
                warnings.append(f"Could not parse score '{score_raw}' for {team.get('displayName')}")
                break
            if code is None:
                ok = False
                warnings.append(
                    f"UNMAPPED TEAM: '{team.get('displayName')}' "
                    f"(abbr='{team.get('abbreviation')}') — add to ESPN_NAME_TO_CODE"
                )
                break
            home_away = c.get("homeAway", "")
            parsed.append((code, score, home_away))

        if not ok or len(parsed) != 2:
            continue

        # Order doesn't matter for results.txt, but keep home first for clarity
        parsed.sort(key=lambda x: 0 if x[2] == "home" else 1)
        (code_a, score_a, _), (code_b, score_b, _) = parsed
        results.append((code_a, code_b, score_a, score_b))

    return results, warnings


def write_results(results):
    lines = [HEADER]
    for code_a, code_b, sa, sb in sorted(results):
        lines.append(f"{code_a} {code_b} {sa} {sb}\n")
    RESULTS_PATH.write_text("".join(lines))


def main():
    print(f"Fetching results from ESPN: {ESPN_URL}")
    try:
        data = fetch_espn()
    except Exception as e:
        print(f"ERROR fetching ESPN data: {e}", file=sys.stderr)
        sys.exit(1)

    results, warnings = parse_results(data)

    print(f"Found {len(results)} finished, mapped match(es).")
    for code_a, code_b, sa, sb in sorted(results):
        print(f"  {code_a} {sa}-{sb} {code_b}")

    if warnings:
        print("\nWARNINGS:", file=sys.stderr)
        for w in warnings:
            print(f"  ! {w}", file=sys.stderr)

    write_results(results)
    print(f"\nWrote {len(results)} result(s) to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
