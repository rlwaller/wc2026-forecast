"""Load Silver Bulletin CSV exports into clean pandas dataframes.

Files expected in the same directory:
  - data-dxUJw.csv  : per-team PELE & Tilt ratings
  - data-DcqkH.csv  : per-team round-robin GF/GA/W/L/D
  - data-3bTOr.csv  : per-match forecasts (Win%, xG, D%, modal)
  - data-4bcIB.csv  : team metadata (confederation, FIFA rank)
  - data-4oVop.csv  : historical PELE quarters (not used for calibration)
"""

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_team_ratings():
    """Per-team PELE, Tactical Tilt, Personnel Tilt, Total Tilt."""
    df = pd.read_csv(DATA_DIR / "data-dxUJw.csv")
    df = df.rename(columns={
        "Code": "code",
        "PELE": "pele",
        "Tactical": "tilt_tactical",
        "Personnel": "tilt_personnel",
        "total_tilt": "tilt",
    })
    return df[["code", "pele", "tilt_tactical", "tilt_personnel", "tilt"]]


def load_round_robin():
    """Per-team round-robin: avg GF, GA, W%, L%, D% vs all 210 opponents."""
    df = pd.read_csv(DATA_DIR / "data-DcqkH.csv")
    df = df.rename(columns={
        "Team": "code",
        "PELE": "pele",
        "GF": "rr_gf",
        "GA": "rr_ga",
        "W": "rr_w_pct",
        "L": "rr_l_pct",
        "D": "rr_d_pct",
    })
    return df


def load_metadata():
    """Confederation and FIFA rank."""
    df = pd.read_csv(DATA_DIR / "data-4bcIB.csv")
    df = df.rename(columns={
        "code": "code",
        "country": "country",
        "confederation": "confederation",
        "FIFA": "fifa_rank",
        "PELE": "pele_rank",
    })
    return df[["code", "country", "confederation", "fifa_rank", "pele_rank"]]


# Match the team-token format ":xx: ABC" with optional trailing 🏡 home flag
TEAM_TOKEN_RE = re.compile(
    r"^:[a-z\-]+:\s*([A-Z]{2,3})\s*(🏡)?\s*$"
)


def _parse_team_cell(cell):
    """Extract (code, is_home) from a team cell.

    Examples:
        ":mx: MEX 🏡"      -> ("MEX", True)
        ":za: RSA"         -> ("RSA", False)
        ":gb-sct: SCO 🏡"  -> ("SCO", True)
    """
    cell = str(cell).strip()
    m = TEAM_TOKEN_RE.match(cell)
    if not m:
        # fallback: take last alphabetic token of length 2-3
        toks = re.findall(r"[A-Z]{2,3}", cell)
        return (toks[-1] if toks else cell, "🏡" in cell)
    return m.group(1), bool(m.group(2))


def load_matches():
    """Per-match forecasts.

    The CSV has duplicate column names ('Win', 'GF') for the two teams.
    pandas reads the second occurrences as 'Win.1', 'GF.1'.
    """
    # Skip the BOM-prefixed header by reading raw and renaming.
    df = pd.read_csv(DATA_DIR / "data-3bTOr.csv")
    cols = list(df.columns)
    # Expected: Date, Team, Win, GF, Opponent, Win.1, GF.1, Draw, modal_score, comp_tier, notes
    df.columns = [
        "date_raw", "team_a_raw", "win_a", "gf_a",
        "team_b_raw", "win_b", "gf_b",
        "draw", "modal_score", "comp_tier", "notes",
    ]

    # Parse team codes and home flags
    parsed_a = df["team_a_raw"].apply(_parse_team_cell)
    parsed_b = df["team_b_raw"].apply(_parse_team_cell)
    df["team_a"] = parsed_a.apply(lambda t: t[0])
    df["home_a"] = parsed_a.apply(lambda t: t[1])
    df["team_b"] = parsed_b.apply(lambda t: t[0])
    df["home_b"] = parsed_b.apply(lambda t: t[1])

    # Probabilities are given as percentages
    df["p_win_a"] = df["win_a"] / 100.0
    df["p_win_b"] = df["win_b"] / 100.0
    df["p_draw"] = df["draw"] / 100.0

    # Sanity: probabilities should sum to ~1
    df["prob_sum"] = df["p_win_a"] + df["p_win_b"] + df["p_draw"]

    # Date — strip the trophy/index suffix
    df["date"] = df["date_raw"].str.split("@@").str[0].str.strip()

    keep = [
        "date", "team_a", "team_b", "home_a", "home_b",
        "gf_a", "gf_b", "p_win_a", "p_win_b", "p_draw",
        "modal_score", "comp_tier", "notes", "prob_sum",
    ]
    return df[keep]


if __name__ == "__main__":
    teams = load_team_ratings()
    rr = load_round_robin()
    meta = load_metadata()
    matches = load_matches()

    print("=== Team ratings ===")
    print(teams.head())
    print(f"  rows: {len(teams)}")
    print()
    print("=== Round-robin ===")
    print(rr.head())
    print(f"  rows: {len(rr)}")
    print()
    print("=== Metadata ===")
    print(meta.head())
    print(f"  rows: {len(meta)}")
    print()
    print("=== Matches ===")
    print(matches.head(8).to_string())
    print(f"  rows: {len(matches)}")
    print()
    print("=== Match counts by comp_tier ===")
    print(matches["comp_tier"].value_counts().sort_index())
    print()
    print("=== Probability sum sanity (should be ~1.0) ===")
    print(matches["prob_sum"].describe())
    print()
    print("=== Home flag usage ===")
    print(f"  matches with team_a home: {matches['home_a'].sum()}")
    print(f"  matches with team_b home: {matches['home_b'].sum()}")
    print(f"  matches with neither home (neutral): {((~matches['home_a']) & (~matches['home_b'])).sum()}")
