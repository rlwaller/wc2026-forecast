"""Load per-team HFA values from data-uzR80.csv."""

from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def load_hfa():
    """Per-team home-field advantage in PELE points.

    'hfa' column is the total: hfa_base + team_hfa_home + altitude + distance.
    Used as a multiplicative HFA in PELE points when team plays at home.
    """
    df = pd.read_csv(DATA_DIR / "data-uzR80.csv", usecols=[
        "Code", "hfa", "hfa_base", "team_hfa_home", "alt_bonus", "dist_bonus",
    ])
    df = df.rename(columns={
        "Code": "code",
        "hfa": "hfa",
        "hfa_base": "hfa_base",
        "team_hfa_home": "hfa_team",
        "alt_bonus": "hfa_alt",
        "dist_bonus": "hfa_dist",
    })
    return df


if __name__ == "__main__":
    hfa = load_hfa()
    print(f"HFA records: {len(hfa)}")
    print()
    print("=== Distribution of HFA values ===")
    print(hfa["hfa"].describe())
    print()
    print("=== Top 10 by HFA ===")
    print(hfa.nlargest(10, "hfa").to_string(index=False))
    print()
    print("=== Bottom 10 ===")
    print(hfa.nsmallest(10, "hfa").to_string(index=False))
