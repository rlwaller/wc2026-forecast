"""Generate the forecast JSON consumed by the website.

This is the entry point for the daily GitHub Actions update.
Reads team data, runs N simulations, and writes a single forecast.json.
"""

import json
import time
import datetime
import sys
from pathlib import Path
import numpy as np

from bracket import (
    GROUPS, TEAM_GROUP, VENUES,
    ROUND_OF_32, ROUND_OF_16, QUARTERFINALS, SEMIFINALS,
    THIRD_PLACE, FINAL,
)
from simulator_optimized import OptimizedSimulator
from load_data import load_team_ratings, load_matches
from load_hfa import load_hfa


def slot_spec_to_dict(spec):
    """Convert a slot spec tuple to a dict for JSON serialization."""
    kind = spec[0]
    if kind == "W":
        return {"type": "group_winner", "group": spec[1]}
    elif kind == "R":
        return {"type": "group_runner_up", "group": spec[1]}
    elif kind == "3":
        return {"type": "third_place", "possible_groups": list(spec[1])}
    elif kind == "M":
        return {"type": "match_winner", "from_match": spec[1]}
    elif kind == "ML":
        return {"type": "match_loser", "from_match": spec[1]}
    return {"type": "unknown", "raw": str(spec)}


def build_forecast_json(summary, slot_probs, team_data, hfa_data, n_sims):
    """Build the complete forecast JSON structure."""
    # Knockout match metadata
    knockout_matches = []
    for matches, round_name in [
        (ROUND_OF_32, "R32"),
        (ROUND_OF_16, "R16"),
        (QUARTERFINALS, "QF"),
        (SEMIFINALS, "SF"),
        ([THIRD_PLACE], "3rd_place"),
        ([FINAL], "Final"),
    ]:
        for mid, slot_a, slot_b, venue in matches:
            knockout_matches.append({
                "match_id": mid,
                "round": round_name,
                "venue": venue,
                "country": VENUES.get(venue, "?"),
                "slot_a": slot_spec_to_dict(slot_a),
                "slot_b": slot_spec_to_dict(slot_b),
            })

    # Per-team summary
    team_summary = {}
    for _, r in summary.iterrows():
        t = r["team"]
        team_summary[t] = {
            "team": t,
            "group": r["group"],
            "pele": round(r["pele"], 1),
            "tilt": round(float(team_data.loc[t, "tilt"]), 3),
            "hfa": round(float(hfa_data.loc[t, "hfa"]) if t in hfa_data.index else 0, 0),
            "reach_r32":   round(r["reach_r32"], 4),
            "reach_r16":   round(r["reach_r16"], 4),
            "reach_qf":    round(r["reach_qf"], 4),
            "reach_sf":    round(r["reach_sf"], 4),
            "reach_final": round(r["reach_final"], 4),
            "p_third":     round(r["p_third"], 4),
            "p_runner_up": round(r["p_runner_up"], 4),
            "p_champion":  round(r["p_champion"], 4),
        }

    # Per-team match slot probabilities (only non-zero)
    team_slot_probs = {}
    for team, probs in slot_probs.items():
        team_slot_probs[team] = {
            str(mid): round(prob, 4)
            for mid, prob in probs.items() if prob >= 0.0001  # >= 0.01%
        }

    # Compute inverse: per-match team probabilities
    # match_team_probs[match_id][team] = prob team plays in that match
    match_team_probs = {}
    for team, slot_dict in slot_probs.items():
        for mid, prob in slot_dict.items():
            if prob < 0.0001:
                continue
            mid_str = str(mid)
            if mid_str not in match_team_probs:
                match_team_probs[mid_str] = {}
            match_team_probs[mid_str][team] = round(prob, 4)

    return {
        "metadata": {
            "n_simulations": n_sims,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": "PELE v5 (split-HFA) + Dixon-Coles NegBin",
            "format": "FIFA 2026 World Cup (48 teams, 12 groups of 4)",
        },
        "groups": {g: teams for g, teams in GROUPS.items()},
        "venues": VENUES,
        "knockout_matches": knockout_matches,
        "team_summary": team_summary,
        "team_slot_probabilities": team_slot_probs,
        "match_team_probabilities": match_team_probs,
    }


def main(n_sims=20000, output_path=None, seed=42):
    # Default output: write into ../docs/forecast.json (where the website reads from)
    if output_path is None:
        output_path = str(Path(__file__).parent.parent / "docs" / "forecast.json")
    print("=" * 60)
    print(f"WC2026 PELE Forecast Generator")
    print(f"  N simulations: {n_sims:,}")
    print(f"  Output:        {output_path}")
    print(f"  Seed:          {seed}")
    print("=" * 60)
    print()

    print("[1/3] Loading data...")
    teams = load_team_ratings().set_index("code")
    hfa = load_hfa().set_index("code")
    all_matches = load_matches()
    group_matches = all_matches[all_matches["comp_tier"] == 9].copy().reset_index(drop=True)
    group_matches["group"] = group_matches["team_a"].map(TEAM_GROUP)

    print(f"  Loaded {len(teams)} teams, {len(hfa)} HFA values, {len(group_matches)} group matches")
    print()

    print("[2/3] Running Monte Carlo simulation...")
    sim = OptimizedSimulator(teams, hfa, group_matches)
    summary, slot_probs = sim.run_monte_carlo(n_sims, seed=seed, verbose=True)
    print()

    print("[3/3] Building JSON output...")
    forecast = build_forecast_json(summary, slot_probs, teams, hfa, n_sims)
    with open(output_path, "w") as f:
        json.dump(forecast, f, separators=(",", ":"))  # compact
    file_size = Path(output_path).stat().st_size
    print(f"  Wrote {output_path} ({file_size/1024:.1f} KB)")
    print()

    # Print top 10
    print("=" * 60)
    print("=== TOP 10 CHAMPIONSHIP CONTENDERS ===")
    print("=" * 60)
    for i, r in summary.head(10).iterrows():
        print(f"  {i+1:>2}. {r['team']:>3}  ({r['group']})  "
              f"PELE {r['pele']:.0f}   "
              f"R16 {r['reach_r16']*100:>4.1f}%   "
              f"QF {r['reach_qf']*100:>4.1f}%   "
              f"SF {r['reach_sf']*100:>4.1f}%   "
              f"FIN {r['reach_final']*100:>4.1f}%   "
              f"WIN {r['p_champion']*100:>4.1f}%")
    print()

    return forecast


if __name__ == "__main__":
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    main(n_sims=n_sims)
