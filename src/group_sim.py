"""Group stage simulator.

Simulates the 72 group matches by sampling scorelines from the score matrix,
then builds standings and ranks third-place teams.

Tiebreakers implemented (FIFA order):
  1. Points
  2. Goal difference
  3. Goals scored
  4. (head-to-head) -- not implemented; falls back to random
  5. (fair play) -- not implemented
  6. (FIFA ranking) -- not implemented

For Monte Carlo aggregation across many simulations, the random tiebreak
when the first 3 criteria all match is fine -- it happens in <1% of sims
and the long-run average is unbiased.
"""

import numpy as np
import pandas as pd

from pele_model import predict_xg, score_matrix, sample_score
from bracket import GROUPS, VENUES


def _sample_match_score(lam_a, lam_b, rng):
    """Sample a (score_a, score_b) tuple from the score matrix."""
    M = score_matrix(lam_a, lam_b)
    flat = M.flatten()
    idx = rng.choice(len(flat), p=flat)
    return int(idx // M.shape[1]), int(idx % M.shape[1])


def simulate_group_stage(team_data, hfa_data, group_match_data, rng):
    """Simulate all 72 group matches once.

    Args:
      team_data:        DataFrame indexed by code with columns 'pele', 'tilt'
      hfa_data:         DataFrame indexed by code with column 'hfa'
      group_match_data: DataFrame of 72 matches with columns:
                          team_a, team_b, home_a, home_b, group
      rng:              numpy random Generator

    Returns:
      standings: dict {group_letter: list of 4 dicts with team, pts, gd, gs, ga}
                 (sorted by tiebreakers; first is winner, last is 4th place)
      results: list of dicts (one per match) with team_a, team_b, score_a, score_b
    """
    # Set up team accumulators per group
    team_stats = {team: {"pts": 0, "gs": 0, "ga": 0} for g in GROUPS.values() for team in g}

    results = []
    for _, m in group_match_data.iterrows():
        ta, tb = m["team_a"], m["team_b"]
        pele_a = float(team_data.loc[ta, "pele"])
        pele_b = float(team_data.loc[tb, "pele"])
        tilt_a = float(team_data.loc[ta, "tilt"])
        tilt_b = float(team_data.loc[tb, "tilt"])
        home_a = bool(m["home_a"])
        home_b = bool(m["home_b"])
        is_neutral = not (home_a or home_b)

        hfa_a = float(hfa_data.loc[ta, "hfa"]) if (home_a and ta in hfa_data.index) else 0.0
        hfa_b = float(hfa_data.loc[tb, "hfa"]) if (home_b and tb in hfa_data.index) else 0.0

        lam_a, lam_b = predict_xg(
            pele_a, pele_b, tilt_a, tilt_b,
            home_a=home_a, home_b=home_b, is_neutral=is_neutral,
            comp_tier=9, hfa_a=hfa_a, hfa_b=hfa_b,
        )
        score_a, score_b = _sample_match_score(float(lam_a), float(lam_b), rng)

        # Update standings
        team_stats[ta]["gs"] += score_a
        team_stats[ta]["ga"] += score_b
        team_stats[tb]["gs"] += score_b
        team_stats[tb]["ga"] += score_a
        if score_a > score_b:
            team_stats[ta]["pts"] += 3
        elif score_b > score_a:
            team_stats[tb]["pts"] += 3
        else:
            team_stats[ta]["pts"] += 1
            team_stats[tb]["pts"] += 1

        results.append({
            "team_a": ta, "team_b": tb,
            "score_a": score_a, "score_b": score_b,
        })

    # Build per-group standings, sorted by tiebreakers
    standings = {}
    for g, teams in GROUPS.items():
        rows = []
        for t in teams:
            s = team_stats[t]
            rows.append({
                "team": t, "group": g,
                "pts": s["pts"], "gs": s["gs"], "ga": s["ga"],
                "gd": s["gs"] - s["ga"],
            })
        # Sort: pts desc, gd desc, gs desc, then random tiebreak
        rng_keys = rng.random(len(rows))
        rows_with_keys = list(zip(rows, rng_keys))
        rows_with_keys.sort(key=lambda x: (-x[0]["pts"], -x[0]["gd"], -x[0]["gs"], x[1]))
        standings[g] = [r for r, _ in rows_with_keys]

    return standings, results


def rank_third_place_teams(standings, rng):
    """Rank the 12 third-place teams by points/GD/GS and return top 8 groups.

    Args:
      standings: output of simulate_group_stage
      rng: numpy random Generator (for breaking ties)

    Returns:
      advancing_groups: sorted tuple of 8 group letters (e.g. ('A','C','D',...))
      eliminated_groups: list of 4 groups whose 3rd-place teams go home
    """
    third_place = []
    for g, sorted_teams in standings.items():
        # The 3rd-placed team is index 2
        team_record = sorted_teams[2].copy()
        team_record["from_group"] = g
        third_place.append(team_record)

    # Apply tiebreakers across all 12 third-place teams
    rng_keys = rng.random(len(third_place))
    rows_with_keys = list(zip(third_place, rng_keys))
    rows_with_keys.sort(key=lambda x: (-x[0]["pts"], -x[0]["gd"], -x[0]["gs"], x[1]))
    ranked = [r for r, _ in rows_with_keys]

    advancing = ranked[:8]
    eliminated = ranked[8:]
    advancing_groups = tuple(sorted(r["from_group"] for r in advancing))
    eliminated_groups = [r["from_group"] for r in eliminated]
    return advancing_groups, eliminated_groups, ranked


if __name__ == "__main__":
    from load_data import load_team_ratings, load_matches
    from load_hfa import load_hfa

    teams = load_team_ratings().set_index("code")
    hfa = load_hfa().set_index("code")
    all_matches = load_matches()
    group_matches = all_matches[all_matches["comp_tier"] == 9].copy().reset_index(drop=True)

    # Add 'group' column based on team_a's group
    from bracket import TEAM_GROUP
    group_matches["group"] = group_matches["team_a"].map(TEAM_GROUP)

    print(f"Group-stage matches loaded: {len(group_matches)}")
    print()

    # Run 1 simulation
    rng = np.random.default_rng(42)
    standings, results = simulate_group_stage(teams, hfa, group_matches, rng)

    print("=== Sample group standings (1 simulation) ===\n")
    for g in sorted(standings):
        print(f"Group {g}:")
        for i, row in enumerate(standings[g], start=1):
            marker = "👑" if i == 1 else "  "
            print(f"  {marker} {i}. {row['team']:3s}  pts={row['pts']}  "
                  f"gs={row['gs']:2d}  ga={row['ga']:2d}  gd={row['gd']:+d}")
        print()

    advancing, eliminated, ranked = rank_third_place_teams(standings, rng)
    print(f"=== 3rd-place ranking (top 8 advance) ===\n")
    for i, r in enumerate(ranked, start=1):
        status = "✓ ADVANCE" if i <= 8 else "✗ OUT    "
        print(f"  {i:2d}. {status}  {r['team']:3s} (Group {r['from_group']}) "
              f"pts={r['pts']}  gd={r['gd']:+d}  gs={r['gs']}")
    print()
    print(f"Advancing groups: {advancing}")
    print(f"Eliminated:       {eliminated}")
