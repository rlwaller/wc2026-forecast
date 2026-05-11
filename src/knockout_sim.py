"""Knockout-stage simulator.

Given group standings and the advancing 3rd-place teams, slot teams into
the bracket via Annex C, then simulate every knockout match through to the
Final using:

  - regulation: sample a scoreline from the score matrix
  - extra time: if drawn, sample a smaller-lambda scoreline (~30 min worth of football)
  - penalty kicks: if still drawn, the favorite by PELE wins ~55% of the time
                   (capped at 60/40 per Silver's methodology)

Host nation gets HFA when playing in their own country at a known venue.
"""

import numpy as np

from pele_model import predict_xg, score_matrix
from bracket import (
    GROUPS, VENUES, TEAM_GROUP,
    ROUND_OF_32, ROUND_OF_16, QUARTERFINALS, SEMIFINALS,
    THIRD_PLACE, FINAL,
    lookup_annex_c, ANNEX_C_COLUMNS_TO_MATCH,
)


# Extra-time goal scaling: 30 min vs 90 min = 1/3, with mild reduction for
# the more cautious play that's typical in extra time
ET_LAMBDA_SCALE = 0.30

# Penalty shootout: favorite (by PELE) wins this fraction of the time
PK_FAVORITE_BASE = 0.55
PK_FAVORITE_CAP = 0.60   # never exceed this even for huge gaps
PK_PELE_PER_PCT = 200.0  # roughly 200 PELE pts gap = 5pp boost above 50/50


def _sample_score(lam_a, lam_b, rng):
    """Sample a scoreline from the score matrix."""
    M = score_matrix(lam_a, lam_b)
    flat = M.flatten()
    idx = rng.choice(len(flat), p=flat)
    return int(idx // M.shape[1]), int(idx % M.shape[1])


def _resolve_pk(pele_a, pele_b, rng):
    """Penalty-shootout resolver. Returns 'A' or 'B'."""
    diff = pele_a - pele_b
    p_a_wins = 0.5 + (diff / PK_PELE_PER_PCT) * 0.05
    # Cap at 60/40 either way
    p_a_wins = max(1 - PK_FAVORITE_CAP, min(PK_FAVORITE_CAP, p_a_wins))
    return "A" if rng.random() < p_a_wins else "B"


def _knockout_match(team_a, team_b, venue, team_data, hfa_data, rng):
    """Simulate one knockout match: regulation -> ET -> PK as needed.

    Returns (winner_code, score_a_reg, score_b_reg, decided_in)
    where decided_in is 'reg', 'ET', or 'PK'.
    """
    pele_a = float(team_data.loc[team_a, "pele"])
    pele_b = float(team_data.loc[team_b, "pele"])
    tilt_a = float(team_data.loc[team_a, "tilt"])
    tilt_b = float(team_data.loc[team_b, "tilt"])

    # Determine HFA: host nation playing in their own country?
    venue_country = VENUES.get(venue)
    home_a = home_b = False
    hfa_a_val = hfa_b_val = 0.0
    is_neutral = True

    if venue_country == "USA" and team_a == "USA":
        home_a, is_neutral = True, False
        hfa_a_val = float(hfa_data.loc["USA", "hfa"]) if "USA" in hfa_data.index else 0.0
    elif venue_country == "USA" and team_b == "USA":
        home_b, is_neutral = True, False
        hfa_b_val = float(hfa_data.loc["USA", "hfa"]) if "USA" in hfa_data.index else 0.0
    elif venue_country == "MEX" and team_a == "MEX":
        home_a, is_neutral = True, False
        hfa_a_val = float(hfa_data.loc["MEX", "hfa"]) if "MEX" in hfa_data.index else 0.0
    elif venue_country == "MEX" and team_b == "MEX":
        home_b, is_neutral = True, False
        hfa_b_val = float(hfa_data.loc["MEX", "hfa"]) if "MEX" in hfa_data.index else 0.0
    elif venue_country == "CAN" and team_a == "CAN":
        home_a, is_neutral = True, False
        hfa_a_val = float(hfa_data.loc["CAN", "hfa"]) if "CAN" in hfa_data.index else 0.0
    elif venue_country == "CAN" and team_b == "CAN":
        home_b, is_neutral = True, False
        hfa_b_val = float(hfa_data.loc["CAN", "hfa"]) if "CAN" in hfa_data.index else 0.0

    lam_a, lam_b = predict_xg(
        pele_a, pele_b, tilt_a, tilt_b,
        home_a=home_a, home_b=home_b, is_neutral=is_neutral,
        comp_tier=9, hfa_a=hfa_a_val, hfa_b=hfa_b_val,
    )
    lam_a = float(lam_a)
    lam_b = float(lam_b)

    # Regulation
    score_a_reg, score_b_reg = _sample_score(lam_a, lam_b, rng)
    if score_a_reg != score_b_reg:
        winner = team_a if score_a_reg > score_b_reg else team_b
        return winner, score_a_reg, score_b_reg, "reg"

    # Extra time
    et_lam_a = lam_a * ET_LAMBDA_SCALE
    et_lam_b = lam_b * ET_LAMBDA_SCALE
    et_a, et_b = _sample_score(et_lam_a, et_lam_b, rng)
    if et_a != et_b:
        winner = team_a if et_a > et_b else team_b
        # Total scoreline reported includes ET goals
        return winner, score_a_reg + et_a, score_b_reg + et_b, "ET"

    # Penalty shootout
    pk_winner = _resolve_pk(pele_a, pele_b, rng)
    winner = team_a if pk_winner == "A" else team_b
    return winner, score_a_reg + et_a, score_b_reg + et_b, "PK"


def _resolve_slot(spec, group_winners, group_runners, third_place_map, match_results):
    """Resolve a slot specification to a team code.

    spec: tuple like ("W", "A"), ("R", "A"), ("3", source_group),
          ("M", match_id), or ("ML", match_id).
    group_winners: {group_letter: team_code}
    group_runners: {group_letter: team_code}
    third_place_map: {match_id: team_code} (for R32 third-place slots)
    match_results: {match_id: {"winner": ..., "loser": ..., ...}}
    """
    kind = spec[0]
    if kind == "W":
        return group_winners[spec[1]]
    elif kind == "R":
        return group_runners[spec[1]]
    elif kind == "3":
        # spec[1] is the source group, looked up via Annex C
        return third_place_map[spec[1]]  # third_place_map keyed by source group
    elif kind == "M":
        return match_results[spec[1]]["winner"]
    elif kind == "ML":
        return match_results[spec[1]]["loser"]
    else:
        raise ValueError(f"Unknown slot spec: {spec}")


def simulate_knockout(standings, advancing_groups, team_data, hfa_data, rng):
    """Simulate the full knockout stage.

    Args:
      standings:        output of simulate_group_stage
      advancing_groups: tuple of 8 group letters whose 3rd-place team advances
      team_data, hfa_data: as for predict_xg
      rng:              numpy random Generator

    Returns:
      results: dict {match_id: {team_a, team_b, score_a, score_b, winner, loser, decided_in, venue}}
      progression: dict {team: deepest_round} where deepest_round is one of:
                   'group', 'r32', 'r16', 'qf', 'sf', 'final', 'champion', 'runner_up', 'third'
    """
    # Extract group winners, runners-up, third-place teams from standings
    group_winners = {g: s[0]["team"] for g, s in standings.items()}
    group_runners = {g: s[1]["team"] for g, s in standings.items()}
    group_thirds = {g: s[2]["team"] for g, s in standings.items()}

    # Look up Annex C mapping: which match's 3rd-place slot gets which group's 3rd-placer
    annex_c_map = lookup_annex_c(advancing_groups)
    # annex_c_map: {match_id: source_group}
    # Convert to {match_id: team} for easier resolution
    third_place_for_match = {mid: group_thirds[src_g] for mid, src_g in annex_c_map.items()}

    # Now also build a map keyed by source group, for slot resolution
    # The "3" slot specs in ROUND_OF_32 give a tuple of allowed source groups.
    # We need to figure out which one was actually selected for that slot.
    third_slot_team = {}  # {match_id: team}
    for match_id, slot_a, slot_b, _ in ROUND_OF_32:
        if slot_a[0] == "3":
            third_slot_team[match_id] = third_place_for_match[match_id]
        if slot_b[0] == "3":
            third_slot_team[match_id] = third_place_for_match[match_id]

    # Helper to resolve slots, with R32-specific handling for "3" slots
    def resolve(spec, match_id, match_results):
        kind = spec[0]
        if kind == "3":
            return third_slot_team[match_id]
        return _resolve_slot(spec, group_winners, group_runners, {}, match_results)

    # Track results and deepest round per team
    match_results = {}
    progression = {}

    # Initialize all 32 advancing teams to 'r32' as their deepest reach
    advancing_teams = set()
    for g in GROUPS:
        advancing_teams.add(group_winners[g])
        advancing_teams.add(group_runners[g])
    for g in advancing_groups:
        advancing_teams.add(group_thirds[g])

    for t in advancing_teams:
        progression[t] = "r32"

    # Group-stage exits
    eliminated_groups = [g for g in GROUPS if g not in advancing_groups]
    for g in eliminated_groups:
        progression[group_thirds[g]] = "group"
    # 4th-placed teams in every group
    for g, s in standings.items():
        progression[s[3]["team"]] = "group"

    # --- Round of 32 ---
    for match_id, slot_a, slot_b, venue in ROUND_OF_32:
        team_a = resolve(slot_a, match_id, match_results)
        team_b = resolve(slot_b, match_id, match_results)
        winner, sa, sb, decided = _knockout_match(team_a, team_b, venue,
                                                   team_data, hfa_data, rng)
        loser = team_b if winner == team_a else team_a
        match_results[match_id] = {
            "team_a": team_a, "team_b": team_b,
            "score_a": sa, "score_b": sb,
            "winner": winner, "loser": loser,
            "decided_in": decided, "venue": venue,
        }
        progression[winner] = "r16"  # advanced to R16

    # --- Round of 16 ---
    for match_id, slot_a, slot_b, venue in ROUND_OF_16:
        team_a = resolve(slot_a, match_id, match_results)
        team_b = resolve(slot_b, match_id, match_results)
        winner, sa, sb, decided = _knockout_match(team_a, team_b, venue,
                                                   team_data, hfa_data, rng)
        loser = team_b if winner == team_a else team_a
        match_results[match_id] = {
            "team_a": team_a, "team_b": team_b,
            "score_a": sa, "score_b": sb,
            "winner": winner, "loser": loser,
            "decided_in": decided, "venue": venue,
        }
        progression[winner] = "qf"

    # --- Quarterfinals ---
    for match_id, slot_a, slot_b, venue in QUARTERFINALS:
        team_a = resolve(slot_a, match_id, match_results)
        team_b = resolve(slot_b, match_id, match_results)
        winner, sa, sb, decided = _knockout_match(team_a, team_b, venue,
                                                   team_data, hfa_data, rng)
        loser = team_b if winner == team_a else team_a
        match_results[match_id] = {
            "team_a": team_a, "team_b": team_b,
            "score_a": sa, "score_b": sb,
            "winner": winner, "loser": loser,
            "decided_in": decided, "venue": venue,
        }
        progression[winner] = "sf"

    # --- Semifinals ---
    for match_id, slot_a, slot_b, venue in SEMIFINALS:
        team_a = resolve(slot_a, match_id, match_results)
        team_b = resolve(slot_b, match_id, match_results)
        winner, sa, sb, decided = _knockout_match(team_a, team_b, venue,
                                                   team_data, hfa_data, rng)
        loser = team_b if winner == team_a else team_a
        match_results[match_id] = {
            "team_a": team_a, "team_b": team_b,
            "score_a": sa, "score_b": sb,
            "winner": winner, "loser": loser,
            "decided_in": decided, "venue": venue,
        }
        progression[winner] = "final"

    # --- Third-place playoff ---
    match_id, slot_a, slot_b, venue = THIRD_PLACE
    team_a = resolve(slot_a, match_id, match_results)
    team_b = resolve(slot_b, match_id, match_results)
    winner, sa, sb, decided = _knockout_match(team_a, team_b, venue,
                                               team_data, hfa_data, rng)
    loser = team_b if winner == team_a else team_a
    match_results[match_id] = {
        "team_a": team_a, "team_b": team_b,
        "score_a": sa, "score_b": sb,
        "winner": winner, "loser": loser,
        "decided_in": decided, "venue": venue,
    }
    progression[winner] = "third"
    progression[loser] = "fourth"

    # --- Final ---
    match_id, slot_a, slot_b, venue = FINAL
    team_a = resolve(slot_a, match_id, match_results)
    team_b = resolve(slot_b, match_id, match_results)
    winner, sa, sb, decided = _knockout_match(team_a, team_b, venue,
                                               team_data, hfa_data, rng)
    loser = team_b if winner == team_a else team_a
    match_results[match_id] = {
        "team_a": team_a, "team_b": team_b,
        "score_a": sa, "score_b": sb,
        "winner": winner, "loser": loser,
        "decided_in": decided, "venue": venue,
    }
    progression[winner] = "champion"
    progression[loser] = "runner_up"

    return match_results, progression


if __name__ == "__main__":
    from load_data import load_team_ratings, load_matches
    from load_hfa import load_hfa
    from group_sim import simulate_group_stage, rank_third_place_teams

    teams = load_team_ratings().set_index("code")
    hfa = load_hfa().set_index("code")
    all_matches = load_matches()
    group_matches = all_matches[all_matches["comp_tier"] == 9].copy().reset_index(drop=True)
    group_matches["group"] = group_matches["team_a"].map(TEAM_GROUP)

    rng = np.random.default_rng(42)
    standings, _ = simulate_group_stage(teams, hfa, group_matches, rng)
    advancing, eliminated, ranked = rank_third_place_teams(standings, rng)

    print(f"Advancing 3rd-place groups: {advancing}\n")

    results, progression = simulate_knockout(standings, advancing, teams, hfa, rng)

    print("=== Knockout matches (1 simulation) ===\n")
    print(f"{'ID':>4}  {'Match':<14}  {'Score':<8}  {'Decided':<5}  {'Venue':<22}")
    for mid in sorted(results):
        r = results[mid]
        match_str = f"{r['team_a']} vs {r['team_b']}"
        score_str = f"{r['score_a']}-{r['score_b']}"
        print(f"{mid:>4}  {match_str:<14}  {score_str:<8}  {r['decided_in']:<5}  {r['venue']:<22}")
    print()

    print("=== Final placements ===\n")
    rounds_order = ["group", "r32", "r16", "qf", "sf", "fourth", "third",
                    "runner_up", "champion"]
    rounds_label = {
        "group":    "Group stage exit",
        "r32":      "Round of 32 exit",
        "r16":      "Round of 16 exit",
        "qf":       "Quarterfinal exit",
        "sf":       "Semifinal exit (4th place playoff)",
        "fourth":   "4th place",
        "third":    "3rd place",
        "runner_up": "Runner-up",
        "champion": "🏆 CHAMPION",
    }
    for r_key in reversed(rounds_order):
        teams_at_level = [t for t, p in progression.items() if p == r_key]
        if teams_at_level:
            print(f"{rounds_label[r_key]}: {sorted(teams_at_level)}")
