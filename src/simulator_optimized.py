"""Optimized Monte Carlo simulator.

Strategy:
  1. Pre-compute the 72 group-stage score matrices (deterministic given inputs).
     Stack into shape (72, 121) and use vectorized searchsorted to sample all
     72 matches across all N sims at once.
  2. Resolve group standings vectorized.
  3. Rank 3rd-place teams.
  4. Look up Annex C scenarios.
  5. Knockout simulation: pairings vary across sims, so we batch by
     (team_a, team_b, venue) tuples and cache score matrices on demand.

Target: 10,000+ sims/sec for the full tournament including knockouts.
"""

import time
from collections import defaultdict
import numpy as np
import pandas as pd

from pele_model import predict_xg, score_matrix
from bracket import (
    GROUPS, TEAM_GROUP, VENUES,
    ROUND_OF_32, ROUND_OF_16, QUARTERFINALS, SEMIFINALS,
    THIRD_PLACE, FINAL, ANNEX_C_ROWS, ANNEX_C_COLUMNS_TO_MATCH,
)
from load_data import load_team_ratings, load_matches
from load_hfa import load_hfa


# Knockout match IDs by round
R32_MATCHES = [m[0] for m in ROUND_OF_32]
R16_MATCHES = [m[0] for m in ROUND_OF_16]
QF_MATCHES = [m[0] for m in QUARTERFINALS]
SF_MATCHES = [m[0] for m in SEMIFINALS]

# ET / PK parameters (matching knockout_sim.py)
ET_LAMBDA_SCALE = 0.30
PK_FAVORITE_CAP = 0.60
PK_PELE_PER_PCT = 200.0


def _annex_c_lookup_table():
    """Pre-build a dict mapping sorted-tuple-of-8-groups to slot dict.

    Returns:
      {frozenset of 8 group letters: dict {match_id: source_group}}
    """
    table = {}
    for advancing, slots in ANNEX_C_ROWS:
        key = frozenset(advancing)
        slot_dict = dict(zip(ANNEX_C_COLUMNS_TO_MATCH, slots))
        table[key] = slot_dict
    return table


ANNEX_C_TABLE = _annex_c_lookup_table()


class OptimizedSimulator:
    """Pre-builds the data structures needed for fast batch simulation."""

    def __init__(self, team_data, hfa_data, group_match_data, fixed_results=None):
        self.team_data = team_data
        self.hfa_data = hfa_data
        self.group_match_data = group_match_data
        self.all_teams = [t for g in GROUPS.values() for t in g]
        self.team_to_idx = {t: i for i, t in enumerate(self.all_teams)}

        # Pre-compute group-stage score matrices and CDFs
        self._build_group_cdfs()

        # Fixed (already-played) group results: {match_index: (score_a, score_b)}
        # built from the team-keyed dict passed in
        self.fixed_scores = self._resolve_fixed_results(fixed_results or {})

        # Cache for knockout score matrices: key = (lam_a_rounded, lam_b_rounded)
        # since we get many repeated matchups across sims
        self._knockout_cache = {}

        # Pre-extract team rating arrays for vectorized lookups
        self._build_team_arrays()

    def _resolve_fixed_results(self, fixed_results):
        """Map team-pair-keyed results onto group-match indices.

        fixed_results: {(team_a, team_b): (score_a, score_b)} in either orientation.
        Returns: {match_index: (score_a_oriented, score_b_oriented)} aligned to
        the simulator's internal team_a/team_b ordering for that match.
        """
        resolved = {}
        for i in range(len(self.group_team_a)):
            ta = self.group_team_a[i]
            tb = self.group_team_b[i]
            if (ta, tb) in fixed_results:
                resolved[i] = tuple(fixed_results[(ta, tb)])
            elif (tb, ta) in fixed_results:
                # Flip the score to match internal ordering
                sb, sa = fixed_results[(tb, ta)]
                resolved[i] = (sa, sb)
        return resolved

    def _build_group_cdfs(self):
        """Pre-compute CDF for each of the 72 group matches."""
        cdfs = []
        for _, m in self.group_match_data.iterrows():
            ta, tb = m["team_a"], m["team_b"]
            pele_a = float(self.team_data.loc[ta, "pele"])
            pele_b = float(self.team_data.loc[tb, "pele"])
            tilt_a = float(self.team_data.loc[ta, "tilt"])
            tilt_b = float(self.team_data.loc[tb, "tilt"])
            home_a = bool(m["home_a"])
            home_b = bool(m["home_b"])
            is_neutral = not (home_a or home_b)
            hfa_a = float(self.hfa_data.loc[ta, "hfa"]) if (home_a and ta in self.hfa_data.index) else 0.0
            hfa_b = float(self.hfa_data.loc[tb, "hfa"]) if (home_b and tb in self.hfa_data.index) else 0.0
            lam_a, lam_b = predict_xg(
                pele_a, pele_b, tilt_a, tilt_b,
                home_a=home_a, home_b=home_b, is_neutral=is_neutral,
                comp_tier=9, hfa_a=hfa_a, hfa_b=hfa_b,
            )
            M = score_matrix(float(lam_a), float(lam_b))
            cdfs.append(np.cumsum(M.flatten()))
        self.group_cdfs = np.array(cdfs)  # shape (72, 121)
        # Pre-store team identifiers per match
        self.group_team_a = self.group_match_data["team_a"].tolist()
        self.group_team_b = self.group_match_data["team_b"].tolist()
        self.group_group = [TEAM_GROUP[t] for t in self.group_team_a]

    def _build_team_arrays(self):
        """Build numpy arrays of team features for vectorized matchups."""
        T = len(self.all_teams)
        self.pele = np.array([float(self.team_data.loc[t, "pele"]) for t in self.all_teams])
        self.tilt = np.array([float(self.team_data.loc[t, "tilt"]) for t in self.all_teams])
        self.hfa = np.array([
            float(self.hfa_data.loc[t, "hfa"]) if t in self.hfa_data.index else 0.0
            for t in self.all_teams
        ])

    def _get_knockout_cdf(self, team_a, team_b, venue):
        """Get the CDF of the score matrix for a knockout match (with HFA)."""
        venue_country = VENUES.get(venue)
        home_a = (
            (venue_country == "USA" and team_a == "USA") or
            (venue_country == "MEX" and team_a == "MEX") or
            (venue_country == "CAN" and team_a == "CAN")
        )
        home_b = (
            (venue_country == "USA" and team_b == "USA") or
            (venue_country == "MEX" and team_b == "MEX") or
            (venue_country == "CAN" and team_b == "CAN")
        )
        is_neutral = not (home_a or home_b)
        key = (team_a, team_b, venue, home_a, home_b)
        if key in self._knockout_cache:
            return self._knockout_cache[key]

        idx_a = self.team_to_idx[team_a]
        idx_b = self.team_to_idx[team_b]
        hfa_a = self.hfa[idx_a] if home_a else 0.0
        hfa_b = self.hfa[idx_b] if home_b else 0.0

        lam_a, lam_b = predict_xg(
            self.pele[idx_a], self.pele[idx_b],
            self.tilt[idx_a], self.tilt[idx_b],
            home_a=home_a, home_b=home_b, is_neutral=is_neutral,
            comp_tier=9, hfa_a=hfa_a, hfa_b=hfa_b,
        )
        lam_a = float(lam_a)
        lam_b = float(lam_b)
        M = score_matrix(lam_a, lam_b)
        cdf = np.cumsum(M.flatten())

        # Also pre-compute ET CDF
        M_et = score_matrix(lam_a * ET_LAMBDA_SCALE, lam_b * ET_LAMBDA_SCALE)
        cdf_et = np.cumsum(M_et.flatten())

        # PK probability for A to win
        diff = self.pele[idx_a] - self.pele[idx_b]
        p_a_pk = 0.5 + (diff / PK_PELE_PER_PCT) * 0.05
        p_a_pk = max(1 - PK_FAVORITE_CAP, min(PK_FAVORITE_CAP, p_a_pk))

        result = (cdf, cdf_et, p_a_pk, lam_a, lam_b)
        self._knockout_cache[key] = result
        return result

    def _sample_match(self, team_a, team_b, venue, rng):
        """Sample one knockout match. Returns (winner, loser)."""
        cdf, cdf_et, p_a_pk, _, _ = self._get_knockout_cdf(team_a, team_b, venue)

        # Regulation
        u = rng.random()
        idx = np.searchsorted(cdf, u)
        score_a = idx // 11
        score_b = idx % 11
        if score_a != score_b:
            return (team_a, team_b) if score_a > score_b else (team_b, team_a)

        # ET
        u = rng.random()
        idx_et = np.searchsorted(cdf_et, u)
        et_a = idx_et // 11
        et_b = idx_et % 11
        if et_a != et_b:
            return (team_a, team_b) if et_a > et_b else (team_b, team_a)

        # PK
        if rng.random() < p_a_pk:
            return team_a, team_b
        return team_b, team_a

    def simulate_one(self, rng):
        """Simulate one full tournament. Returns (progression, match_results).

        progression: {team: deepest_round_label}
        match_results: {match_id: {"team_a": ..., "team_b": ..., "winner": ..., "loser": ...}}
        """
        # --- Group stage ---
        u = rng.random(72)
        idx = np.array([np.searchsorted(self.group_cdfs[i], u[i]) for i in range(72)])
        score_a = idx // 11
        score_b = idx % 11

        # Override sampled scores with real results for already-played matches
        if self.fixed_scores:
            score_a = score_a.copy()
            score_b = score_b.copy()
            for i, (sa, sb) in self.fixed_scores.items():
                score_a[i] = sa
                score_b[i] = sb

        # Build standings per group
        team_pts = {}
        team_gs = {}
        team_ga = {}
        for t in self.all_teams:
            team_pts[t] = 0
            team_gs[t] = 0
            team_ga[t] = 0
        for i in range(72):
            ta = self.group_team_a[i]
            tb = self.group_team_b[i]
            sa, sb = int(score_a[i]), int(score_b[i])
            team_gs[ta] += sa
            team_ga[ta] += sb
            team_gs[tb] += sb
            team_ga[tb] += sa
            if sa > sb:
                team_pts[ta] += 3
            elif sb > sa:
                team_pts[tb] += 3
            else:
                team_pts[ta] += 1
                team_pts[tb] += 1

        # Per-group standings
        standings = {}
        for g, teams in GROUPS.items():
            rows = []
            for t in teams:
                rows.append({
                    "team": t,
                    "pts": team_pts[t],
                    "gd": team_gs[t] - team_ga[t],
                    "gs": team_gs[t],
                })
            rng_keys = rng.random(4)
            rows_keyed = sorted(zip(rows, rng_keys),
                                 key=lambda x: (-x[0]["pts"], -x[0]["gd"], -x[0]["gs"], x[1]))
            standings[g] = [r for r, _ in rows_keyed]

        # --- 3rd-place ranking ---
        third_place = []
        for g, srt in standings.items():
            row = srt[2].copy()
            row["from_group"] = g
            third_place.append(row)
        rng_keys = rng.random(12)
        ranked = sorted(zip(third_place, rng_keys),
                        key=lambda x: (-x[0]["pts"], -x[0]["gd"], -x[0]["gs"], x[1]))
        advancing_groups = frozenset(r["from_group"] for r, _ in ranked[:8])
        eliminated_groups = [r["from_group"] for r, _ in ranked[8:]]

        # --- Annex C lookup ---
        annex_map = ANNEX_C_TABLE[advancing_groups]  # {match_id: source_group}

        # --- R32 and beyond ---
        group_winners = {g: s[0]["team"] for g, s in standings.items()}
        group_runners = {g: s[1]["team"] for g, s in standings.items()}
        group_thirds = {g: s[2]["team"] for g, s in standings.items()}

        # Build a dict of match_id -> 3rd-place team for R32 third-place slots
        third_for_match = {mid: group_thirds[src_g] for mid, src_g in annex_map.items()}

        match_results = {}
        progression = {}

        # Initialize: group-stage exits
        for g in GROUPS:
            if g in eliminated_groups:
                progression[group_thirds[g]] = "group"
            progression[standings[g][3]["team"]] = "group"

        def resolve(spec, match_id):
            kind = spec[0]
            if kind == "W":
                return group_winners[spec[1]]
            elif kind == "R":
                return group_runners[spec[1]]
            elif kind == "3":
                return third_for_match[match_id]
            elif kind == "M":
                return match_results[spec[1]]["winner"]
            elif kind == "ML":
                return match_results[spec[1]]["loser"]
            raise ValueError(f"Unknown spec: {spec}")

        # R32
        for mid, slot_a, slot_b, venue in ROUND_OF_32:
            ta = resolve(slot_a, mid)
            tb = resolve(slot_b, mid)
            winner, loser = self._sample_match(ta, tb, venue, rng)
            match_results[mid] = {"team_a": ta, "team_b": tb,
                                  "winner": winner, "loser": loser,
                                  "venue": venue}
            progression[ta] = "r32"
            progression[tb] = "r32"
            progression[winner] = "r16"

        # R16
        for mid, slot_a, slot_b, venue in ROUND_OF_16:
            ta = resolve(slot_a, mid)
            tb = resolve(slot_b, mid)
            winner, loser = self._sample_match(ta, tb, venue, rng)
            match_results[mid] = {"team_a": ta, "team_b": tb,
                                  "winner": winner, "loser": loser,
                                  "venue": venue}
            progression[winner] = "qf"

        # QF
        for mid, slot_a, slot_b, venue in QUARTERFINALS:
            ta = resolve(slot_a, mid)
            tb = resolve(slot_b, mid)
            winner, loser = self._sample_match(ta, tb, venue, rng)
            match_results[mid] = {"team_a": ta, "team_b": tb,
                                  "winner": winner, "loser": loser,
                                  "venue": venue}
            progression[winner] = "sf"

        # SF
        for mid, slot_a, slot_b, venue in SEMIFINALS:
            ta = resolve(slot_a, mid)
            tb = resolve(slot_b, mid)
            winner, loser = self._sample_match(ta, tb, venue, rng)
            match_results[mid] = {"team_a": ta, "team_b": tb,
                                  "winner": winner, "loser": loser,
                                  "venue": venue}
            progression[winner] = "final"

        # Third place
        mid, slot_a, slot_b, venue = THIRD_PLACE
        ta = resolve(slot_a, mid)
        tb = resolve(slot_b, mid)
        winner, loser = self._sample_match(ta, tb, venue, rng)
        match_results[mid] = {"team_a": ta, "team_b": tb,
                              "winner": winner, "loser": loser, "venue": venue}
        progression[winner] = "third"
        progression[loser] = "fourth"

        # Final
        mid, slot_a, slot_b, venue = FINAL
        ta = resolve(slot_a, mid)
        tb = resolve(slot_b, mid)
        winner, loser = self._sample_match(ta, tb, venue, rng)
        match_results[mid] = {"team_a": ta, "team_b": tb,
                              "winner": winner, "loser": loser, "venue": venue}
        progression[winner] = "champion"
        progression[loser] = "runner_up"

        return progression, match_results

    def run_monte_carlo(self, n_sims, seed=None, verbose=True):
        """Run N tournament simulations and aggregate results."""
        rng = np.random.default_rng(seed)

        # Counters
        slot_counts = defaultdict(lambda: defaultdict(int))
        reach_counts = {
            "reach_r32": defaultdict(int),
            "reach_r16": defaultdict(int),
            "reach_qf":  defaultdict(int),
            "reach_sf":  defaultdict(int),
            "reach_final": defaultdict(int),
            "p_third":   defaultdict(int),
            "p_runner_up": defaultdict(int),
            "p_champion": defaultdict(int),
        }

        start = time.time()
        progress_step = max(1, n_sims // 10)

        for i in range(n_sims):
            progression, match_results = self.simulate_one(rng)

            for mid, res in match_results.items():
                slot_counts[res["team_a"]][mid] += 1
                slot_counts[res["team_b"]][mid] += 1

            for team, deepest in progression.items():
                level = {"group": 0, "r32": 1, "r16": 2, "qf": 3, "sf": 4,
                         "fourth": 4, "third": 4, "runner_up": 5, "champion": 5}
                r = level.get(deepest, 0)
                if r >= 1: reach_counts["reach_r32"][team] += 1
                if r >= 2: reach_counts["reach_r16"][team] += 1
                if r >= 3: reach_counts["reach_qf"][team] += 1
                if r >= 4: reach_counts["reach_sf"][team] += 1
                if r >= 5: reach_counts["reach_final"][team] += 1
                if deepest == "third":     reach_counts["p_third"][team] += 1
                if deepest == "runner_up": reach_counts["p_runner_up"][team] += 1
                if deepest == "champion":  reach_counts["p_champion"][team] += 1

            if verbose and (i + 1) % progress_step == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (n_sims - (i + 1)) / rate
                print(f"  [{i+1:>6d}/{n_sims}]  {elapsed:.1f}s elapsed, "
                      f"~{eta:.1f}s remaining ({rate:.0f} sims/s)")

        # Build outputs
        rows = []
        for t in self.all_teams:
            rows.append({
                "team": t, "group": TEAM_GROUP[t],
                "pele": float(self.team_data.loc[t, "pele"]),
                "reach_r32":   reach_counts["reach_r32"][t]   / n_sims,
                "reach_r16":   reach_counts["reach_r16"][t]   / n_sims,
                "reach_qf":    reach_counts["reach_qf"][t]    / n_sims,
                "reach_sf":    reach_counts["reach_sf"][t]    / n_sims,
                "reach_final": reach_counts["reach_final"][t] / n_sims,
                "p_third":     reach_counts["p_third"][t]     / n_sims,
                "p_runner_up": reach_counts["p_runner_up"][t] / n_sims,
                "p_champion":  reach_counts["p_champion"][t]  / n_sims,
            })
        summary = pd.DataFrame(rows).sort_values("p_champion", ascending=False).reset_index(drop=True)

        slot_probs = {}
        for t in self.all_teams:
            slot_probs[t] = {mid: cnt / n_sims for mid, cnt in slot_counts[t].items()}

        elapsed = time.time() - start
        if verbose:
            print(f"\n  Completed {n_sims} sims in {elapsed:.1f}s ({n_sims/elapsed:.0f} sims/s)")
            print(f"  Knockout score-matrix cache size: {len(self._knockout_cache)}")

        return summary, slot_probs


if __name__ == "__main__":
    import sys
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 10000

    print("Loading data...")
    teams = load_team_ratings().set_index("code")
    hfa = load_hfa().set_index("code")
    all_matches = load_matches()
    group_matches = all_matches[all_matches["comp_tier"] == 9].copy().reset_index(drop=True)
    group_matches["group"] = group_matches["team_a"].map(TEAM_GROUP)

    print("Initializing optimized simulator...")
    sim = OptimizedSimulator(teams, hfa, group_matches)

    print(f"\nRunning {n_sims:,} simulations...\n")
    summary, slot_probs = sim.run_monte_carlo(n_sims, seed=42)

    print("\n=== Top 15 contenders ===\n")
    display = summary.copy()
    for c in ["reach_r32", "reach_r16", "reach_qf", "reach_sf", "reach_final", "p_champion"]:
        display[c] = (display[c] * 100).round(2)
    print(display.head(15)[["team", "group", "pele", "reach_r32", "reach_r16",
                              "reach_qf", "reach_sf", "reach_final", "p_champion"]].to_string(index=False))
