"""Locked PELE-style score-matrix and rating-to-xG models.

Parameters were calibrated against Silver Bulletin's published values:
  - Problem A (score matrix): phi=12.1, rho=-0.118 fit to 342 matches'
    W/D/L probabilities. RMS error ~0.5-0.9 pp in W/D/L.
  - Problem B (rating-to-xG): v5 with per-team HFA from Silver's atlas data.
    HFA split into offensive boost for home team + defensive suppression
    for away team. RMS error ~0.11-0.15 goals on World Cup matches.

End-to-end (rating -> xG -> W/D/L) RMS error: ~4 pp in win probability
on the 72 World Cup group-stage matches.
"""

import numpy as np
from scipy.stats import nbinom, poisson


# ============================================================
# Score matrix (Problem A) — locked parameters
# ============================================================

PHI = 12.1
RHO = -0.118
MAX_GOALS = 10


def _negbin_pmf(lam, phi, max_goals=MAX_GOALS):
    """P(X=k) for k=0..max_goals, X ~ NegBin(mean=lam, dispersion=phi).

    Variance is lam + lam^2/phi. As phi -> infinity, this collapses to Poisson.
    """
    if phi >= 1e6:
        return poisson.pmf(np.arange(max_goals + 1), lam)
    n = phi
    p = phi / (phi + lam)
    return nbinom.pmf(np.arange(max_goals + 1), n, p)


def score_matrix(lam_a, lam_b, phi=PHI, rho=RHO, max_goals=MAX_GOALS):
    """Joint probability matrix P(A=x, B=y) for x,y in 0..max_goals."""
    pa = _negbin_pmf(lam_a, phi, max_goals)
    pb = _negbin_pmf(lam_b, phi, max_goals)
    M = np.outer(pa, pb)

    # Dixon-Coles low-score correction
    M[0, 0] *= max(0.0, 1.0 - lam_a * lam_b * rho)
    M[0, 1] *= max(0.0, 1.0 + lam_a * rho)
    M[1, 0] *= max(0.0, 1.0 + lam_b * rho)
    M[1, 1] *= max(0.0, 1.0 - rho)

    s = M.sum()
    if s > 0:
        M /= s
    return M


def wdl_probabilities(M):
    """Return (P_win_A, P_draw, P_win_B) from a score matrix."""
    diag = np.trace(M)
    upper = np.triu(M, k=1).sum()  # B beats A
    lower = np.tril(M, k=-1).sum()  # A beats B
    return float(lower), float(diag), float(upper)


def sample_score(lam_a, lam_b, rng=None):
    """Draw a random scoreline (x, y) from the score matrix."""
    if rng is None:
        rng = np.random.default_rng()
    M = score_matrix(lam_a, lam_b)
    flat = M.flatten()
    idx = rng.choice(len(flat), p=flat)
    return int(idx // M.shape[1]), int(idx % M.shape[1])


# ============================================================
# Rating-to-xG (Problem B v5: split HFA) — locked parameters
# ============================================================

# Order:
#   c0, c_pele_gap_lin, c_pele_gap_sq,
#   c_tilt_sum, c_neutral,
#   c_pele_diff_lin, c_pele_diff_sq, c_tilt_diff,
#   c_hfa_off, c_hfa_def,
#   tier4_offset, tier6_offset, tier9_offset
PARAMS_B = np.array([
    2.490135,    # c0_baseline
    0.020453,    # c_pele_gap_lin
    0.069215,    # c_pele_gap_sq
    1.002712,    # c_tilt_sum
    0.255199,    # c_neutral
    0.657836,    # c_pele_diff_lin
    0.015934,    # c_pele_diff_sq
    0.058289,    # c_tilt_diff
    0.425496,    # c_hfa_off  (home team's offense boost per 100 PELE pts of HFA)
    0.185336,    # c_hfa_def  (away team's offense suppression per 100 pts)
    0.031862,    # tier4_offset
    -0.014015,   # tier6_offset
    -0.012058,   # tier9_offset
])


def predict_xg(pele_a, pele_b, tilt_a, tilt_b,
               *, home_a=False, home_b=False,
               is_neutral=True, comp_tier=9,
               hfa_a=0.0, hfa_b=0.0,
               params=PARAMS_B):
    """Predict (lam_a, lam_b) for a single match or vectors of matches.

    HFA is split: home team gets offensive boost, away team gets defensive
    suppression. The 2:1 split (offense:defense) is fitted, not assumed.

    Args:
      pele_a, pele_b:   PELE ratings (scalar or array)
      tilt_a, tilt_b:   Total Tilt values
      home_a, home_b:   home flags (mutually exclusive)
      is_neutral:       True for neutral-site, False otherwise
      comp_tier:        1=friendly, 4=regional, 6=continental, 9=World Cup
      hfa_a, hfa_b:     per-team HFA in PELE points. Used only when team is home.

    Returns (lam_a, lam_b), each scalar or array depending on inputs.
    """
    p = params
    pele_a = np.asarray(pele_a, dtype=float)
    pele_b = np.asarray(pele_b, dtype=float)
    tilt_a = np.asarray(tilt_a, dtype=float)
    tilt_b = np.asarray(tilt_b, dtype=float)
    home_a = np.asarray(home_a)
    home_b = np.asarray(home_b)
    is_neutral = np.asarray(is_neutral)
    comp_tier = np.asarray(comp_tier)
    hfa_a = np.asarray(hfa_a, dtype=float)
    hfa_b = np.asarray(hfa_b, dtype=float)

    p_diff = (pele_a - pele_b) / 100.0
    abs_p_diff = np.abs(p_diff)
    sign_p_diff = np.sign(p_diff)
    tilt_sum = tilt_a + tilt_b
    tilt_diff = tilt_a - tilt_b

    tier_offset = np.where(comp_tier == 4, p[10],
                  np.where(comp_tier == 6, p[11],
                  np.where(comp_tier == 9, p[12], 0.0)))

    total = (p[0]
             + p[1] * abs_p_diff
             + p[2] * abs_p_diff**2
             + p[3] * tilt_sum
             + p[4] * is_neutral.astype(float)
             + tier_offset)

    diff = (p[5] * p_diff
            + p[6] * sign_p_diff * p_diff**2
            + p[7] * tilt_diff)

    lam_a_base = (total + diff) / 2.0
    lam_b_base = (total - diff) / 2.0

    # HFA: scaled to per-100-PELE-point units
    home_a_f = home_a.astype(float)
    home_b_f = home_b.astype(float)
    hfa_pts_a = hfa_a / 100.0
    hfa_pts_b = hfa_b / 100.0

    lam_a = (lam_a_base
             + p[8] * hfa_pts_a * home_a_f      # offensive boost when A home
             - p[9] * hfa_pts_b * home_b_f)     # defensive suppression when B home
    lam_b = (lam_b_base
             + p[8] * hfa_pts_b * home_b_f
             - p[9] * hfa_pts_a * home_a_f)

    lam_a = np.maximum(lam_a, 0.05)
    lam_b = np.maximum(lam_b, 0.05)
    return lam_a, lam_b


# ============================================================
# Convenience: full match prediction
# ============================================================

def predict_match(team_a, team_b, *,
                  home_team=None,
                  comp_tier=9,
                  team_data,
                  hfa_data):
    """High-level helper for a single match.

    Args:
      team_a, team_b: 3-letter team codes
      home_team:      None for neutral, or the team code that's home
      team_data:      DataFrame indexed by code with columns 'pele', 'tilt'
      hfa_data:       DataFrame indexed by code with column 'hfa'

    Returns:
      dict with team_a, team_b, lam_a, lam_b, p_win_a, p_draw, p_win_b,
      score_matrix.
    """
    pele_a = float(team_data.loc[team_a, "pele"])
    pele_b = float(team_data.loc[team_b, "pele"])
    tilt_a = float(team_data.loc[team_a, "tilt"])
    tilt_b = float(team_data.loc[team_b, "tilt"])

    home_a = (home_team == team_a)
    home_b = (home_team == team_b)
    neutral = (home_team is None)

    hfa_a_val = (float(hfa_data.loc[team_a, "hfa"])
                 if home_a and team_a in hfa_data.index else 0.0)
    hfa_b_val = (float(hfa_data.loc[team_b, "hfa"])
                 if home_b and team_b in hfa_data.index else 0.0)

    lam_a, lam_b = predict_xg(
        pele_a, pele_b, tilt_a, tilt_b,
        home_a=home_a, home_b=home_b, is_neutral=neutral,
        comp_tier=comp_tier, hfa_a=hfa_a_val, hfa_b=hfa_b_val,
    )
    lam_a = float(lam_a)
    lam_b = float(lam_b)

    M = score_matrix(lam_a, lam_b)
    p_win_a, p_draw, p_win_b = wdl_probabilities(M)

    return {
        "team_a": team_a, "team_b": team_b,
        "lam_a": lam_a, "lam_b": lam_b,
        "p_win_a": p_win_a, "p_draw": p_draw, "p_win_b": p_win_b,
        "score_matrix": M,
    }


if __name__ == "__main__":
    # Smoke tests against Silver's published values
    from load_data import load_team_ratings
    from load_hfa import load_hfa
    teams = load_team_ratings().set_index("code")
    hfa = load_hfa().set_index("code")

    cases = [
        ("ARG", "JOR", None, "89.1% / 9.4% / 1.5%   (xG 3.3 - 0.3)"),
        ("MEX", "RSA", "MEX", "74.6% / 19.9% / 5.5%  (xG 2.1 - 0.4)"),
        ("USA", "PAR", "USA", "39.0% / 30.8% / 30.2% (xG 1.3 - 1.1)"),
        ("CAN", "BIH", "CAN", "65.8% / 22.8% / 11.5% (xG 2.0 - 0.7)"),
        ("ESP", "CPV", None, "89.8% / 8.9% / 1.4%   (xG 3.4 - 0.3)"),
        ("MEX", "CZE", "MEX", "68.6% / 22.3% / 9.1%  (xG 2.0 - 0.6)"),
    ]
    print("=== Smoke tests (vs. Silver Bulletin's published forecasts) ===")
    print()
    for ta, tb, host, silver in cases:
        host_str = f"{host} HOME" if host else "neutral"
        out = predict_match(ta, tb, home_team=host, comp_tier=9,
                            team_data=teams, hfa_data=hfa)
        wdl = (f"{out['p_win_a']*100:.1f}% / {out['p_draw']*100:.1f}% / "
               f"{out['p_win_b']*100:.1f}%")
        xg = f"(xG {out['lam_a']:.1f} - {out['lam_b']:.1f})"
        print(f"  {ta} vs {tb} ({host_str:>10}):")
        print(f"    Ours:    {wdl} {xg}")
        print(f"    Silver:  {silver}")
        print()
