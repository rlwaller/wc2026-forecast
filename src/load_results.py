"""Load already-played group-stage results from data/results.txt."""

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_results(path=None):
    """Parse results.txt into {(team_a, team_b): (score_a, score_b)}.

    Returns an empty dict if the file doesn't exist or has no results.
    Lines are: TEAM_A TEAM_B score_a score_b
    Comments (#) and blank lines are ignored.
    """
    if path is None:
        path = DATA_DIR / "results.txt"
    path = Path(path)
    results = {}
    if not path.exists():
        return results

    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(
                    f"results.txt line {lineno}: expected 'TEAM_A TEAM_B score_a score_b', "
                    f"got: {raw.rstrip()}"
                )
            ta, tb, sa, sb = parts
            try:
                sa, sb = int(sa), int(sb)
            except ValueError:
                raise ValueError(
                    f"results.txt line {lineno}: scores must be integers, got '{sa}' '{sb}'"
                )
            results[(ta.upper(), tb.upper())] = (sa, sb)
    return results


if __name__ == "__main__":
    r = load_results()
    if not r:
        print("No results recorded yet (data/results.txt is empty or missing).")
    else:
        print(f"Loaded {len(r)} played match(es):")
        for (ta, tb), (sa, sb) in r.items():
            print(f"  {ta} {sa}-{sb} {tb}")
