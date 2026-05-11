# 2026 World Cup PELE Forecast

A reverse-engineered version of Nate Silver's PELE-based tournament forecast for the 2026 FIFA World Cup. Runs 20,000 Monte Carlo simulations using team ratings from Silver Bulletin, then publishes per-team probabilities for each round of the tournament — plus an interactive bracket showing where each team is most likely to play.

**Live site**: https://YOUR-USERNAME.github.io/wc2026-forecast/

## What this is

Silver Bulletin publishes the *inputs* to its forecasting model (team ratings, expected goals per match, home-field advantages) but not the *tournament-wide outputs* (championship probabilities, advancement by round, bracket paths). This project fills that gap. It uses a two-stage model — calibrated against Silver's published match forecasts — to simulate the tournament end to end.

The end-to-end accuracy vs Silver's published win/draw/loss probabilities is about 4 percentage points RMS on the 72 group-stage matches.

## How to use

The website has two main views:

**Forecast Table** — All 48 teams ranked by championship probability, with their probabilities of reaching each round.

**Interactive Bracket** — Pick a team to see a heat-mapped bracket showing where that team is most likely to play in each round. Click any specific match slot to flip the view and see which teams are most likely to play in that match (useful for fans planning trips to specific cities).

## Project structure

```
wc2026-forecast/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── data/                           # Input CSVs from Silver Bulletin
│   ├── data-dxUJw.csv              # Team ratings (PELE, Tilt)
│   ├── data-DcqkH.csv              # Round-robin GF/GA
│   ├── data-3bTOr.csv              # Per-match forecasts
│   ├── data-4bcIB.csv              # Team metadata
│   ├── data-4oVop.csv              # Historical PELE
│   └── data-uzR80.csv              # Per-team home-field advantage
├── src/                            # Python source
│   ├── pele_model.py               # Locked rating-to-xG + score matrix
│   ├── bracket.py                  # Groups, venues, Annex C table
│   ├── group_sim.py                # Group-stage simulator
│   ├── knockout_sim.py             # Knockout-stage simulator
│   ├── simulator_optimized.py      # Fast Monte Carlo engine
│   ├── generate_forecast.py        # Main entry point (writes JSON)
│   ├── load_data.py                # CSV loaders
│   └── load_hfa.py                 # HFA loader
├── docs/                           # GitHub Pages serves from here
│   ├── index.html                  # The website
│   └── forecast.json               # Output from generate_forecast.py
└── .github/workflows/
    └── update_forecast.yml         # Daily automatic regeneration
```

## Running locally

Requires Python 3.9+ and the dependencies in `requirements.txt`.

```bash
pip install -r requirements.txt
cd src
python3 generate_forecast.py 20000      # 20,000 simulations, ~12 seconds
# Output: docs/forecast.json
```

To view the website locally, you need a tiny local server (modern browsers block `fetch()` from `file://`):

```bash
cd docs
python3 -m http.server 8000
# Then open http://localhost:8000 in a browser
```

## Deploying to GitHub Pages

1. Push this repository to GitHub
2. Go to repository **Settings → Pages**
3. Set **Source** to "Deploy from a branch"
4. Set **Branch** to `main` and **Folder** to `/docs`
5. Save. Your site goes live at `https://YOUR-USERNAME.github.io/REPO-NAME/` within a minute.

## Updating the forecast

When Silver Bulletin's ratings change:

1. Re-export the CSV files from your Silver Bulletin subscription
2. Replace the files in `data/` (keep the same filenames)
3. Run `cd src && python3 generate_forecast.py 20000` to regenerate `docs/forecast.json`
4. Commit and push — GitHub Pages will redeploy automatically

Or, set up the included GitHub Actions workflow to run this on a schedule.

## The model

**Rating-to-expected-goals** (Problem B). Predicts each team's expected goals given:
- Both teams' PELE ratings (overall strength)
- Both teams' Tilt ratings (attacking vs. defensive style)
- Home-field flags + per-team HFA values
- Competition tier (World Cup, friendly, etc.)

Fitted form has 13 coefficients including a quadratic correction for extreme rating gaps and a split offense/defense home-field-advantage term. Calibrated against 342 published match forecasts. RMS error on World Cup matches: ~0.12 goals.

**Expected-goals-to-scoreline** (Problem A). Builds a probability distribution over every possible scoreline using negative-binomial marginals × Dixon-Coles low-score correction. Two parameters: dispersion `φ=12.1` and Dixon-Coles `ρ=-0.118`. Reproduces Silver's W/D/L probabilities within 0.5-0.9 percentage points given his expected goals.

**Tournament simulation**. Plays out the 2026 format: 12 groups of 4 → 8 advancing third-place teams (using FIFA's 495-scenario Annex C lookup table) → standard R32 → R16 → QF → SF → F bracket. Knockouts include extra time and penalty shootouts. Host nations get HFA when playing at their own venues.

## Credits

- **Model methodology**: Nate Silver, [Silver Bulletin](https://www.natesilver.net/)
- **Implementation**: This is an independent reproduction; not affiliated with Silver Bulletin

## License

MIT
