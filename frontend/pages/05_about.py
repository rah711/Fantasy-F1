from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components import inject_theme
from frontend.state import require_auth


require_auth()
inject_theme()

st.title("About & FAQ")
st.caption("How this experiment works, what the model sees, and what to read into the results.")

st.markdown("---")


st.markdown(
    """
<div style="
    border-left: 4px solid #E10600;
    background: linear-gradient(135deg, rgba(225,6,0,0.18), rgba(225,6,0,0.06));
    padding: 1rem 1.1rem;
    border-radius: 8px;
    color: #F0F0F0;
">
  <div style="font-size: 1rem; font-weight: 800; color: #E10600; margin-bottom: 0.45rem;">
    What is this?
  </div>
  <div>
    Three Fantasy F1 teams running side-by-side across the full 2026 F1 season,
    each picking lineups using a <strong>different decision-making approach</strong>.
    The question is whether AI-assisted human play beats pure human judgement —
    and whether either beats pure AI.
    <br><br>
    The leaderboard, charts, and per-round breakdowns on this site let you
    watch all three teams race each other through the season. New races land
    each weekend; this app updates within a day.
  </div>
</div>
    """,
    unsafe_allow_html=True,
)


with st.expander("**How does Fantasy F1 work?**"):
    st.markdown(
        """
Each week you pick **5 drivers + 2 constructors** within a **£100M budget**.
Each pick scores fantasy points based on what happens at the race weekend.

A few things that surprise newcomers:

- **Fantasy points ≠ F1 championship points.** Fantasy scoring counts a lot
  more than just finishing position — overtakes, fastest lap, driver of the
  day, position gained from grid, constructor pitstop tier, qualifying
  Q2/Q3 bonuses, DNF penalties. So a P8 driver who started P15, made
  overtakes, and grabbed fastest lap can outscore a P5 driver who started
  P3 and just held position.

- **The DRS Boost.** Each weekend you pick one driver to get **2× points**.
  Optional special "extra DRS Boost" chip adds a second slot at 3×.

- **Transfers.** You get 2 free transfers per race. Extra transfers cost
  -10 pts each.

- **Price changes.** Driver and constructor prices fluctuate every week
  based on how they performed. Cheap drivers who suddenly score well *gain
  price* — which can then be banked or used to afford an expensive
  superstar later. This opens a second tactic alongside "pick the best
  weekly scorers": the **"budget builder"** play, where you intentionally
  pick rising cheap assets to grow your team's value over time. Most
  successful Fantasy players blend both.
        """
    )


with st.expander("**Who are the three teams + how do they pick lineups?**"):
    st.markdown(
        """
1. **Pure human judgement (teal)** — Sarah picking the way she normally
   plays, using gut feel + the usual budget-builder community sources
   for likely weekly price changes.

2. **Pure-AI Claude chat (orange)** — A Claude chat given only the
   minimal inputs (race results + the official fantasy scoring rules)
   and asked to pick a team weekly. No engineered features, no
   domain-specific tooling. Just an LLM doing its best.

3. **Vibe-coded data science model (red)** — This repo. A LightGBM
   model trained on 2018–2025 F1 data, ingesting 2026 results as the
   season progresses, plus engineered features for what Sarah believes
   matters (track characteristics, driver skill independent of car,
   team-year performance, weather, sprint flags, etc.). The
   recommendation that gets locked in is whatever the model produces —
   on-the-spot manual overrides are avoided so the experiment measures
   the model's actual decisions, not a hybrid. When the model gets
   something obviously wrong, it's a signal to fix the model (retrain,
   add a feature, tune a knob), not to paper over it at lock-in.
        """
    )


with st.expander("**When does the model make its prediction?**"):
    st.markdown(
        """
**Before qualifying.** The team is locked in by the Friday of the race
weekend, so the model can't react to anything that happens in qualifying
or practice. That's intentional — the experiment is about predicting
ahead of time, not reacting to live information.

Practically this means: even if you watch quali and see something
obvious (a driver crashes out, a team unexpectedly tops the times), the
model doesn't get to use that signal. Whatever you read on a Sunday
recap was decided based on what was known Friday morning.
        """
    )


with st.expander("**What information does the model get?**"):
    st.markdown(
        """
**Historical:** Every race weekend from 2018 onwards — finishing positions,
grid positions, overtakes, fastest laps, Driver of the Day, pitstop
performance, weather, qualifying times. Sourced from a Kaggle F1 dataset
+ a TracingInsights historical archive.

**2026 live:** As races happen, the OpenF1 API ingests them within a day
or so. After each race weekend, a scheduled GitHub Actions job retrains
the model with the new data.

**Circuit metadata:** Each track has hand-coded properties — overtake
difficulty, DRS zone count, safety-car probability, downforce demand,
circuit type (street / permanent / hybrid).

**Configuration knobs:** Sarah sets a small handful of values per race —
weather forecast (rain probability, temperature) and an "era weight"
slider that tells the model how much to lean on 2026 data vs 2018–2025
history. The era weight starts low (R1 = 5%) and climbs through the
season as the picture clarifies (R7 ≈ 65%).

**What the model does *not* see:** anything from qualifying or practice,
team-radio gossip, driver health rumours, paddock news, weather changes
between Friday and Sunday, mid-race incidents. Humans can read all of
that and adjust; the model can't.
        """
    )


with st.expander("**What variables does this model think about?**"):
    st.markdown(
        """
The model is a gradient-boosted regression (LightGBM) trained to predict
each driver's fantasy points for a race. The features it considers:

- **Track characteristics**: overtake difficulty, DRS zones, safety car
  probability, sprint-round flag.
- **Context**: era weight (current season vs history), rainfall flag.
- **Driver form**: rolling fantasy points over the last 3 and 5 races,
  average finishing position at this circuit, career overtake rate, DNF
  rate.
- **Driver skill residual**: a rolling average of how the driver
  out-finishes (or under-finishes) their own teammate, persistent across
  team changes. Captures *intrinsic skill independent of the car* —
  helps the model not punish a great driver who's stuck in a slow car.
- **Driver cold-start**: previous-season aggregates so rookies and
  team-switchers aren't penalized for having no rolling history.
- **Team-year baseline**: data-derived current-season finishing
  performance per constructor. Replaces what used to be a hand-rated
  "development score." Updates automatically as the season progresses.
- **Pitstop performance**: rolling fastest and average pit stop times
  per team.
- **Car-track interactions**: how well this team has historically (and
  this season) done at this *type* of circuit (street / permanent /
  hybrid) and at this downforce level.
- **Driver-track interactions**: same, per driver.

After training, the optimizer takes the per-driver predictions plus
current prices and finds the best 5-driver / 2-constructor combo within
budget — accounting for transfer penalties and the DRS Boost
multiplier.
        """
    )


with st.expander("**Why does the model sometimes make weird picks?**"):
    st.markdown(
        """
Three main reasons it gets things wrong:

1. **It hasn't seen 2026 driver-team combos yet.** When a driver moves
   teams (e.g. Perez to Cadillac for 2026), the model has lots of data
   on the driver and zero data on the new car. Early in the season it
   tends to project the driver's *historical team performance* onto the
   new combo. The team-year baseline + driver skill residual features
   are designed to fix this, and do — but only as the season produces
   more data.

2. **It's static between retrains.** New model trains every Tuesday.
   Anything that emerged Tuesday–Friday (a development upgrade, a
   driver penalty, a news story) the model can't use until next week.

3. **It can't see qualifying.** Locked in Friday. So if a team
   unexpectedly aces or bombs qualifying, the model's predictions are
   based on a stale view.

When you see a recommendation that obviously contradicts a real-world
fact, that's a signal that something needs fixing in the model
itself — a missing feature, a stale prediction, a tuning knob in the
wrong place. The fix lands in the next retrain, not as a manual
override at lock-in. Keeping the model's decisions intact (even the
bad ones) is how the experiment honestly measures the AI approach
against the human and pure-AI ones.
        """
    )


with st.expander("**Why choose 2026, and why does the data science model look rough early?**"):
    st.markdown(
        """
Short answer: **2026 is the hardest possible year to start this experiment**, by design.

**Why regulations change in F1 (plain English):**
- F1 periodically rewrites technical rules to reset competition, improve safety,
  control costs, and push engineering in specific directions (for example:
  cleaner power units, aero changes, or easier overtaking).
- When rules change, teams effectively build new cars under new constraints.
  Even top teams can rise or fall unexpectedly.

**Why this hurts historical models:**
- Most models learn from patterns in past data ("this team tends to be strong
  here", "this driver-team combo behaves like X").
- In 2026, many of those patterns break. It's not a small setup tweak year;
  it's closer to a **full-stack reset** (car concept + aero + powertrain era
  context), so past seasons are less reliable as a guide.
- Bottom line: early-season model performance is expected to be noisy.
  The first few rounds are about collecting fresh signal, then improving as
  retrains absorb real 2026 race outcomes.

**Also important: this project currently does *not* ingest qualitative intel.**
- The model and pure-chat AI mostly see structured race data.
- They do **not** directly read preseason reports, testing analysis, paddock
  rumours, or expert commentary unless that is manually encoded.
- Human judgement *does* use that info. Example:
  - Williams preseason issues (including missed safety-test running) implied
    less prep data than the historical trend suggested.
  - Aston Martin reports suggested severe drivability/bump issues over long
    race stints, which made reliability/performance riskier than historical
    numbers alone would imply.

So if the human team avoids something that looks statistically okay on paper,
that can be because it used context that the current model stack never saw.
        """
    )


with st.expander("**Why are R4 and R5 missing / dim red on the charts?**"):
    st.markdown(
        """
**R4 Bahrain** and **R5 Saudi Arabia** were cancelled mid-season due to
escalating conflict in the region. The calendar in the app keeps the
round numbers stable (R6 is still Miami, R7 still Canada) but flags
those two rounds as cancelled. Charts skip over them; the driver-tenure
grid renders them as dim red cells so the gap is explained rather than
hidden.
        """
    )


with st.expander("**Sprint weekends + chips?**"):
    st.markdown(
        """
**Sprint weekends** add a short Saturday race (with its own qualifying
session) on top of the usual Sunday race. More scoring sessions = higher
fantasy totals overall. Sprint rounds in 2026: R2 China, R6 Miami, R7
Canada, R11 Great Britain, R14 Netherlands, R18 Singapore. The app
marks these with a ★ on the round labels.

**Chips** are one-shot abilities you can play once per season (max 1 per
weekend):
- **Wildcard** — unlimited free transfers for this race.
- **Limitless** — unlimited transfers AND no budget cap for this race.
- **No Negative** — floors your score at 0 per scoring category. Useful
  in wet/chaotic races where DNFs are likely.
- **Extra DRS Boost** — adds a second DRS slot at 3× alongside the
  regular 2× slot (so two drivers get boosted).
- **Autopilot** — retroactively gives DRS to your highest scorer.
- **Final Fix** — one penalty-free transfer between qualifying end and
  race start.

The wizard makes a single recommendation per weekend: "play X chip" or
"hold all chips," with a confidence score.
        """
    )


with st.expander("**Can I see the code / follow along myself?**"):
    st.markdown(
        """
Everything's open source: **[github.com/rah711/Fantasy-F1](https://github.com/rah711/Fantasy-F1)**.

- Model code, feature engineering, training pipeline — all there.
- Weekly retrain runs automatically via GitHub Actions every Tuesday
  morning UTC; the bot commits the new model + features back to the
  repo. You can watch the [Actions tab](https://github.com/rah711/Fantasy-F1/actions)
  to see when it last ran.
- Each round's predictions are committed too (under
  `data/fantasy/predictions/`), so the "Prediction vs actual" view on
  the Performance page has a full historical record.
- Locked-in teams and round-by-round scores auto-PR back to the repo,
  so the season is fully version-controlled.

If you want to fork it and run the same experiment for your own
fantasy league, the README walks through it.
        """
    )

st.markdown("---")
st.caption(
    "Spotted something wrong? File an issue on "
    "[GitHub](https://github.com/rah711/Fantasy-F1/issues)."
)
