15-MINUTE DIRECTION REFINEMENT

Run from November/Week 3:
python Code/fit_direction_15m_refined.py

Optional no-fit input audit:
python Code/fit_direction_15m_refined.py --check-inputs

Focused follow-up to the 52.07% historical direction result. Improvement is
not established until results are reviewed. No training executed by assistant.

Six candidate specifications:
  BCE and Brier MLP controls: original384 inventory, 2000 uniformly spaced
  training rows, hidden16, 16 epochs, seed48 (same original neural settings).
  BCE and Brier expanded MLPs on original384 and combined478 (384+94 new).
  Expanded models: up to12000 uniformly spaced mature rows, hidden32,
  32 epochs, average probabilities of three seeds48/149/250.
All use AdamW learning rate .0005, weight decay .1, batch256, clip1,
prior-logit output initialization, training-only median/IQR/missing flags.
Availability/constant filtering can reduce input counts. No class balancing.
--row-cap changes expanded candidates only; controls always cap at2000.
--models mlp_brier runs the three Brier specifications only.
--threads defaults to2. Uses the original runner's installed dependencies.

Same canonical15m endpoint UP target, zero treated as non-UP; checksum and
source crossing checks, strict label maturity, original half-hour origin grid.
Four quarterly issues and four earlier three-month validation windows per issue.
Same dual logloss/Brier guard and .25/.5/.75/1 prevalence-blend weights.
Selection does not use outer evaluation outcomes. Failure to pass the guard
falls back to training prevalence; candidate remains diagnostic. Acceptance
is against prevalence, not a guarantee to beat the previous saved forecast.

Default workload: 96 specification/window fits, comprising224 individual
neural fits before outer fitting; up to16 distinct outer specification fits
(up to48 individual networks). No runtime promise or resume. More expensive
than the original small MLP. Earlier histories with fewer rows use all rows.

Previous run direction_15m_classifiers_v1/20261116T132527199849Z is required.
Its completion receipt/hash, timestamps, labels, signed returns and prevalence
are checked. Previous forecasts are reporting controls only, never features
or selection inputs. Same --data-dir, --direction-template, --canonical-root,
--canonical-manifest overrides as the original runner.

Outputs (new timestamped directories, prior results preserved):
Code/experiments/direction_15m_refined_v1/<timestamp>/
Report/direction_15m_refined_v1/<timestamp>/
Includes previous_selected vs new p_up metrics (logloss/Brier/accuracy/AUC),
family policies, diagnostic candidate, saved models and replay/hash checks,
validation ledger, feature-pack comparison, selections, formulas and contract.

Historical evaluation has already informed this design. Reusing it is an
exploratory comparison, not independent confirmation. No new input audit,
training, serialization replay or end-to-end execution performed yet.
