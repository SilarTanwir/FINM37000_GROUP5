# The variance risk premium in E-mini S&P 500 futures options

**ES options are priced above the volatility that follows, but the premium sits in the skew: delta-hedged
near-the-money options earn about nothing, and a monthly short straddle loses money after costs.**

Using three years of CME data from Databento (October 2023 to September 2026), we measure the variance
risk premium in end-of-month E-mini S&P 500 options two independent ways and then test whether a
systematic option seller could have collected it.

## Headline results

| Hypothesis | Result |
|---|---|
| **H1** Implied variance exceeds realized (VRP < 0) | **Supported in log terms.** On the 30-day series options implied 17.9% volatility against 15.7% realized, and the premium was negative on 86% of days. Mean ln(RV/SW) = −0.52, Newey–West t = −6.3. In variance units the mean is −0.0076 but not significant (t = −1.7): one episode, April 2025, dominates |
| **H2** Delta-hedged option gains are negative | **Supported, strongest away from the money.** Per unit of vega, hedged gains are negative in every moneyness and maturity bucket: −1.80 vol points for 14–30 day out-of-the-money puts (t = −3.4), −2.13 for out-of-the-money calls (t = −4.7), but only −1.08 near the money (t = −1.7). No clear pattern across volatility regimes |
| **H3** The two measures agree in sign and co-move | **Partly.** Signs agree in 89% of months and the monthly correlation is 0.95, but 0.45 without April 2025. Gamma-weighted hedged gains average +0.0007 while the VRP is −0.0057; the gap is the skew |
| **H4** A short delta-hedged straddle earns the premium after costs | **Not supported.** 36 monthly trades: −0.17 vol points per unit vega before costs, −0.99 after; net Sharpe −0.75; worst month April 2025 (−$12,242 on one straddle) |

![VRP against the hedged gain](reports/figures/h3.png)

*Each point is one option expiry on one day: the Carr–Wu premium (x) against the gamma-weighted hedged
gain (y). Points above the 45° line are expiries where hedged options lost less than the strip implies.*

## What we found

**The premium is real, but it is a premium for crash protection.** The swap rate weights every strike by
1/K², so out-of-the-money puts carry much of it. Hedged options near the money only collect the
difference between at-the-money implied and realized variance. Splitting the 0.0063 monthly gap between the
two measures, the skew term SW − IV²_gw averages 0.0079, more than the whole gap; the realized-variance
term offsets −0.0016. The bucket table (H2) agrees: per unit of vega, out-of-the-money options lose about
two to five times more than near-the-money ones.

**One month can decide a variance statistic.** The 30-day windows that start in late March 2025 run into
the April 2025 selloff, when realized volatility reached about 49% against roughly 17% implied. That
episode makes the variance-unit premium insignificant, inflates the H3 correlation and produces the
strategy's worst month. Log premia and rank correlations are much less sensitive to it, which is why we
report both.

**Selling at-the-money straddles is not the way to harvest it.** Before costs the straddle earns about
zero per unit of vega; spreads, hedge slippage and fees turn that negative, and the return is left-skewed
(skew −1.8). A seller who wants the premium has to sell the wings, and carry their tail risk.

**Overlap matters.** Daily 30-day windows share most of their returns: the naive t-statistic for the
variance premium is −5.5, the Newey–West one −1.7. Every test here uses Newey–West errors or
non-overlapping monthly samples.

## How it works

1. **Data.** Databento `GLBX.MDP3`: option definitions, 1-minute best bid and offer at 15:00 CT each day,
   and 1-minute bars and settlements for the ES futures each option is written on. End-of-month (EW)
   options are European, so Black-76 prices them exactly.
2. **Chains.** Quotes are cleaned (crossed, zero-bid, below-intrinsic quotes and thin chains removed),
   the forward is implied from put–call parity, and implied volatilities and Greeks are computed.
3. **Carr–Wu (2009).** A model-free variance swap rate SW is built from out-of-the-money options and
   compared with realized variance RV from 5-minute returns: VRP = RV − SW.
4. **Bakshi–Kapadia (2003).** Every option within 10% of the money and 14–60 days from expiry is bought
   and delta-hedged daily with its own futures contract to expiry.
5. **Strategy.** Each month, sell one delta-hedged at-the-money straddle, charging the bid–ask spread,
   hedge slippage and fees.

Derivations are in [docs/theory.md](docs/theory.md); each stage is documented in [docs/stages/](docs/stages/).

## Limitations

- Three years of data, dominated by one volatility episode.
- One snapshot a day at 15:00 CT, the ES settlement minute, when near-the-money quotes are wider and more
  often one-sided than a few minutes earlier; four months needed a later entry for the straddle.
- The strip uses quoted strikes only, without Carr–Wu's implied-volatility extrapolation, and leaves 21%
  of chains without a swap rate when no quote sits near the forward.
- The forward comes from a single strike; strike coverage is checked as a fixed percentage of F; the
  realized-variance window ends at Databento's expiry timestamp, not verified against CME's settlement time.
- Hedge costs are conservative: one futures contract of slippage per rebalance, each leg hedged separately.
- No systematic robustness table (cost level, bid/ask strip, snapshot time); the strangle, margin and H3
  sensitivities are reported.

## How to run

You need Python 3.11+ and a Databento API key with access to `GLBX.MDP3`.

```bash
git clone https://github.com/SilarTanwir/FINM37000_GROUP5.git && cd FINM37000_GROUP5
python -m venv .venv && source .venv/bin/activate   # keep this environment active: doit runs `python`
pip install -e ".[dev]"
cp .env.example .env              # then set DATABENTO_API_KEY in .env
pytest -q                         # offline tests, no key needed
doit universe quotes underlying chain vrp hedging analysis report
```

- **Cost.** Every request logs Databento's cost estimate before pulling and stops if it exceeds
  `data.budget_usd` in `config/default.yaml` (default $50 per request). On our Databento Plus plan every
  pull was estimated at $0; we have not checked the price on other plans.
- **Runtime.** A first run downloads everything; the futures bars alone (stage 3) took about 50 minutes.
  Pulls are cached in `data/`, so a rerun of all eight stages takes about 3 minutes.
- **Outputs.** Intermediate tables in `data/` (Parquet, not committed: Databento data is licensed);
  results in `reports/tables/` (CSV; `headline.csv` collects every number above) and `reports/figures/`
  (PNG). The committed `reports/` files are from the run of 25 October 2026. Parameters are in
  `config/default.yaml`; `doit list` shows every stage.

## Repository layout

```
src/esvrp/      data/ (Databento client, universe, quotes, futures) · pricing/ (Black-76, forward, chain)
                vrp/ (strip, realized variance, premium) · hedging/ (simulator, costs)
                analysis/ (inference, H2 buckets, H3, strategy) · reporting/ (headline table)
config/         default.yaml, every tunable parameter
tests/          offline tests (pytest)
docs/           theory.md, data.md, stages/, PLANNING.md and DEVELOPER_NOTES.md (the original plan)
reports/        tables/ and figures/ written by the pipeline
```

FINM 37000 Group 5: Tanwir Silar, Alex Nikolaev, Aryaa Gunavante, Andrew Yang.
