# Stage 2: snapshots

See `docs/PLANNING.md` §6 for inputs, outputs and the cleaning rules. Code: `src/esvrp/data/quotes.py`.
Run with `doit quotes` or `python -m esvrp.stages 2` (needs `data/universe.parquet` from stage 1).

## What is pulled per business day

- `bbo-1m` for the option parent (`EW.OPT`) over `[snapshot - backfill_minutes, snapshot + 1 min)`.
- `bbo-1m` for the specific futures contracts the live options are written on (raw symbols such as
  `ESZ6`, not the `ES.FUT` parent, which also carries every calendar spread and is slow to price).
- The snapshot is `data.snapshot_time_ct` (15:00 CT) converted to UTC per day, so CST/CDT is handled.

All of this is covered by the program's Plus plan ($0 estimated cost).

## Snapshot and back-fill rule

A `bbo-1m` record stamped T carries the best bid/offer as of T. For each option-day we keep the last
record at or before the snapshot. If an option has no record in the snapshot minute, the most recent record
within `cleaning.backfill_minutes` (default 15) is used and its age is kept in `quote_age_s`. Older quotes
count as missing.

Checked on 2026-09-01: 3,044 of about 3,100 live EW options had a record in the 15:00 minute itself;
a 15-minute look-back reached 3,102 and longer look-backs added almost nothing. Options expiring at or
before the snapshot are excluded.

`F` is the futures mid of each option's own underlying at the same minute (same back-fill rule).

## Cleaning rules (`clean_quotes`, every rule counted in `data/quotes_cleaning_log.csv`)

Row rules, in order:
1. `no_quote`: no bid and no ask.
2. `zero_bid`: bid zero or missing (switch: `cleaning.drop_zero_bid`).
3. `crossed` (bid > ask) and `locked` (bid = ask) (switch: `cleaning.drop_crossed`).
4. `no_futures_price`: the underlying future had no quote near the snapshot. In practice this is the
   furthest-dated contracts (e.g. `ESZ7`, `ESH8` in Sep 2026), whose options are far outside the
   14–60 day maturities used in the analysis.
5. `below_intrinsic`: mid below the discounted intrinsic value `e^{-r tau} max(±(F - K), 0)`.

A quote with a valid bid and no ask is kept and flagged `one_sided` for diagnostics.

Chain rules, per (date, expiry):
6. `chain_min_strikes`: fewer than `cleaning.min_strikes` strikes left.
7. `chain_min_range`: strikes do not reach both `F(1 - min_range_pct)` and `F(1 + min_range_pct)`.

Typical result: about 8 usable chains per day (the monthly expiries out to about a year).

## Chain-quality diagnostics (roadmap 20)

The row rules above decide whether one quote is usable. `data/quality.py` decides whether the cleaned quotes
of one chain (date, expiry) are collectively usable. It runs at the start of stage 4, on `quotes.parquet`, and
does not repeat the row rules. Why it matters: a short strike range biases the variance-swap rate downward,
and stale or non-synchronous quotes add noise to implied volatilities and the parity forward.

Per chain, `data/chain_quality.parquet` records:

| Diagnostic | Columns |
|---|---|
| Completeness | `n_calls`, `n_puts` (two-sided only), `n_strikes`, `n_both_sides` |
| Strike coverage | `strike_min`, `strike_max`, `log_range_lo`, `log_range_hi` (ln(K/F) at the ends, F the futures mid) |
| Quote freshness | `age_median`, `age_p90`, `age_max`, `stale_share` (share older than `max_quote_age_s`) |
| Synchronization | `F_age_s` (snapshot minus the futures record used), `sync_gap_s` (median abs. option-quote time minus futures time) |
| Verdict | `reasons` (semicolon-joined), `fail_<reason>` booleans, `status` |

Rules (`quality:` in `config/default.yaml`; each failure adds a reason):

| Reason | Fails when |
|---|---|
| `few_calls`, `few_puts` | fewer than `min_calls` / `min_puts` two-sided quotes |
| `few_strikes` | fewer than `min_strikes` distinct strikes |
| `narrow_range` | strikes do not reach both F(1 - `min_range_pct`) and F(1 + `min_range_pct`) |
| `stale_quotes` | more than `max_stale_share` of the quotes are older than `max_quote_age_s` |
| `stale_futures` | `F_age_s` above `max_futures_age_s` |
| `unsynchronized` | `sync_gap_s` above `max_sync_gap_s` |

`quality.action: flag` (default) keeps every chain and marks the failures; `reject` drops the failing chains
before stage 4 builds them. Either way the reasons stay in `chain_quality.parquet`, so no observation
disappears silently. `min_range_pct` here (0.15) is deliberately stricter than the stage 2 chain rule (0.10).

Outputs: `reports/tables/chain_quality_summary.csv` (chains failing each rule), `reports/tables/quote_age_by_day.csv`
(age distribution per day) and `reports/figures/chain_quality.png` (strikes, stale share and synchronization
through time, failing days marked).

Stage 2 now also keeps `F_age_s` next to `F`. Quote files pulled before this change lack it; the chain is then
not judged on the two futures rules and `sync_measured` is False (rerun stage 2 to fill it in).

Checked on the 2026 Q3 sample (452 chains, 64 days; no `F_age_s`): 25 chains fail, none on freshness (99.9% of
quotes are in the snapshot minute; the oldest is 14 minutes). 24 fail `narrow_range`, all short-dated chains
(under about a month) whose call wing stops at 10-14% above F; one far-dated chain (Jan 2027 on 2026-08-03) has
only 11 strikes. Databento's degraded-quality days (see below) are not in this sample.

## Known data issues

- Databento marks some days as degraded quality (seen so far: 2024-09-18, 2025-09-17, 2025-09-24).
  The chain-quality diagnostics above show their effect on the chains; they do not flag the degraded days directly.
- On early-close days (e.g. the day after Thanksgiving) there is no 15:00 CT quote; those days drop out.
