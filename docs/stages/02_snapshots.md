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

## Known data issues

- Databento marks some days as degraded quality (seen so far: 2024-09-18, 2025-09-17, 2025-09-24).
  Diagnostics for these belong in the chain-quality issue (roadmap 20).
- On early-close days (e.g. the day after Thanksgiving) there is no 15:00 CT quote; those days drop out.
