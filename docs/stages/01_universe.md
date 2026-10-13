# Stage 1: universe

See `docs/PLANNING.md` §6 for inputs, outputs and the theory section. Code: `src/esvrp/data/universe.py`.
Run with `doit universe` or `python -m esvrp.stages 1`.

## Series choice

The project uses the **end-of-month (`monthly_eom`) ES options**, which are **European** exercise, so
Black-76 applies exactly (§4.1, §15.1). CME lists them under the root **`EW`**, not `ES`: `ES` carries the
futures and the quarterly (American) options. The universe is pulled with the parent symbol `EW.OPT`
(`stype_in="parent"`); the mapping lives in `SERIES_PARENT`.

## What the definition records contain (checked on 2026-09-01)

- `instrument_class`: `C` and `P` are single options; `T` (exchange-listed strategies) and `M`
  (user-defined multi-leg spreads) are dropped. About 3.8k of 13k records that day were single options.
- `cfi`: six-letter code; the third letter is the exercise style (`E` European, `A` American).
  `OCEFPS` = option, call, European. All EW single options are European.
- `underlying`: the futures contract the option is written on. EW options are written on the **next
  quarterly** ES contract (for example an October EW option has underlying `ESZ6`), so the hedge and the
  forward must use this field, never the same-month future (§5.3).
- `user_defined_instrument = Y`: strikes created at a trader's request. Kept, flagged in `user_defined`.
- `expiration`: 15:00 CT on the last business day of the month (stored in UTC).
- The same instrument can appear more than once in a day (updates); the latest record wins.

## Output: `data/universe.parquet`

One row per option: `instrument_id`, `raw_symbol`, `cp` (`c`/`p`), `strike`, `expiry` (UTC),
`underlying`, `style`, `user_defined`, `activation`.

## How the sample is covered

Definitions are read from **weekly snapshots (Tuesdays)** over `data.start`–`data.end` and unioned by
`instrument_id`. Weekly rather than monthly because CME adds strikes as the market moves; a monthly
snapshot would miss strikes listed mid-month, which would truncate the variance-swap strip (§4.3).
A strike listed and expiring between two Tuesdays would still be missed; this only affects the last
week before an expiry. All pulls are covered by the program's Plus plan ($0 estimated cost).

## Strike sanity check

`check_strikes_bracket_futures(universe, F, underlying=...)` raises if the strikes written on a futures
contract do not bracket its price. A strike scale error (Databento's display-factor issue, §5.3) would put
every strike on one side of `F`. ES is not on Databento's affected list; the check stays as a guard and is
run against ES futures settlements in stage 3.
