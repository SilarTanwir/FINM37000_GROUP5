# Stage 3: underlying

See `docs/PLANNING.md` §6 for inputs, outputs and the theory section (§4.2). Code:
`src/esvrp/data/underlying.py`. Run with `doit underlying` or `python -m esvrp.stages 3` (needs
`data/universe.parquet` from stage 1).

## What is pulled per futures contract

- `ohlcv-1m` and `statistics` for each futures contract that appears as an `underlying` in the universe
  (raw symbols such as `ESZ6`), one request per contract. Never a continuous or front-month series: the
  realized variance and the hedge of an option use the contract it is written on (§4.2, §5.3).
- A contract is pulled for the trade dates on which any option written on it is live: from the first
  option listing (`activation`) to the last option expiry, clipped to `data.start`–`data.end`
  (`contract_windows`). Requests cover whole trade dates, from the 17:00 CT session open of the first to
  the session open after the last.

Checked on 2026-10-10 with a universe built from every eighth weekly snapshot: 19 contracts (`ESZ3` to
`ESM8`), 1.2 million 1-minute records, all covered by the program's Plus plan ($0 estimated cost). The
pull took about 6 minutes.

## Bars and the gap rule

A 1-minute record is stamped with its opening time and exists only if the contract traded in that
minute. `build_bars` groups these into 5-minute or daily bars and makes a bar only from minutes that
traded. Nothing is filled in during the 16:00–17:00 CT maintenance halt, weekends, holidays, or the idle
minutes of a thin contract, so the move from one bar's close to the next bar's close is **one return
spanning the gap** (§15.3). Summing squared close-to-close returns over consecutive rows therefore
neither drops nor double counts a gap.

- `ts` is the **end** of the bar in UTC: the bar stamped 15:00 CT holds the minutes 14:55 to 14:59, so
  its close is the last trade before the 15:00 CT snapshot.
- `date` is the CME trade date: the session that opens at 17:00 CT belongs to the next day, so Sunday
  evening is Monday.
- Session (`realized.session`): `full_globex` keeps every minute traded (decided, §15.3). `rth` keeps
  08:30–15:00 CT only, ending at the snapshot time.

A front contract has 276 five-minute bars on a full day (23 hours).

## Settlements

`settlements` reads the settlement price (`stat_type` 3) from `statistics`. CME publishes a preliminary
value about 20 seconds after 15:00 CT and the final one in the evening, both with `ts_ref` set to the
trade date; the last one published is kept. The settlement is close to the last trade before 15:00 CT
(mean gap of about 1 index point for `ESZ5` as front contract in autumn 2025), which matches the
snapshot time used for the option quotes (§15.4).

## Outputs

- `data/underlying_5m.parquet`: `underlying`, `date`, `ts`, `open`, `high`, `low`, `close`, `volume`.
- `data/underlying_daily.parquet`: the same columns per trade date (`ts` is the end of the last minute
  traded) plus `settle`. Daily realized variance uses settlement-to-settlement returns (§4.2).

Two kinds of daily row are incomplete on purpose:
- **Settled but not traded** (`close` empty, `volume` 0): back-month contracts settle every day but
  trade rarely. About half of all contract-days; `ESH8` and `ESM8` had no trade at all in the sample.
- **Traded but not settled** (`settle` empty): exchange holidays with a shortened Globex session (23
  dates in the sample, e.g. Thanksgiving, 2025-01-09). CME counts these trades toward the next trade
  date; here they keep the calendar date and simply have no settlement.

## Strike sanity check

The stage runs `check_strikes_bracket_futures` for every contract against its last settlement (the guard
described in `docs/stages/01_universe.md`). It passed for all 19 contracts.

## Known data issues

- A single off-session record falls inside the maintenance halt (`ESZ3`, 2023-10-30 16:00 CT, 3,000
  contracts at one price). It is kept as its own bar.
- Databento marks some days as degraded quality (seen in these pulls: 2024-09-18, 2025-09-17,
  2025-09-24, 2026-04-10, 2026-05-24, 2026-08-29). Diagnostics belong in the chain-quality issue
  (roadmap 20).
