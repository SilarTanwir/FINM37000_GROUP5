# Data notes

Source: Databento `GLBX.MDP3`. See `docs/PLANNING.md` §5 for schemas, known issues and cost control.
Raw Databento data must never be committed (`data/` and `*.parquet` are gitignored); use
`src/esvrp/synthetic.py` for tests and demos.

The stage documents in `docs/stages/` hold the details (columns, counts, checks); this file collects what
the pipeline assumes about the data and where each assumption is applied.

## Sources and what each is used for

| Schema (`GLBX.MDP3`) | Symbols | Pulled by | Used for |
|---|---|---|---|
| `definition` | `EW.OPT` (parent), weekly snapshots on Tuesdays | stage 1, `data/universe.py` | The option universe: strike, call/put, expiry, exercise style, underlying future |
| `bbo-1m` | `EW.OPT` and each underlying future (raw symbols such as `ESZ6`), 14:45–15:00 CT each business day | stage 2, `data/quotes.py` | One bid/ask per option per day, and the futures mid at the same minute |
| `ohlcv-1m` | Each underlying future, while any option on it is live, plus 30 days | stage 3, `data/underlying.py` | 5-minute bars for realized variance |
| `statistics` (`stat_type` 3) | Each underlying future | stage 3 | Daily settlements: daily realized variance, the strike check, the expiry price of the hedge |
| Config constant | — | `data/rates.py` | Risk-free rate `r` (4%, continuously compounded) |

All of these are estimated at $0 on the program's Plus plan for Oct 2023–Sep 2026 (open item 8). Every
request goes through `data/client.py`: cost check, per-request budget stop, retries, and a Parquet cache in
`data/` keyed by a hash of the request, so a rerun pulls nothing that is already cached.

## Assumptions and how the pipeline applies them

**Instruments**
- **Series.** End-of-month options (`EW` root), European exercise, so Black-76 is exact (§4.1). The
  quarterly `ES` options are American and not used. Applied in stage 1 (`SERIES_PARENT`, `cfi` filter).
- **Underlying.** Each EW option is written on the next quarterly ES future, not the same-month one. The
  forward, the hedge and realized variance all use the `underlying` field of the option's definition,
  never a front-month or continuous series. Applied in stages 2–6.
- **Strike listings.** CME adds strikes as the market moves. Definitions are unioned over weekly snapshots
  so strikes listed mid-month enter the strip; a strike listed and expiring between two Tuesdays is
  missed (last week before expiry only). Stage 1.
- **Strike scale.** ES is not affected by Databento's `strike_price` display-factor issue. The check that
  strikes bracket the futures price still runs in stage 3 against each contract's last settlement.

**Prices**
- **Snapshot time.** One observation per day at 15:00 CT, the last quote before the cash close
  (Bakshi–Kapadia's rule), converted to UTC per day so CST/CDT is handled. Stage 2.
- **Back-fill.** An option with no `bbo-1m` record in the 15:00 minute uses its last record within 15
  minutes (`cleaning.backfill_minutes`); its age is kept in `quote_age_s`. Older quotes count as missing.
- **Fair value is the mid.** `(bid + ask) / 2` is the price for implied vol, the strip and hedged-gain
  entries. Bid and ask strips are bounds; costs charge the half-spread separately.
- **Forward.** The pricing forward `F` is the put–call-parity forward at the strike where call and put
  mids are closest, because option and futures quotes are not synchronous. The futures mid `F_fut` is the
  fallback and is the price the hedge trades. Stage 4.
- **Cleaning** (stage 2, every rule counted in `data/quotes_cleaning_log.csv`): drop quotes with no bid
  and no ask, zero bids, crossed or locked quotes, no futures price, or a mid below discounted intrinsic
  value; drop chains with fewer than 10 strikes or strikes not reaching ±10% of `F`. One-sided quotes are
  kept and flagged.
- **Implied-vol bounds.** Quotes with implied vol outside 1%–100% (Bakshi–Kapadia's screen) are left out
  of the strip, where one junk far-from-the-money quote can otherwise dominate the sum. Stage 5.

**Time and returns**
- **Day count.** ACT/365 from the snapshot to expiry everywhere, so implied and realized variance are
  annualized on the same basis.
- **Expiry.** EW options expire at 15:00 CT on the last business day of the month. Realized variance runs
  to the last bar at or before expiry; the hedged option pays intrinsic value at the future's settlement
  on the expiry date (open item 9 in #43).
- **Session.** Realized variance uses full Globex (nearly 23 hours a day). A return across the daily halt,
  a weekend or a holiday is one return spanning the gap, never dropped or filled. Stage 3 builds bars only
  from minutes that traded.
- **Incomplete windows.** If the bars miss either end of a realized-variance window by more than
  `realized.max_gap_days` (2), that RV is left empty instead of understated. Stage 5.
- **Rate.** A constant 4%. At 30-day horizons the rate moves option prices and the strip by a factor of
  about `e^{0.04 × 30/365} ≈ 1.003`, so its error is small; FRED rates are not implemented.

## Known data issues

- **Degraded days** flagged by Databento: 2024-09-18, 2025-09-17, 2025-09-24, 2026-04-10, 2026-05-24,
  2026-08-29 (chain-quality diagnostics, roadmap 20, #44).
- **Early closes** (e.g. the day after Thanksgiving) have no 15:00 CT quote and drop out.
- **Back-month futures** are often unquoted at 15:00 CT (their options drop as `no_futures_price`) and
  settle without trading. Their parity forwards are noisier (forward gaps up to 15 points beyond 120 days).
- **Holes near the money** push `K0` far below `F` and the strip's correction term then dominates; such
  chains get no swap rate (`strip.max_k0_gap_pct`, 1%; #42).
- **Off-session record:** one `ESZ3` trade inside the maintenance halt (2023-10-30), kept as its own bar.

## Open verification items

Items marked **[VERIFY]** or **[DECISION]** in `docs/PLANNING.md` that are still unresolved, plus the open
decisions in §15. Each has a `TODO` comment at the point of use in the code.

| # | Type | Item | Plan § | Where in code |
|---|---|---|---|---|
| 8 | DECISION | Budget semantics on the program's Plus plan: what Databento's cost estimate means when data is covered by the subscription, and whether `data.budget_usd` is per pull or cumulative. `fetch` currently treats it as a per-pull hard stop. **Evidence (Oct 9, Alex's key):** estimates are plan-aware: EW.OPT `definition` and `bbo-1m` for the full Oct 2023–Sep 2026 sample are $0.00; ES.FUT `mbp-10` from Jan 2024 is $1.20 (outside the plan's book-depth history); non-CME data is charged. Proposed: keep the per-pull hard stop as a guard against uncovered pulls, and close this item. | 5.4 | `data/client.py`, `config/default.yaml` |
| 9 | VERIFY | Expiry price of EW options. Stage 6 pays each option intrinsic value at its futures contract's daily settlement on the expiry date, assuming EW options (expiring 15:00 CT on the last business day) are exercised against that 15:00 CT price. Check against the CME contract specs; if exercise uses a different reference, `F_T` in `build_hedged_gains` changes. | 4.5 | `hedging/simulator.py` |

## Resolved since the first draft

| Item | Resolution |
|---|---|
| ES option series | `monthly_eom`; analysis restricted to European options (§4.1, §15.1) |
| Exercise style per series | Follows from the series choice above |
| Parent symbols | Underlying futures: `ES.FUT` (`data.product`). End-of-month options are under the root `EW`, so the `monthly_eom` option universe is pulled with the `EW` parent, not `ES` (§5.3) |
| Snapshot time (was item 4) | `15:00` CT, checked against ES contract specs; strike coverage at that minute checked in #5 (§15.4) |
| Realized-variance session (was item 3) | Full Globex. Returns across the maintenance halt, weekends and holidays are kept as one return spanning the gap (gap rule proposed in the README PR, §15.3) |
| Sample period (was item 2) | Oct 2023 to Sep 2026; 5-minute data cost checked before the first pull (roadmap #3, §15.2) |
| Databento access, history, cost; schema names; settlement stat type; data licensing | Verified by the team; the `[VERIFY]` tags were removed from the plan |
| Hedged-gain identity and financing terms | Tag removed from the plan; the identity is documented as the standard Black-Scholes result, distinct from Bakshi-Kapadia's vega-weighted link (§4.5) |
| "Exploring the Variance Risk Premium Across Assets" | Heston and Todorov (2023), in `docs/references/` |
| Snapshot coverage (stage 2) | `bbo-1m` has a record for nearly every live EW option in the 15:00 CT minute; a 15-minute back-fill (`cleaning.backfill_minutes`) covers the rest. Back-month futures (furthest one or two contracts) can be unquoted; their options are dropped as `no_futures_price`. See `docs/stages/02_snapshots.md` |
| Underlying bars and settlements (stage 3) | `ohlcv-1m` and `statistics` (settlement is `stat_type` 3) per underlying contract, $0 on the Plus plan. Bars are made only from minutes that traded, so a gap is one return. Back-month contracts often settle without trading, and holiday sessions trade without a settlement. See `docs/stages/03_underlying.md` |
| Forward for pricing (stage 4) | The chain's `F` is the put-call-parity forward, with the futures mid as fallback; the gap between the two is plotted as the stage diagnostic (median absolute gap about 1 index point on Jul–Sep 2026). `rate_source: fred` is not implemented yet. See `docs/stages/04_chain.md` |
| Degraded-quality days | Databento flags some days as degraded (2024-09-18, 2025-09-17, 2025-09-24 so far); their chains are covered by the chain-quality diagnostics (roadmap 20, `docs/stages/02_snapshots.md`); not flagged by date |
| `strike_price` display-factor issue (was item 1) | ES is not in the list of affected symbols on Databento's issue tracker. The strikes-bracket-futures check stays in `data/universe.py` as a guard (§5.3) |
| Strategy return denominator (was item 5) | Entry vega is primary: trades sized to constant entry vega, P&L per unit vega. Assumed margin fraction is the secondary view; `Π/F` and `Π/O₀` kept for Bakshi-Kapadia comparability only (§4.7, §15.5) |
| Primary VRP definition (was item 6) | The difference `RV - SW` in variance units (Carr-Wu). Log ratio reported alongside for every significance test; swap return used in the strategy section (§4.4, §15.6) |
