# Data notes

Source: Databento `GLBX.MDP3`. See `docs/PLANNING.md` §5 for schemas, known issues and cost control.
Raw Databento data must never be committed (`data/` and `*.parquet` are gitignored); use
`src/esvrp/synthetic.py` for tests and demos.

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
| Degraded-quality days | Databento flags some days as degraded (2024-09-18, 2025-09-17, 2025-09-24 so far); to be surfaced by the chain-quality diagnostics (roadmap 20) |
| `strike_price` display-factor issue (was item 1) | ES is not in the list of affected symbols on Databento's issue tracker. The strikes-bracket-futures check stays in `data/universe.py` as a guard (§5.3) |
| Strategy return denominator (was item 5) | Entry vega is primary: trades sized to constant entry vega, P&L per unit vega. Assumed margin fraction is the secondary view; `Π/F` and `Π/O₀` kept for Bakshi-Kapadia comparability only (§4.7, §15.5) |
| Primary VRP definition (was item 6) | The difference `RV - SW` in variance units (Carr-Wu). Log ratio reported alongside for every significance test; swap return used in the strategy section (§4.4, §15.6) |
