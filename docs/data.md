# Data notes

Source: Databento `GLBX.MDP3`. See `docs/PLANNING.md` §5 for schemas, known issues and cost control.
Raw Databento data must never be committed (`data/` and `*.parquet` are gitignored); use
`src/esvrp/synthetic.py` for tests and demos.

## Open verification items

Items marked **[VERIFY]** or **[DECISION]** in `docs/PLANNING.md` that are still unresolved, plus the open
decisions in §15. Each has a `TODO` comment at the point of use in the code.

| # | Type | Item | Plan § | Where in code |
|---|---|---|---|---|
| 2 | DECISION | Sample period; whether a longer daily-only sample exists | 15.2 | `config/default.yaml` |
| 3 | DECISION | Session for realized variance (full Globex vs RTH) and maintenance-gap treatment | 4.2, 15.3 | `data/underlying.py`, `vrp/realized.py`, `config.py` |
| 4 | DECISION | Snapshot time and ES settlement-time convention (config currently `15:00` CT) | 15.4 | `data/quotes.py`, `config.py` |
| 7 | DECISION | Whether the CL/EIA extension is in scope | 15.7 | issue #20 |

## Resolved since the first draft

| Item | Resolution |
|---|---|
| ES option series | `monthly_eom`; analysis restricted to European options (§4.1, §15.1 still lists it) |
| Exercise style per series | Follows from the series choice above |
| Parent symbols | Underlying futures: `ES.FUT` (`data.product`). End-of-month options are under the root `EW`, so the `monthly_eom` option universe is pulled with the `EW` parent, not `ES` (§5.3) |
| Snapshot cutoff | `15:00` CT in config (convention still open, item 4) |
| Databento access, history, cost; schema names; settlement stat type; data licensing | Verified by the team; the `[VERIFY]` tags were removed from the plan |
| Hedged-gain identity and financing terms | Tag removed from the plan; the identity is documented as the standard Black-Scholes result, distinct from Bakshi-Kapadia's vega-weighted link (§4.5) |
| "Exploring the Variance Risk Premium Across Assets" | Heston and Todorov (2023), in `docs/references/` |
| `strike_price` display-factor issue (was item 1) | ES is not in the list of affected symbols on Databento's issue tracker. The strikes-bracket-futures check stays in `data/universe.py` as a guard (§5.3) |
| Strategy return denominator (was item 5) | Entry vega is primary: trades sized to constant entry vega, P&L per unit vega. Assumed margin fraction is the secondary view; `Π/F` and `Π/O₀` kept for Bakshi-Kapadia comparability only (§4.7, §15.5) |
| Primary VRP definition (was item 6) | The difference `RV - SW` in variance units (Carr-Wu). Log ratio reported alongside for every significance test; swap return used in the strategy section (§4.4, §15.6) |
