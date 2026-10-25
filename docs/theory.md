# Theory

The math behind each stage, expanded from `docs/PLANNING.md` §4. PLANNING.md states the plan; this file
derives it; `docs/stages/` documents how each piece is implemented. Theory changes go through review of
this file.

The variance swap rate (§4) and the hedging identity (§6) are derived step by step, because the two
measures of the premium and their reconciliation (H3) rest on them. The other sections are condensed and
cite their sources. Sections 6–7 and 9 describe stages 6–7 (merged in #33, #43 and #45).

Contents: 1 Setup · 2 Black-76 and the parity forward · 3 Realized variance · 4 The variance swap rate ·
5 The variance risk premium · 6 Delta-hedged gains · 7 Reconciling the two measures (H3) · 8 Inference ·
9 Strategy and costs · 10 From math to code

---

## 1. Setup and notation

| Symbol | Meaning |
|---|---|
| $F_t$ | Price at $t$ of the ES futures contract the option is written on (the `underlying` field, e.g. `ESZ6`) |
| $K$, $T$ | Strike and expiry of the option |
| $\tau = (T-t)/365$ | Time to expiry in years, ACT/365 everywhere |
| $r$ | Continuously compounded risk-free rate (constant, `pricing.rate_constant`) |
| $\sigma_t$ | Instantaneous volatility of $F$; may itself be random |
| $N$, $\phi$ | Standard normal CDF and PDF |
| $C(K)$, $P(K)$ | Today's call and put prices at strike $K$ |

**Dynamics.** We model the futures price as

$$dF_t = \mu_t F_t\,dt + \sigma_t F_t\,dW_t ,$$

which allows stochastic volatility. Section 4 adds jumps as an error term.

**Futures are martingales under $Q$.** A futures position costs nothing to enter and is marked to market
daily, so its risk-neutral expected gain is zero: $\mathbb{E}^Q_t[F_T] = F_t$, i.e. $\mu_t = 0$ under $Q$.
With a deterministic rate, the futures and forward prices coincide. Two consequences are used throughout:
a futures hedge has no financing term, and any $Q$-expectation of $\int dF/F$ is zero.

**Pricing.** A European payoff $g(F_T)$ paid at $T$ is worth $e^{-r\tau}\,\mathbb{E}^Q_t[g(F_T)]$ today.
The option premium is paid upfront, which is the CME convention for EW options.

---

## 2. Black-76 and the parity forward

*Code: `pricing/black76.py`, `pricing/forward.py`, `pricing/chain.py` (stage 4).*

**Price.** If $\sigma$ is constant, $F_T$ is lognormal under $Q$ with mean $F_t$, and

$$C = e^{-r\tau}\big[F N(d_1) - K N(d_2)\big],\qquad P = e^{-r\tau}\big[K N(-d_2) - F N(-d_1)\big],$$

$$d_1 = \frac{\ln(F/K) + \tfrac12\sigma^2\tau}{\sigma\sqrt\tau},\qquad d_2 = d_1 - \sigma\sqrt\tau .$$

This is Black–Scholes with $F$ in place of $Se^{(r-q)\tau}$. It is exact for European exercise, which is
why the project uses the end-of-month (`EW`) series and not the American quarterly options (§15.1).

**Greeks** (with respect to $F$): $\Delta_C = e^{-r\tau}N(d_1)$, $\Delta_P = \Delta_C - e^{-r\tau}$,
$\Gamma = e^{-r\tau}\phi(d_1)/(F\sigma\sqrt\tau)$, $\text{Vega} = Fe^{-r\tau}\phi(d_1)\sqrt\tau$.

**Implied volatility** is the $\sigma$ at which the model price equals the quote, found by Brent's method.
A root exists only inside the no-arbitrage bounds

$$e^{-r\tau}(F-K)^+ \le C \le e^{-r\tau}F,\qquad e^{-r\tau}(K-F)^+ \le P \le e^{-r\tau}K ,$$

so quotes outside them get no implied vol.

**Put–call parity.** $C - P = e^{-r\tau}\,\mathbb{E}^Q[(F_T-K)^+ - (K-F_T)^+] = e^{-r\tau}\,\mathbb{E}^Q[F_T - K]
= e^{-r\tau}(F - K)$, using the martingale property. Solving for $F$:

$$F = K + e^{r\tau}(C - P).$$

Stage 4 applies this at the strike where $|C - P|$ is smallest (the Cboe VIX rule). The choice is about
quote quality, not the formula: an error $\varepsilon$ in $C - P$ moves $F$ by $e^{r\tau}\varepsilon$ at any
strike, but the strike nearest the money has the most liquid, tightest quotes, so $\varepsilon$ is smallest
there. This parity forward is used for pricing instead of the futures mid because the
option and futures quotes are not synchronous: an error in $F$ pushes call and put implied vols in
opposite directions. The futures mid `F_fut` remains the price the hedge trades (§6).

---

## 3. Realized variance

*Code: `vrp/realized.py` (stage 5), bars from `data/underlying.py` (stage 3).*

**Estimator.** With log returns $r_i = \ln(F_{t_i}/F_{t_{i-1}})$ of the option's own futures contract over
$[t, T]$,

$$RV_{t,T} = \frac{1}{\tau}\sum_{i=1}^n r_i^2 .$$

**Why it measures variance.** As sampling gets finer, $\sum r_i^2$ converges to the quadratic variation of
$\ln F$: $\int_t^T \sigma_s^2\,ds$ plus the sum of squared jumps. Dividing by $\tau$ annualizes it on the
same calendar-year basis as implied variance, so $RV$ and $SW$ (§4) are directly comparable.

**Sampling frequency.** Observed prices carry microstructure noise (bid–ask bounce, discreteness). With
i.i.d. noise of variance $\omega^2$ per price, $\mathbb{E}[\sum r_i^2] \approx QV + 2n\omega^2$, which grows with
the number of returns $n$. Daily settlement-to-settlement returns are nearly noise-free but use few
observations; 5-minute returns are the standard compromise. Both are computed and compared (planned
output 3).

**Gaps.** ES trades nearly 23 hours a day (full Globex, §15.3). The return across the daily halt, a
weekend or a holiday is kept as one return spanning the gap. Because each return runs from one bar's close
to the next, the returns telescope: $\sum r_i = \ln(F_{\text{end}}/F_{\text{start}})$, so no gap is dropped or
counted twice. A window the bars do not cover is left empty rather than understated.

**Log vs simple returns.** $(\Delta F/F)^2$ and $(\ln(1+\Delta F/F))^2$ differ at third order in the return;
Carr–Wu use simple returns and report no difference. Log is the baseline, simple a variant.

---

## 4. The variance swap rate

*Code: `vrp/strip.py` (stage 5). Sources: Carr and Wu (2009) Proposition 1 and Eq. (5); Cboe VIX
methodology for the discrete sum.*

### 4.1 What is being priced

A variance swap pays $RV_{t,T} - K_{\text{var}}$ at $T$ and costs nothing to enter. Zero cost means
$e^{-r\tau}\mathbb{E}^Q[RV_{t,T} - K_{\text{var}}] = 0$, so the fair strike, the **swap rate**, is

$$SW_{t,T} = \mathbb{E}^Q_t[RV_{t,T}] .$$

It is the market's risk-neutral forecast of variance. Variance swaps on ES are not in our data, so the rate
is rebuilt from listed options, in four steps.

### 4.2 Step 1: variance as a hedging gain minus a log payoff

Apply Itô's lemma to $\ln F$. With $f(F) = \ln F$, $f' = 1/F$ and $f'' = -1/F^2$:

$$d\ln F_t = \frac{1}{F_t}dF_t + \frac12\Big(-\frac{1}{F_t^2}\Big)(dF_t)^2 .$$

Under the Itô rules $(dW)^2 = dt$ and $dt\,dW = (dt)^2 = 0$, so $(dF_t)^2 = \sigma_t^2F_t^2\,dt$ and

$$d\ln F_t = \frac{dF_t}{F_t} - \frac12\sigma_t^2\,dt .$$

The log return falls short of the simple return by half the variance. Rearranging and integrating from
$t$ to $T$:

$$\int_t^T \sigma_s^2\,ds \;=\; 2\int_t^T \frac{dF_s}{F_s} \;-\; 2\ln\frac{F_T}{F_t}. \tag{1}$$

Total variance equals a continuously rebalanced futures position (holding $2/F_s$ contracts) minus two
log contracts.

### 4.3 Step 2: take risk-neutral expectations

$F$ is a $Q$-martingale, so the first term has zero expectation, and

$$\mathbb{E}^Q_t\Big[\int_t^T \sigma_s^2\,ds\Big] = -2\,\mathbb{E}^Q_t\Big[\ln\frac{F_T}{F_t}\Big]. \tag{2}$$

Pricing variance reduces to pricing the log payoff.

### 4.4 Step 3: span the log payoff with options

**Spanning formula** (Carr–Madan). For any twice-differentiable $f$ and any $a > 0$,

$$f(x) = f(a) + f'(a)(x-a) + \int_0^a f''(K)(K-x)^+\,dK + \int_a^\infty f''(K)(x-K)^+\,dK .$$

*Proof.* For $x \ge a$, Taylor's theorem with integral remainder gives
$f(x) = f(a) + f'(a)(x-a) + \int_a^x f''(K)(x-K)\,dK$. Since $(x-K) = (x-K)^+$ for $K \le x$ and
$(x-K)^+ = 0$ for $K > x$, the last integral equals $\int_a^\infty f''(K)(x-K)^+dK$, and the put integral
is zero because $(K-x)^+ = 0$ for $K \le a \le x$. The case $x < a$ is symmetric. $\square$

Any smooth payoff is therefore a bond, $f'(a)$ futures, and a strip of puts below $a$ and calls above
$a$, each strike held in quantity $f''(K)\,dK$.

**Apply it to the log.** Take $f = \ln$, $a = F_t$, $x = F_T$. Then $f''(K) = -1/K^2$ and

$$\ln\frac{F_T}{F_t} = \frac{F_T - F_t}{F_t} - \int_0^{F_t}\frac{(K-F_T)^+}{K^2}dK - \int_{F_t}^\infty \frac{(F_T-K)^+}{K^2}dK . \tag{3}$$

**The $1/K^2$ weight is the curvature of the log.** Low strikes get more weight, which is why
out-of-the-money puts, and the skew, matter so much for the swap rate.

### 4.5 Step 4: convert to today's option prices

Take $Q$-expectations of (3). The futures term has zero expectation, and
$\mathbb{E}^Q[(K-F_T)^+] = e^{r\tau}P(K)$, $\mathbb{E}^Q[(F_T-K)^+] = e^{r\tau}C(K)$. Substituting into (2)
and dividing by $\tau$:

$$\boxed{\;SW_{t,T} = \frac{2e^{r\tau}}{\tau}\left[\int_0^{F_t}\frac{P(K)}{K^2}dK + \int_{F_t}^\infty \frac{C(K)}{K^2}dK\right]\;} \tag{4}$$

The swap rate is a $1/K^2$-weighted strip of out-of-the-money options. No model of $\sigma_t$ was assumed:
the result holds under any continuous stochastic volatility. This is Carr and Wu's Eq. (5).

### 4.6 Discretization and the $K_0$ correction

Listed strikes are discrete and rarely equal $F$. The Cboe discrete version, used as the baseline
(`strip.method: vix_discrete`), is

$$SW_{t,T} = \frac{2}{\tau}e^{r\tau}\sum_i \frac{\Delta K_i}{K_i^2}Q(K_i) - \frac{1}{\tau}\Big(\frac{F}{K_0} - 1\Big)^2 , \tag{5}$$

with $K_0$ the highest strike at or below $F$; $Q(K_i)$ the put below $K_0$, the call above, and their
average at $K_0$; and $\Delta K_i = (K_{i+1} - K_{i-1})/2$, one-sided at the ends.

**Where the correction comes from.** The sum switches from puts to calls at $K_0$, not at $F$. Write the
integral in (4) as $I$ and the version split at $K_0$ as $I_0$:

$$I - I_0 = \int_{K_0}^{F}\frac{P(K) - C(K)}{K^2}dK = e^{-r\tau}\int_{K_0}^{F}\frac{K - F}{K^2}dK
= e^{-r\tau}\Big[\ln\frac{F}{K_0} - \Big(\frac{F}{K_0} - 1\Big)\Big],$$

using parity, $P - C = e^{-r\tau}(K - F)$. With $x = F/K_0 - 1 \ge 0$, $\ln(1+x) - x \approx -x^2/2$, so
$I \approx I_0 - e^{-r\tau}x^2/2$. Multiplying by $2e^{r\tau}/\tau$ gives the correction
$-\frac{1}{\tau}(F/K_0 - 1)^2$ in (5).

### 4.7 Approximation errors and how each is handled

| Error | Direction | Handling |
|---|---|---|
| **Truncation:** no strikes beyond the quoted range | $SW$ biased down (missing tail options all have positive value) | `truncation_flag` when strikes do not span $F(1\pm 10\%)$; Carr–Wu's extrapolating `iv_grid` variant as robustness (#31) |
| **Discretization:** strikes spaced $\Delta K$ apart | Shrinks as the grid refines | Tested on a flat-vol grid (`test_strip_flat_vol`); `max_strike_gap` recorded |
| **Jumps:** (1) assumes continuous paths | With a jump $J$ in $\ln F$, (1) picks up $2\sum(e^J - 1 - J)$ while realized variance picks up $\sum J^2$. The strip then exceeds $\mathbb{E}^Q[QV]$ by about $\mathbb{E}^Q[\sum J^3]/3$, so downward jumps make it understate. Third order, small in Carr–Wu's calibrations | Stated as a limitation |
| **Holes near the money:** $K_0$ far below $F$ | The correction term dominates; swap rates can turn negative | No swap rate if $(F-K_0)/F > 1\%$ (`strip.max_k0_gap_pct`, #42) |
| **Junk quotes far out of the money** | A single bad quote can dominate the sum | Quotes with implied vol outside 1%–100% are dropped |

### 4.8 Constant 30-day maturity

The baseline matches each expiry's own $\tau$. For comparison with Carr–Wu, a 30-day rate is
interpolated linearly in **total** variance $SW\cdot\tau$ between the two expiries bracketing 30 days
(their Eq. 10), skipping expiries within 8 days:

$$SW_{30} = \frac{1}{\tau}\left[SW_1\tau_1\frac{\tau_2 - \tau}{\tau_2 - \tau_1} + SW_2\tau_2\frac{\tau - \tau_1}{\tau_2 - \tau_1}\right],\qquad \tau = 30/365 .$$

Interpolating total variance, not the rate, keeps the interpolated value free of calendar arbitrage:
total variance must rise with maturity, and a weighted average of two increasing values stays between
them. That guarantee holds only for interpolation. With monthly expiries only, late in a month both valid
expiries can lie beyond 30 days; the same formula then extrapolates (flagged `extrapolated`), and from far
expiries it can give a negative total variance. On the real data this produced four negative swap rates,
so extrapolation is capped (#55).

---

## 5. The variance risk premium

*Code: `vrp/premium.py` (stage 5).*

$$VRP_{t,T} = RV_{t,T} - SW_{t,T} \qquad \text{(primary, variance units; decided §15.6)}$$

It is the payoff per unit notional of a long variance swap: **negative means buyers of variance pay a
premium and sellers earn it.** H1 is $\mathbb{E}[VRP] < 0$.

Two transformations are reported alongside:
- $\ln(RV/SW)$, the log premium. Variance is right-skewed and the difference scales with the level of
  volatility, so a few spikes can dominate its mean; the log is closer to normal and is reported with every
  significance test. Carr–Wu's Sharpe ratio for shorting variance is the mean of $-\ln(RV/SW)$ over its
  Newey–West standard deviation, annualized by $\sqrt{365/30}$.
- $RV/SW - 1$, the excess return of a long variance swap.

**Timing.** $SW_{t,T}$ is known at $t$; $RV_{t,T}$ only at $T$. A premium observation exists only after
expiry, and any conditioning variable (a regime) must use information at $t$ only (§8).

---

## 6. Delta-hedged gains

*Code: `hedging/simulator.py` (merged in #33 and #43). Source: Bakshi and Kapadia
(2003), §1.*

### 6.1 Definition

Buy an option at $t_0$ for $O_0$, financed at $r$. At each rebalance date $t_n$ ($n = 0, \dots, N-1$), short
$\Delta_n$ futures contracts. Hold to $t_N = T$:

$$\Pi_{t_0,T} = O_T - e^{r\tau}O_0 - \sum_{n=0}^{N-1}\Delta_n\,(F_{t_{n+1}} - F_{t_n}). \tag{6}$$

$O_T$ is intrinsic value at expiry. The futures hedge has no financing term because it costs nothing to
hold (§1). A seller earns $-\Pi$. Gains are reported in dollars ($50 per index point), as $\Pi/F_{t_0}$ and
as $\Pi/O_0$, as in Bakshi–Kapadia.

If volatility carries no risk premium and the hedge were continuous with the right volatility, $\Pi$ would
be zero on average. A mean of $\Pi$ below zero means options were priced above the variance that followed.

### 6.2 The gamma-weighted identity

Let $V(F, t)$ be the Black-76 price at a fixed implied volatility $\sigma_i$. It satisfies the Black-76
PDE

$$V_t + \tfrac12\sigma_i^2F^2V_{FF} - rV = 0 . \tag{7}$$

Over a short interval, the hedged and financed position (long $V$, borrowing $V$ at $r$, short
$\Delta = V_F$ futures) changes by

$$dV - rV\,dt - V_F\,dF = V_t\,dt + V_F\,dF + \tfrac12 V_{FF}(dF)^2 - rV\,dt - V_F\,dF .$$

The $dF$ terms cancel; that is the delta hedge. Using (7) to replace $V_t - rV$:

$$= \tfrac12 V_{FF}\big[(dF)^2 - \sigma_i^2F^2\,dt\big] = \tfrac12\Gamma F^2\Big[\Big(\frac{dF}{F}\Big)^2 - \sigma_i^2\,dt\Big].$$

Summing over the daily rebalance intervals, with $(\Delta F/F)^2 \approx r_{n+1}^2$:

$$\Pi \approx \frac12\sum_{n}\Gamma_nF_n^2\big(r_{n+1}^2 - \sigma_i^2\,\Delta t_n\big). \tag{8}$$

The hedged gain is **realized minus implied variance, weighted by dollar gamma** $\tfrac12\Gamma F^2$. The
weight is largest when the futures price is near the strike and expiry is close.

Approximations in (8): interest on interim P&L is ignored (each step's gain should be compounded to $T$,
a factor near 1 at 30 days); returns enter to second order; and daily rebalancing adds path-by-path noise
but no bias on average (`test_hedge_identity`). This identity is the standard Black–Scholes hedging
result, **not** a result from either source paper.

### 6.3 Bakshi and Kapadia's link

Bakshi and Kapadia work under stochastic volatility, where $\sigma$ has its own shock and a market price of
volatility risk $\lambda$. Their result (Eqs. 10–12) is

$$\mathbb{E}[\Pi] = \mathbb{E}\int_t^T \lambda_u\,\frac{\partial C}{\partial\sigma_u}\,du ,$$

so the mean hedged gain is **vega-weighted**, and negative exactly when $\lambda < 0$, since vega is
positive. It predicts the largest effect at the money, where vega peaks, and a larger effect when
volatility is high (H2). Identity (8) and this result weight the premium differently, gamma versus vega,
which is why H3 tests co-movement, not equality.

### 6.4 Hedge-delta choices

| `hedge_delta` | Black-76 delta evaluated at | Role |
|---|---|---|
| `iv_daily` | The option's own implied vol each day | Baseline (what a trader would do) |
| `iv_fixed` | The entry implied vol, held fixed | The case identity (8) describes; used for H3 |
| `rv` | Trailing 21-day realized vol, lagged one day | Closest to Bakshi–Kapadia's GARCH or rolling delta |

With `iv_daily`, the implied vol changes over the life of the trade, so the gain also picks up a vega term
$\approx \text{Vega}\cdot d\sigma_i$; (8) is exact only for `iv_fixed`.

---

## 7. Reconciling the two measures (H3)

*Code: `analysis/reconcile.py` (stage 7, merged in #43; sensitivity and disagreement cases in #54).*

**Same units.** For each chain (one expiry on one date), sum identity (8) over the chain's options $j$ and
divide by their total gamma weight:

$$HG = \frac{\sum_j \Pi_j}{\sum_j W_j},\qquad W_j = \tfrac12\sum_n \Gamma_{j,n}F_n^2\,\Delta t_n .$$

By (8), $HG \approx RV_{gw} - IV^2_{gw}$: a gamma-weighted realized variance minus a gamma-weighted implied
variance, both annualized, the same units as $RV - SW$. Gains are in index points here (dollars / 50).

**Sum, then divide.** Averaging each option's own ratio $\Pi_j/W_j$ is biased upward, since the mean of a
ratio is not the ratio of means and $W_j$ depends on the path. On simulated paths with realized equal to
implied vol (18%), the mean per-option ratio is +0.0014 at daily hedging, while the ratio of sums is
−0.0003 ± 0.0002.

**What to expect.**

$$HG - VRP = (RV_{gw} - RV) - (IV^2_{gw} - SW).$$

- $IV^2_{gw}$ sits near at-the-money implied variance, while $SW$ weights every strike by $1/K^2$ and so
  picks up the put skew. Hence $SW > IV^2_{gw}$ and **HG should be less negative than VRP**. The gap
  $SW - IV^2_{gw}$ is the part of the premium paid for the skew, which a near-the-money hedged option
  does not earn.
- $RV_{gw}$ weights each day's squared return by that day's gamma; $RV$ weights days equally. They
  differ most when the futures price moves far from the strikes.

**Test.** H3 holds if the means share a sign and the two series co-move: sign agreement, and Pearson and
Spearman correlations whose interval excludes zero, on all chains and on the non-overlapping monthly
subsample. The interval on all chains comes from a block bootstrap over dates (21-day blocks, all chains of
a date together), because overlapping windows share most of their returns. On the monthly subsample it uses
Fisher's $z$: $\operatorname{atanh}\hat\rho \pm 1.96/\sqrt{n-3}$, transformed back.

---

## 8. Inference

*Code: `analysis/inference.py`, `analysis/buckets.py` (merged in #33).*

**Why naive errors are wrong here.** Observations start daily and run about 30 days, so neighbouring
premia share most of their returns. With a horizon of $h$ trading days, the errors follow roughly an
MA($h-1$) process, and the naive $s/\sqrt n$ overstates significance. Options on the same date share the
same shock as well.

**Newey–West (HAC).** For the mean $\bar x$ with autocovariances $\hat\gamma_j$,

$$\widehat{\operatorname{Var}}(\bar x) = \frac{1}{n}\Big[\hat\gamma_0 + 2\sum_{j=1}^{L}\Big(1 - \frac{j}{L+1}\Big)\hat\gamma_j\Big].$$

The Bartlett weights keep the estimate positive. Lag $L$ is at least the horizon in trading days minus
one (`inference.hac_lags: auto`), with 30 lags on daily series and 6 on monthly series for comparison with
Carr–Wu. Without a horizon, the code uses the rule of thumb $\lfloor 4(n/100)^{2/9}\rfloor$.

**Clustering** by entry date, for the hedged-gain cross-section:
$\widehat{\operatorname{Var}}(\bar x) = \frac{1}{n^2}\sum_g\big(\sum_{i\in g}(x_i - \bar x)\big)^2$.

**Block bootstrap.** Resample moving blocks of consecutive observations (default length $n^{1/3}$, fixed
seed) to keep the dependence within blocks.

**Non-overlapping subsample.** One observation per expiry gives about 36 independent monthly observations.
It is cleaner but has less power.

The naive standard error is always reported next to the robust ones, to show how much the overlap
inflates significance.

**Regimes without look-ahead.** A date is classified using only data up to that date. With the regime
variable $x_t$ (`sw_level` or trailing 21-day realized vol), its expanding-window percentile is

$$p_t = \frac{1}{t}\sum_{s\le t}\mathbf 1\{x_s \le x_t\},\qquad \text{label}_t = \lceil p_t\cdot n_{\text{bins}}\rceil - 1 ,$$

left empty for the first 20 observations (the first label is at the 21st, `min_periods=21`). Appending later data never changes an earlier label
(`test_regimes_no_lookahead`). Full-sample percentiles would leak future volatility into the buckets.

---

## 9. Strategy, returns and costs

*Code: `analysis/strategy.py` (merged in #43 and #45), `hedging/costs.py`.*

**Trade.** For each end-of-month expiry, on the first snapshot of its month, sell one delta-hedged
straddle (call and put at the strike nearest the forward), hedge daily with the option's own futures
contract, and hold to expiry. Trades never overlap. Short P&L is $-(\Pi_C + \Pi_P)$ at mid; selling at the
bid is the mid gain less the half-spread.

**Costs** (§4.7), per leg:

$$\text{cost} = \underbrace{h\cdot\$50}_{\text{half-spread }h\text{ in points}} + \underbrace{N\cdot\$6.25}_{0.5\text{ tick per rebalance}} + \underbrace{\$1.50\,(1 + N)}_{\text{fees: option and each hedge trade}},$$

with $N$ rebalances. Net P&L is gross minus cost; `cost_multiplier` scales costs for robustness. Slippage is
one contract per rebalance whatever the change in delta (#41 proposes charging on the change in delta). Each
leg of a straddle is also hedged on its own, so slippage is paid on two contracts even when the legs' deltas
partly cancel; hedging the net delta (#57) would lower costs and leave gross P&L unchanged.

**Returns per unit vega** (primary, §15.5). Divide P&L by the straddle's dollar vega per vol point,
$(\text{Vega}_C + \text{Vega}_P)\times\$50\times 0.01$. To first order a hedged short option earns
$\text{Vega}\cdot(\sigma_i - \sigma_{\text{realized}})$, so this return is roughly implied minus realized vol
in vol points: comparable across strikes, maturities and with the VRP. Notional is not used because a
delta-hedged position has no net exposure to the futures price.

**Return on margin** (secondary): P&L over an assumed 10% of the two legs' futures notional
($0.10\times 2F\times\$50$). This is an assumption, since actual margin is not in the data.

**Risk metrics** per monthly trade: mean with naive and HAC (lag 6) $t$; annualized Sharpe
$\bar x/s\cdot\sqrt{12}$; skew; share of winning months; worst months; maximum drawdown of cumulative P&L;
CVaR, the mean of the worst 5% of months. A short volatility position has negative skew, so the average can
hide rare large losses; the tail metrics are reported, not averaged away.

---

## 10. From math to code

| Section | Quantity | Stage | Module | Output |
|---|---|---|---|---|
| 2 | Parity forward $F$, implied vol, Greeks | 4 | `pricing/forward.py`, `black76.py`, `chain.py` | `data/chain.parquet` |
| 3 | $RV$ (5-minute and daily) | 3 → 5 | `data/underlying.py`, `vrp/realized.py` | `data/rv.parquet` |
| 4 | $SW$ (strip, eq. 5) and diagnostics | 5 | `vrp/strip.py` | `data/strip.parquet` |
| 4.8 | Constant 30-day $SW$, $RV$ | 5 | `vrp/premium.py` | `data/*_cm.parquet` |
| 5 | $VRP$, log premium, swap return | 5 | `vrp/premium.py` | `data/vrp.parquet`, `reports/tables/vrp_descriptives.csv` |
| 6 | $\Pi$, gamma weight $W$ | 6 | `hedging/simulator.py` | `data/hedged_gains.parquet` |
| 7 | $HG$, H3 statistics | 7 | `analysis/reconcile.py` | `data/h3.parquet`, `reports/tables/h3.csv` |
| 8 | Naive, HAC, clustered, bootstrap SEs; regimes | 7 | `analysis/inference.py`, `buckets.py` | used by stage 7 tables |
| 9 | Strategy P&L, costs, risk metrics | 7 | `analysis/strategy.py`, `hedging/costs.py` | `reports/tables/strategy.csv` |

## References

Carr and Wu (2009); Bakshi and Kapadia (2003); Black (1976); Newey and West (1987); Cboe VIX methodology
(`docs/references/`). Full citations in `docs/PLANNING.md` §16.
