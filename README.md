# multidim-screening-plain

<!-- [![Release](https://img.shields.io/github/v/release/bsalanie/multidim-screening-plain)](https://img.shields.io/github/v/release/bsalanie/multidim-screening-plain) -->
[![Build status](https://img.shields.io/github/actions/workflow/status/bsalanie/multidim-screening-plain/main.yml?branch=main)](https://github.com/bsalanie/multidim-screening-plain/actions/workflows/main.yml?query=branch%3Amain)
<!-- [![codecov](https://codecov.io/gh/bsalanie/multidim-screening-plain/branch/main/graph/badge.svg)](https://codecov.io/gh/bsalanie/multidim-screening-plain) -->
[![Commit activity](https://img.shields.io/github/commit-activity/m/bsalanie/multidim-screening-plain)](https://img.shields.io/github/commit-activity/m/bsalanie/multidim-screening-plain)
[![License](https://img.shields.io/github/license/bsalanie/multidim-screening-plain)](https://img.shields.io/github/license/bsalanie/multidim-screening-plain)

Multidimensional screening without Numba

- **Github repository**: <https://github.com/bsalanie/multidim-screening-plain/>
- **Documentation** <https://bsalanie.github.io/multidim-screening-plain/>


## TO DO
- optimize the code.

## Convergence criteria and tolerances

The solver stops when two residuals, one for the contracts and one for the multipliers, are
both below a tolerance computed from the single `TOLERANCE` setting of the configuration file.
Both residuals measure how far the current iterate is from satisfying the optimality conditions.
The projection must also have converged, but that holds automatically.
Everything is in `solve()` in `solver.py`.

### The step sizes

$$\tau = \frac{1}{\sqrt{r}\,\|\Lambda'\|\,\mu}, \qquad \sigma = \frac{\sqrt{r}}{\|\Lambda'\|\,\mu}$$

- $r$ is `STEP_RATIO` (from the configuration file, 1.0 by default) and $\mu$ is `mult_fac` (1 by default).
- $\|\Lambda'\|$ is the largest singular value of the Jacobian of $\Lambda$, computed once at the
  starting contracts. For `insurance d2_m2` on a 10x10 grid it is about 324.
- So $\tau\sigma\|\Lambda'\|^2 = 1/\mu^2$, which is the usual step-size condition of the
  Chambolle–Pock method. Only the ratio $\sigma/\tau = r$ is free.
- For example, at $r = 1{,}000$ in `insurance d2_m2`: $\tau \approx 9.8\times10^{-5}$ and $\sigma \approx 0.098$.

### The residuals

The iteration targets the saddle-point conditions

$$0 \in -\nabla S(y) + \Lambda'(y)^{*} v + N_Y(y), \qquad \Lambda(y) \in N_K(v).$$

Here $N_Y$ is the normal cone of the box of allowed contracts, and $N_K$ is the normal cone of
$K$, the set of allowed multipliers ($v \ge 0$, $D^*v \le f$).
After each iteration the solver computes:

**Primal residual** (on the contracts, a vector of length $mN$):

$$p_k = \Big\| \frac{y_{k-1} - y_k}{\tau} \;-\; \big(\Lambda'(y_{k-1})^{*} v_{k-1} - \Lambda'(y_k)^{*} v_k\big) \Big\|_2$$

**Dual residual** (on the multipliers, a vector of length $N^2$):

$$d_k = \Big\| \frac{v_{k-1} - v_k}{\sigma} \;+\; \big(\Lambda(\bar y_k) - \Lambda(y_k)\big) \Big\|_2$$

In each case the first term divides the step by its step size. That turns it into a
gradient-like quantity, in the units of $\nabla S$ for $p$ and of $\Lambda$ for $d$.
The second term corrects for the multipliers (or the contracts) having moved in the same
iteration. Both residuals are zero at a fixed point, and $p_k$ is the error in the first-order
condition for the contracts.

With a larger step ratio, a given movement of the contracts gives a larger primal residual,
because it is divided by a smaller $\tau$. But the contracts also move proportionally less per
iteration, so $p$ stays in the units of $\nabla S$ and can be compared across step ratios as a
measure of optimality.

### The tolerances

`main.py` passes `TOLERANCE` as both the primal and the dual tolerance, with `scale=True`.
That multiplies each by the square root of the length of its vector:

| | Formula | `insurance d2_m2`, 10x10, `TOLERANCE = 1e-5` |
|---|---|---|
| Primal tolerance | $\sqrt{mN}\times$ `TOLERANCE` | $\sqrt{200}\times10^{-5} = 1.4\times10^{-4}$ |
| Dual tolerance | $\sqrt{N^2}\times$ `TOLERANCE` | $100\times10^{-5} = 10^{-3}$ |

Scaling by $\sqrt{\text{length}}$ turns the test into one on the root-mean-square entry: on
average, each component of the first-order condition must be off by less than `TOLERANCE`.

The solver stops when all three of these hold:

1. $p_k <$ the primal tolerance;
2. $d_k <$ the dual tolerance;
3. the projection onto $K$ converged.

Condition 3 is always true when checked, because `proj_K` aborts the program if it fails to
converge in 100,000 inner iterations. Its own tolerances, `rtol_proj = 1e-4` and
`atol_proj = 1e-6`, apply to the constraint $D^*v \le f$ and to complementary slackness.

The "primal" and "dual" numbers in the progress output and in `log.txt` are these residuals
divided by their tolerances. So "primal residual = 1.08e+03 tol" means $p$ is 1,080 times
the tolerance.

**An example.** After 40,000 iterations of `insurance d2_m2` (10x10) at step ratio 1,000,
$p \approx 0.134$, or about 0.0095 per component. The first-order conditions were off by
about 0.01 units of surplus per unit of contract, while `TOLERANCE = 1e-5` asks for 1,000 times
less. That is very strict next to how precisely the economic quantities are pinned down: the
profit was already stable to about 0.1.

### Other thresholds in the code

| Quantity | Threshold | Where |
|---|---|---|
| IC constraint counted as binding (for `IC_binds.txt`) | $v_{ij} >$ the dual tolerance | `solve()` |
| IR constraint counted as binding | $U_i < 10^{-4}$ | `compute_utilities` |
| Rent fixed point | change $< 10^{-6}$, at most $N$ sweeps | `compute_utilities` |
| Type counted as excluded | copay > 0.999 (twopoint and the deductible models: deductible within 0.001 of its maximum; Mussa–Rosen: SB/FB surplus < 0.001) | `adjust_excluded` |
| Warning that excluded types have rents | rent > 1e-4 | `warn_if_excluded_have_rents` |
| Batched proximal step | step in units of $y$ < 1e-9 | `prox_box_batch` |
| Per-type proximal step | L-BFGS-B defaults | `proximal_operator` |
