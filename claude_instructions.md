# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This folder contains a subfolder @Paper with the draft of a paper (`Paper/MultiInsurance.tex`) that adapts the
primal-dual algorithm described in @CDRT_JMathE_2024.pdf (Carlier–Dupuis–Rochet–Thanassoulis) to a model of
insurance with CARA insurees.

It also has a subfolder @Code (`Code/multidim-screening-plain/`) that contains my implementation of the adapted algorithm.

Standing tasks:

1. Check that all formulae in my draft paper are correct.
2. Check that my code is correct.

When checking, the paper's formulae (especially the Appendix derivations of the integral `I`, and the TPAC
model in the "Insurance" section) should match what the code in `insurance/*/..._values.py` computes.

Other folders: `References/` (literature PDFs), `Figures/` (PNGs used by the paper via `\graphicspath{{../Figures/}}`),
`SlidesSAET2024/` (older slides), `Meetings/` (meeting transcripts/summaries), `ag-competition-master/`
(third-party Azevedo–Gottlieb MATLAB code, reference only).

## Paper

Compile from inside `Paper/` (figure paths are relative): `pdflatex MultiInsurance && bibtex MultiInsurance && pdflatex MultiInsurance`
(uses the local `econometrica.bst`).

## Code (`Code/multidim-screening-plain/`)

uv project (`pyproject.toml` + `uv.lock`, hatchling backend), Python >=3.13 (`.python-version` pins 3.13 for
the local `.venv`; CI also uses 3.13). The `dev` and `docs` dependency groups are installed by default. Run all commands from
`Code/multidim-screening-plain/`, because paths are resolved against the current directory
(`results_dir = Path.cwd() / "results"`, configs under `Path.cwd() / "multidim_screening_plain"`).

- Install: `make install` (`uv sync` + pre-commit hooks); add a dependency with `uv add <pkg>` (`--group dev` for dev tools)
- Run a model: `uv run python -m multidim_screening_plain.main <model_type> <model_instance>`
  e.g. `... main insurance d2_m2`, `... main mussarosen d1_m1`, `... main jointtax d2_m2`
- Lint/type check: `make check` (`uv lock --locked`, ruff + black via pre-commit, then mypy)
- Tests: `make test`; single test: `uv run pytest tests/test_main.py::test_contracts_vector`
- Docs: `make docs` (serve) / `make docs-test` (strict build)

Tests: `tests/test_main.py` covers `utils.py`; `tests/test_<type>_<instance>.py` check analytic gradients of
`b_fun`, `S_fun` and the `val_*` functions against finite differences. Build a model in a test with the
`build_model("insurance", "d2_m2")` fixture from `tests/conftest.py`; type values used in tests must lie in the
config's ranges (e.g. risk location δ ∈ [4, 8]). Every model instance has a test file (`test_mussarosen.py` covers
both Mussa–Rosen instances). In the all-types mode, `y` is laid out by dimension (`split_y`), so the gradient of
the sum is `np.sum(grad, axis=1).reshape(-1)`. `obsolete/` holds superseded model files. `codeflow.png` is a stale call graph.

### Architecture

Generic solver + pluggable model modules.

- `main.py`: reads `multidim_screening_plain/<type>/<instance>/config_<type>_<instance>.env` (all values are quoted
  strings: type grid, parameters, contract bounds, `DO_FIRST_BEST`, `DO_SOLVE`, `START_FROM_FIRST_BEST`, `DO_PLOTS`,
  `TOLERANCE`, `MAX_ITERATIONS`, `FIX_TOP`), then first best → second best → plots.
- `setup.py`: builds a `ScreeningModel` (uniform type grid `theta_mat` of shape `(N, d)`, `N = prod(grid sizes)`,
  weights `f`) and dynamically imports `multidim_screening_plain.<type>.<instance>.<type>_<instance>`.
- `solver.py`: the CDRT primal-dual (Chambolle–Pock style) loop in `solve()`. Primal step `prox_minusS` (uses the
  model's `proximal_operator`); dual step `proj_K` (projection onto the IC/IR cone, iterative with warm start);
  `nlLambda` computes `Lambda_ij(y) = b_i(y_j) - b_j(y_j)` and `JLambda` its Jacobian. Stops when primal residual,
  dual residual, and projection convergence all hold. Writes `current_y.txt`/`current_v.txt` every 100 iterations.
- `classes.py`: `ScreeningModel` and `ScreeningResults` (`output_results` writes CSVs/`IC_binds.txt`/`IR_binds.txt`
  to `results/<type>/<instance>/<type>_<instance>_<grid>/`).
- Contract layout convention: `y` is a flat vector of length `m*N`; `utils.contracts_vector`/`contracts_matrix`
  convert to/from the `(m, N)` matrix (interleaved by type).

Each model module must provide: `b_fun(model, y, theta=None, gr=False)` (value to type i of contract j for all
pairs, plus derivatives), `S_fun(model, y, theta, gr=False)` (joint surplus and gradient), `create_initial_contracts`,
`proximal_operator`, `adjust_excluded`, `add_results`, `add_plots`. Insurance models additionally keep the heavy
math (CARA expected utility integrals `val_A`, `val_BC`, `val_D`, `val_I`, truncated-normal losses) in
`<name>_values.py` and plots in `<name>_plots.py`. Gradients are validated in tests with
`bs_python_utils.bs_opt.check_gradient_scalar_function`.

Model instances: `insurance/{d1_m1_risk_copay, d1_m1_risk_deduc, d2_m1_copay, d2_m1_deduc, d2_m2,
twopoint_d1_m1_risk_deduc}` (d = type dimension, m = contract dimension; d2_m2 types are risk aversion × risk
location, contracts are deductible × copay), plus `mussarosen/` and `jointtax/` used to validate against the two
examples in CDRT.
