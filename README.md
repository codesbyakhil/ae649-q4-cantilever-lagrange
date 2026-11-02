# Q4 plane-stress cantilever with a Lagrange-multiplier clamp

AE 649 course project, Akhil S. Pillai, Department of Aerospace Engineering, IIT Bombay.

A 2D finite-element code (four-node Q4 elements, 2×2 Gauss quadrature) for a plane-stress cantilever
whose clamp is imposed by **Lagrange multipliers**. Every numerical result is validated against an
analytical counterpart, and every pairing is an automated test. The full theory, derivations and results
are in [`THEORY.md`](THEORY.md); the presentation is [`slides/presentation.pdf`](slides/presentation.pdf).

**Problem:** L = 2 m, D = 0.15 m, t = 0.1 m, E = 200 GPa, ν = 0.3, plane stress; clamped at x = 0
(u = v = 0 at every left-edge node); total end load P = −10 N as a uniform shear traction on x = L.

## 1. Setup and running

Requires Python 3.10+ (tested with 3.12). From the repository root:

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
```

| Command | What it does |
|---|---|
| `python -m pytest -q` | runs the 109 validation tests (about 10 s) |
| `python run_all.py` | regenerates every figure and table in `figures/` (about 10 s) |
| `python -m studies.tip_deflection` | runs one study (likewise any file in `studies/`) |
| `python -m checks.derivations` | symbolic (sympy) proofs of every closed form used |
| `python -m checks.beam_theory` | prints the beam-theory reference values |
| `python -m bar1d.bar` | runs the 1D bar warm-up |

**Slides:** from `slides/`, run `pdflatex presentation.tex` twice (needs a LaTeX distribution with beamer,
TikZ and siunitx; figures are read from `../figures/`). The IIT Bombay logo is not included in this
repository; to show it, save the institute logo as `slides/iitb_logo.png`. Without it the slides compile
without the logo.

## 2. Validation plan

| # | The code computes | Analytical counterpart | Measured agreement |
|---|---|---|---|
| V1 | 1D bar, both ends prescribed (Lagrange) | u = qx(L − x)/(2EA) + Ux/L; support forces from N = EAu′ | 4.8e-15; −λ = −700 N, −300 N |
| V2 | Q4 stiffness, 2×2 Gauss | exact integration for a rectangle | 2.3e-16; 3 zero eigenvalues |
| V3 | uniform tension, regular and distorted mesh | u = σ₀x/E, v = −νσ₀y/E (patch test) | 1.1e-13 |
| V4 | Q4 in pure bending | Cook Eq. 3.6-11 | 8.6e-13 |
| V5 | one-element cantilever | complete hand solution | 1.5e-13 |
| V6 | full problem, meshes 3×1 … 384×128 | statics, Clapeyron, minimum potential energy, beam theory, O(h²) | round-off; converged tip deflection −4.75532 µm = 0.99869 × Timoshenko |

Agreement = largest relative difference between code and formula; the numbers come from `figures/*.txt`.

## 3. Folder map

```
FEM_Project/
├── fem/            the finite-element code itself (the "solver")
├── bar1d/          1D warm-up problem (V1), uses the same Lagrange solver
├── checks/         analytical counterparts: formulas, beam theory, symbolic proofs
├── studies/        one script per figure/table → writes to figures/
├── tests/          automated tests (pytest) pairing code with formulas
├── figures/        generated outputs (.png plots, .txt tables)
├── slides/         Beamer presentation
├── run_all.py      runs every study
├── THEORY.md       all derivations + results, section by section
├── requirements.txt, pytest.ini, .gitignore
```

The rule of the layout: **`fem/` computes, `checks/` predicts, `tests/` compares the two automatically,
`studies/` compares them on screen.** `checks/` never imports the solver, so the analytical side is independent.

---

## 4. How the files connect

```
                    fem/material.py ─┐
                    fem/q4.py ───────┼─► fem/assemble.py ─┐
                    fem/mesh.py ─────┘                     │
                    fem/loads.py ──────────────────────────┤
                    fem/lagrange.py ───────────────────────┼─► fem/cantilever.py ─► fem/post.py
                         ▲                                 │      (solve_cantilever)
                         │                                 │
bar1d/bar.py ────────────┘ (V1 uses the same solver)       │
                                                           ▼
checks/closed_form.py, checks/beam_theory.py ──► tests/*.py and studies/*.py ◄── fem/*
checks/derivations.py ──► tests/test_derivations.py (proves the formulas in checks/ are right)
studies/*.py ──► figures/*.png, *.txt ──► THEORY.md and slides/presentation.tex
run_all.py ──► calls every studies/*.py
```

**One solve, step by step** (`fem/cantilever.py:solve_cantilever(nx, ny)`):

1. `mesh.structured_mesh` → node coordinates and element connectivity (4 nodes each, counter-clockwise)
2. `material.plane_stress_D` → the 3×3 matrix D
3. `assemble.assemble_stiffness` → calls `q4.element_stiffness` for all elements, sums them into the sparse K
4. `loads.edge_traction_load` → load vector f (P/ny per interior end node, P/2ny at the corners)
5. `mesh.edge_nodes` + `lagrange.constraint_matrix` → the clamp matrix C (one row per clamped DOF)
6. `lagrange.solve_lagrange` → solves [[K, sCᵀ],[sC, 0]] [d; λ] = [f; sg], returns d, λ, R = −Cᵀλ
7. `post.py` → tip deflection v(L,0), Gauss-point stresses, resultants, convergence order

---

## 5. Every file

### `fem/` — the solver

| File | What it does | Main functions |
|---|---|---|
| `material.py` | plane-stress constitutive matrix D (from 3D Hooke's law with σz = 0) | `plane_stress_D` |
| `q4.py` | the Q4 element: shape functions, Jacobian, B matrix, element stiffness by Gauss quadrature. Works on one element or a stack of all elements at once (same formulas, faster) | `shape_functions`, `shape_derivatives`, `jacobian`, `strain_displacement`, `gauss_rule`, `element_stiffness` |
| `mesh.py` | structured nx × ny grid; nodes on a vertical line; a randomly distorted copy for the patch test | `structured_mesh`, `edge_nodes`, `distort_interior` |
| `assemble.py` | DOF numbering (node n → 2n, 2n+1) and global sparse K (COO triplets → CSR, duplicates summed) | `element_dofs`, `assemble_stiffness` |
| `loads.py` | consistent nodal loads of a uniform edge traction (t h τ̄ / 2 per edge end) | `edge_traction_load` |
| `lagrange.py` | **the method of the project**: constraint matrix C, the scaled saddle-point system, sparse LU solve, reactions R = −Cᵀλ | `constraint_matrix`, `augmented_matrix`, `solve_lagrange` |
| `cantilever.py` | problem data (L, D, t, E, ν, P) and the full solve | `clamp_dofs`, `solve_cantilever` |
| `post.py` | Gauss-point stresses, tip deflection, force/moment resultants, observed order, Richardson extrapolation | `gauss_point_stresses`, `tip_deflection`, `force_resultants`, `observed_order`, `richardson` |

### `bar1d/` — V1 warm-up

| File | What it does |
|---|---|
| `bar.py` | 1D bar EAu″ + q = 0 with u(0) = 0, u(L) = U: two-node elements, consistent loads, ends imposed with `fem/lagrange.py`; exact solution and support forces. Runnable: `python -m bar1d.bar` |

### `checks/` — analytical counterparts (never call the solver)

| File | What it does |
|---|---|
| `beam_theory.py` | I = tD³/12, Euler–Bernoulli and Timoshenko tip deflection, bending stress −P(L−x)y/I, shear stress, pure-bending rotation. Runnable: prints −4.74074 / −4.76154 µm |
| `closed_form.py` | hand-derived formulas: exact 8×8 stiffness of a rectangle (V2), Cook Eq. 3.6-11 (V4), one-element cantilever solution (V5), uniform-tension field (V3) |
| `derivations.py` | sympy derivations of every formula above from first principles (plus Lagrange = elimination, scaling, Richardson). Runnable: prints "proved" for each |

### `tests/` — automated validation (`python -m pytest -q`)

| File | Validates |
|---|---|
| `test_bar1d.py` | V1: nodal values exact, −λ = support forces, Lagrange = elimination |
| `test_q4.py` | V2 and V4 (one element): shape functions, Jacobian, all 64 stiffness entries vs exact, 3 rigid modes, constant strain, Cook's ratio |
| `test_exact_field.py` | V3: uniform tension exact on regular and distorted meshes, also with non-zero prescribed values |
| `test_one_element.py` | V5: 1×1 mesh equals the hand solution and statics |
| `test_cantilever.py` | V4 strips and V6 (a–c): loads, clamp, statics, Clapeyron, monotone compliance, elimination check, matrix inertia and conditioning |
| `test_derivations.py` | every formula in `checks/` equals its symbolic derivation |

### `studies/` — figures and tables (`python run_all.py`)

| File | Output in `figures/` | Validation |
|---|---|---|
| `common.py` | — (shared colours, plot style, mesh list, table writer) | — |
| `bar1d.py` | `bar1d.png/.txt` | V1 |
| `element_checks.py` | `element_checks.png/.txt` | V2 |
| `exact_field.py` | `exact_field.png/.txt`, `exact_field_scaling.png` | V3 + scaling, condition numbers |
| `bending_error.py` | `bending_error.png/.txt` | V4 |
| `one_element.py` | `one_element.png/.txt` | V5 |
| `tip_deflection.py` | `tip_deflection.png/.txt`, `convergence_rate.png` | V6 (c–e) |
| `midspan_stress.py` | `midspan_stress.png/.txt` | V6 (d) stresses |
| `reactions.py` | `reactions.png/.txt` | V6 (a, b) statics, reaction distribution |

`run_all.py` imports and runs all eight in order.

### Documents and configuration

| File | Purpose |
|---|---|
| `THEORY.md` | every equation the code uses, derived in order, with the measured results and a code map |
| `slides/presentation.tex` / `.pdf` | Beamer deck (compile from `slides/` with `pdflatex presentation.tex`, twice) |
| `requirements.txt` | Python packages needed |
| `pytest.ini` | tells pytest to look in `tests/` and to put the project root on the import path |
| `.gitignore` | keeps caches and LaTeX build files out of git |
