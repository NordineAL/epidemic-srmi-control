# Epidemic Modelling & Control — SRMI Model

Stochastic network + deterministic ODE modelling of infectious-disease spread
in a city, with control strategies that bring the basic reproduction number
**R₀ below 1**.

![SRMI cases](outputs/srmi_cases.png)

## The model

Four compartments — **S** (susceptible), **I** (infected), **R** (recovered),
**M** (deceased) — with three control levers in [0, 1]:

| Control | Meaning |
|---|---|
| `u1` | Confinement / safety measures (reduces contacts) |
| `u2` | Treatment & care of the infected (boosts recovery) |
| `u3` | Prevention of deaths (critical-case intervention) |

```
dS/dt = -(1-u1)·β·S·I
dI/dt =  (1-u1)·β·S·I - (1-u3)·μ·I - (1+u2)·γ·I
dR/dt =  (1+u2)·γ·I
dM/dt =  (1-u3)·μ·I
```

Next-generation-matrix result:

```
R0 = (1-u1)·β·S* / ((1+u2)·γ + (1-u3)·μ)
```

If **R₀ < 1**, the disease dies out; if **R₀ ≥ 1**, it stays endemic.

Two layers are implemented:

1. **ODE layer** (`src/epidemic_srmi/ode.py`) — RK4 integration, population
   conservation guaranteed, R₀ computation.
2. **Stochastic network layer** (`src/epidemic_srmi/network.py`) — spread on a
   contact graph (Erdős–Rényi or Barabási–Albert). A susceptible node with
   infected neighbours is infected with probability `1 - exp(-τ·k)`, the
   discrete form of `T_ij(t) = 1 - exp(-τ_ij·t)`; infected nodes recover or
   die after a characteristic infectious period.

`control.py` grid-searches `(u1, u2, u3)` to find which combinations control
the epidemic, and computes the minimal confinement level that alone brings
R₀ below 1.

## Results

Population 110,000 (10,000 initially infected), β = 7·10⁻⁸, γ = 5·10⁻⁴,
μ = 2·10⁻⁴:

| Scenario | Controls | R₀ | Outcome |
|---|---|---|---|
| Case 1 — no control | 0, 0, 0 | **10.0** | Epidemic takes off |
| Case 2 — controlled | 0.88, 0.70, 0.65 | **0.91** | Epidemic dies out |

## Quickstart

```bash
pip install -r requirements.txt
python scripts/run_all.py   # CSV + figures in outputs/
pytest tests/               # conservation, R0 cases, network counts
```

## About

TIPE project — Modelling and Control of Infectious Diseases in a City (2023),
rebuilt here as a runnable Python portfolio project: stochastic simulation,
ODE integration, control analysis and testing.

## Skills demonstrated

Python · NumPy · Matplotlib · stochastic simulation · ODE integration (RK4) ·
network science · control theory · parameter sweeps · testing (pytest)
