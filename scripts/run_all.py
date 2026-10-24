"""Reproduce the two TIPE scenarios and a stochastic network run, saving CSVs + figures.

Case 1: no control  -> R0 = 10   (epidemic takes off)
Case 2: u1=0.88, u2=0.70, u3=0.65 -> R0 = 0.91 (epidemic controlled)

Run from the repo root:  python scripts/run_all.py
Outputs: outputs/srmi_cases.csv, outputs/srmi_cases.png, outputs/network.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from epidemic_srmi import srmi_ode, reproduction_number, ContactNetwork, run_network_simulation

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
BETA, GAMMA, MU = 7e-8, 5e-4, 2e-4
S0, I0 = 100_000, 10_000

# --- ODE cases ---------------------------------------------------------------
cases = {
    "case1_no_control": dict(u1=0.0, u2=0.0, u3=0.0),
    "case2_controlled": dict(u1=0.88, u2=0.70, u3=0.65),
}
rows = []
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (name, controls) in zip(axes, cases.items()):
    r0 = reproduction_number(BETA, GAMMA, MU, S0, **controls)
    t, traj = srmi_ode(S0, I0, beta=BETA, gamma=GAMMA, mu=MU, days=300, **controls)
    print(f"{name}: R0 = {r0:.2f}, peak infected = {traj[:, 1].max():,.0f}, "
          f"final R = {traj[-1, 2]:,.0f}, final M = {traj[-1, 3]:,.0f}")
    for ti, (s, i, r, m) in zip(t[::10], traj[::10]):
        rows.append({"case": name, "day": ti, "S": s, "I": i, "R": r, "M": m})
    for col, label in [(1, "I"), (2, "R"), (3, "M")]:
        ax.plot(t, traj[:, col], label=label)
    ax.set_title(f"{name}\nR0 = {r0:.2f}")
    ax.set_xlabel("days"); ax.legend()
axes[0].set_ylabel("people")
fig.tight_layout(); fig.savefig(OUT / "srmi_cases.png", dpi=150)
import csv
with open(OUT / "srmi_cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["case", "day", "S", "I", "R", "M"]); w.writeheader(); w.writerows(rows)

# --- Stochastic network (smaller demo population, same mechanics) ------------
net = ContactNetwork(3000, avg_degree=6, model="barabasi_albert", seed=42)
print(f"network: {net.n_nodes} nodes, mean degree {net.mean_degree():.2f}")
hist = run_network_simulation(net, initial_infected=range(30), tau=0.08,
                              infectious_period=14, death_prob=0.02, steps=120, seed=42)
plt.figure(figsize=(8, 4))
for k, v in hist.items():
    plt.plot(v, label=k)
plt.xlabel("steps"); plt.ylabel("people"); plt.legend(); plt.tight_layout()
plt.savefig(OUT / "network.png", dpi=150)
print(f"network final: S={hist['S'][-1]}, I={hist['I'][-1]}, R={hist['R'][-1]}, M={hist['M'][-1]}")
print(f"done. Figures + CSV in {OUT}")
