import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from epidemic_srmi import srmi_ode, reproduction_number, ContactNetwork, run_network_simulation, control_sweep

BETA, GAMMA, MU = 7e-8, 5e-4, 2e-4

def test_population_is_conserved():
    _, traj = srmi_ode(100_000, 10_000, beta=BETA, gamma=GAMMA, mu=MU, days=100)
    assert np.allclose(traj.sum(axis=1), 110_000, atol=1e-6)

def test_r0_matches_tipe_cases():
    # Case 1 (no control): R0 = 10. Case 2 (u = 0.88, 0.70, 0.65): R0 = 0.91.
    assert reproduction_number(BETA, GAMMA, MU, 100_000) == 10.0
    r0 = reproduction_number(BETA, GAMMA, MU, 100_000, u1=0.88, u2=0.70, u3=0.65)
    assert abs(r0 - 0.91) < 0.02

def test_control_brings_epidemic_down():
    _, no_control = srmi_ode(100_000, 10_000, beta=BETA, gamma=GAMMA, mu=MU, days=200)
    _, controlled = srmi_ode(100_000, 10_000, beta=BETA, gamma=GAMMA, mu=MU,
                             u1=0.88, u2=0.70, u3=0.65, days=200)
    assert controlled[:, 1].max() < no_control[:, 1].max()

def test_network_simulation_counts():
    net = ContactNetwork(500, avg_degree=6, seed=1)
    hist = run_network_simulation(net, initial_infected=range(10), steps=30, seed=1)
    for t in range(30):
        assert hist["S"][t] + hist["I"][t] + hist["R"][t] + hist["M"][t] == 500

def test_control_sweep_finds_passing_controls():
    passing, _ = control_sweep(BETA, GAMMA, MU, 100_000)
    assert passing  # at least the strongest controls bring R0 < 1
