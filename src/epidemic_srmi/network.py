"""Stochastic epidemic spread on a contact network (SRMI, individual level).

Each node is in state S, I, R or M. At each discrete step:
  - a susceptible node with infected neighbours is infected with
    probability 1 - exp(-tau * n_infected_neighbours), where tau is the
    per-contact transmission rate (the discrete form of T_ij(t) = 1 - exp(-tau_ij * t));
  - an infected node leaves the infected state after its characteristic
    infectious period, dying with probability m or recovering otherwise.

This mirrors the stochastic model of the TIPE: infection probabilities are
propagated through the contact graph instead of through a well-mixed ODE.
"""
import numpy as np


class ContactNetwork:
    """Random contact graph (Erdos-Renyi by default, or Barabasi-Albert)."""

    def __init__(self, n_nodes, avg_degree=6, model="erdos_renyi", seed=42):
        rng = np.random.default_rng(seed)
        self.n_nodes = n_nodes
        self.neighbours = [[] for _ in range(n_nodes)]
        if model == "erdos_renyi":
            p = avg_degree / max(n_nodes - 1, 1)
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if rng.random() < p:
                        self.neighbours[i].append(j)
                        self.neighbours[j].append(i)
        elif model == "barabasi_albert":
            m = max(1, avg_degree // 2)
            for i in range(m + 1):
                for j in range(i + 1, m + 1):
                    self.neighbours[i].append(j)
                    self.neighbours[j].append(i)
            degrees = np.array([len(nb) for nb in self.neighbours], dtype=float)
            for i in range(m + 1, n_nodes):
                probs = degrees[:i] / degrees[:i].sum()
                targets = rng.choice(i, size=min(m, i), replace=False, p=probs)
                for j in targets:
                    self.neighbours[i].append(int(j))
                    self.neighbours[int(j)].append(i)
                degrees = np.array([len(nb) for nb in self.neighbours], dtype=float)
        else:
            raise ValueError(f"unknown model: {model}")

    def mean_degree(self):
        return float(np.mean([len(nb) for nb in self.neighbours]))


def run_network_simulation(network, initial_infected, tau=0.08, infectious_period=14,
                           death_prob=0.02, steps=120, seed=42):
    """Run the stochastic SRMI simulation. Returns a dict of time series.

    States are encoded 0=S, 1=I, 2=R, 3=M. Each infected node carries a
    countdown; when it reaches zero the node recovers or dies.
    """
    rng = np.random.default_rng(seed)
    n = network.n_nodes
    state = np.zeros(n, dtype=int)
    timer = np.zeros(n, dtype=int)
    state[list(initial_infected)] = 1
    timer[list(initial_infected)] = infectious_period
    history = {"S": [], "I": [], "R": [], "M": []}
    for _ in range(steps):
        for name, code in history.items():
            code.append(int((state == {"S": 0, "I": 1, "R": 2, "M": 3}[name]).sum()))
        new_state = state.copy()
        infected = np.where(state == 1)[0]
        for node in np.where(state == 0)[0]:
            k = sum(1 for nb in network.neighbours[node] if state[nb] == 1)
            if k and rng.random() < 1 - np.exp(-tau * k):
                new_state[node] = 1
                timer[node] = infectious_period
        for node in infected:
            timer[node] -= 1
            if timer[node] <= 0:
                new_state[node] = 3 if rng.random() < death_prob else 2
        state = new_state
    return history
