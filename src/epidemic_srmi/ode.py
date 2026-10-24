"""Deterministic SRMI compartment model with control parameters.

Compartments: S (susceptible), I (infected), R (recovered), M (deceased).
Controls u1, u2, u3 in [0, 1]:
    u1: confinement / safety measures (reduces contacts)
    u2: treatment / care of the infected (boosts recovery)
    u3: prevention of deaths (critical-case intervention)

    dS/dt = -(1 - u1) * beta * S * I
    dI/dt =  (1 - u1) * beta * S * I - (1 - u3) * mu * I - (1 + u2) * gamma * I
    dR/dt =  (1 + u2) * gamma * I
    dM/dt =  (1 - u3) * mu * I

Basic reproduction number (next-generation matrix, disease-free equilibrium):
    R0 = (1 - u1) * beta * S* / ((1 + u2) * gamma + (1 - u3) * mu)
"""
import numpy as np


def reproduction_number(beta, gamma, mu, s_star, u1=0.0, u2=0.0, u3=0.0):
    """Return R0 for given parameters and controls."""
    for name, u in (("u1", u1), ("u2", u2), ("u3", u3)):
        if not 0.0 <= u <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {u}")
    return (1 - u1) * beta * s_star / ((1 + u2) * gamma + (1 - u3) * mu)


def _derivatives(state, beta, gamma, mu, u1, u2, u3):
    s, i, r, m = state
    infection = (1 - u1) * beta * s * i
    recovery = (1 + u2) * gamma * i
    death = (1 - u3) * mu * i
    return np.array([ -infection, infection - recovery - death, recovery, death ])


def srmi_ode(s0, i0, r0=0.0, m0=0.0, beta=7e-8, gamma=5e-4, mu=2e-4,
             u1=0.0, u2=0.0, u3=0.0, days=300, dt=0.5):
    """Integrate the SRMI ODE with a classic RK4 scheme (NumPy only).

    Returns (time, trajectory) where trajectory has shape (n_steps, 4)
    with columns S, I, R, M. Population is conserved: S + I + R + M = N.
    """
    steps = int(days / dt) + 1
    time = np.linspace(0, days, steps)
    traj = np.zeros((steps, 4))
    traj[0] = [s0, i0, r0, m0]
    for k in range(steps - 1):
        y = traj[k]
        k1 = _derivatives(y, beta, gamma, mu, u1, u2, u3)
        k2 = _derivatives(y + 0.5 * dt * k1, beta, gamma, mu, u1, u2, u3)
        k3 = _derivatives(y + 0.5 * dt * k2, beta, gamma, mu, u1, u2, u3)
        k4 = _derivatives(y + dt * k3, beta, gamma, mu, u1, u2, u3)
        traj[k + 1] = np.maximum(y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
    return time, traj
