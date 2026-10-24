"""Control analysis: which (u1, u2, u3) combinations bring R0 below 1?"""
import itertools
import numpy as np
from .ode import reproduction_number


def control_sweep(beta, gamma, mu, s_star, grid=(0.0, 0.25, 0.5, 0.75, 0.9)):
    """Grid-search controls and return (passing, failing) lists of (u1, u2, u3, R0)."""
    passing, failing = [], []
    for u1, u2, u3 in itertools.product(grid, repeat=3):
        r0 = reproduction_number(beta, gamma, mu, s_star, u1, u2, u3)
        (passing if r0 < 1 else failing).append((u1, u2, u3, round(float(r0), 3)))
    return passing, failing


def minimal_confinement(beta, gamma, mu, s_star, u2=0.0, u3=0.0):
    """Smallest u1 (confinement) that alone brings R0 below 1, or None."""
    r0_uncontrolled = reproduction_number(beta, gamma, mu, s_star, 0.0, u2, u3)
    if r0_uncontrolled < 1:
        return 0.0
    # R0(u1) = (1 - u1) * R0(0)  ->  u1 > 1 - 1 / R0(0)
    u1_min = 1 - 1.0 / r0_uncontrolled
    return float(u1_min) if u1_min <= 1 else None
