"""Toy example for the introduction: fraction of a spherical nodule's volume inside the 10 mm ball
and inside the band, for a nodule whose centre is displaced from the proposal.
Uses the same ball/band geometry as the Fig. 1 drawing code (scripts/fig_schematic.py:
R_BALL = 10 mm, BAND = [r_N - d, r_F + d] for 0.70 x 0.70 x 1.25 mm voxels).
Monte Carlo with 10^6 uniform points per nodule, fixed seed."""
import numpy as np
from fig_schematic import R_BALL, BAND

CASES = [(5, 2), (14, 6), (20, 8)]   # (diameter mm, centre offset mm)


def fractions(diam, offset, n=1_000_000, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    x = u * (diam / 2 * rng.random(n) ** (1 / 3))[:, None]
    x[:, 0] += offset
    r = np.linalg.norm(x, axis=1)
    return (r <= R_BALL).mean(), ((r >= BAND[0]) & (r <= BAND[1])).mean()


if __name__ == "__main__":
    print(f"band = [{BAND[0]:.1f}, {BAND[1]:.1f}] mm")
    for d, o in CASES:
        b, a = fractions(d, o)
        print(f"{d:>2} mm nodule, offset {o} mm: ball {100 * b:.0f}%, band {100 * a:.0f}%")
