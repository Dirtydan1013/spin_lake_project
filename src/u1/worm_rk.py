"""Classical worm Monte Carlo for the RK state on the medial honeycomb lattice.

Uniform sampling over perfect dimer coverings (= the diagonal ensemble of the
RK wavefunction, paper/u1 Methods Eq. 6) via the worm update of paper/u1
Methods Eq. 16: remove a dimer, propagate the head with non-backtracking
transition probability 1/2, terminate when the loop closes.

Graph representation comes from ``KagomeU1Geometry``: honeycomb vertices are
kagome triangles, honeycomb edges are atoms.  A configuration is the atom
occupation n ∈ {0,1}^N with exactly one excited atom per triangle.

Reference values (their Fig. 5c): ⟨V⟩ = ⟨T⟩ ≈ 0.29 for large lattices, where
⟨V⟩ is the flippable-plaquette fraction; ⟨T⟩ = ⟨V⟩ holds exactly for the RK
state (Eq. 15 pairs each flippable configuration with its flip partner).

Usage:
    python -m src.u1.worm_rk --nx 12 --ny 12 --n-sweeps 20000
"""
from __future__ import annotations

import argparse

import numpy as np
from numba import njit

from src.u1.honeycomb_dimer import KagomeU1Geometry


def columnar_covering(geo: KagomeU1Geometry, k: int = 0) -> np.ndarray:
    """A valid initial covering: excite every basis-k atom (each triangle
    contains exactly one atom of each sublattice)."""
    n = (geo.site_basis == k).astype(np.int8)
    occ = n[geo.triangles].sum(axis=1)
    assert np.all(occ == 1), "columnar covering violates the dimer constraint"
    return n


@njit(cache=True)
def _worm_update(n, tri_dimer, tri_atoms, atom_tris, u0):
    """One worm update starting from triangle u0.  Mutates n / tri_dimer.
    Returns the loop length (number of head steps)."""
    e = tri_dimer[u0]                       # dimer incident on the tail
    v = atom_tris[e, 0] if atom_tris[e, 1] == u0 else atom_tris[e, 1]
    n[e] = 0
    tri_dimer[u0] = -1
    tri_dimer[v] = -1
    entering = e
    steps = 0
    while True:
        steps += 1
        # pick one of v's two other edges with probability 1/2
        k0 = -1
        r = np.random.randint(2)            # 0 or 1: which of the two others
        cnt = 0
        for idx in range(3):
            a = tri_atoms[v, idx]
            if a == entering:
                continue
            if cnt == r:
                k0 = a
                break
            cnt += 1
        # insert dimer k0 from v to u
        n[k0] = 1
        tri_dimer[v] = k0
        u = atom_tris[k0, 0] if atom_tris[k0, 1] == v else atom_tris[k0, 1]
        if tri_dimer[u] == -1:              # u is the tail monomer → close
            tri_dimer[u] = k0
            return steps
        # u temporarily double-covered: remove its old dimer, advance head
        e = tri_dimer[u]
        n[e] = 0
        tri_dimer[u] = k0
        v = atom_tris[e, 0] if atom_tris[e, 1] == u else atom_tris[e, 1]
        tri_dimer[v] = -1
        entering = e


@njit(cache=True)
def _run_sweeps(n, tri_dimer, tri_atoms, atom_tris, n_sweeps, worms_per_sweep,
                seed):
    np.random.seed(seed)
    T = tri_dimer.shape[0]
    for _ in range(n_sweeps * worms_per_sweep):
        _worm_update(n, tri_dimer, tri_atoms, atom_tris, np.random.randint(T))


class WormRK:
    """Uniform sampler over perfect dimer coverings of the periodic lattice."""

    def __init__(self, geo: KagomeU1Geometry, seed: int = 1):
        if geo.box is None:
            raise ValueError("worm sampler requires the periodic geometry")
        self.geo = geo
        self.tri_atoms = np.ascontiguousarray(geo.triangles, dtype=np.int64)
        atom_tris = np.full((geo.N, 2), -1, dtype=np.int64)
        for s in range(geo.N):
            ts = geo.site_tris[s]
            if len(ts) != 2:
                raise ValueError("every atom must join exactly 2 triangles")
            atom_tris[s] = ts
        self.atom_tris = np.ascontiguousarray(atom_tris)
        self.n = columnar_covering(geo)
        self.tri_dimer = np.empty(geo.T, dtype=np.int64)
        for t in range(geo.T):
            tri = self.tri_atoms[t]
            self.tri_dimer[t] = tri[np.argmax(self.n[tri])]
        self._seed = seed
        self._seed_used = False

    def sweep(self, n_sweeps: int = 1, worms_per_sweep: int | None = None):
        """Run worm updates; one sweep defaults to T worm updates."""
        wps = worms_per_sweep or self.geo.T
        seed = self._seed if not self._seed_used else np.random.randint(2**31)
        self._seed_used = True
        _run_sweeps(self.n, self.tri_dimer, self.tri_atoms, self.atom_tris,
                    int(n_sweeps), int(wps), int(seed))

    def snapshot(self) -> np.ndarray:
        return self.n.copy()

    def check(self):
        occ = self.n[self.geo.triangles].sum(axis=1)
        assert np.all(occ == 1), "dimer constraint violated"

    def sample(self, n_samples: int, sweeps_between: int = 1) -> np.ndarray:
        """(n_samples, N) snapshots, `sweeps_between` sweeps apart."""
        out = np.empty((n_samples, self.geo.N), dtype=np.int8)
        for i in range(n_samples):
            self.sweep(sweeps_between)
            out[i] = self.n
        return out


def main():
    ap = argparse.ArgumentParser(
        description="Classical worm MC reference for the RK dimer state")
    ap.add_argument("--nx", type=int, default=12)
    ap.add_argument("--ny", type=int, default=12)
    ap.add_argument("--n-thermalize", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--sweeps-between", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default=None,
                    help="optional .npz for snapshots + observables")
    args = ap.parse_args()

    geo = KagomeU1Geometry(args.nx, args.ny, a=2.0, boundary='periodic')
    w = WormRK(geo, seed=args.seed)
    w.sweep(args.n_thermalize)
    snaps = w.sample(args.n_samples, args.sweeps_between)
    w.check()

    V = geo.flippable_fraction(snaps)
    dens = snaps.mean()
    frac = geo.defect_fractions(snaps)
    print(f"RK worm MC {args.nx}x{args.ny} (P={geo.P}, N={geo.N}): "
          f"{args.n_samples} samples")
    print(f"  density={dens:.6f} (exact 1/3)  defects={frac}")
    print(f"  <V> = <T> = {V:.5f}  (paper: ≈0.29 at large size)")

    rs, cd = geo.dimer_dimer_corr(snaps)
    rr, cr = geo.rectified_dimer_corr(snaps)
    if args.out:
        np.savez(args.out, snaps=snaps, V=V, r_dd=rs, C_dd=cd,
                 r_rect=rr, C_rect=cr)
        print(f"  saved → {args.out}")


if __name__ == "__main__":
    main()
