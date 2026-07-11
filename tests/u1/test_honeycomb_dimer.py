from __future__ import annotations

import itertools

import numpy as np
import pytest

from src.u1.honeycomb_dimer import KagomeU1Geometry
from src.u1.worm_rk import WormRK, columnar_covering


@pytest.fixture(scope="module")
def geo66():
    return KagomeU1Geometry(6, 6, a=2.0, boundary='periodic')


def test_periodic_counts(geo66):
    g = geo66
    assert g.N == 3 * 36
    assert g.P == 36
    assert g.T == 2 * 36
    # every atom joins exactly 2 triangles and 2 plaquettes
    assert all(len(t) == 2 for t in g.site_tris)
    assert all(len(p) == 2 for p in g.site_plaqs)
    # every triangle has one atom of each sublattice
    assert np.all(np.sort(g.site_basis[g.triangles], axis=1) == [0, 1, 2])


def test_plaquette_ring_is_cyclic_nn(geo66):
    g = geo66
    for ring in g.plaq_sites:
        for k in range(6):
            d = np.linalg.norm(g.disp(g.pos[ring[k]],
                                      g.pos[ring[(k + 1) % 6]]))
            assert np.isclose(d, g.a / 2.0)


def test_plaquette_coloring_tripartite(geo66):
    g = geo66
    for p in range(g.P):
        for q in range(g.P):
            if p < q and set(map(int, g.plaq_sites[p])) & \
                         set(map(int, g.plaq_sites[q])):
                assert g.plaq_color[p] != g.plaq_color[q]


def test_straight_strings(geo66):
    g = geo66
    sets, meta = g.straight_string_sets()
    assert [m["size"] for m in meta] == [1, 2, 3, 4, 5]
    # every length: 3 directions × 36 plaquettes
    assert all(m["n_copies"] == 3 * 36 for m in meta)
    # consecutive plaquettes share exactly one atom; string atoms distinct
    for st in sets:
        assert len(set(st)) == len(st)


def test_columnar_covering_and_observables(geo66):
    g = geo66
    n = columnar_covering(g, 0)
    assert np.isclose(n.mean(), 1.0 / 3.0)
    frac = g.defect_fractions(n)
    assert frac == dict(monomer=0.0, dimer=1.0, multi=0.0)
    # columnar covering: 2 same-sublattice dimers per hexagon → not flippable
    assert g.flippable_fraction(n) == 0.0
    # fully sublattice-polarized covering → |Φ| = number of triangles
    phi = g.nematic_phi(n)[0]
    assert np.isclose(abs(phi), g.T)


def _enumerate_coverings(geo):
    """All perfect dimer coverings by backtracking over triangles."""
    tri_atoms = geo.triangles
    atom_tris = geo.site_tris
    T = geo.T
    covered = np.zeros(T, dtype=bool)
    n = np.zeros(geo.N, dtype=np.int8)
    out = []

    def rec():
        free = np.where(~covered)[0]
        if len(free) == 0:
            out.append(n.copy())
            return
        t = free[0]
        for a in tri_atoms[t]:
            u, v = atom_tris[a]
            other = v if u == t else u
            if not covered[other]:
                covered[t] = covered[other] = True
                n[a] = 1
                rec()
                covered[t] = covered[other] = False
                n[a] = 0

    rec()
    return out


def test_worm_uniform_vs_exact_enumeration():
    geo = KagomeU1Geometry(3, 3, a=2.0, boundary='periodic')
    coverings = _enumerate_coverings(geo)
    n_cov = len(coverings)
    assert n_cov > 1
    keys = {tuple(map(int, c)): i for i, c in enumerate(coverings)}

    w = WormRK(geo, seed=7)
    w.sweep(200)
    counts = np.zeros(n_cov)
    n_samp = 20000
    for _ in range(n_samp):
        w.sweep(1)
        counts[keys[tuple(map(int, w.n))]] += 1
    w.check()
    # every covering visited, frequencies uniform within 5 sigma
    assert np.all(counts > 0)
    expect = n_samp / n_cov
    sd = np.sqrt(expect)
    assert np.all(np.abs(counts - expect) < 5 * sd + 0.05 * expect), (
        counts.min(), counts.max(), expect)

    # exact <V> over the uniform ensemble matches the sampled estimate
    V_exact = np.mean([geo.flippable_fraction(c) for c in coverings])
    snaps = w.sample(4000, 1)
    assert abs(geo.flippable_fraction(snaps) - V_exact) < 0.01


def test_open_patch_geometry():
    g = KagomeU1Geometry(4, 4, a=2.0, boundary='open')
    assert g.P == 16
    # interior atoms join 2 triangles / 2 plaquettes; boundary fewer
    n2 = sum(len(t) == 2 for t in g.site_tris)
    assert 0 < n2 < g.N
    sets, meta = g.straight_string_sets()
    assert len(sets) > 0
    # rectified correlator runs and skips boundary triangles
    rng = np.random.default_rng(0)
    snaps = (rng.random((20, g.N)) < 0.3).astype(np.int8)
    rs, cd = g.rectified_dimer_corr(snaps)
    assert len(rs) == len(cd) > 0


def test_subsystem_histogram(geo66):
    g = geo66
    n = columnar_covering(g, 0)
    inner = [int(s) for s in g.plaq_sites[0]]
    boundary = sorted({int(s) for p in range(g.P) if p != 0
                       for s in g.plaq_sites[p]
                       if int(s) not in inner and
                       any(int(q) in inner
                           for t in g.site_tris[int(s)]
                           for q in g.triangles[t])})
    hist = g.subsystem_histogram(np.array([n, n]), inner, boundary)
    total = sum(hist.values())
    assert total in (0, 2)   # kept iff the boundary happens to be all-0


def test_gauss_law_closed_loop_parity_and_fm_strings(geo66):
    """On perfect dimer coverings the closed dual-loop parity is fixed (+1)
    by the emergent Gauss law; the FM denominator is therefore exact."""
    g = geo66
    w = WormRK(g, seed=11)
    w.sweep(100)
    snaps = w.sample(200, 1)
    sz = 2.0 * snaps.astype(float) - 1.0
    for L in (1, 2):
        up, lo, closed = g.zigzag_string_pair(2, 3, L)
        assert len(up) == len(lo) == 2 * L and len(closed) == 4 * L
        par = np.prod(sz[:, closed], axis=1)
        assert np.all(par == 1.0)
    lengths, cs, diag = g.fm_string_corr(snaps, max_len=2)
    assert list(lengths) == [1, 2]
    assert all(np.isclose(dg["S_closed"], 1.0) for dg in diag)
    assert np.all(np.isfinite(cs))
