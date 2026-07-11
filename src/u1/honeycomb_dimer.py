"""Kagome-vertex ↔ medial-honeycomb dimer geometry and observables (paper/u1).

Atoms sit on the vertices of the kagome lattice (``generate_kagome_lattice``);
a Rydberg excitation encodes a dimer on the medial honeycomb lattice:

    honeycomb vertex    = kagome triangle  (3 mutually-blockaded atoms)
    honeycomb edge      = atom             (each atom joins its 2 triangles)
    honeycomb plaquette = kagome hexagonal void

The perfect-dimer constraint "every honeycomb vertex touched by exactly one
dimer" is "every triangle holds exactly one excitation" (n = 1/3 filling).

Everything here is built geometrically from the atom positions (with
minimum-image convention on the torus), so it works for both the open patch
and the periodic 3-site-basis lattice, and is independent of site-index
conventions.

Sign convention: the C++ ``measure_loops``/``measure_strings`` compute
``∏(1 - 2 n_i)``; the paper uses σᶻ = 2n − 1, so engine values differ from
the paper's by (−1)^len.  The helpers in this module use σᶻ = 2n − 1.
"""
from __future__ import annotations

import numpy as np

from src.rydberg.lattices import generate_kagome_lattice, kagome_hex_centers

OMEGA3 = np.exp(2j * np.pi / 3.0)


class KagomeU1Geometry:
    """Dimer-model geometry for an nx×ny (hex-void cells) vertex kagome lattice.

    Attributes
    ----------
    pos : (N, 2) atom positions.
    plaq_sites : (P, 6) int — atoms around each hexagonal void, in cyclic
        (counter-clockwise angular) order; P == nx*ny, plaquette p == cell
        (i, j) with p = j*nx + i (``kagome_hex_centers`` order).
    triangles : (T, 3) int — blockade triangles (honeycomb vertices), sorted
        site triples.  Periodic: T == 2*nx*ny.  Open: only complete triangles.
    site_tris : list[N] of triangle ids per atom (2 in the bulk).
    site_plaqs : list[N] of plaquette ids per atom (2 in the bulk).
    site_basis : (N,) int — kagome sublattice A/B/C = 0/1/2 (basis offset
        v1/2, v2/2, (v1+v2)/2 modulo the Bravais lattice).
    plaq_color : (P,) int — tripartite plaquette sublattice (i - j) mod 3.
        Only globally consistent on the torus when nx % 3 == ny % 3 == 0.
    """

    def __init__(self, nx: int, ny: int, a: float = 2.0,
                 boundary: str = 'periodic'):
        if boundary == 'periodic' and (nx < 3 or ny < 3):
            raise ValueError("periodic U(1) geometry needs nx, ny >= 3 "
                             "(smaller tori alias plaquette adjacency)")
        self.nx, self.ny, self.a, self.boundary = nx, ny, float(a), boundary
        self.pos = np.asarray(
            generate_kagome_lattice(nx, ny, a, boundary=boundary),
            dtype=np.float64)
        self.N = len(self.pos)
        v1 = np.array([a, 0.0])
        v2 = np.array([a / 2.0, a * np.sqrt(3) / 2.0])
        self._v1, self._v2 = v1, v2
        self.box = (np.array([nx * v1, ny * v2])
                    if boundary == 'periodic' else None)
        self._nn = a / 2.0

        self._build_plaquettes()
        self._build_triangles()
        self._build_site_basis()

    # ── minimum-image displacement ───────────────────────────────────────────
    def disp(self, from_xy: np.ndarray, to_xy: np.ndarray) -> np.ndarray:
        """to - from, minimum-image on the torus (identity when open)."""
        d = np.asarray(to_xy, dtype=np.float64) - np.asarray(from_xy,
                                                             dtype=np.float64)
        if self.box is None:
            return d
        frac = np.linalg.solve(self.box.T, np.atleast_2d(d).T).T
        frac -= np.round(frac)
        out = frac @ self.box
        return out[0] if np.ndim(to_xy) == 1 and np.ndim(from_xy) == 1 else out

    def _neighbors_within(self, center: np.ndarray, radius: float) -> list:
        d = self.disp(center, self.pos)
        r = np.linalg.norm(d, axis=1)
        return [int(i) for i in np.where(r < radius)[0]]

    # ── construction ─────────────────────────────────────────────────────────
    def _build_plaquettes(self):
        centers = kagome_hex_centers(self.nx, self.ny, self.a)
        if self.box is not None:
            # keep centers inside the torus cell (they already are)
            pass
        plaq_sites = []
        for c in centers:
            sites = self._neighbors_within(c, self._nn * 1.0 + 1e-9)
            if len(sites) != 6:
                raise RuntimeError(
                    f"hex void at {c} has {len(sites)} vertices (expected 6)")
            ang = [np.arctan2(*self.disp(c, self.pos[s])[::-1]) for s in sites]
            plaq_sites.append([s for _, s in sorted(zip(ang, sites))])
        self.plaq_centers = centers
        self.plaq_sites = np.asarray(plaq_sites, dtype=np.int64)
        self.P = len(plaq_sites)
        self.site_plaqs = [[] for _ in range(self.N)]
        for p, sites in enumerate(self.plaq_sites):
            for s in sites:
                self.site_plaqs[s].append(p)
        # tripartite coloring by cell index (p = j*nx + i)
        idx = np.arange(self.P)
        i, j = idx % self.nx, idx // self.nx
        self.plaq_color = ((i - j) % 3).astype(np.int64)

    def _build_triangles(self):
        # triangles = 3-cliques of the nn graph (kagome has no other 3-cliques)
        nn_of = [set() for _ in range(self.N)]
        for s in range(self.N):
            for t in self._neighbors_within(self.pos[s], self._nn + 1e-9):
                if t != s:
                    nn_of[s].add(t)
        tris = set()
        for s in range(self.N):
            for t in nn_of[s]:
                if t < s:
                    continue
                for u in (nn_of[s] & nn_of[t]):
                    if u > t:
                        tris.add((s, t, u))
        self.triangles = np.asarray(sorted(tris), dtype=np.int64)
        self.T = len(self.triangles)
        self.site_tris = [[] for _ in range(self.N)]
        for t, tri in enumerate(self.triangles):
            for s in tri:
                self.site_tris[s].append(t)

    def _build_site_basis(self):
        # fractional coords in (v1, v2); basis A=(.5,0), B=(0,.5), C=(.5,.5)
        B = np.array([self._v1, self._v2])
        frac = np.linalg.solve(B.T, self.pos.T).T % 1.0
        basis = np.full(self.N, -1, dtype=np.int64)
        for k, off in enumerate(((0.5, 0.0), (0.0, 0.5), (0.5, 0.5))):
            hit = np.all(np.abs((frac - off + 0.5) % 1.0 - 0.5) < 1e-6, axis=1)
            basis[hit] = k
        if np.any(basis < 0):
            raise RuntimeError("atom off the kagome vertex family")
        self.site_basis = basis

    # ── plaquette walk (strings) ─────────────────────────────────────────────
    def _plaq_id(self, i: int, j: int):
        """Plaquette id of cell (i, j); None when off the open patch."""
        if self.box is not None:
            return (j % self.ny) * self.nx + (i % self.nx)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return j * self.nx + i
        return None

    def shared_atom(self, p: int, q: int) -> int:
        """The single atom shared by adjacent plaquettes p and q."""
        common = set(map(int, self.plaq_sites[p])) & \
                 set(map(int, self.plaq_sites[q]))
        if len(common) != 1:
            raise ValueError(f"plaquettes {p},{q} share {len(common)} atoms")
        return common.pop()

    #: plaquette-lattice steps (cell-index increments) for the 3 directions
    DIRECTIONS = ((1, 0), (0, 1), (1, -1))          # +v1, +v2, +(v1-v2)

    def straight_string(self, i: int, j: int, direction: tuple,
                        length: int) -> list | None:
        """Atoms crossed walking `length` plaquette steps from cell (i, j).

        Returns the list of `length` atom ids (the dimer-parity string between
        plaquette (i,j) and the plaquette `length` steps away), or None when
        the walk leaves the open patch / wraps past half the torus.
        """
        di, dj = direction
        if self.box is not None:
            span = self.nx if dj == 0 else (self.ny if di == 0 else
                                            min(self.nx, self.ny))
            if length >= span:
                return None                          # would wrap onto itself
        atoms, p = [], self._plaq_id(i, j)
        if p is None:
            return None
        for step in range(1, length + 1):
            q = self._plaq_id(i + step * di, j + step * dj)
            if q is None:
                return None
            atoms.append(self.shared_atom(p, q))
            p = q
        return atoms

    # ── engine observable site-lists ─────────────────────────────────────────
    def hexagon_loop_sets(self) -> list:
        """All hexagon Z loops (size 6) — closed-loop parity / FM denominators."""
        return [[int(s) for s in row] for row in self.plaq_sites]

    def straight_string_sets(self, max_len: int | None = None):
        """Dimer-parity strings, grouped by length across all 3 directions.

        Returns (string_sets, string_meta) in the ``_lattice_observables``
        convention: string_meta = [{'size': s, 'n_copies': n, 'offset': o}].
        The C++ engine groups sets by size, so all translations and directions
        of a given length are averaged together (C3 + translation symmetry).
        """
        if max_len is None:
            max_len = (min(self.nx, self.ny) - 1 if self.box is not None
                       else min(self.nx, self.ny))
        sets, meta = [], []
        for s in range(1, max_len + 1):
            offset = len(sets)
            copies = []
            for j in range(self.ny):
                for i in range(self.nx):
                    for d in self.DIRECTIONS:
                        st = self.straight_string(i, j, d, s)
                        if st is not None:
                            copies.append(st)
            if copies:
                sets.extend(copies)
                meta.append(dict(size=s, n_copies=len(copies), offset=offset))
        return sets, meta

    def zigzag_string_pair(self, i: int, j: int, length: int):
        """Two mirror-symmetric dual paths from plaquette (i, j) to
        (i+length, j), and their closed loop (Fig. 6c geometry).

        upper path: +v2, +(v1-v2), +v2, ...  (2·length atoms)
        lower path: +(v1-v2), +v2, ...       (2·length atoms)
        closed loop = upper ∪ lower (4·length atoms); on a perfect dimer
        covering its parity is fixed by the emergent Gauss law:
        ∏σᶻ = (−1)^(# enclosed honeycomb vertices) = (−1)^(2·length) = +1.

        Returns (upper, lower, closed) atom-id lists, or None when the walk
        leaves the open patch / wraps (needs length < nx/2 on the torus).
        """
        if self.box is not None and not (0 < length < self.nx / 2):
            return None

        def walk(steps):
            atoms, p, ci, cj = [], self._plaq_id(i, j), i, j
            if p is None:
                return None
            for di, dj in steps:
                ci, cj = ci + di, cj + dj
                q = self._plaq_id(ci, cj)
                if q is None:
                    return None
                atoms.append(self.shared_atom(p, q))
                p = q
            return atoms

        up_steps, lo_steps = [], []
        for _ in range(length):
            up_steps += [(0, 1), (1, -1)]           # +v2 then +(v1-v2)
            lo_steps += [(1, -1), (0, 1)]           # +(v1-v2) then +v2
        upper, lower = walk(up_steps), walk(lo_steps)
        if upper is None or lower is None:
            return None
        return upper, lower, upper + lower

    def fm_string_corr(self, snaps: np.ndarray, max_len: int | None = None):
        """FM-normalized dimer-string correlator along rows (Fig. 6d).

        For each separation L (in plaquette steps along v1) averages over all
        translations:  C_s(L) = geomean(⟨S_up⟩, ⟨S_lo⟩) / ⟨S_closed⟩,
        with S = ∏σᶻ (σᶻ = 2n − 1) and the signed geometric mean
        sgn(⟨S_up⟩⟨S_lo⟩ ≥ 0)·sqrt(|⟨S_up⟩⟨S_lo⟩|).

        Returns (lengths, C_s, diagnostics) where diagnostics carries the raw
        ⟨S_up⟩, ⟨S_lo⟩, ⟨S_closed⟩ per length."""
        snaps = np.atleast_2d(np.asarray(snaps)).astype(np.float64)
        sz = 2.0 * snaps - 1.0
        if max_len is None:
            max_len = (int(np.ceil(self.nx / 2)) - 1 if self.box is not None
                       else self.nx - 1)
        lengths, cs, diag = [], [], []
        for L in range(1, max_len + 1):
            su, sl, sc = [], [], []
            for j in range(self.ny):
                for i in range(self.nx):
                    trio = self.zigzag_string_pair(i, j, L)
                    if trio is None:
                        continue
                    up, lo, closed = trio
                    su.append(np.prod(sz[:, up], axis=1).mean())
                    sl.append(np.prod(sz[:, lo], axis=1).mean())
                    sc.append(np.prod(sz[:, closed], axis=1).mean())
            if not su:
                continue
            mu, ml, mc = np.mean(su), np.mean(sl), np.mean(sc)
            num = np.sign(mu * ml) * np.sqrt(abs(mu * ml))
            lengths.append(L)
            cs.append(num / mc if mc != 0 else np.nan)
            diag.append(dict(S_up=mu, S_lo=ml, S_closed=mc))
        return np.array(lengths), np.array(cs), diag

    # ── snapshot (Z-basis bitstring) analysis; σᶻ = 2n − 1 ──────────────────
    def triangle_occupation(self, snaps: np.ndarray) -> np.ndarray:
        """(n_snap, T) number of excited atoms per triangle (0 = monomer,
        1 = dimer, >=2 = blockade violation)."""
        snaps = np.atleast_2d(np.asarray(snaps))
        return snaps[:, self.triangles].sum(axis=2)

    def defect_fractions(self, snaps: np.ndarray) -> dict:
        """Monomer / dimer / multi-dimer fractions per honeycomb vertex."""
        occ = self.triangle_occupation(snaps)
        return dict(monomer=float((occ == 0).mean()),
                    dimer=float((occ == 1).mean()),
                    multi=float((occ >= 2).mean()))

    def flippable_fraction(self, snaps: np.ndarray) -> float:
        """Bare RK potential energy ⟨V⟩: fraction of plaquettes whose 6 atoms
        alternate 101010 / 010101 (three parallel dimers)."""
        snaps = np.atleast_2d(np.asarray(snaps))
        ring = snaps[:, self.plaq_sites]              # (n_snap, P, 6)
        alt0 = np.array([1, 0, 1, 0, 1, 0])
        flip = (np.all(ring == alt0, axis=2) |
                np.all(ring == 1 - alt0, axis=2))
        return float(flip.mean())

    def nematic_phi(self, snaps: np.ndarray) -> np.ndarray:
        """Complex nematic order parameter Φ per snapshot (Fig. 4g):
        Φ = Σ_triangles Σ_{s∈tri} n_s ω^{basis(s)}  (bulk triangles only)."""
        snaps = np.atleast_2d(np.asarray(snaps))
        w = OMEGA3 ** self.site_basis
        tri_sites = self.triangles.reshape(-1)
        vals = (snaps[:, tri_sites] *
                w[tri_sites]).reshape(len(snaps), self.T, 3).sum(axis=(1, 2))
        return vals

    def dimer_dimer_corr(self, snaps: np.ndarray, r_decimals: int = 6):
        """Connected σᶻσᶻ correlator binned by (min-image) pair distance.

        Returns (r, C_d(r)) with C_d = ⟨σᶻiσᶻj⟩ − ⟨σᶻi⟩⟨σᶻj⟩ averaged over
        pairs at equal distance."""
        snaps = np.atleast_2d(np.asarray(snaps)).astype(np.float64)
        sz = 2.0 * snaps - 1.0
        m = sz.mean(axis=0)
        G = sz.T @ sz / len(sz)                       # ⟨σᶻiσᶻj⟩
        C = G - np.outer(m, m)
        iu = np.triu_indices(self.N, k=1)
        d = np.linalg.norm(
            self.disp(self.pos[iu[0]], self.pos[iu[1]]), axis=-1)
        r = np.round(d, r_decimals)
        rs = np.unique(r)
        cd = np.array([C[iu][r == rv].mean() for rv in rs])
        return rs, cd

    def rectified_dimer_corr(self, snaps: np.ndarray, r_decimals: int = 6):
        """Rectified dimer-dimer correlator C̃_d = ⟨φ*(ri) φ(rj)⟩ binned by
        triangle-centroid distance.  φ(tri) = Σ_{s∈tri} σᶻ_s ω^{pair(s)} with
        the phase set by the colors of the atom's two adjacent plaquettes:
        {b,c}→1, {a,c}→ω, {a,b}→ω² (i.e. ω^{missing color}).  Triangles with
        a boundary atom (single adjacent plaquette) are skipped."""
        snaps = np.atleast_2d(np.asarray(snaps)).astype(np.float64)
        sz = 2.0 * snaps - 1.0
        # per-atom phase (None → exclude)
        phase = np.zeros(self.N, dtype=np.complex128)
        ok = np.zeros(self.N, dtype=bool)
        for s in range(self.N):
            ps = self.site_plaqs[s]
            if len(ps) == 2:
                colors = {int(self.plaq_color[ps[0]]),
                          int(self.plaq_color[ps[1]])}
                missing = ({0, 1, 2} - colors).pop()
                phase[s] = OMEGA3 ** missing
                ok[s] = True
        tri_ok = np.array([all(ok[s] for s in tri) for tri in self.triangles])
        tris = self.triangles[tri_ok]
        phi = (sz[:, tris.reshape(-1)] *
               phase[tris.reshape(-1)]).reshape(len(sz), -1, 3).sum(axis=2)
        Gm = (phi.conj().T @ phi) / len(phi)          # ⟨φ*_t φ_u⟩
        cent = np.array([
            self.pos[tri[0]] + self.disp(self.pos[tri[0]],
                                         self.pos[tri[1]]) / 3.0
            + self.disp(self.pos[tri[0]], self.pos[tri[2]]) / 3.0
            for tri in tris])
        iu = np.triu_indices(len(tris), k=1)
        d = np.linalg.norm(self.disp(cent[iu[0]], cent[iu[1]]), axis=-1)
        r = np.round(d, r_decimals)
        rs = np.unique(r)
        cd = np.array([Gm[iu][r == rv].mean() for rv in rs])
        return rs, cd

    def subsystem_histogram(self, snaps: np.ndarray, inner_sites: list,
                            boundary_sites: list) -> dict:
        """Fig. 5a analysis: histogram of inner-site bitstrings over snapshots
        whose boundary sites are all 0.  Returns {bitstring tuple: count}."""
        snaps = np.atleast_2d(np.asarray(snaps))
        keep = np.all(snaps[:, list(boundary_sites)] == 0, axis=1)
        hist = {}
        for row in snaps[keep][:, list(inner_sites)]:
            key = tuple(int(x) for x in row)
            hist[key] = hist.get(key, 0) + 1
        return hist
