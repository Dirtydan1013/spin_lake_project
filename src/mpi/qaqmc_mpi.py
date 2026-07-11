"""
MPI-parallel QAQMC: each rank runs an independent Markov chain.
Rank 0 gathers samples and writes a single HDF5 file.

Usage:
    mpirun -np 4 python -m src.mpi.qaqmc_mpi --config config.json
    mpirun -np 40 python -m src.mpi.qaqmc_mpi --N 384 --M 5000 ...

Requires: mpi4py, h5py, numpy
"""

import numpy as np
import h5py
import time
import datetime
import argparse
import json
import os

try:
    from mpi4py import MPI
except ImportError:
    raise ImportError("mpi4py is required for MPI mode. Install with: pip install mpi4py")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from src.engines.qaqmc import QAQMC_Rydberg
from src.mpi.equil_progress import run_equil_with_progress
from src.rydberg.lattices import (kagome_loop_string_translations,
                          kagome_multi_size_translations, kagome_bulk_sites,
                          kagome_vertex_sites)


def _build_vbs_triangles(nx, ny):
    """Up-triangles for the VBS/SS order parameters (paper Eq. 5-6).
    Triangle (n1,n2)=(i,j) has corners {(i,j)k0, (i+1,j)k2, (i,j+1)k4},
    for i in 0..nx-2, j in 0..ny-2.  Returns (corners_flat, n1_parity,
    vbs_sign, ss_sign, ref00, ref10) or None if the lattice is too small."""
    if nx < 3 or ny < 2:
        return None
    def site(i, j, k): return (j * nx + i) * 6 + k
    corners = []; par = []; vbs = []; ss = []
    idx_by_ij = {}
    t = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            corners += [site(i, j, 0), site(i + 1, j, 2), site(i, j + 1, 4)]
            par.append(i % 2)
            vbs.append((-1) ** (i + j))
            ss.append((-1) ** i)
            idx_by_ij[(i, j)] = t
            t += 1
    ref00 = idx_by_ij[(0, 0)]
    ref10 = idx_by_ij[(1, 0)]
    return (np.array(corners, np.int32), np.array(par, np.int32),
            np.array(vbs, np.int32), np.array(ss, np.int32), ref00, ref10)


def _build_occ_sf_geometry(nx, ny, a, boundary='open'):
    """Per-site (cell Bravais position R, sublattice α, bulk-complete-cell flag)
    for the kagome bond lattice.  site = (j*nx+i)*6+k → α=k, cell (i,j).
    A cell is 'bulk' if ALL 6 of its atoms are in kagome_bulk_sites.  With
    boundary='periodic' the torus has no edge, so EVERY cell is bulk-complete
    (S_bulk == S_full)."""
    from src.rydberg.lattices import kagome_bulk_sites
    v1 = np.array([a, 0.0]); v2 = np.array([a/2.0, a*np.sqrt(3)/2.0])
    N = 6 * nx * ny
    basis = np.empty(N, dtype=np.int32)
    cell_R = np.empty((N, 2), dtype=np.float64)
    for s in range(N):
        k = s % 6; cell = s // 6; i = cell % nx; j = cell // nx
        basis[s] = k
        cell_R[s] = i * v1 + j * v2
    if boundary == 'periodic':
        in_bulk = np.ones(N, dtype=np.int32)
        return cell_R, basis, in_bulk
    bulk = set(kagome_bulk_sites(nx, ny))
    # cell (i,j) is bulk-complete iff all 6 atoms are bulk
    in_bulk = np.zeros(N, dtype=np.int32)
    for cell in range(nx * ny):
        atoms = [cell*6 + k for k in range(6)]
        if all(s in bulk for s in atoms):
            for s in atoms:
                in_bulk[s] = 1
    return cell_R, basis, in_bulk


def _build_occ2_sf_geometry(nx, ny, a):
    """Alternate occ-SF unit cell: an up+down triangle pair (6 atoms) tiling
    the *bulk* of the kagome bond lattice.  Cell (I,J), I∈0..nx-2, J∈0..ny-2:
        up-triangle   : {(I,J)k0, (I+1,J)k2, (I,J+1)k4}   → α=0,1,2
        down-triangle : {(I+1,J)k1, (I,J+1)k5, (I+1,J+1)k3} → α=3,4,5
    Same triangular Bravais lattice as the hexagon cell (cell_R = I·v1 + J·v2),
    so it is gauge-comparable, but covers (nx-1)(ny-1) cells using every bulk
    atom (25 cells / 150 atoms for 6×6).  Atoms not in any pair get basis=-1.
    Returns (cell_R (N,2), basis (N,))."""
    v1 = np.array([a, 0.0]); v2 = np.array([a/2.0, a*np.sqrt(3)/2.0])
    N = 6 * nx * ny
    basis = np.full(N, -1, dtype=np.int32)
    cell_R = np.zeros((N, 2), dtype=np.float64)
    def site(i, j, k): return (j * nx + i) * 6 + k
    for J in range(ny - 1):
        for I in range(nx - 1):
            R = I * v1 + J * v2
            members = [site(I, J, 0), site(I + 1, J, 2), site(I, J + 1, 4),
                       site(I + 1, J, 1), site(I, J + 1, 5), site(I + 1, J + 1, 3)]
            for alpha, s in enumerate(members):
                basis[s] = alpha
                cell_R[s] = R
    return cell_R, basis


def _build_vbs_triangles_tri(nx, ny, ijk_map):
    """kagome_bond_triangle version of _build_vbs_triangles: same up-triangle
    construction in void coordinates, enumerated over the oversized
    (nx+1)x(ny+1) void grid and kept only when all 3 corner atoms survive the
    crop.  Reference triangles (even/odd column gauge) are chosen as the
    horizontally-adjacent kept pair closest to the lattice centre."""
    corners = []; par = []; vbs = []; ss = []
    idx_by_ij = {}
    t = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            tri = [ijk_map.get((i, j, 0)), ijk_map.get((i + 1, j, 2)),
                   ijk_map.get((i, j + 1, 4))]
            if any(v is None for v in tri):
                continue
            corners += tri
            par.append(i % 2)
            vbs.append((-1) ** (i + j))
            ss.append((-1) ** i)
            idx_by_ij[(i, j)] = t
            t += 1
    if t < 2:
        return None
    # Adjacent (even, odd) reference pair nearest the centre of the void grid.
    ci, cj = nx / 2.0, ny / 2.0
    best = None
    for (i, j), idx in idx_by_ij.items():
        if i % 2 == 0 and (i + 1, j) in idx_by_ij:
            d = (i - ci) ** 2 + (j - cj) ** 2
            if best is None or d < best[0]:
                best = (d, idx, idx_by_ij[(i + 1, j)])
    if best is None:
        return None
    ref00, ref10 = best[1], best[2]
    return (np.array(corners, np.int32), np.array(par, np.int32),
            np.array(vbs, np.int32), np.array(ss, np.int32), ref00, ref10)


def _build_occ_sf_geometry_tri(nx, ny, a, ijk_map, bulk_sites):
    """kagome_bond_triangle version of _build_occ_sf_geometry: hexagon-void
    unit cell on the oversized void grid.  Every atom has a unique (i,j,k)
    label; α = k, cell_R = i·v1 + j·v2.  A cell is bulk-complete iff all 6 of
    its atoms survive the crop AND all 6 are bulk."""
    v1 = np.array([a, 0.0]); v2 = np.array([a/2.0, a*np.sqrt(3)/2.0])
    N = len(ijk_map)
    basis = np.empty(N, dtype=np.int32)
    cell_R = np.empty((N, 2), dtype=np.float64)
    for (i, j, k), s in ijk_map.items():
        basis[s] = k
        cell_R[s] = i * v1 + j * v2
    bulk = set(bulk_sites)
    in_bulk = np.zeros(N, dtype=np.int32)
    for j in range(ny + 1):
        for i in range(nx + 1):
            atoms = [ijk_map.get((i, j, k)) for k in range(6)]
            if all(s is not None and s in bulk for s in atoms):
                for s in atoms:
                    in_bulk[s] = 1
    return cell_R, basis, in_bulk


def _build_occ2_sf_geometry_tri(nx, ny, a, ijk_map):
    """kagome_bond_triangle version of _build_occ2_sf_geometry: up+down
    triangle-pair cells over the oversized void grid, kept when all 6 member
    atoms survive the crop.  Atoms not in any kept pair get basis = -1."""
    v1 = np.array([a, 0.0]); v2 = np.array([a/2.0, a*np.sqrt(3)/2.0])
    N = len(ijk_map)
    basis = np.full(N, -1, dtype=np.int32)
    cell_R = np.zeros((N, 2), dtype=np.float64)
    for J in range(ny + 1):
        for I in range(nx + 1):
            members = [ijk_map.get(q) for q in (
                (I, J, 0), (I + 1, J, 2), (I, J + 1, 4),
                (I + 1, J, 1), (I, J + 1, 5), (I + 1, J + 1, 3))]
            if any(s is None for s in members):
                continue
            R = I * v1 + J * v2
            for alpha, s in enumerate(members):
                basis[s] = alpha
                cell_R[s] = R
    return cell_R, basis


def _lattice_observables(lattice, nx, ny, loop_sizes=None, string_sizes=None,
                         boundary='open', N=None):
    """bulk sites, loop/string translation sets (+meta), and A_v vertex sets
    for either kagome bond lattice.  Returns (bulk, loop_sets, string_sets,
    loop_meta, string_meta, vertex_sets, ijk_map) — ijk_map is None for
    kagome_bond and the (i,j,k)->index map for kagome_bond_triangle.

    With boundary='periodic' (kagome_bond only) there is no edge, so the bulk
    restriction is dropped and density/SF use ALL sites — same as the triangle
    lattice's every-atom convention.

    The vertex 'kagome' lattice (U(1) QDM, paper/u1) uses the medial-honeycomb
    dimer geometry from src.u1.honeycomb_dimer instead of the bond-lattice
    builders: loops = the hexagon Z loops (size 6, closed dimer parity / FM
    denominators), strings = straight dimer-parity strings between plaquette
    centres grouped by length (C_s numerators; all translations and the three
    directions share a group).  bulk = ALL sites, no A_v vertex sets (Z2-only).
    NOTE the engine measures ∏(1-2n) = (-1)^len · ∏σᶻ_paper."""
    if lattice == 'kagome':
        from src.u1.honeycomb_dimer import KagomeU1Geometry
        geo = KagomeU1Geometry(nx, ny, a=1.0, boundary=boundary)
        loop_sets = geo.hexagon_loop_sets()
        loop_meta = [dict(size=6, n_copies=len(loop_sets), offset=0)]
        max_len = max(string_sizes) if string_sizes else None
        string_sets, string_meta = geo.straight_string_sets(max_len)
        return (list(range(geo.N)), loop_sets, string_sets,
                loop_meta, string_meta, [], None)
    if lattice == 'kagome_bond_triangle':
        from src.rydberg.lattices import (
            kagome_triangle_bulk_sites,
            kagome_triangle_ijk_map,
            kagome_triangle_multi_size_translations,
            kagome_triangle_vertex_sites,
        )
        ijk_map = kagome_triangle_ijk_map(nx, ny, 1.0)
        # The oversized void grid admits one more valid size than the plain
        # bond lattice; sizes that end up with zero surviving copies are
        # dropped by the builder.
        if loop_sizes is None:
            loop_sizes = list(range(2, min(nx, ny) + 1))
        if string_sizes is None:
            string_sizes = list(range(1, min(nx, ny) + 1))
        # No bulk restriction on the cropped triangle lattice: every atom
        # (including boundary ones) belongs to at least one complete blockade
        # triangle, so density / SF measurements use ALL sites.
        bulk = sorted(ijk_map.values())
        loop_sets, string_sets, loop_meta, string_meta = (
            kagome_triangle_multi_size_translations(
                nx, ny, loop_sizes=loop_sizes, string_sizes=string_sizes,
                a=1.0, ijk_map=ijk_map))
        vertex_sets = kagome_triangle_vertex_sites(nx, ny, 1.0, ijk_map=ijk_map)
        return bulk, loop_sets, string_sets, loop_meta, string_meta, vertex_sets, ijk_map

    if loop_sizes is None:
        loop_sizes = list(range(2, min(nx, ny)))
    if string_sizes is None:
        string_sizes = list(range(1, min(nx, ny)))
    # Periodic torus has no boundary → every atom is a valid bulk site (density
    # and structure factors over ALL sites); open patch keeps interior-only bulk.
    if boundary == 'periodic':
        bulk = list(range(6 * nx * ny))
    else:
        bulk = kagome_bulk_sites(nx, ny)
    loop_sets, string_sets, loop_meta, string_meta = kagome_multi_size_translations(
        nx, ny, loop_sizes=loop_sizes, string_sizes=string_sizes)
    vertex_sets = kagome_vertex_sites(nx, ny)
    return bulk, loop_sets, string_sets, loop_meta, string_meta, vertex_sets, None


def _build_occ_q_grid(grid_n, a):
    """2D q-grid over one BZ of the triangular Bravais lattice.
    Returns (q_points (grid_n², 2), frac (grid_n², 2) fractional coords for plotting)."""
    b1 = (2*np.pi/a) * np.array([1.0, -1.0/np.sqrt(3)])
    b2 = (2*np.pi/a) * np.array([0.0,  2.0/np.sqrt(3)])
    qs = []; frac = []
    for m in range(grid_n):
        for n in range(grid_n):
            fm, fn = m/grid_n, n/grid_n
            qs.append(fm*b1 + fn*b2)
            frac.append((fm, fn))
    return np.array(qs, dtype=np.float64), np.array(frac, dtype=np.float64)


def _snap_delta_points(prof_delta, requested, sweep='forward'):
    """Profile-point indices nearest each requested δ, restricted to one ramp.

    The δ profile goes up then down, so a requested value exists on BOTH
    ramps; an unrestricted argmin can land on either branch depending on
    sub-grid float offsets (a ~1e-6 coin flip).  Restricting the search to the
    chosen ramp ('forward' = δ-increasing, default) makes the selection
    deterministic; pass sweep='backward' to deliberately sample the down-ramp
    (e.g. for hysteresis comparisons).
    """
    prof_delta = np.asarray(prof_delta, dtype=np.float64)
    turn = np.where(np.diff(prof_delta) < 0)[0]
    n_fwd = int(turn[0]) + 1 if len(turn) else len(prof_delta)
    if sweep == 'backward' and n_fwd < len(prof_delta):
        offset, seg = n_fwd, prof_delta[n_fwd:]
    else:
        offset, seg = 0, prof_delta[:n_fwd]
    req = np.asarray(requested, dtype=np.float64)
    return np.unique(np.array(
        [offset + int(np.argmin(np.abs(seg - d))) for d in req], dtype=np.int32))


def _build_sf_q_points(sf_q_points_file=None, sf_q_path=None, sf_q_n=20, a=1.0):
    """Return q-points array of shape (n_q, 2), or None if SF not requested.

    Sources (first match wins):
      - sf_q_points_file: path to .npy or .json giving an (n_q, 2) array.
      - sf_q_path: named high-symmetry path for the kagome BZ
                   (triangular Bravais, lattice constant `a`).
        Currently supported:
          'GammaMKGamma'  : Γ → M → K → Γ
          'GammaKMGamma'  : Γ → K → M → Γ
          'GammaM'        : Γ → M
          'GammaK'        : Γ → K
        sf_q_n = points per segment (endpoints repeated between segments).
    """
    if sf_q_points_file:
        ext = os.path.splitext(sf_q_points_file)[1].lower()
        if ext == '.npy':
            q = np.load(sf_q_points_file)
        elif ext in ('.json', '.txt'):
            with open(sf_q_points_file) as f:
                q = np.array(json.load(f), dtype=np.float64)
        else:
            q = np.loadtxt(sf_q_points_file)
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        if q.shape[1] != 2:
            raise ValueError(f"sf_q_points file must give (n_q, 2); got {q.shape}")
        return q

    if not sf_q_path:
        return None

    # Triangular Bravais lattice (v1=(a,0), v2=(a/2, a√3/2)) reciprocal:
    #   b1 = (2π/a)(1, -1/√3),   b2 = (2π/a)(0, 2/√3)
    Gamma = np.array([0.0, 0.0])
    M_pt  = np.array([np.pi / a, np.pi / (a * np.sqrt(3))])
    K_pt  = np.array([4.0 * np.pi / (3.0 * a), 0.0])

    paths = {
        'GammaMKGamma': [Gamma, M_pt, K_pt, Gamma],
        'GammaKMGamma': [Gamma, K_pt, M_pt, Gamma],
        'GammaM':       [Gamma, M_pt],
        'GammaK':       [Gamma, K_pt],
    }
    if sf_q_path not in paths:
        raise ValueError(f"unknown sf_q_path '{sf_q_path}'; choices={list(paths)}")
    nodes = paths[sf_q_path]
    n_seg = max(1, int(sf_q_n))

    pts = []
    for k in range(len(nodes) - 1):
        q0, q1 = nodes[k], nodes[k + 1]
        for t in np.linspace(0.0, 1.0, n_seg, endpoint=False):
            pts.append((1 - t) * q0 + t * q1)
    pts.append(np.asarray(nodes[-1], dtype=np.float64))
    return np.asarray(pts, dtype=np.float64)


def _parse_measure_every(me, default=1):
    """Parse measure_every into (me_density, me_zl, me_cml).

    Accepts:
      - int:  all three observables use the same interval
      - dict: keys 'density', 'Z_l', 'C_m_l' (missing keys default to `default`)
    Returns (me_density, me_zl, me_cml) as positive ints.
    """
    if isinstance(me, int):
        v = max(1, me)
        return v, v, v
    d = me
    me_d = max(1, int(d.get('density', default)))
    me_z = max(1, int(d.get('Z_l',     default)))
    me_c = max(1, int(d.get('C_m_l',   default)))
    return me_d, me_z, me_c


def _make_tqdm_callback(total, desc):
    """Create a progress callback compatible with C++ run(..., callback, ...)."""
    if not HAS_TQDM:
        return None, None

    bar = tqdm(total=total, desc=desc, unit='step', leave=True)

    def _cb(i_done, _n_total, _stage):
        # C++ callback reports cumulative completed steps.
        bar.update(max(0, int(i_done) - bar.n))

    return _cb, bar


def run_mpi(*, N, M, Omega=1.0, Rb=1.2, delta_min=0.0, delta_max=1.0,
            pos=None, epsilon=0.01, seed=42, n_equil=5000, n_samples=10000,
            measure_every=1, filepath='data/qaqmc_mpi.h5', neighbor_cutoff=None,
            delta_groups=600, omp_threads=0,
            compression='gzip', compression_opts=4,
            checkpoint_every=0, verbose=True):
    """
    MPI-parallel QAQMC simulation.

    Each rank runs an independent Markov chain with a different seed.
    Rank 0 creates the HDF5 file and gathers samples from all ranks.

    Parameters
    ----------
    N, M, Omega, Rb, delta_min, delta_max, pos, epsilon, seed:
        Same as QAQMC_Rydberg.__init__
    n_equil, n_samples:
        Per-rank equilibration and total samples (split across ranks)
    measure_every:
        Record one operator-string sample every this many MC steps (default 1).
        Useful to reduce autocorrelation at the cost of fewer samples per wall time.
    filepath:
        Output HDF5 path (written by rank 0)
    neighbor_cutoff, delta_groups, omp_threads:
        Engine options passed to QAQMC_Rydberg (delta_groups must be > 0)
    compression, compression_opts:
        HDF5 compression settings
    checkpoint_every:
        Save checkpoint every N samples (per-rank, 0 = disabled).
    verbose:
        Print progress info (only rank 0 prints)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # Split total samples across ranks
    base = n_samples // n_ranks
    rem = n_samples % n_ranks
    my_n_samples = base + (1 if rank < rem else 0)

    # Compute write offsets for each rank
    counts = np.array([base + (1 if r < rem else 0) for r in range(n_ranks)])
    offsets = np.zeros(n_ranks, dtype=int)
    for r in range(1, n_ranks):
        offsets[r] = offsets[r - 1] + counts[r - 1]
    my_offset = offsets[rank]

    if verbose and rank == 0:
        print(f"[MPI] {n_ranks} ranks, {n_samples} total samples "
              f"({my_n_samples} per rank ± 1), measure_every={measure_every}")

    # Each rank gets a different seed
    rank_seed = seed + rank * 9973  # large prime to avoid correlation

    # Create the engine
    engine = QAQMC_Rydberg(
        N=N, M=M, Omega=Omega, Rb=Rb,
        delta_min=delta_min, delta_max=delta_max,
        pos=pos, epsilon=epsilon, seed=rank_seed,
        verbose=False, use_cpp=True, omp_threads=omp_threads,
        neighbor_cutoff=neighbor_cutoff, delta_groups=delta_groups,
    )

    M2 = engine.M_total

    # Equilibration
    comm.Barrier()
    t0 = time.perf_counter()
    eq_cb = None
    eq_bar = None
    if verbose and rank == 0 and n_equil > 0:
        eq_cb, eq_bar = _make_tqdm_callback(n_equil, "[MPI] Equilibration (rank0)")

    if engine._cpp_engine is not None:
        try:
            engine._cpp_engine.run(n_equil, 0, eq_cb, 1000 if eq_cb is not None else 1)
        except TypeError:
            engine._cpp_engine.run(n_equil, 0)
    else:
        it = range(n_equil)
        if verbose and rank == 0 and HAS_TQDM and n_equil > 0:
            it = tqdm(it, total=n_equil, desc="[MPI] Equilibration (rank0)", unit='step', leave=True)
        for _ in it:
            engine.mc_step()
    if eq_bar is not None:
        eq_bar.close()

    t_equil = time.perf_counter() - t0
    comm.Barrier()

    if verbose and rank == 0:
        max_eq = comm.reduce(t_equil, op=MPI.MAX, root=0)
        print(f"[MPI] Equilibration done in {max_eq:.1f}s (slowest rank)")
    else:
        comm.reduce(t_equil, op=MPI.MAX, root=0)

    # Sampling
    t0 = time.perf_counter()
    sa_cb = None
    sa_bar = None
    if verbose and rank == 0 and my_n_samples > 0:
        sa_cb, sa_bar = _make_tqdm_callback(my_n_samples, "[MPI] Sampling (rank0)")

    if engine._cpp_engine is not None:
        try:
            my_types, my_sites = engine._cpp_engine.run(
                0, my_n_samples,
                sa_cb, 1000 if sa_cb is not None else 1,
                measure_every)
        except TypeError:
            my_types, my_sites = engine._cpp_engine.run(0, my_n_samples)
    else:
        my_types = np.empty((my_n_samples, M2), dtype=np.int8)
        my_sites = np.empty((my_n_samples, M2), dtype=np.int32)
        it = range(my_n_samples)
        if verbose and rank == 0 and HAS_TQDM and my_n_samples > 0:
            it = tqdm(it, total=my_n_samples, desc="[MPI] Sampling (rank0)", unit='sample', leave=True)
        for i in it:
            engine.mc_step()
            my_types[i] = engine.op_types[:M2].astype(np.int8)
            my_sites[i] = engine.op_sites[:M2].astype(np.int32)
    if sa_bar is not None:
        sa_bar.close()

    t_sample = time.perf_counter() - t0
    comm.Barrier()

    if verbose and rank == 0:
        max_sa = comm.reduce(t_sample, op=MPI.MAX, root=0)
        print(f"[MPI] Sampling done in {max_sa:.1f}s (slowest rank)")
    else:
        comm.reduce(t_sample, op=MPI.MAX, root=0)

    # ── Rank 0 writes HDF5 ───────────────────────────────────────────────
    if rank == 0:
        kw = dict(compression=compression, compression_opts=compression_opts) \
             if compression == 'gzip' else dict(compression=compression) \
             if compression else {}

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            # metadata
            pg = f.create_group('params')
            pg.attrs['N'] = N
            pg.attrs['Omega'] = Omega
            pg.attrs['Rb'] = Rb
            pg.attrs['delta_min'] = delta_min
            pg.attrs['delta_max'] = delta_max
            pg.attrs['M'] = M
            pg.attrs['epsilon'] = epsilon
            pg.attrs['seed'] = seed
            pg.attrs['n_equil'] = n_equil
            pg.attrs['n_samples'] = n_samples
            pg.attrs['measure_every'] = measure_every
            pg.attrs['n_ranks'] = n_ranks
            pg.attrs['timestamp'] = datetime.datetime.utcnow().isoformat()

            if neighbor_cutoff is not None:
                pg.attrs['neighbor_cutoff'] = neighbor_cutoff

            # geometry
            gg = f.create_group('geometry')
            pos_stored = pos if pos is not None \
                         else np.arange(N).reshape(-1, 1).astype(np.float64)
            gg.create_dataset('pos', data=pos_stored.astype(np.float64))

            # schedule
            sg = f.create_group('schedule')
            delta_sched = np.array(engine._cpp_engine.delta_schedule
                                   if engine._cpp_engine else
                                   np.zeros(M2), dtype=np.float64)
            sg.create_dataset('delta_schedule', data=delta_sched)

            # sample datasets
            smg = f.create_group('samples')
            ds_types = smg.create_dataset('op_types', shape=(n_samples, M2),
                                          dtype='int8', **kw)
            ds_sites = smg.create_dataset('op_sites', shape=(n_samples, M2),
                                          dtype='int32', **kw)

            # Write rank 0's data
            ds_types[my_offset:my_offset + my_n_samples] = my_types
            ds_sites[my_offset:my_offset + my_n_samples] = my_sites

            # Receive and write data from other ranks
            for src in range(1, n_ranks):
                src_count = int(counts[src])
                src_offset = int(offsets[src])
                buf_t = np.empty((src_count, M2), dtype=np.int8)
                buf_s = np.empty((src_count, M2), dtype=np.int32)
                comm.Recv(buf_t, source=src, tag=100 + src)
                comm.Recv(buf_s, source=src, tag=200 + src)
                ds_types[src_offset:src_offset + src_count] = buf_t
                ds_sites[src_offset:src_offset + src_count] = buf_s

        if verbose:
            print(f"[MPI] Saved {n_samples} samples → {filepath}")
    else:
        # Non-root ranks send their data to rank 0
        comm.Send(np.ascontiguousarray(my_types), dest=0, tag=100 + rank)
        comm.Send(np.ascontiguousarray(my_sites), dest=0, tag=200 + rank)

    comm.Barrier()
    return filepath


def run_mpi_onthefly(*, N, M, Omega=1.0, Rb=1.2, delta_min=0.0, delta_max=1.0,
                     pos=None, epsilon=0.01, seed=42, n_equil=5000, n_samples=10000,
                     measure_every=1, filepath='data/qaqmc_onthefly.h5',
                     neighbor_cutoff=None, delta_groups=600, omp_threads=0,
                     nx=6, ny=6,
                     loop_sizes=None, string_sizes=None,
                     lattice='kagome_bond', verbose=True):
    """
    MPI-parallel QAQMC with on-the-fly observable computation.

    Each rank runs an independent chain. Rank 0 gathers per-sample scalars
    (density, Z_l, C_m_l) and writes them to HDF5.

    measure_every : int or dict
        int:  all observables use the same interval.
        dict: {'density': 300, 'Z_l': 20, 'C_m_l': 70} — per-observable intervals.
        n_samples refers to the finest (smallest) interval.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # Parse per-observable measure_every
    me_density, me_zl, me_cml = _parse_measure_every(measure_every)
    min_me = min(me_density, me_zl, me_cml)

    # n_samples is defined for the finest observable
    base = n_samples // n_ranks
    rem  = n_samples % n_ranks
    my_n_samples = base + (1 if rank < rem else 0)   # finest samples for this rank

    # Actual per-observable sample counts per rank
    def _rank_count(me):
        return (my_n_samples * min_me) // me

    if verbose and rank == 0:
        n_d = (n_samples * min_me) // me_density
        n_z = (n_samples * min_me) // me_zl
        n_c = (n_samples * min_me) // me_cml
        print(f"[MPI-OTF] {n_ranks} ranks, n_samples(finest)={n_samples}, "
              f"me={{density:{me_density}, Z_l:{me_zl}, C_m_l:{me_cml}}}")
        print(f"[MPI-OTF] actual samples: density={n_d}, Z_l={n_z}, C_m_l={n_c}")

    # Each rank gets a different seed
    rank_seed = seed + rank * 9973

    # Create the engine
    engine = QAQMC_Rydberg(
        N=N, M=M, Omega=Omega, Rb=Rb,
        delta_min=delta_min, delta_max=delta_max,
        pos=pos, epsilon=epsilon, seed=rank_seed,
        verbose=False, use_cpp=True, omp_threads=omp_threads,
        neighbor_cutoff=neighbor_cutoff, delta_groups=delta_groups,
    )

    if engine._cpp_engine is None:
        raise RuntimeError("On-the-fly mode requires the C++ backend")

    # Lattice-aware bulk sites + loop/string observable sites (multi-size).
    (bulk, loop_sets, string_sets, loop_meta, string_meta,
     _vertex_sets, _tri_map) = _lattice_observables(
        lattice, nx, ny, loop_sizes=loop_sizes, string_sizes=string_sizes)
    engine._cpp_engine.set_bulk_sites(bulk)
    engine._cpp_engine.set_observable_sites(loop_sets, string_sets)
    loop_sizes = [m['size'] for m in loop_meta]
    string_sizes = [m['size'] for m in string_meta]

    if verbose and rank == 0:
        print(f"[MPI-OTF] bulk sites: {len(bulk)} / {N} (interior atoms only)")
        print(f"[MPI-OTF] loop sizes auto-selected: {list(loop_sizes)}")
        print(f"[MPI-OTF] string sizes auto-selected: {list(string_sizes)}")
        for m in loop_meta:
            print(f"[MPI-OTF]   loop size={m['size']}: {m['n_copies']} copies "
                  f"(offset {m['offset']})")
        for m in string_meta:
            print(f"[MPI-OTF]   string size={m['size']}: {m['n_copies']} copies "
                  f"(offset {m['offset']})")

    # Equilibration
    comm.Barrier()
    t0 = time.perf_counter()
    eq_cb = None
    eq_bar = None
    if verbose and rank == 0 and n_equil > 0:
        eq_cb, eq_bar = _make_tqdm_callback(n_equil, "[MPI-OTF] Equilibration (rank0)")

    try:
        engine._cpp_engine.run(n_equil, 0, eq_cb, 1000 if eq_cb is not None else 1)
    except TypeError:
        engine._cpp_engine.run(n_equil, 0)
    if eq_bar is not None:
        eq_bar.close()

    t_equil = time.perf_counter() - t0
    comm.Barrier()

    if verbose and rank == 0:
        max_eq = comm.reduce(t_equil, op=MPI.MAX, root=0)
        print(f"[MPI-OTF] Equilibration done in {max_eq:.1f}s (slowest rank)")
    else:
        comm.reduce(t_equil, op=MPI.MAX, root=0)

    # Sampling with on-the-fly measurement
    t0 = time.perf_counter()
    sa_cb = None
    sa_bar = None
    if verbose and rank == 0 and my_n_samples > 0:
        sa_cb, sa_bar = _make_tqdm_callback(my_n_samples, "[MPI-OTF] Sampling (rank0)")

    try:
        result = engine._cpp_engine.run_onthefly(
            0, my_n_samples, me_density, me_zl, me_cml,
            sa_cb, 1000 if sa_cb is not None else 1)
    except TypeError:
        result = engine._cpp_engine.run_onthefly(
            0, my_n_samples, me_density, me_zl, me_cml)
    if sa_bar is not None:
        sa_bar.close()

    my_density = np.ascontiguousarray(result['density'], dtype=np.float64)
    # Z_l / C_m_l now (n_zl, n_loop_size_groups) and (n_cml, n_string_size_groups):
    # C++ already averaged over copies within each size group (signed, no |·|).
    my_z_l   = np.ascontiguousarray(result['Z_l'],   dtype=np.float64)
    my_c_m_l = np.ascontiguousarray(result['C_m_l'], dtype=np.float64)
    n_loop_sg   = my_z_l.shape[1]   if my_z_l.ndim   == 2 else 1
    n_string_sg = my_c_m_l.shape[1] if my_c_m_l.ndim == 2 else 1

    t_sample = time.perf_counter() - t0
    comm.Barrier()

    if verbose and rank == 0:
        max_sa = comm.reduce(t_sample, op=MPI.MAX, root=0)
        print(f"[MPI-OTF] Sampling done in {max_sa:.1f}s (slowest rank)")
    else:
        comm.reduce(t_sample, op=MPI.MAX, root=0)

    # ── Gather and write HDF5 ────────────────────────────────────────────
    fine_counts = np.array([base + (1 if r < rem else 0) for r in range(n_ranks)])

    def _obs_counts(me):
        return np.array([(fine_counts[r] * min_me) // me for r in range(n_ranks)])

    def _obs_offsets(counts_arr):
        off = np.zeros(n_ranks, dtype=int)
        for r in range(1, n_ranks):
            off[r] = off[r-1] + counts_arr[r-1]
        return off

    counts_d = _obs_counts(me_density)
    counts_z = _obs_counts(me_zl)
    counts_c = _obs_counts(me_cml)
    off_d = _obs_offsets(counts_d)
    off_z = _obs_offsets(counts_z)
    off_c = _obs_offsets(counts_c)

    n_total_d = int(counts_d.sum())
    n_total_z = int(counts_z.sum())
    n_total_c = int(counts_c.sum())

    if rank == 0:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        all_density = np.empty(n_total_d,                  dtype=np.float64)
        all_z_l     = np.empty((n_total_z, n_loop_sg),     dtype=np.float64)
        all_c_m_l   = np.empty((n_total_c, n_string_sg),   dtype=np.float64)

        all_density[off_d[0]:off_d[0] + counts_d[0]] = my_density
        all_z_l    [off_z[0]:off_z[0] + counts_z[0]] = my_z_l
        all_c_m_l  [off_c[0]:off_c[0] + counts_c[0]] = my_c_m_l

        for src in range(1, n_ranks):
            buf_d = np.empty(counts_d[src],                  dtype=np.float64)
            buf_z = np.empty((counts_z[src], n_loop_sg),     dtype=np.float64)
            buf_c = np.empty((counts_c[src], n_string_sg),   dtype=np.float64)
            comm.Recv(buf_d, source=src, tag=300 + src)
            comm.Recv(buf_z, source=src, tag=400 + src)
            comm.Recv(buf_c, source=src, tag=500 + src)
            all_density[off_d[src]:off_d[src] + counts_d[src]] = buf_d
            all_z_l    [off_z[src]:off_z[src] + counts_z[src]] = buf_z
            all_c_m_l  [off_c[src]:off_c[src] + counts_c[src]] = buf_c

        with h5py.File(filepath, 'w') as f:
            pg = f.create_group('params')
            pg.attrs['N']            = N
            pg.attrs['Omega']        = Omega
            pg.attrs['Rb']           = Rb
            pg.attrs['delta_min']    = delta_min
            pg.attrs['delta_max']    = delta_max
            pg.attrs['M']            = M
            pg.attrs['epsilon']      = epsilon
            pg.attrs['seed']         = seed
            pg.attrs['n_equil']      = n_equil
            pg.attrs['n_samples']    = n_samples
            pg.attrs['me_density']   = me_density
            pg.attrs['me_zl']        = me_zl
            pg.attrs['me_cml']       = me_cml
            pg.attrs['n_ranks']      = n_ranks
            pg.attrs['nx']           = nx
            pg.attrs['ny']           = ny
            pg.attrs['timestamp']    = datetime.datetime.utcnow().isoformat()
            if neighbor_cutoff is not None:
                pg.attrs['neighbor_cutoff'] = neighbor_cutoff

            gg = f.create_group('geometry')
            pos_stored = pos if pos is not None \
                         else np.arange(N).reshape(-1, 1).astype(np.float64)
            gg.create_dataset('pos', data=pos_stored.astype(np.float64))

            sg = f.create_group('schedule')
            delta_sched = np.array(engine._cpp_engine.delta_schedule, dtype=np.float64)
            sg.create_dataset('delta_schedule', data=delta_sched)

            og = f.create_group('observables')
            og.create_dataset('density', data=all_density)
            # Z_l / C_m_l: (n_total, n_size_groups) — signed mean over copies within each size group.
            # <Z(l)>_size = all_z_l[:, size_index].mean(axis=0)
            og.create_dataset('Z_l',   data=all_z_l)
            og.create_dataset('C_m_l', data=all_c_m_l)

            lg = f.create_group('observable_sites')
            for idx, lp in enumerate(loop_sets):
                lg.create_dataset(f'loop_{idx}', data=np.array(lp, dtype=np.int32))
            for idx, st in enumerate(string_sets):
                lg.create_dataset(f'string_{idx}', data=np.array(st, dtype=np.int32))
            # Size metadata: loop_size{s}_index maps the logical loop_size to its
            # column index in Z_l (C++ groups by first-occurrence of set length,
            # which matches loop_meta order). _n_copies is diagnostic (count of
            # copies averaged into the group).
            mg = f.create_group('size_meta')
            for i, m in enumerate(loop_meta):
                mg.attrs[f'loop_size{m["size"]}_index']    = i
                mg.attrs[f'loop_size{m["size"]}_n_copies'] = m['n_copies']
            for i, m in enumerate(string_meta):
                mg.attrs[f'string_size{m["size"]}_index']    = i
                mg.attrs[f'string_size{m["size"]}_n_copies'] = m['n_copies']

        if verbose:
            print(f"[MPI-OTF] Saved {n_samples} samples → {filepath}")
    else:
        comm.Send(my_density, dest=0, tag=300 + rank)
        comm.Send(np.ascontiguousarray(my_z_l),   dest=0, tag=400 + rank)
        comm.Send(np.ascontiguousarray(my_c_m_l), dest=0, tag=500 + rank)

    comm.Barrier()
    return filepath


# Occupation-SF matrices dominate the on-disk chunk size.  They are accumulated
# in float64 inside the C++ engine (full precision through the sum), but stored
# as float32 — a structure-factor bin doesn't need double precision, and this
# halves the largest datasets.  The raw occ_state snapshot (int8) is untouched.
_OCC_FLOAT32_KEYS = frozenset({
    'occ_S_full_re', 'occ_S_full_im', 'occ_S_bulk_re', 'occ_S_bulk_im',
    'occ_nprof', 'occ2_S_re', 'occ2_S_im',
})


def _store_array(key, value):
    """Cast occ-SF float64 matrices to float32 for storage; pass others through."""
    arr = np.ascontiguousarray(value)
    if key in _OCC_FLOAT32_KEYS and arr.dtype == np.float64:
        return arr.astype(np.float32)
    return arr


def _write_result_h5(path, arrays, attrs=None):
    """Atomically write a dict of arrays to a self-contained HDF5 file.

    Writes to ``path + '.tmp'`` first, then ``os.replace()`` into place, so a
    crash mid-write can never corrupt a previously-completed checkpoint file.
    Arrays larger than 1 MiB are gzip-compressed (op-type-like data shrinks a
    lot; batch means less so, but the cost is negligible at this cadence).
    """
    tmp = path + '.tmp'
    with h5py.File(tmp, 'w') as f:
        if attrs:
            for k, v in attrs.items():
                f.attrs[k] = v
        for k, v in arrays.items():
            arr = np.ascontiguousarray(v)
            if arr.nbytes > (1 << 20):
                f.create_dataset(k, data=arr, compression='gzip', compression_opts=4)
            else:
                f.create_dataset(k, data=arr)
    os.replace(tmp, path)


def run_mpi_profile(*, N, M, Omega=1.0, Rb=1.2, delta_min=0.0, delta_max=1.0,
                    pos=None, epsilon=0.01, seed=42, n_equil=0, n_samples=1000,
                    measure_every=1, profile_step=10000, batch_size=1000,
                    checkpoint=0, checkpoint_every_batches=0,
                    filepath='data/qaqmc_profile.h5',
                    neighbor_cutoff=None,
                    loop_sizes=None, string_sizes=None,
                    delta_groups=600, omp_threads=0, nx=6, ny=6,
                    sf_q_points=None, sf_delta_points=None,
                    snapshot_deltas=None, n_snapshots=0, snapshot_sweep='forward',
                    occ_sf_delta_points=None, occ_sf_grid_n=0, occ_sf_nbatch=4,
                    occ_sf_sweep='forward',
                    lattice='kagome_bond', boundary='open', box_vectors=None,
                    config_in=None, config_out=None,
                    equil_progress_every=500,
                    permute_site_labels=True,
                    verbose=True):
    """
    MPI-parallel QAQMC asymmetric profile measurement.

    Each rank runs an independent chain. For each sample, the full operator
    string is traversed and observables (density, Z_l, C_m_l) are recorded
    every `profile_step` imaginary-time slices, giving the asymmetric profile
    as a function of the annealing parameter delta(p).

    Parameters
    ----------
    measure_every : int or dict
        int:  all observables use the same interval.
        dict: {'density': 300, 'Z_l': 20, 'C_m_l': 70} — per-observable intervals.
        n_samples refers to the finest (smallest) interval.
    profile_step : int
        Sample one point along the path every this many slices (default 10000).
        n_points = M_total / profile_step.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    # Parse per-observable measure_every
    me_density, me_zl, me_cml = _parse_measure_every(measure_every)
    min_me = min(me_density, me_zl, me_cml)

    base = n_samples // n_ranks
    rem  = n_samples % n_ranks
    my_n_samples = base + (1 if rank < rem else 0)   # finest samples for this rank

    if verbose and rank == 0:
        n_d = (n_samples * min_me) // me_density
        n_z = (n_samples * min_me) // me_zl
        n_c = (n_samples * min_me) // me_cml
        print(f"[MPI-PROF] {n_ranks} ranks, n_samples(finest)={n_samples}, "
              f"me={{density:{me_density}, Z_l:{me_zl}, C_m_l:{me_cml}}}, "
              f"profile_step={profile_step}")
        print(f"[MPI-PROF] actual samples: density={n_d}, Z_l={n_z}, C_m_l={n_c}")

    rank_seed = seed + rank * 9973

    if verbose and rank == 0:
        print(f"[MPI-PROF] spatial boundary: {boundary}")

    # ── Per-rank site-label permutation (scan-order decorrelation) ──────────
    # The updates visit sites in label order, and that shared fixed order was
    # shown to deterministically select the ordered-phase domain pattern:
    # independent-seed chains freeze into the SAME real-space stripe (see
    # scripts/experiments/scan_order_bias_probe.py; permuting labels restores
    # domain diversity and the C3 orientation balance).  With this option each
    # rank runs the engine on randomly relabelled sites (identical physics,
    # different scan geometry); every site-resolved input is mapped into the
    # engine labelling and every site-resolved output is mapped back, so the
    # stored files stay in canonical labels.  Warm-start configs record the
    # permutation and continue under it.
    cfg = None
    if config_in:
        from src.mpi.chunk_io import check_config_compat, load_warm_config
        cfg = load_warm_config(config_in, rank, verbose=(rank == 0 and verbose))
        if cfg is None:
            raise FileNotFoundError(f"[warm-start] no rank*.h5 files in {config_in}")

    from src.mpi.site_permutation import resolve_site_permutation
    site_perm, inv_perm = resolve_site_permutation(
        N, rank_seed, permute_site_labels, cfg=cfg, label="MPI-PROF")
    pos_engine = (pos if site_perm is None
                  else np.ascontiguousarray(np.asarray(pos)[site_perm]))

    def _to_engine(idx):
        """Map canonical site indices to engine labels."""
        arr = np.asarray(idx, dtype=np.int64)
        return arr if inv_perm is None else inv_perm[arr]

    def _unpermute_res(res):
        """Map the site-resolved outputs of run_profile back to canonical labels."""
        if inv_perm is None:
            return res
        for key in ('snapshots', 'occ_nprof'):
            if key in res:
                res[key] = np.ascontiguousarray(np.asarray(res[key])[..., inv_perm])
        return res

    engine = QAQMC_Rydberg(
        N=N, M=M, Omega=Omega, Rb=Rb,
        delta_min=delta_min, delta_max=delta_max,
        pos=pos_engine, epsilon=epsilon, seed=rank_seed,
        verbose=False, use_cpp=True, omp_threads=omp_threads,
        neighbor_cutoff=neighbor_cutoff, delta_groups=delta_groups,
        box_vectors=box_vectors,
    )

    if engine._cpp_engine is None:
        raise RuntimeError("Profile mode requires the C++ backend")

    # ── Warm start: load a previously saved final configuration ─────────────
    # When config_in is given, each rank installs its saved operator string
    # (+ RNG state) and thermalization is skipped (n_equil forced to 0).
    if cfg is not None:
        check_config_compat(cfg, dict(N=N, M_total=2 * M, lattice=str(lattice),
                                      boundary=str(boundary)),
                            f"profile rank {rank}")
        types = np.ascontiguousarray(cfg["op_types"], dtype=np.int32)
        sites = np.ascontiguousarray(cfg["op_sites"], dtype=np.int32)
        if len(types) != 2 * M:
            raise ValueError(f"[warm-start] op string length {len(types)} != M_total {2*M}")
        engine._cpp_engine.set_op_string(types, sites)
        rng_state = cfg["attrs"].get("rng_state")
        if rng_state is not None and int(cfg["attrs"].get("rank", -1)) == rank:
            engine._cpp_engine.set_rng_state(rng_state)
        n_equil = 0
        if rank == 0 and verbose:
            print(f"[MPI-PROF] warm start from {config_in} — n_equil forced to 0",
                  flush=True)

    # ── Warm start: save the final configuration at every exit ──────────────
    # Each rank stashes its final operator string + RNG state so a later run
    # can `config_in` it and skip thermalization entirely.  Configs live in
    # their OWN dir (one rank{r}.h5 each), never inside the observable chunk
    # files — so the whole set is reclaimable with a single `rm -rf` and the
    # data files never need an h5repack.  In the chunked run the config dir is
    # <run_dir>/configs (default_dir); the legacy single-call path uses
    # config_out or <filepath>_configs.  An explicit --config_out always wins.
    def _save_final_config(default_dir=None):
        out_dir = config_out or default_dir
        if not out_dir and filepath:
            base = str(filepath)
            out_dir = (base[:-3] if base.endswith('.h5') else base) + '_configs'
        if not out_dir:
            return
        config_datasets = dict(
            op_types=np.asarray(engine._cpp_engine.op_types, dtype=np.int32),
            op_sites=np.asarray(engine._cpp_engine.op_sites, dtype=np.int32))
        if site_perm is not None:
            config_datasets['site_perm'] = np.asarray(site_perm, dtype=np.int32)
        config_attrs = dict(N=int(N), M_total=int(2 * M), lattice=str(lattice),
                            boundary=str(boundary), seed=int(rank_seed),
                            rng_state=engine._cpp_engine.get_rng_state())
        from src.mpi.chunk_io import RankChunkWriter
        with RankChunkWriter(out_dir, rank) as w:
            w.write_final_config(datasets=config_datasets, attrs=config_attrs)
        if rank == 0 and verbose:
            print(f"[MPI-PROF] final configs saved → {out_dir}", flush=True)

    # Lattice-aware observable geometry (kagome_bond or kagome_bond_triangle).
    # periodic kagome_bond drops the bulk restriction (density/SF over ALL sites).
    (bulk, loop_sets, string_sets, loop_meta, string_meta,
     vertex_sets, tri_ijk_map) = _lattice_observables(
        lattice, nx, ny, loop_sizes=loop_sizes, string_sizes=string_sizes,
        boundary=boundary, N=N)
    engine._cpp_engine.set_bulk_sites([int(s) for s in _to_engine(bulk)])

    # A_v vertex operator: append to loop_sets, track offset
    n_vertex    = len(vertex_sets)
    vertex_offset = len(loop_sets)
    all_loop_sets = loop_sets + vertex_sets

    engine._cpp_engine.set_observable_sites(
        [[int(s) for s in _to_engine(st)] for st in all_loop_sets],
        [[int(s) for s in _to_engine(st)] for st in string_sets])

    # VBS/SS order parameters (paper Eq. 5-6): measured at every profile point,
    # batched like Z_l.  Always on for kagome BOND lattices large enough; the
    # vertex 'kagome' lattice has no bond-lattice VBS geometry.
    if lattice == 'kagome':
        vbs_tri = None
    elif lattice == 'kagome_bond_triangle':
        vbs_tri = _build_vbs_triangles_tri(nx, ny, tri_ijk_map)
    else:
        vbs_tri = _build_vbs_triangles(nx, ny)
    do_vbs = vbs_tri is not None
    if do_vbs:
        corners, par, vsign, ssign, ref00, ref10 = vbs_tri
        engine._cpp_engine.set_vbs_triangles(
            np.asarray(_to_engine(corners), dtype=np.int32),
            par, vsign, ssign, ref00, ref10)

    # Optional: dimer structure factor S_d(q).  Setting q-points enables the
    # SF accumulator inside measure_profile (same forward walk, no extra MC).
    n_q = 0
    if sf_q_points is not None and len(sf_q_points) > 0:
        sf_q_points = np.asarray(sf_q_points, dtype=np.float64)
        n_q = sf_q_points.shape[0]
        engine._cpp_engine.set_dimer_sf_q_points(sf_q_points)

    # Profile grid (point k → slice p_indices[k], δ = delta_sched[p_indices[k]])
    M_total_  = engine._cpp_engine.M_total
    n_points_ = M_total_ // profile_step
    delta_sched_full = np.array(engine._cpp_engine.delta_schedule, dtype=np.float64)
    prof_p_idx   = np.array([(k + 1) * profile_step - 1 for k in range(n_points_)],
                            dtype=np.int64)
    prof_delta   = delta_sched_full[prof_p_idx]   # δ at each profile point

    # Optional: full-state snapshots at chosen δ values (snapped to the chosen
    # ramp of the profile grid; forward by default).
    snap_pt_indices = np.array([], dtype=np.int32)
    if snapshot_deltas is not None and len(snapshot_deltas) > 0 and n_snapshots > 0:
        snap_pt_indices = _snap_delta_points(prof_delta, snapshot_deltas,
                                             sweep=snapshot_sweep)
        engine._cpp_engine.set_snapshot_point_indices(snap_pt_indices)
    n_snap_pts = len(snap_pt_indices)

    # Optional: sublattice-resolved occupation SF matrix at chosen δ on a 2D q-grid.
    occ_pt_indices = np.array([], dtype=np.int32)
    occ_q_points = None; occ_q_frac = None
    occ_cell_R = occ_basis = occ_in_bulk = None
    occ2_cell_R = occ2_basis = None
    do_occ = (occ_sf_delta_points is not None and len(occ_sf_delta_points) > 0
              and occ_sf_grid_n > 0 and occ_sf_nbatch > 0)
    if do_occ and lattice == 'kagome':
        # The occ-SF cell decomposition is written in kagome_bond 6-atom void
        # coordinates; the vertex lattice needs its own 3-site-basis geometry
        # (U(1) module, not yet wired).  Fail loudly rather than mis-index.
        raise ValueError("--occ-sf-delta-points is not supported yet for "
                         "lattice='kagome' (vertex lattice)")
    if do_occ:
        # q·R phases are dimensionless (the lattice constant cancels between the
        # reciprocal vectors and the cell positions), so use a=1.0 internally.
        if lattice == 'kagome_bond_triangle':
            occ_cell_R, occ_basis, occ_in_bulk = _build_occ_sf_geometry_tri(
                nx, ny, 1.0, tri_ijk_map, bulk)
        else:
            occ_cell_R, occ_basis, occ_in_bulk = _build_occ_sf_geometry(
                nx, ny, 1.0, boundary=boundary)
        occ_q_points, occ_q_frac = _build_occ_q_grid(int(occ_sf_grid_n), 1.0)
        occ_pt_indices = _snap_delta_points(prof_delta, occ_sf_delta_points,
                                            sweep=occ_sf_sweep)
        # engine site i == canonical site site_perm[i]: feed the engine the
        # permuted views; meta.h5 keeps the canonical arrays.
        occ_sel = site_perm if site_perm is not None else slice(None)
        engine._cpp_engine.set_occ_sf_site_map(
            np.ascontiguousarray(np.asarray(occ_cell_R)[occ_sel]),
            np.ascontiguousarray(np.asarray(occ_basis)[occ_sel]),
            np.ascontiguousarray(np.asarray(occ_in_bulk)[occ_sel]), 6)
        engine._cpp_engine.set_occ_sf_q_points(occ_q_points)
        engine._cpp_engine.set_occ_sf_point_indices(occ_pt_indices)
        # Second unit cell: up+down triangle pair (bulk tiling), same q-grid.
        if lattice == 'kagome_bond_triangle':
            occ2_cell_R, occ2_basis = _build_occ2_sf_geometry_tri(
                nx, ny, 1.0, tri_ijk_map)
        else:
            occ2_cell_R, occ2_basis = _build_occ2_sf_geometry(nx, ny, 1.0)
        engine._cpp_engine.set_occ2_sf_site_map(
            np.ascontiguousarray(np.asarray(occ2_cell_R)[occ_sel]),
            np.ascontiguousarray(np.asarray(occ2_basis)[occ_sel]), 6)
    n_occ_pts = len(occ_pt_indices)

    if verbose and rank == 0:
        M_total  = engine._cpp_engine.M_total
        n_points = M_total // profile_step
        print(f"[MPI-PROF] M_total={M_total}, profile_step={profile_step}, "
              f"n_points={n_points}")
        if lattice == 'kagome_bond_triangle':
            bulk_note = "all atoms; complete blockade triangles everywhere"
        elif boundary == 'periodic':
            bulk_note = "all atoms; periodic torus (no boundary)"
        else:
            bulk_note = "interior atoms only"
        print(f"[MPI-PROF] bulk={len(bulk)}/{N} ({bulk_note}), "
              f"{len(loop_sets)} loop copies / {len(string_sets)} string copies / "
              f"{n_vertex} A_v vertices")
        for m in loop_meta:
            print(f"[MPI-PROF]   loop size={m['size']}: {m['n_copies']} copies "
                  f"(offset {m['offset']})")
        for m in string_meta:
            print(f"[MPI-PROF]   string size={m['size']}: {m['n_copies']} copies "
                  f"(offset {m['offset']})")
        print(f"[MPI-PROF]   A_v vertices: {n_vertex} (offset {vertex_offset})")
        if n_q > 0:
            print(f"[MPI-PROF]   dimer SF: {n_q} q-points")
        if n_snap_pts > 0:
            print(f"[MPI-PROF]   snapshots: {n_snapshots}/rank at {n_snap_pts} "
                  f"δ-points (δ≈{np.round(prof_delta[snap_pt_indices], 3).tolist()})")
        if do_occ:
            n_bulk_cells = int(occ_in_bulk.sum() // 6)
            n_cells_total = ((nx + 1) * (ny + 1)
                             if lattice == 'kagome_bond_triangle' else nx * ny)
            print(f"[MPI-PROF]   occ-SF matrix: {occ_sf_grid_n}×{occ_sf_grid_n} q-grid, "
                  f"{n_occ_pts} δ-points (δ≈{np.round(prof_delta[occ_pt_indices],3).tolist()}), "
                  f"{occ_sf_nbatch} super-bins/rank, bulk cells={n_bulk_cells}/{n_cells_total}")

    # Equilibration
    comm.Barrier()

    def _advance_equil(n):
        try:
            engine._cpp_engine.run(n, 0)
        except TypeError:
            for _ in range(n):
                engine._cpp_engine.mc_step()

    t_equil = run_equil_with_progress(
        _advance_equil, n_equil, label="MPI-PROF", rank=rank,
        print_every=equil_progress_every, verbose=verbose)
    comm.Barrier()
    if verbose and rank == 0:
        max_eq = comm.reduce(t_equil, op=MPI.MAX, root=0)
        print(f"[MPI-PROF] Equilibration done in {max_eq:.1f}s (slowest rank)")
    else:
        comm.reduce(t_equil, op=MPI.MAX, root=0)

    # Sampling
    t0 = time.perf_counter()

    # ── Merged bin==chunk checkpointing ──────────────────────────────────────
    # When checkpoint > 0 it sets the merged bin==flush size: every `checkpoint`
    # samples each rank forms ONE bin (run_profile with batch_size=checkpoint →
    # a single batch row) and appends it as the next chunk{i} group in its
    # single file  data/M={M}_{nx}x{ny}_{timestamp}/rank{r}.h5  (flat, plus a
    # final_config group for warm start).  A deprecated --checkpoint_every_batches
    # still works (bin size stays --batch_size).  Repeated run_profile(0, ...)
    # calls continue the SAME Markov chain, so this is statistically identical
    # to one long call.  Rank 0 also writes a one-time meta.h5 (geometry/params).
    use_checkpoint = (checkpoint and checkpoint > 0) or \
                     (checkpoint_every_batches and checkpoint_every_batches > 0)
    if use_checkpoint:
        from src.mpi.chunk_io import RankChunkWriter
        if checkpoint and checkpoint > 0:
            bin_size = int(checkpoint)          # merged: one bin per chunk
        else:
            bin_size = int(checkpoint_every_batches) * batch_size
        bin_size = max(1, bin_size)

        if rank == 0:
            base_dir = os.path.dirname(filepath) or '.'
            stamp = time.strftime('%Y%m%d_%H%M%S')
            run_dir = os.path.join(base_dir, f'qaqmc_profile_M={M}_{nx}x{ny}_{stamp}')
            os.makedirs(run_dir, exist_ok=True)
        else:
            run_dir = None
        run_dir = comm.bcast(run_dir, root=0)

        n_chunks = ((my_n_samples + bin_size - 1) // bin_size
                    if my_n_samples > 0 else 0)

        samples_done = 0
        c = 0
        with RankChunkWriter(run_dir, rank,
                             run_attrs=dict(N=int(N), M=int(M), nx=int(nx), ny=int(ny),
                                            lattice=str(lattice), boundary=str(boundary),
                                            seed=int(seed),
                                            profile_step=int(profile_step),
                                            samples_per_bin=int(bin_size),
                                            my_n_samples=int(my_n_samples))) as writer:
            while samples_done < my_n_samples:
                n = min(bin_size, my_n_samples - samples_done)
                res = _unpermute_res(engine._cpp_engine.run_profile(
                    0, n, me_density, me_zl, me_cml, profile_step,
                    n,                       # batch_size = bin size → one bin/chunk
                    None, 1,
                    n_snapshots if n_snap_pts > 0 else 0,
                    occ_sf_nbatch if do_occ else 0))
                writer.write_chunk(
                    c, {k: _store_array(k, v) for k, v in res.items()},
                    attrs=dict(n_samples=n, samples_cumulative=samples_done + n))
                samples_done += n
                c += 1
                if verbose and rank == 0:
                    el = time.perf_counter() - t0
                    print(f"[MPI-PROF] rank0 chunk {c}/{n_chunks} written "
                          f"({samples_done}/{my_n_samples} samples, {el:.0f}s)", flush=True)

        # Warm-start config → <run_dir>/configs/rank{r}.h5 (separate from the
        # observable chunk files, so it can be reclaimed independently).
        _save_final_config(default_dir=os.path.join(run_dir, 'configs'))

        # One-time run metadata (geometry + params) from rank 0.
        if rank == 0:
            meta = {
                'delta_schedule': np.array(engine._cpp_engine.delta_schedule,
                                           dtype=np.float64),
                'p_indices': np.array([(k + 1) * profile_step - 1
                                       for k in range(M_total_ // profile_step)],
                                      dtype=np.int64),
            }
            if pos is not None:
                meta['pos'] = np.asarray(pos, dtype=np.float64)
            if n_snap_pts > 0:
                meta['snap_pt_indices'] = np.asarray(snap_pt_indices, dtype=np.int32)
            if do_occ:
                meta['occ_pt_indices'] = np.asarray(occ_pt_indices, dtype=np.int32)
                meta['occ_basis'] = np.asarray(occ_basis, dtype=np.int32)
                meta['occ_cell_R'] = np.asarray(occ_cell_R, dtype=np.float64)
                if occ_q_points is not None:
                    meta['occ_q_points'] = np.asarray(occ_q_points, dtype=np.float64)
            _write_result_h5(
                os.path.join(run_dir, 'meta.h5'), meta,
                attrs=dict(N=N, M=M, nx=nx, ny=ny, Omega=Omega, Rb=Rb,
                           delta_min=delta_min, delta_max=delta_max, epsilon=epsilon,
                           seed=seed, profile_step=profile_step,
                           samples_per_bin=int(bin_size),
                           n_samples=n_samples, n_ranks=n_ranks,
                           permute_site_labels=bool(site_perm is not None)))
            print(f"[MPI-PROF] checkpoint run dir: {run_dir}", flush=True)

        comm.Barrier()
        t_sample = time.perf_counter() - t0
        if verbose and rank == 0:
            max_sa = comm.reduce(t_sample, op=MPI.MAX, root=0)
            print(f"[MPI-PROF] Sampling done in {max_sa:.1f}s (slowest rank); "
                  f"{n_chunks} chunk(s)/rank written.")
        else:
            comm.reduce(t_sample, op=MPI.MAX, root=0)
        return run_dir

    # ── Legacy single-call path (checkpointing disabled) ─────────────────────
    sa_cb = None
    sa_bar = None
    if verbose and rank == 0 and my_n_samples > 0:
        sa_cb, sa_bar = _make_tqdm_callback(my_n_samples, "[MPI-PROF] Sampling (rank0)")

    result = _unpermute_res(engine._cpp_engine.run_profile(
        0, my_n_samples, me_density, me_zl, me_cml, profile_step,
        batch_size,
        sa_cb, 1000 if sa_cb is not None else 1,
        n_snapshots if n_snap_pts > 0 else 0,
        occ_sf_nbatch if do_occ else 0))
    if sa_bar is not None:
        sa_bar.close()

    # density: (n_batches_d_rank, n_points)  — batch means (was per-sample)
    # Z_l:     (n_batches_z_rank, n_points, n_loop_size_groups + 1)
    #          last group = A_v vertices (set length 4, distinct from loop set lengths
    #          so it always forms its own trailing size group).
    # C_m_l:   (n_batches_c_rank, n_points, n_string_size_groups)
    my_density = np.ascontiguousarray(result['density'], dtype=np.float64)
    my_z_raw   = np.ascontiguousarray(result['Z_l'],     dtype=np.float64)
    my_c_m_l   = np.ascontiguousarray(result['C_m_l'],   dtype=np.float64)
    p_indices  = np.asarray(result['p_indices'], dtype=np.int32)
    n_points   = len(p_indices)

    # split Z_l size groups from A_v vertex group.
    # Regular loop size groups are first (one per entry of loop_meta with n_copies>0),
    # vertex group is last.  Zero-copy loop_meta entries (e.g. loop_size=4,5 on a
    # 4x4 lattice → (nx-s)(ny-s)=0 copies) are NOT registered as C++ size groups
    # because no copies exist, so we filter them out here too.
    loop_meta = [m for m in loop_meta if m['n_copies'] > 0]
    n_loop_sg = len(loop_meta)
    if engine._cpp_engine.n_loop_size_groups != n_loop_sg + 1:
        raise RuntimeError(
            f"size-group layout mismatch: expected {n_loop_sg}+1 (loops+vertex), "
            f"got {engine._cpp_engine.n_loop_size_groups}. "
            f"Did vertex_sets collide with a loop_size?")
    my_z_l = np.ascontiguousarray(my_z_raw[:, :, :n_loop_sg])    # (batches, pts, n_loop_sg)
    my_a_v = np.ascontiguousarray(my_z_raw[:, :, n_loop_sg])     # (batches, pts) — C++ already avg'd vertices

    # VBS/SS order parameters (batched (batches, pts) means of M and M²).
    my_mvbs = my_mss = my_mvbs2 = my_mss2 = None
    if do_vbs and 'M_vbs' in result:
        my_mvbs  = np.ascontiguousarray(result['M_vbs'],  dtype=np.float64)
        my_mss   = np.ascontiguousarray(result['M_ss'],   dtype=np.float64)
        my_mvbs2 = np.ascontiguousarray(result['M_vbs2'], dtype=np.float64)
        my_mss2  = np.ascontiguousarray(result['M_ss2'],  dtype=np.float64)

    # Full-state snapshots (only present if snapshot points were set).
    # Shape per rank: (n_snap_collected, n_snap_pts, N) int8.
    my_snap = None
    if n_snap_pts > 0 and 'snapshots' in result:
        my_snap = np.ascontiguousarray(result['snapshots'], dtype=np.int8)

    # Occupation-SF matrices (only if configured). Shapes per rank:
    #   occ_S_*: (occ_nbatch, n_occ_pt, n_q, 6, 6),  occ_nprof: (occ_nbatch, n_occ_pt, N)
    my_occ = None
    if do_occ and 'occ_S_full_re' in result:
        keys = ['occ_S_full_re','occ_S_full_im','occ_S_bulk_re',
                'occ_S_bulk_im','occ_nprof']
        if 'occ2_S_re' in result:      # triangle-pair unit cell present
            keys += ['occ2_S_re','occ2_S_im']
        my_occ = {k: np.ascontiguousarray(result[k], dtype=np.float64)
                  for k in keys}

    # Dimer SF (only present if q-points were set above).  Shape: (batches, pts, n_q).
    my_sf_re = None; my_sf_im = None
    my_sf_a1 = None; my_sf_a2 = None; my_sf_a3 = None; my_sf_a4 = None
    if n_q > 0 and 's_q_real' in result:
        my_sf_re = np.ascontiguousarray(result['s_q_real'], dtype=np.float64)
        my_sf_im = np.ascontiguousarray(result['s_q_imag'], dtype=np.float64)
        my_sf_a1 = np.ascontiguousarray(result['s_q_abs1'], dtype=np.float64)
        my_sf_a2 = np.ascontiguousarray(result['s_q_abs2'], dtype=np.float64)
        my_sf_a3 = np.ascontiguousarray(result['s_q_abs3'], dtype=np.float64)
        my_sf_a4 = np.ascontiguousarray(result['s_q_abs4'], dtype=np.float64)

    # Filter zero-copy string_meta entries for the same reason as loop_meta above:
    # C++ only registers size groups for sizes with actual copies.
    string_meta = [m for m in string_meta if m['n_copies'] > 0]
    n_string_sg = engine._cpp_engine.n_string_size_groups

    t_sample = time.perf_counter() - t0
    comm.Barrier()
    if verbose and rank == 0:
        max_sa = comm.reduce(t_sample, op=MPI.MAX, root=0)
        print(f"[MPI-PROF] Sampling done in {max_sa:.1f}s (slowest rank)")
    else:
        comm.reduce(t_sample, op=MPI.MAX, root=0)

    # ── Gather and write HDF5 ─────────────────────────────────────────────
    fine_counts = np.array([base + (1 if r < rem else 0) for r in range(n_ranks)])

    def _obs_counts(me):
        return np.array([(fine_counts[r] * min_me) // me for r in range(n_ranks)])

    def _batch_counts(me):
        # Number of complete batches per rank (mirrors C++ logic: max(1, n_samples//batch_size))
        return np.array([max(1, int((fine_counts[r] * min_me) // me) // batch_size)
                         for r in range(n_ranks)])

    def _obs_offsets(counts_arr):
        off = np.zeros(n_ranks, dtype=int)
        for r in range(1, n_ranks):
            off[r] = off[r-1] + counts_arr[r-1]
        return off

    # density is now BATCHED (same batch_size as Z_l/C_m_l), so it uses a batch count.
    counts_d  = _batch_counts(me_density)
    batches_z = _batch_counts(me_zl)
    batches_c = _batch_counts(me_cml)
    off_d  = _obs_offsets(counts_d)
    off_bz = _obs_offsets(batches_z)
    off_bc = _obs_offsets(batches_c)

    n_total_d  = int(counts_d.sum())
    n_total_bz = int(batches_z.sum())
    n_total_bc = int(batches_c.sum())

    # Gather full-state snapshots (collective; modest size → pickle gather is fine).
    all_snap = None
    all_snap_rank = None
    if n_snap_pts > 0:
        gathered = comm.gather(my_snap, root=0)
        if rank == 0:
            parts, rank_lbls = [], []
            for r, g in enumerate(gathered):
                if g is not None and len(g) > 0:
                    parts.append(g)
                    rank_lbls.append(np.full(len(g), r, dtype=np.int16))
            all_snap      = np.concatenate(parts, axis=0)        # (total_snaps, n_snap_pts, N)
            all_snap_rank = np.concatenate(rank_lbls)            # (total_snaps,) source rank

    # Gather occ-SF matrices (super-bins concatenate along axis 0 across ranks).
    all_occ = None
    if do_occ:
        gathered_occ = comm.gather(my_occ, root=0)
        if rank == 0:
            all_occ = {}
            present = [g for g in gathered_occ if g is not None]
            keys = ['occ_S_full_re','occ_S_full_im','occ_S_bulk_re',
                    'occ_S_bulk_im','occ_nprof']
            if present and 'occ2_S_re' in present[0]:
                keys += ['occ2_S_re','occ2_S_im']
            for key in keys:
                all_occ[key] = np.concatenate([g[key] for g in present], axis=0)

    # Gather VBS/SS (batched (batches_z, n_points), concatenate along axis 0).
    all_vbs = None
    if do_vbs:
        my_v = dict(M_vbs=my_mvbs, M_ss=my_mss, M_vbs2=my_mvbs2, M_ss2=my_mss2)
        gathered_v = comm.gather(my_v, root=0)
        if rank == 0:
            all_vbs = {k: np.concatenate([g[k] for g in gathered_v if g is not None], axis=0)
                       for k in ('M_vbs','M_ss','M_vbs2','M_ss2')}

    if rank == 0:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        all_density = np.empty((n_total_d,  n_points),                  dtype=np.float64)
        all_z_l     = np.empty((n_total_bz, n_points, n_loop_sg),       dtype=np.float64)
        all_a_v     = np.empty((n_total_bz, n_points),                  dtype=np.float64)
        all_c_m_l   = np.empty((n_total_bc, n_points, n_string_sg),     dtype=np.float64)
        if n_q > 0:
            all_sf_re = np.empty((n_total_bz, n_points, n_q), dtype=np.float64)
            all_sf_im = np.empty((n_total_bz, n_points, n_q), dtype=np.float64)
            all_sf_a1 = np.empty((n_total_bz, n_points, n_q), dtype=np.float64)
            all_sf_a2 = np.empty((n_total_bz, n_points, n_q), dtype=np.float64)
            all_sf_a3 = np.empty((n_total_bz, n_points, n_q), dtype=np.float64)
            all_sf_a4 = np.empty((n_total_bz, n_points, n_q), dtype=np.float64)

        all_density[off_d[0] :off_d[0] +counts_d[0] ] = my_density
        all_z_l    [off_bz[0]:off_bz[0]+batches_z[0]] = my_z_l
        all_a_v    [off_bz[0]:off_bz[0]+batches_z[0]] = my_a_v
        all_c_m_l  [off_bc[0]:off_bc[0]+batches_c[0]] = my_c_m_l
        if n_q > 0:
            all_sf_re[off_bz[0]:off_bz[0]+batches_z[0]] = my_sf_re
            all_sf_im[off_bz[0]:off_bz[0]+batches_z[0]] = my_sf_im
            all_sf_a1[off_bz[0]:off_bz[0]+batches_z[0]] = my_sf_a1
            all_sf_a2[off_bz[0]:off_bz[0]+batches_z[0]] = my_sf_a2
            all_sf_a3[off_bz[0]:off_bz[0]+batches_z[0]] = my_sf_a3
            all_sf_a4[off_bz[0]:off_bz[0]+batches_z[0]] = my_sf_a4

        for src in range(1, n_ranks):
            buf_d  = np.empty((counts_d[src],  n_points),                dtype=np.float64)
            buf_z  = np.empty((batches_z[src], n_points, n_loop_sg),     dtype=np.float64)
            buf_av = np.empty((batches_z[src], n_points),                dtype=np.float64)
            buf_c  = np.empty((batches_c[src], n_points, n_string_sg),   dtype=np.float64)
            comm.Recv(buf_d,  source=src, tag=300 + src)
            comm.Recv(buf_z,  source=src, tag=400 + src)
            comm.Recv(buf_av, source=src, tag=450 + src)
            comm.Recv(buf_c,  source=src, tag=500 + src)
            all_density[off_d[src] :off_d[src] +counts_d[src] ] = buf_d
            all_z_l    [off_bz[src]:off_bz[src]+batches_z[src]] = buf_z
            all_a_v    [off_bz[src]:off_bz[src]+batches_z[src]] = buf_av
            all_c_m_l  [off_bc[src]:off_bc[src]+batches_c[src]] = buf_c
            if n_q > 0:
                buf_sf_re = np.empty((batches_z[src], n_points, n_q), dtype=np.float64)
                buf_sf_im = np.empty((batches_z[src], n_points, n_q), dtype=np.float64)
                buf_sf_a1 = np.empty((batches_z[src], n_points, n_q), dtype=np.float64)
                buf_sf_a2 = np.empty((batches_z[src], n_points, n_q), dtype=np.float64)
                buf_sf_a3 = np.empty((batches_z[src], n_points, n_q), dtype=np.float64)
                buf_sf_a4 = np.empty((batches_z[src], n_points, n_q), dtype=np.float64)
                comm.Recv(buf_sf_re, source=src, tag=600 + src)
                comm.Recv(buf_sf_im, source=src, tag=700 + src)
                comm.Recv(buf_sf_a1, source=src, tag=750 + src)
                comm.Recv(buf_sf_a2, source=src, tag=800 + src)
                comm.Recv(buf_sf_a3, source=src, tag=820 + src)
                comm.Recv(buf_sf_a4, source=src, tag=850 + src)
                all_sf_re[off_bz[src]:off_bz[src]+batches_z[src]] = buf_sf_re
                all_sf_im[off_bz[src]:off_bz[src]+batches_z[src]] = buf_sf_im
                all_sf_a1[off_bz[src]:off_bz[src]+batches_z[src]] = buf_sf_a1
                all_sf_a2[off_bz[src]:off_bz[src]+batches_z[src]] = buf_sf_a2
                all_sf_a3[off_bz[src]:off_bz[src]+batches_z[src]] = buf_sf_a3
                all_sf_a4[off_bz[src]:off_bz[src]+batches_z[src]] = buf_sf_a4

        delta_sched = np.array(engine._cpp_engine.delta_schedule, dtype=np.float64)

        with h5py.File(filepath, 'w') as f:
            pg = f.create_group('params')
            pg.attrs['N']            = N
            pg.attrs['Omega']        = Omega
            pg.attrs['Rb']           = Rb
            pg.attrs['delta_min']    = delta_min
            pg.attrs['delta_max']    = delta_max
            pg.attrs['M']            = M
            pg.attrs['epsilon']      = epsilon
            pg.attrs['seed']         = seed
            pg.attrs['n_equil']      = n_equil
            pg.attrs['n_samples']    = n_samples
            pg.attrs['me_density']   = me_density
            pg.attrs['me_zl']        = me_zl
            pg.attrs['me_cml']       = me_cml
            pg.attrs['profile_step'] = profile_step
            pg.attrs['batch_size']   = batch_size
            pg.attrs['n_points']     = n_points
            pg.attrs['n_ranks']      = n_ranks
            pg.attrs['nx']           = nx
            pg.attrs['ny']           = ny
            pg.attrs['timestamp']    = datetime.datetime.utcnow().isoformat()
            if neighbor_cutoff is not None:
                pg.attrs['neighbor_cutoff'] = neighbor_cutoff

            gg = f.create_group('geometry')
            pos_stored = pos if pos is not None \
                         else np.arange(N).reshape(-1, 1).astype(np.float64)
            gg.create_dataset('pos', data=pos_stored.astype(np.float64))

            sg = f.create_group('schedule')
            sg.create_dataset('delta_schedule', data=delta_sched)
            sg.create_dataset('p_indices',      data=p_indices)
            sg.create_dataset('delta_sampled',  data=delta_sched[p_indices])

            # ── Provenance: which rank + which within-rank step each row came from ──
            # Rows of every gathered array are concatenated in rank order, and are
            # time-ordered within each rank's block. We store explicit per-row rank
            # ids and within-rank indices so post-processing can drop burn-in
            # without relying on n_samples being divisible by n_ranks.
            #
            # All gathered arrays (density, Z_l, A_v, C_m_l, SF) are now BATCH means
            # concatenated in rank order, time-ordered within each rank's block.
            # We store explicit per-row rank ids and within-rank batch indices so
            # post-processing can drop burn-in without relying on divisibility.
            # batch b of rank r spans recorded samples [b*batch_size, (b+1)*batch_size),
            # i.e. MC steps n_equil + (b*batch_size + 1 .. (b+1)*batch_size) * measure_every.
            pv = f.create_group('provenance')
            pv.attrs['n_ranks']  = n_ranks
            pv.attrs['n_equil']  = n_equil
            pv.attrs['me_density'] = me_density
            pv.attrs['me_zl']      = me_zl
            pv.attrs['me_cml']     = me_cml
            pv.attrs['batch_size'] = batch_size
            # density: batch-level (matches all_density rows, which are batch means)
            dens_rank = np.concatenate([np.full(int(counts_d[r]), r, dtype=np.int16)
                                        for r in range(n_ranks)])
            dens_step = np.concatenate([np.arange(int(counts_d[r]), dtype=np.int32)
                                        for r in range(n_ranks)])
            pv.create_dataset('density_rank', data=dens_rank)   # (n_total_d,) source rank
            pv.create_dataset('density_batch_index', data=dens_step)  # within-rank batch index
            pv.create_dataset('density_rank_offsets', data=off_d.astype(np.int64))
            pv.create_dataset('density_rank_counts',  data=counts_d.astype(np.int64))
            # Z_l / A_v / dimer-SF batches (share the me_zl batch cadence)
            zb_rank = np.concatenate([np.full(int(batches_z[r]), r, dtype=np.int16)
                                      for r in range(n_ranks)])
            zb_idx  = np.concatenate([np.arange(int(batches_z[r]), dtype=np.int32)
                                      for r in range(n_ranks)])
            pv.create_dataset('zbatch_rank', data=zb_rank)      # (n_total_bz,)
            pv.create_dataset('zbatch_index', data=zb_idx)      # within-rank batch index
            pv.create_dataset('zbatch_rank_offsets', data=off_bz.astype(np.int64))
            pv.create_dataset('zbatch_rank_counts',  data=batches_z.astype(np.int64))
            # C_m_l batches (me_cml cadence)
            cb_rank = np.concatenate([np.full(int(batches_c[r]), r, dtype=np.int16)
                                      for r in range(n_ranks)])
            cb_idx  = np.concatenate([np.arange(int(batches_c[r]), dtype=np.int32)
                                      for r in range(n_ranks)])
            pv.create_dataset('cbatch_rank', data=cb_rank)      # (n_total_bc,)
            pv.create_dataset('cbatch_index', data=cb_idx)
            pv.create_dataset('cbatch_rank_offsets', data=off_bc.astype(np.int64))
            pv.create_dataset('cbatch_rank_counts',  data=batches_c.astype(np.int64))

            og = f.create_group('profiles')
            og.create_dataset('density', data=all_density)  # (n_batches_d, n_points) batch means
            og.create_dataset('Z_l',     data=all_z_l)       # (n_batches, n_points, n_loop_size_groups)
            og.create_dataset('A_v',     data=all_a_v)        # (n_batches, n_points) — mean over vertices (C++)
            og.create_dataset('C_m_l',   data=all_c_m_l)     # (n_batches, n_points, n_string_size_groups)
            og.attrs['n_vertices'] = n_vertex
            if do_vbs and all_vbs is not None:
                # VBS/SS order parameters (paper Eq. 5-6), batched (n_batches, n_points).
                # Post-process:  M_order = sqrt(<M²>),  <M²> = mean over batches of M_vbs2.
                og.create_dataset('M_vbs',  data=all_vbs['M_vbs'])   # batch means of M_VBS
                og.create_dataset('M_ss',   data=all_vbs['M_ss'])
                og.create_dataset('M_vbs2', data=all_vbs['M_vbs2'])  # batch means of M_VBS²
                og.create_dataset('M_ss2',  data=all_vbs['M_ss2'])
                og.attrs['n_up_triangles'] = int(vbs_tri[0].shape[0] // 3)
            if n_q > 0:
                # Optional δ-subsetting: keep only the profile points nearest to
                # the user-specified δ values; otherwise save SF at every profile
                # point (same δ grid as density/Z/C).
                delta_at_pts = delta_sched[p_indices]
                if sf_delta_points is not None and len(sf_delta_points) > 0:
                    req = np.asarray(sf_delta_points, dtype=np.float64)
                    pick = np.array([int(np.argmin(np.abs(delta_at_pts - d))) for d in req])
                    pick = np.unique(pick)  # drop duplicates if two reqs snap to same point
                    sf_re_out = all_sf_re[:, pick, :]
                    sf_im_out = all_sf_im[:, pick, :]
                    sf_a1_out = all_sf_a1[:, pick, :]
                    sf_a2_out = all_sf_a2[:, pick, :]
                    sf_a3_out = all_sf_a3[:, pick, :]
                    sf_a4_out = all_sf_a4[:, pick, :]
                    sf_delta_used = delta_at_pts[pick]
                    sf_p_used     = p_indices[pick]
                else:
                    sf_re_out = all_sf_re
                    sf_im_out = all_sf_im
                    sf_a1_out = all_sf_a1
                    sf_a2_out = all_sf_a2
                    sf_a3_out = all_sf_a3
                    sf_a4_out = all_sf_a4
                    sf_delta_used = delta_at_pts
                    sf_p_used     = p_indices

                sfg = f.create_group('dimer_sf')
                sfg.create_dataset('q_points',  data=sf_q_points)
                sfg.create_dataset('delta',     data=sf_delta_used)
                sfg.create_dataset('p_indices', data=sf_p_used)
                sfg.create_dataset('s_q_real',  data=sf_re_out)  # (n_batches, n_sf_pts, n_q)
                sfg.create_dataset('s_q_imag',  data=sf_im_out)
                sfg.create_dataset('s_q_abs1',  data=sf_a1_out)  # per-sample |s_q|
                sfg.create_dataset('s_q_abs2',  data=sf_a2_out)  # per-sample |s_q|²
                sfg.create_dataset('s_q_abs3',  data=sf_a3_out)  # per-sample |s_q|³
                sfg.create_dataset('s_q_abs4',  data=sf_a4_out)  # per-sample |s_q|⁴
                # Post-processing (let N := N_bulk):
                #   m_q  = <s_q_abs1>          / N
                #   m_q² = <s_q_abs2>          / N²
                #   m_q⁴ = <s_q_abs4>          / N⁴
                #   χ_q  = N * (m_q² - m_q²)              [Var of |s_q|, per-site]
                #   U_q  = 1 - <s_q_abs4>/(3 * <s_q_abs2>²)         (Binder cumulant)
                #   S_q_conn = (<s_q_abs2> - |<s_q>|²)   (connected SF; divide by N_d).
            # Post-processing:
            #   <Z(l)>(δ, size_s)   = all_z_l[..., size_meta_attrs[f'loop_size{s}_index']].mean(axis=0)
            #   <C_m(l)>(δ, size_s) = all_c_m_l[..., size_meta_attrs[f'string_size{s}_index']].mean(axis=0)

            if n_snap_pts > 0 and all_snap is not None and len(all_snap) > 0:
                # Full-state snapshots for real-space phase visualisation.
                #   configs[s, k, i] = n_i (0/1) of sample s at snapshot δ-point k
                # Pair with geometry/pos[i] to plot real-space occupation patterns.
                sng = f.create_group('snapshots')
                sng.create_dataset('configs', data=all_snap)              # (n_snaps, n_snap_pts, N) int8
                sng.create_dataset('delta',     data=prof_delta[snap_pt_indices])
                sng.create_dataset('p_indices', data=prof_p_idx[snap_pt_indices].astype(np.int32))
                sng.create_dataset('source_rank', data=all_snap_rank)     # (n_snaps,) which rank
                sng.attrs['n_snapshots_per_rank'] = n_snapshots
                sng.attrs['n_snapshots_total']    = int(all_snap.shape[0])

            if do_occ and all_occ is not None:
                # Sublattice-resolved occupation structure-factor matrix S_αβ(q).
                #   S_*_re/im: (n_superbins, n_δ, n_q, 6, 6) UNCONNECTED ⟨s_α s*_β⟩
                #   nprof: (n_superbins, n_δ, N) per-site ⟨n_i⟩ (for connected subtraction)
                # Connected (post): S^c_αβ = S_αβ - (1/N_c) F_α F*_β,
                #   F_α(q) = Σ_{cells in set} e^{iq·R} ⟨n_{R,α}⟩  (built from nprof + geometry).
                # Two cell sets: 'full' (all cells) and 'bulk' (bulk-complete cells).
                ocg = f.create_group('occ_sf')
                ocg.create_dataset('q_points',  data=occ_q_points)      # (n_q, 2) cartesian (a=1)
                ocg.create_dataset('q_frac',    data=occ_q_frac)        # (n_q, 2) fractional (m/G, n/G)
                ocg.create_dataset('delta',     data=prof_delta[occ_pt_indices])
                ocg.create_dataset('p_indices', data=prof_p_idx[occ_pt_indices].astype(np.int32))
                ocg.create_dataset('S_full_re', data=_store_array('occ_S_full_re', all_occ['occ_S_full_re']))
                ocg.create_dataset('S_full_im', data=_store_array('occ_S_full_im', all_occ['occ_S_full_im']))
                ocg.create_dataset('S_bulk_re', data=_store_array('occ_S_bulk_re', all_occ['occ_S_bulk_re']))
                ocg.create_dataset('S_bulk_im', data=_store_array('occ_S_bulk_im', all_occ['occ_S_bulk_im']))
                ocg.create_dataset('nprof',     data=_store_array('occ_nprof', all_occ['occ_nprof']))  # (n_sb, n_δ, N) all sites
                # geometry for reconstructing F_α(q)
                ocg.create_dataset('cell_R',       data=occ_cell_R)      # (N,2) per-site cell pos (a=1)
                ocg.create_dataset('site_basis',   data=occ_basis)       # (N,) α index
                ocg.create_dataset('site_in_bulk', data=occ_in_bulk)     # (N,) bulk-cell flag
                ocg.attrs['grid_n']  = int(occ_sf_grid_n)
                ocg.attrs['n_basis'] = 6
                ocg.attrs['nbatch_per_rank'] = int(occ_sf_nbatch)
                # Alternate triangle-pair unit cell (bulk tiling), same q-grid/δ.
                if 'occ2_S_re' in all_occ:
                    ocg.create_dataset('S_tri_re',   data=_store_array('occ2_S_re', all_occ['occ2_S_re']))
                    ocg.create_dataset('S_tri_im',   data=_store_array('occ2_S_im', all_occ['occ2_S_im']))
                    ocg.create_dataset('tri_cell_R', data=occ2_cell_R)    # (N,2) per-site cell pos
                    ocg.create_dataset('tri_basis',  data=occ2_basis)     # (N,) α (−1 if excluded)

            lg = f.create_group('observable_sites')
            for idx, lp in enumerate(loop_sets):
                lg.create_dataset(f'loop_{idx}',   data=np.array(lp, dtype=np.int32))
            for idx, st in enumerate(string_sets):
                lg.create_dataset(f'string_{idx}', data=np.array(st, dtype=np.int32))
            mg = f.create_group('size_meta')
            for i, m in enumerate(loop_meta):
                mg.attrs[f'loop_size{m["size"]}_index']    = i
                mg.attrs[f'loop_size{m["size"]}_n_copies'] = m['n_copies']
            for i, m in enumerate(string_meta):
                mg.attrs[f'string_size{m["size"]}_index']    = i
                mg.attrs[f'string_size{m["size"]}_n_copies'] = m['n_copies']

        if verbose:
            print(f"[MPI-PROF] Saved {n_samples} samples × {n_points} points → {filepath}")
    else:
        comm.Send(np.ascontiguousarray(my_density), dest=0, tag=300 + rank)
        comm.Send(np.ascontiguousarray(my_z_l),     dest=0, tag=400 + rank)
        comm.Send(np.ascontiguousarray(my_a_v),     dest=0, tag=450 + rank)
        comm.Send(np.ascontiguousarray(my_c_m_l),   dest=0, tag=500 + rank)
        if n_q > 0:
            comm.Send(np.ascontiguousarray(my_sf_re), dest=0, tag=600 + rank)
            comm.Send(np.ascontiguousarray(my_sf_im), dest=0, tag=700 + rank)
            comm.Send(np.ascontiguousarray(my_sf_a1), dest=0, tag=750 + rank)
            comm.Send(np.ascontiguousarray(my_sf_a2), dest=0, tag=800 + rank)
            comm.Send(np.ascontiguousarray(my_sf_a3), dest=0, tag=820 + rank)
            comm.Send(np.ascontiguousarray(my_sf_a4), dest=0, tag=850 + rank)

    _save_final_config()
    comm.Barrier()
    return filepath


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MPI-parallel QAQMC simulation')
    parser.add_argument('--config', type=str, help='JSON config file')
    parser.add_argument('--N', type=int, default=6)
    parser.add_argument('--M', type=int, default=1280)
    parser.add_argument('--Omega', type=float, default=1.0)
    parser.add_argument('--Rb', type=float, default=2.4)
    parser.add_argument('--delta_min', type=float, default=0.0)
    parser.add_argument('--delta_max', type=float, default=8.0)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_equil', type=int, default=5000)
    parser.add_argument('--equil_progress_every', type=int, default=500,
                        help='(profile mode) print rank-0 equilibration progress '
                             'every N steps (<= 0 disables intermediate prints)')
    parser.add_argument('--n_samples', type=int, default=30000)
    parser.add_argument('--neighbor_cutoff', type=int, default=None)
    parser.add_argument('--delta_groups', type=int, default=600,
                        help='Number of delta groups for shared alias tables (must be > 0).')
    parser.add_argument('--omp_threads', type=int, default=1)
    parser.add_argument('--filepath', type=str, default='data/qaqmc_mpi.h5')
    parser.add_argument('--lattice', type=str, default='kagome_bond',
                        help='Lattice type: kagome_bond | kagome_bond_triangle | '
                             'kagome (vertex lattice, U(1) QDM; nn distance = a/2 '
                             'so use --a 2.0 for nn units)')
    parser.add_argument('--config_in', type=str, default=None,
                        help='(profile mode) warm-start directory of rank{r}.h5 final '
                             'configurations from a previous run; when given, '
                             'n_equil is forced to 0 (thermalization skipped)')
    parser.add_argument('--config_out', type=str, default=None,
                        help='(profile mode) where to save final configurations '
                             '(default: <filepath minus .h5>_configs)')
    parser.add_argument('--nx', type=int, default=1)
    parser.add_argument('--ny', type=int, default=1)
    parser.add_argument('--a', type=float, default=4.0, help='Lattice constant')
    parser.add_argument('--mode', type=str, default='store',
                        choices=['store', 'onthefly', 'profile'],
                        help='store: save full operator strings; '
                             'onthefly: save scalar observables at symmetry point; '
                             'profile: save asymmetric observable profiles along the path')
    parser.add_argument('--measure_every', type=str, default='1',
                        help='Record one sample every N MC steps. '
                             'Int (same for all) or JSON dict per observable: '
                             '\'{"density":300,"Z_l":20,"C_m_l":70}\'. '
                             'n_samples refers to the finest interval.')
    parser.add_argument('--profile_step', type=int, default=10000,
                        help='(profile mode) Sample one point along the path every N slices. '
                             'Default 10000. n_points = M_total / profile_step.')
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='(profile mode) Number of Z_l/C_m_l samples per batch for '
                             'memory-efficient per-copy storage. Default 1000.')
    parser.add_argument('--checkpoint', type=int, default=0,
                        help='(profile mode) Merged bin==flush size: every N samples each '
                             'rank forms ONE bin and appends it as the next chunk{i} group '
                             'in data/M=..._<nx>x<ny>_<timestamp>/rank{r}.h5 (flat, one '
                             'file per rank, plus a final_config group for warm start). '
                             '0 = disabled (single combined file). Supersedes '
                             '--checkpoint_every_batches / --batch_size.')
    parser.add_argument('--checkpoint_every_batches', type=int, default=0,
                        help='(profile mode) DEPRECATED alias — use --checkpoint. Legacy '
                             'chunk size in batches; still writes the new flat rank{r}.h5 '
                             'layout with bin size = --batch_size.')
    # Loop / string sizes are auto-selected from (nx, ny):
    #   loop sizes   = 2 .. min(nx, ny) - 1
    #   string sizes = 1 .. min(nx, ny) - 1
    # No CLI knobs.
    parser.add_argument('--sf_q_points_file', type=str, default=None,
                        help='(profile mode) Path to .npy/.json/.txt giving an (n_q, 2) '
                             'array of q-points for dimer S_d(q) measurement. '
                             'Overrides --sf_q_path if both are set.')
    parser.add_argument('--sf_q_path', type=str, default=None,
                        choices=['GammaMKGamma', 'GammaKMGamma', 'GammaM', 'GammaK'],
                        help='(profile mode) Named high-symmetry path in the kagome BZ '
                             'for dimer S_d(q). Uses --a for the lattice constant.')
    parser.add_argument('--sf_q_n', type=int, default=20,
                        help='(profile mode) Points per segment of --sf_q_path. Default 20.')
    parser.add_argument('--sf_delta_points', type=float, nargs='+', default=None,
                        help='(profile mode) Optional list of physical δ values at which to '
                             'keep the SF in HDF5. Each value is snapped to the nearest '
                             'profile-grid δ. If omitted, SF is saved at every profile point '
                             '(same grid as density/Z/C).')
    parser.add_argument('--snapshot_deltas', type=float, nargs='+', default=None,
                        help='(profile mode) List of physical δ values at which to dump full '
                             'spin/occupation configurations for real-space phase visualisation. '
                             'Each δ is snapped to the nearest profile-grid point. Requires '
                             '--n_snapshots > 0.')
    parser.add_argument('--n_snapshots', type=int, default=0,
                        help='(profile mode) Number of full-state configs to save PER RANK at '
                             'each --snapshot_deltas point. Total = n_snapshots × n_ranks. '
                             'Default 0 (disabled).')
    parser.add_argument('--snapshot_sweep', type=str, default='forward',
                        choices=['forward', 'backward'],
                        help='(profile mode) which δ ramp --snapshot_deltas snap to '
                             '(default forward = up-ramp).')
    parser.add_argument('--occ_sf_delta_points', type=float, nargs='+', default=None,
                        help='(profile mode) δ values at which to measure the sublattice-resolved '
                             'occupation structure-factor matrix S_αβ(q). Snapped to profile grid.')
    parser.add_argument('--occ_sf_grid_n', type=int, default=0,
                        help='(profile mode) Side length of the 2D BZ q-grid (grid_n × grid_n) for '
                             'the occ-SF matrix. 0 = disabled.')
    parser.add_argument('--occ_sf_nbatch', type=int, default=4,
                        help='(profile mode) Coarse super-bins PER RANK for the occ-SF matrix '
                             '(controls storage; total = nbatch × n_ranks). Default 4.')
    parser.add_argument('--occ_sf_sweep', type=str, default='forward',
                        choices=['forward', 'backward'],
                        help='(profile mode) which δ ramp --occ_sf_delta_points snap to '
                             '(default forward = up-ramp).')
    parser.add_argument('--permute_site_labels',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='(profile mode) give each rank a random site-label '
                             'permutation (identical physics, different update scan '
                             'order). Decorrelates the ordered-phase domain pattern '
                             'across ranks — without this, independent chains freeze '
                             'into the SAME stripe pattern selected by the shared '
                             'scan order. Outputs are stored in canonical labels.')
    parser.add_argument('--boundary', type=str, default='open',
                        choices=['open', 'periodic'],
                        help='Spatial lattice boundary: open (finite cropped patch) or '
                             'periodic (torus via minimum-image). periodic is undefined '
                             'for kagome_bond_triangle.')
    args = parser.parse_args()

    # Load config file if provided (overrides CLI args)
    config = vars(args)
    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    # Parse measure_every: string → int or dict
    me_raw = config.get('measure_every', '1')
    if isinstance(me_raw, str):
        try:
            config['measure_every'] = json.loads(me_raw)
        except (json.JSONDecodeError, ValueError):
            config['measure_every'] = int(me_raw)

    # Generate lattice
    pos = None
    if config.get('lattice') == 'kagome_bond':
        from src.rydberg.lattices import generate_kagome_bond_lattice
        pos = generate_kagome_bond_lattice(
            nx=config.get('nx', 1),
            ny=config.get('ny', 1),
            a=config.get('a', 4.0)
        )
        config['N'] = len(pos)
    elif config.get('lattice') == 'kagome_bond_triangle':
        from src.rydberg.lattices import generate_kagome_bond_triangle_lattice
        pos = generate_kagome_bond_triangle_lattice(
            nx=config.get('nx', 1),
            ny=config.get('ny', 1),
            a=config.get('a', 4.0)
        )
        config['N'] = len(pos)
    elif config.get('lattice') == 'kagome':
        # Vertex kagome (U(1) QDM, paper/u1).  nn distance = a/2: pass a=2.0
        # to work in nn-distance units (paper's Rb/a_nn = 1.32 → Rb = 1.32).
        from src.rydberg.lattices import generate_kagome_lattice
        pos = generate_kagome_lattice(
            nx=config.get('nx', 1),
            ny=config.get('ny', 1),
            a=config.get('a', 2.0),
            boundary=config.get('boundary', 'open')
        )
        config['N'] = len(pos)

    mode = config.get('mode', 'store')

    # Periodic supercell vectors (open → None); raises for the cropped triangle.
    boundary = config.get('boundary', 'open')
    box_vectors = None
    if boundary == 'periodic':
        from src.rydberg.lattices import lattice_box_vectors
        box_vectors = lattice_box_vectors(
            config.get('lattice', 'kagome_bond'),
            config.get('nx', 1), config.get('ny', 1),
            config.get('a', 4.0), N=config.get('N'))

    if mode == 'profile':
        sf_q_points = _build_sf_q_points(
            sf_q_points_file=config.get('sf_q_points_file'),
            sf_q_path=config.get('sf_q_path'),
            sf_q_n=config.get('sf_q_n', 20),
            a=config.get('a', 1.0),
        )
        run_mpi_profile(
            N=config['N'], M=config['M'],
            Omega=config['Omega'], Rb=config['Rb'],
            delta_min=config['delta_min'], delta_max=config['delta_max'],
            pos=pos, epsilon=config['epsilon'], seed=config['seed'],
            n_equil=config['n_equil'], n_samples=config['n_samples'],
            measure_every=config.get('measure_every', 1),
            profile_step=config.get('profile_step', 10000),
            batch_size=config.get('batch_size', 1000),
            checkpoint=config.get('checkpoint', 0),
            checkpoint_every_batches=config.get('checkpoint_every_batches', 0),
            filepath=config['filepath'],
            neighbor_cutoff=config.get('neighbor_cutoff'),
            delta_groups=config.get('delta_groups', 600),
            omp_threads=config.get('omp_threads', 1),
            nx=config.get('nx', 1), ny=config.get('ny', 1),
            sf_q_points=sf_q_points,
            sf_delta_points=config.get('sf_delta_points'),
            snapshot_deltas=config.get('snapshot_deltas'),
            n_snapshots=config.get('n_snapshots', 0),
            snapshot_sweep=config.get('snapshot_sweep', 'forward'),
            occ_sf_delta_points=config.get('occ_sf_delta_points'),
            occ_sf_grid_n=config.get('occ_sf_grid_n', 0),
            occ_sf_sweep=config.get('occ_sf_sweep', 'forward'),
            occ_sf_nbatch=config.get('occ_sf_nbatch', 4),
            lattice=config.get('lattice', 'kagome_bond'),
            boundary=boundary, box_vectors=box_vectors,
            config_in=config.get('config_in'),
            config_out=config.get('config_out'),
            equil_progress_every=config.get('equil_progress_every', 500),
            permute_site_labels=config.get('permute_site_labels', True),
        )
    elif mode == 'onthefly':
        run_mpi_onthefly(
            N=config['N'], M=config['M'],
            Omega=config['Omega'], Rb=config['Rb'],
            delta_min=config['delta_min'], delta_max=config['delta_max'],
            pos=pos, epsilon=config['epsilon'], seed=config['seed'],
            n_equil=config['n_equil'], n_samples=config['n_samples'],
            measure_every=config.get('measure_every', 1),
            filepath=config['filepath'],
            neighbor_cutoff=config.get('neighbor_cutoff'),
            delta_groups=config.get('delta_groups', 600),
            omp_threads=config.get('omp_threads', 1),
            nx=config.get('nx', 1), ny=config.get('ny', 1),
            lattice=config.get('lattice', 'kagome_bond'),
        )
    else:
        run_mpi(
            N=config['N'], M=config['M'],
            Omega=config['Omega'], Rb=config['Rb'],
            delta_min=config['delta_min'], delta_max=config['delta_max'],
            pos=pos, epsilon=config['epsilon'], seed=config['seed'],
            n_equil=config['n_equil'], n_samples=config['n_samples'],
            measure_every=config.get('measure_every', 1),
            filepath=config['filepath'],
            neighbor_cutoff=config.get('neighbor_cutoff'),
            delta_groups=config.get('delta_groups', 600),
            omp_threads=config.get('omp_threads', 1),
        )


if __name__ == '__main__':
    main()
