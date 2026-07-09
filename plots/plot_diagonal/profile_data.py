"""Shared loaders for the diagonal-profile chunked run format.

Data layout (written by ``src.mpi.qaqmc_mpi --mode profile`` with
``--checkpoint > 0``)::

    data/M=<M>_<nx>x<ny>_<stamp>/
        meta.h5           delta_schedule, p_indices, pos,
                          [snap_pt_indices, occ_pt_indices,]  (newer runs)
                          occ_basis, occ_cell_R, occ_q_points  + params attrs
        rank{r}.h5        run attrs + chunk{i} groups (one bin each):
            density  (nb, n_pts)          bin mean
            Z_l      (nb, n_pts, n_lg+1)  signed copy-mean per loop size group;
                                          LAST column = A_v vertex operator
            C_m_l    (nb, n_pts, n_sg)    signed copy-mean per string size group
            M_vbs/M_ss/M_vbs2/M_ss2 (nb, n_pts)
            snapshots (n_snap, n_snap_pts, N) int8
            occ_S_{full,bulk}_{re,im}, occ2_S_{re,im}
                     (occ_nbatch, n_occ_pt, n_q, 6, 6)
                     super-bin means of s_a(q) s_b(q)* — UNCONNECTED, not
                     normalised by cell count
            occ_nprof (occ_nbatch, n_occ_pt, N)  super-bin mean occupation
        configs/          warm-start files (ignored here)

δ at profile point k is ``delta_schedule[p_indices[k]]``.
"""

from __future__ import annotations

import glob
import os
import re
import sys

import h5py
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Same default δ list as run_kagome_otf.sh --snapshot_deltas / --occ_sf_delta_points;
# used to reconstruct point indices for runs whose meta.h5 predates
# snap_pt_indices / occ_pt_indices.
DEFAULT_POINT_DELTAS = (-1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5)


def latest_run_dir(base="data"):
    """Most recently modified profile chunked run directory.

    Matches the engine-tagged name (qaqmc_profile_M=*) and the legacy
    untagged one (M=*_*).
    """
    dirs = [d for pat in ("qaqmc_profile_M=*", "M=*_*")
            for d in glob.glob(os.path.join(base, pat)) if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(
            f"no profile run dirs (qaqmc_profile_M=* / M=*_*) under {base!r}")
    return max(dirs, key=os.path.getmtime)


def default_fig_path(data_path, name):
    """Default figure path: figures/<data basename (minus .h5)>/<name>.png.

    Every plot script mirrors its data directory's name under the top-level
    ``figures/`` tree, so figures from different runs never collide.
    """
    base = os.path.basename(os.path.normpath(str(data_path)))
    if base.endswith(".h5"):
        base = base[:-3]
    return os.path.join("figures", base, f"{name}.png")


def load_meta(run_dir):
    """meta.h5 datasets + attrs (+ derived prof_delta) as a dict."""
    meta = {}
    with h5py.File(os.path.join(run_dir, "meta.h5"), "r") as f:
        for key in f.keys():
            meta[key] = f[key][:]
        meta["attrs"] = dict(f.attrs)
    meta["prof_delta"] = meta["delta_schedule"][meta["p_indices"]]
    return meta


def load_run_attrs(run_dir):
    """Shared run attrs (lattice, boundary, nx, ny, ...) from the first rank file."""
    files = sorted(glob.glob(os.path.join(run_dir, "rank*.h5")),
                   key=lambda p: int(re.search(r"rank(\d+)\.h5$", p).group(1)))
    if not files:
        raise FileNotFoundError(f"no rank*.h5 files in {run_dir!r}")
    with h5py.File(files[0], "r") as f:
        return dict(f.attrs)


def stack_chunks(run_dir, keys, burn_in_fraction=0.0):
    """Concatenate the requested chunk datasets along axis 0 over all ranks/chunks.

    Returns {key: (total_bins, ...) array or None if absent}.  For the occ_*
    datasets the leading axis already holds occ_nbatch super-bins per chunk, so
    the concatenated axis is simply "all bins from everywhere".
    """
    from src.mpi.chunk_io import iter_rank_chunks

    out = {k: [] for k in keys}
    for _rank, _idx, grp in iter_rank_chunks(run_dir, burn_in_fraction):
        for k in keys:
            if k in grp:
                out[k].append(grp[k][:])
    return {k: (np.concatenate(v, axis=0) if v else None) for k, v in out.items()}


def stack_chunk_bins(run_dir, keys, burn_in_fraction=0.0):
    """Like :func:`stack_chunks` but for the SSE layout where each chunk IS one
    bin (scalar or single-bin array, no leading bin axis): each chunk's value
    becomes one row, so scalars stack to (B,) and arrays to (B, ...)."""
    from src.mpi.chunk_io import iter_rank_chunks

    out = {k: [] for k in keys}
    for _rank, _idx, grp in iter_rank_chunks(run_dir, burn_in_fraction):
        for k in keys:
            if k in grp:
                out[k].append(np.asarray(grp[k][()]))
    return {k: (np.stack(v, axis=0) if v else None) for k, v in out.items()}


def resolve_point_indices(meta, kind, requested_deltas=None):
    """Profile-point indices of the snapshot/occ-SF δ points, sorted ascending.

    ``kind`` is 'snap' or 'occ'.  Prefers the indices stored in meta.h5 (newer
    runs); otherwise re-derives them from ``requested_deltas`` with exactly the
    driver's snapping rule (argmin |prof_delta - d|, then unique-sort) — the
    stored per-point axes are ordered by these sorted indices.
    """
    key = f"{kind}_pt_indices"
    if key in meta:
        return np.asarray(meta[key], dtype=int)
    req = DEFAULT_POINT_DELTAS if requested_deltas is None else requested_deltas
    prof_delta = meta["prof_delta"]
    return np.unique(np.array(
        [int(np.argmin(np.abs(prof_delta - float(d)))) for d in req], dtype=int))


def sweep_split(prof_delta):
    """(forward_slice, backward_slice) of the up-then-down δ profile."""
    diffs = np.diff(prof_delta)
    turn = np.where(diffs < 0)[0]
    n_fwd = int(turn[0]) + 1 if len(turn) else len(prof_delta)
    return slice(0, n_fwd), slice(n_fwd, len(prof_delta))


def bin_mean_sem(bins):
    """(mean, sem) over the leading bin axis."""
    arr = np.asarray(bins, dtype=np.float64)
    n = arr.shape[0]
    return arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(max(n, 1))


def loop_string_sizes(attrs):
    """([loop sizes], [string sizes]) matching the stored size-group columns.

    Rebuilt with the driver's own _lattice_observables (zero-copy groups are
    dropped there before the C++ layout, so the filtered order matches the
    stored columns exactly).  Falls back to generic labels on import failure.
    """
    lattice = str(attrs.get("lattice", "kagome_bond"))
    nx, ny = int(attrs["nx"]), int(attrs["ny"])
    boundary = str(attrs.get("boundary", "open"))
    try:
        from src.mpi.qaqmc_mpi import _lattice_observables
        (_bulk, _ls, _ss, loop_meta, string_meta,
         _vs, _ijk) = _lattice_observables(lattice, nx, ny, boundary=boundary)
        loops = [m["size"] for m in loop_meta if m["n_copies"] > 0]
        strings = [m["size"] for m in string_meta if m["n_copies"] > 0]
        return loops, strings
    except Exception:
        return None, None


def occ_geometry(attrs, meta, cell):
    """(cell_R (N,2), basis (N,), site_mask (N,) bool, n_cells) for one occ cell.

    ``cell`` ∈ {'full', 'bulk', 'tri'}.  'full'/'bulk' use the hexagon-void
    unit cell (meta stores its cell_R/basis; the bulk mask is rebuilt with the
    driver's geometry builder), 'tri' is the up+down triangle-pair cell.
    """
    lattice = str(attrs.get("lattice", "kagome_bond"))
    nx, ny = int(attrs["nx"]), int(attrs["ny"])
    boundary = str(attrs.get("boundary", "open"))

    from src.mpi.qaqmc_mpi import (
        _build_occ2_sf_geometry,
        _build_occ2_sf_geometry_tri,
        _build_occ_sf_geometry,
        _build_occ_sf_geometry_tri,
        _lattice_observables,
    )

    ijk_map = bulk_sites = None
    if lattice == "kagome_bond_triangle" or cell == "bulk":
        (bulk_sites, _ls, _ss, _lm, _sm, _vs, ijk_map) = _lattice_observables(
            lattice, nx, ny, boundary=boundary)

    if cell in ("full", "bulk"):
        if "occ_cell_R" in meta and cell == "full":
            cell_R = np.asarray(meta["occ_cell_R"], dtype=np.float64)
            basis = np.asarray(meta["occ_basis"], dtype=int)
            mask = basis >= 0
        else:
            if lattice == "kagome_bond_triangle":
                cell_R, basis, in_bulk = _build_occ_sf_geometry_tri(
                    nx, ny, 1.0, ijk_map, bulk_sites)
            else:
                cell_R, basis, in_bulk = _build_occ_sf_geometry(
                    nx, ny, 1.0, boundary=boundary)
            basis = np.asarray(basis, dtype=int)
            mask = (basis >= 0) if cell == "full" else \
                   (np.asarray(in_bulk, dtype=bool) & (basis >= 0))
    elif cell == "tri":
        if lattice == "kagome_bond_triangle":
            cell_R, basis = _build_occ2_sf_geometry_tri(nx, ny, 1.0, ijk_map)
        else:
            cell_R, basis = _build_occ2_sf_geometry(nx, ny, 1.0)
        basis = np.asarray(basis, dtype=int)
        mask = basis >= 0
    else:
        raise ValueError(f"unknown occ cell {cell!r}; expected full|bulk|tri")

    n_cells = int(mask.sum()) // 6
    return np.asarray(cell_R, dtype=np.float64), basis, mask, n_cells


def occ_sf_matrices(run_dir, attrs, meta, cell="full", connected=True,
                    burn_in_fraction=0.0):
    """Bin-averaged occ-SF matrices S[pt, q, a, b] (complex, per-cell normalised).

    ``cell`` selects the dataset pair ('full' → occ_S_full, 'bulk' → occ_S_bulk,
    'tri' → occ2_S).  With ``connected=True`` the disconnected piece
    ⟨s_a(q)⟩⟨s_b(q)⟩* (from the bin-averaged occupation profile occ_nprof and
    the cell geometry) is subtracted before normalising by the cell count.
    """
    ds = {"full": ("occ_S_full_re", "occ_S_full_im"),
          "bulk": ("occ_S_bulk_re", "occ_S_bulk_im"),
          "tri": ("occ2_S_re", "occ2_S_im")}[cell]
    data = stack_chunks(run_dir, [ds[0], ds[1], "occ_nprof"],
                        burn_in_fraction=burn_in_fraction)
    if data[ds[0]] is None:
        raise KeyError(f"run has no {ds[0]} datasets (occ-SF not enabled?)")

    s_re = data[ds[0]].astype(np.float64).mean(axis=0)   # (pt, q, 6, 6)
    s_im = data[ds[1]].astype(np.float64).mean(axis=0)
    s_mat = s_re + 1j * s_im

    cell_R, basis, mask, n_cells = occ_geometry(attrs, meta, cell)
    if connected:
        nbar = data["occ_nprof"].astype(np.float64).mean(axis=0)  # (pt, N)
        q_points = np.asarray(meta["occ_q_points"], dtype=np.float64)
        phase = np.exp(1j * (cell_R[mask] @ q_points.T))          # (n_sel, q)
        b_sel = basis[mask]
        for pt in range(s_mat.shape[0]):
            w = nbar[pt][mask]                                    # ⟨n_i⟩
            m = np.zeros((q_points.shape[0], 6), dtype=np.complex128)
            for a in range(6):
                sel = b_sel == a
                # ⟨s_a(q)⟩ = Σ_{i: α(i)=a} ⟨n_i⟩ e^{i q·R_i}
                m[:, a] = (w[sel][:, None] * phase[sel, :]).sum(axis=0)
            s_mat[pt] -= np.einsum("qa,qb->qab", m, m.conj())

    return s_mat / max(n_cells, 1)
