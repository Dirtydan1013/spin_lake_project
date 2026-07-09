"""Per-rank site-label permutation (update scan-order decorrelation).

The engines' updates visit sites in label order; that fixed, rank-shared scan
order was shown to deterministically select the ordered-phase domain pattern
(scripts/experiments/scan_order_bias_probe.py).  Running each rank on randomly
relabelled sites keeps the physics identical while decorrelating the scan
geometry.  Conventions used by every driver:

    engine site i  ==  canonical site site_perm[i]
    canonical s    ==  engine site inv_perm[s]

so canonical→engine index maps go through inv_perm, engine→canonical arrays
through ``arr[..., inv_perm]``.  Warm-start configs must record ``site_perm``
(op strings are engine-labelled) and are continued under it.
"""

from __future__ import annotations

import numpy as np


def resolve_site_permutation(N, rank_seed, requested, cfg=None, label=""):
    """(site_perm, inv_perm) for this rank, or (None, None) when unpermuted.

    A warm-start ``cfg``'s labelling ALWAYS wins (its op strings are
    engine-labelled): a saved ``site_perm`` is continued, and a canonical
    config forces canonical labels even when a permutation was requested
    (with a warning) — labellings can never be mixed.
    """
    site_perm = None
    if cfg is not None and "site_perm" in cfg:
        site_perm = np.asarray(cfg["site_perm"], dtype=np.int64)
        if site_perm.size != N:
            raise ValueError(f"[{label}] warm-start site_perm has length "
                             f"{site_perm.size}, expected N={N}")
    elif requested:
        if cfg is not None:
            print(f"[{label}] warm-start config has no site_perm (canonical "
                  f"labels) — continuing WITHOUT site permutation", flush=True)
        else:
            site_perm = np.random.RandomState(
                104729 + int(rank_seed)).permutation(N)
    if site_perm is None:
        return None, None
    return site_perm, np.argsort(site_perm)


def to_engine(idx, inv_perm):
    """Canonical site indices → engine labels (identity when unpermuted)."""
    arr = np.asarray(idx, dtype=np.int64)
    return arr if inv_perm is None else inv_perm[arr]


def permute_rows(arr, site_perm):
    """Per-site array in canonical order → engine order (identity when None)."""
    a = np.asarray(arr)
    return a if site_perm is None else np.ascontiguousarray(a[site_perm])


def unpermute_last_axis(arr, inv_perm):
    """Engine-labelled site axis (last) → canonical (identity when None)."""
    a = np.asarray(arr)
    return a if inv_perm is None else np.ascontiguousarray(a[..., inv_perm])
