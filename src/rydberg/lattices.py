import numpy as np

def generate_1d_chain(N: int, a: float = 1.0) -> np.ndarray:
    """Generate coordinates for a 1D chain of atoms.

    Args:
        N: Number of atoms.
        a: Lattice spacing.

    Returns:
        (N, 2) array of atomic coordinates.
    """
    pos = np.zeros((N, 2))
    pos[:, 0] = np.arange(N) * a
    return pos


def generate_ruby_lattice(nx: int, ny: int, a: float = 1.0) -> np.ndarray:
    """Generate coordinates for a Ruby lattice.

    The Ruby lattice is a triangular Bravais lattice with a 6-atom basis.
    It can be constructed by placing atoms on the links of a Kagome lattice.

    Basis sites in fractional coordinates of the primitive vectors (v1, v2):
        c1 = (1/4, 0)
        c2 = (0, 1/4)
        c3 = (1/4, 1/4)
        c4 = (3/4, 0)
        c5 = (0, 3/4)
        c6 = (3/4, 3/4)

    Args:
        nx: Number of unit cells along the a1 direction.
        ny: Number of unit cells along the a2 direction.
        a: Lattice constant of the underlying triangular lattice.

    Returns:
        (6 * nx * ny, 2) array of atomic coordinates.
    """
    # Primitive vectors of the triangular lattice
    v1 = np.array([a, 0.0])
    v2 = np.array([a/2, a * np.sqrt(3)/2])

    # 6-atom unit cell basis in fractional coordinates
    basis_frac = np.array([
        [1/4, 0],
        [0, 1/4],
        [1/4, 1/4],
        [3/4, 0],
        [0, 3/4],
        [3/4, 3/4]
    ])

    # Convert basis to Cartesian coordinates
    basis_cart = basis_frac[:, 0:1] * v1 + basis_frac[:, 1:2] * v2

    pos = []
    for i in range(nx):
        for j in range(ny):
            cell_origin = i * v1 + j * v2
            for b in basis_cart:
                pos.append(cell_origin + b)

    return np.array(pos)


def lattice_box_vectors(lattice: str, nx: int, ny: int, a: float = 1.0,
                        N: int | None = None) -> np.ndarray:
    """Periodic supercell (torus) vectors for a given lattice.

    Returns an ``(n_box, dim)`` array of the vectors that tile the finite cell
    into an infinite periodic system — passed to the C++ engines as
    ``box_vectors`` to switch from open to periodic boundaries via the
    minimum-image convention.

    - ``1d_chain``:      one vector ``[N*a]`` (a ring of circumference N*a).
    - ``kagome_bond``:   ``(nx*v1, ny*v2)`` on the triangular Bravais lattice
      (v1=(a,0), v2=(a/2, a√3/2)) — a proper torus of the nx×ny void tiling.
    - ``kagome_bond_triangle``:  raises — the cropped protruding-boundary patch
      is inherently open (no commensurate torus), so periodic BC is undefined.
    """
    if lattice == "1d_chain":
        if N is None or N <= 0:
            raise ValueError("periodic 1d_chain needs N > 0")
        return np.ascontiguousarray([[N * a]], dtype=np.float64)
    if lattice == "kagome_bond":
        v1 = np.array([a, 0.0])
        v2 = np.array([a / 2.0, a * np.sqrt(3) / 2.0])
        return np.ascontiguousarray([nx * v1, ny * v2], dtype=np.float64)
    if lattice == "kagome_bond_triangle":
        raise ValueError(
            "periodic boundary is undefined for the cropped "
            "kagome_bond_triangle patch (no commensurate torus); "
            "use open boundary for this lattice")
    raise ValueError(f"unknown lattice {lattice!r} for box vectors")


def generate_kagome_bond_lattice(nx: int, ny: int, a: float = 1.0) -> np.ndarray:
    """Generate atoms at the midpoint of every Kagome hexagonal-void edge.

    The Kagome lattice has large hexagonal voids tiled on a triangular Bravais
    lattice.  This function places ONE atom at the midpoint of each of the 6
    bonds that form the perimeter of each void, giving 6 atoms per void and
    6*nx*ny atoms total.

    Void centres lie at:  C(i,j) = i*v1 + j*v2
    The 6 atom offsets from each void centre are at radius r = sqrt(3)/4 * a
    at angles 30°, 90°, 150°, 210°, 270°, 330°.

    Indexing convention (matches user request):
        - j outer loop  (row 0 = bottom, row ny-1 = top)
        - i inner loop  (column 0 = left, column nx-1 = right)
        - within each void: atoms 0-5 go counterclockwise from 30°

    Args:
        nx: Number of hexagonal voids along v1 direction.
        ny: Number of hexagonal voids along v2 direction.
        a:  Triangular Bravais lattice constant (= hexagon circumradius × 2).

    Returns:
        (6*nx*ny, 2) Cartesian coordinate array.
    """
    v1 = np.array([a,   0.0])
    v2 = np.array([a/2, a * np.sqrt(3) / 2])

    # Atom offsets from void centre: r = sqrt(3)/4 * a  at 30°+k*60°
    r = a * np.sqrt(3) / 4.0
    offsets = np.array([
        [r * np.cos(np.radians(30 + k * 60)),
         r * np.sin(np.radians(30 + k * 60))]
        for k in range(6)
    ])

    pos = []
    for j in range(ny):          # row: bottom (0) → top (ny-1)
        for i in range(nx):      # column: left (0) → right (nx-1)
            centre = i * v1 + j * v2
            for offset in offsets:
                pos.append(centre + offset)

    return np.array(pos)


def kagome_edge_patch_graph(
    nx: int,
    ny: int,
    a: float = 1.0,
    trim_edge_layers: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the cropped kagome graph used by the protruding-boundary patch.

    The construction follows the geometric recipe used for the paper-style
    boundary: first build an oversized ``(nx+1) * (ny+1)`` kagome patch, then
    keep only the kagome edges whose midpoints lie inside the polygon formed
    by the outer unit-cell centers.  The returned graph has exactly
    ``6*nx*ny`` retained edges, so the corresponding bond lattice has
    ``6*nx*ny`` atoms.

    Args:
        nx: Number of kagome unit cells along v1.
        ny: Number of kagome unit cells along v2.
        a:  Triangular Bravais lattice constant.
        trim_edge_layers: Remove this many outer unit-cell layers before
            generating the finite graph.

    Returns:
        ``(vertices, edges)`` where ``vertices`` is an ``(Nv, 2)`` Cartesian
        coordinate array and ``edges`` is an ``(Ne, 2)`` integer array of
        nearest-neighbour kagome vertex pairs.
    """
    trim = int(trim_edge_layers)
    if trim < 0:
        raise ValueError("trim_edge_layers must be non-negative")
    if 2 * trim >= nx or 2 * trim >= ny:
        raise ValueError("trim_edge_layers removes all kagome unit cells")

    v1 = np.array([a, 0.0])
    v2 = np.array([a / 2.0, a * np.sqrt(3) / 2.0])
    vertex_offsets = np.column_stack([
        (a / 2.0) * np.cos(np.radians(np.arange(6) * 60.0)),
        (a / 2.0) * np.sin(np.radians(np.arange(6) * 60.0)),
    ])
    crop_corners = np.array([
        trim * v1 + trim * v2,
        (nx - trim) * v1 + trim * v2,
        (nx - trim) * v1 + (ny - trim) * v2,
        trim * v1 + (ny - trim) * v2,
    ])

    def inside_crop(points, eps=1e-10):
        keep = np.ones(len(points), dtype=bool)
        for idx in range(len(crop_corners)):
            start = crop_corners[idx]
            end = crop_corners[(idx + 1) % len(crop_corners)]
            edge = end - start
            rel = points - start
            keep &= edge[0] * rel[:, 1] - edge[1] * rel[:, 0] >= -eps
        return keep

    def key(point):
        return tuple(np.round(point, 10))

    vertices = []
    index_by_pos = {}

    def vertex_index(point):
        k = key(point)
        idx = index_by_pos.get(k)
        if idx is None:
            idx = len(vertices)
            index_by_pos[k] = idx
            vertices.append(point)
        return idx

    edges = set()
    for j in range(trim, ny - trim + 1):
        for i in range(trim, nx - trim + 1):
            center = i * v1 + j * v2
            ids = [vertex_index(center + offset) for offset in vertex_offsets]
            for sub in range(6):
                edge = tuple(sorted((ids[sub], ids[(sub + 1) % 6])))
                midpoint = 0.5 * (vertices[edge[0]] + vertices[edge[1]])
                if inside_crop(midpoint[None, :])[0]:
                    edges.add(edge)

    if not edges:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 2), dtype=np.int32)

    edges = sorted(edges)
    used = sorted({idx for edge in edges for idx in edge})
    remap = {old: new for new, old in enumerate(used)}
    compact_vertices = np.array([vertices[idx] for idx in used])
    compact_edges = np.array(
        [[remap[i], remap[j]] for i, j in edges],
        dtype=np.int32,
    )
    return compact_vertices, compact_edges


def generate_kagome_bond_triangle_lattice(
    nx: int,
    ny: int,
    a: float = 1.0,
    trim_edge_layers: int = 0,
) -> np.ndarray:
    """Generate atoms on the cropped finite kagome-edge patch.

    Atoms are placed at the midpoints of the retained edges returned by
    ``kagome_edge_patch_graph``.

    Args:
        nx: Number of kagome unit cells along v1.
        ny: Number of kagome unit cells along v2.
        a:  Triangular Bravais lattice constant.
        trim_edge_layers: Remove this many outer cell layers before placing
            atoms.

    Returns:
        Cartesian coordinate array with one site per retained kagome edge.
    """
    vertices, edges = kagome_edge_patch_graph(
        nx,
        ny,
        a,
        trim_edge_layers=trim_edge_layers,
    )
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=float)
    return 0.5 * (vertices[edges[:, 0]] + vertices[edges[:, 1]])


def generate_kagome_lattice(nx: int, ny: int, a: float = 1.0) -> np.ndarray:
    """Generate the original kagome-lattice vertices for the same void tiling.

    The existing ``generate_kagome_bond_lattice`` places atoms at the midpoints
    of the edges of the hexagonal voids.  This helper instead returns the
    unique *vertices* of those hexagonal void outlines, i.e. the intersection
    points of the grey kagome network in the user's sketches.

    Geometry:
        - Void centres are tiled on the same triangular Bravais lattice as in
          ``generate_kagome_bond_lattice``.
        - Around each void centre, the six kagome vertices lie at radius a/2
          and angles 0° + 60° * k.
        - With this convention, the bond-lattice sites are the midpoints of
          nearest-neighbour kagome edges on the void perimeter.
        - Shared vertices between neighbouring voids are deduplicated.

    Args:
        nx: Number of hexagonal voids along v1.
        ny: Number of hexagonal voids along v2.
        a:  Triangular Bravais lattice constant.

    Returns:
        (N, 2) Cartesian coordinate array of unique kagome vertices.
    """
    centers = kagome_hex_centers(nx, ny, a)
    angles = np.radians(np.arange(6) * 60.0)
    offsets = np.column_stack([
        (a / 2.0) * np.cos(angles),
        (a / 2.0) * np.sin(angles),
    ])

    raw = np.vstack([c + offsets for c in centers])
    rounded = np.round(raw, 10)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx.sort()
    return raw[unique_idx]


def kagome_bulk_sites(nx: int, ny: int) -> list:
    """Return site indices for interior atoms of an nx×ny kagome bond lattice.

    An atom is interior if ALL of its inter-void partners (atoms at d≈1.0
    from adjacent voids) exist within the grid.  Each sublattice k has two
    required partner voids; if either is outside the grid the atom is on the
    boundary and excluded.

    Partner void requirements per sublattice k (di, dj offsets from own void):
        k=0: (i+1,j) and (i,j+1)
        k=1: (i-1,j+1) and (i,j+1)
        k=2: (i-1,j) and (i-1,j+1)
        k=3: (i-1,j) and (i,j-1)
        k=4: (i,j-1) and (i+1,j-1)
        k=5: (i+1,j) and (i+1,j-1)

    Each k contributes 25 interior atoms for nx=ny=6, giving 150 total.

    Site index convention: site = (j * nx + i) * 6 + k
    """
    requirements = [
        [(+1,  0), ( 0, +1)],  # k=0
        [(-1, +1), ( 0, +1)],  # k=1
        [(-1,  0), (-1, +1)],  # k=2
        [(-1,  0), ( 0, -1)],  # k=3
        [( 0, -1), (+1, -1)],  # k=4
        [(+1,  0), (+1, -1)],  # k=5
    ]

    def in_grid(ii, jj):
        return 0 <= ii < nx and 0 <= jj < ny

    sites = []
    for j in range(ny):
        for i in range(nx):
            for k in range(6):
                if all(in_grid(i + di, j + dj) for di, dj in requirements[k]):
                    sites.append((j * nx + i) * 6 + k)
    return sites


def kagome_vertex_sites(nx: int, ny: int, interior_only: bool = True) -> list:
    """Return all A_v vertex site-quadruples for an nx×ny kagome bond lattice.

    Each A_v vertex is the product of (1-2n_i) over 4 atoms forming a bowtie
    (hourglass) shape between two adjacent voids.  There are 3 bond directions:
        H : (i,j)k=0,5  -- (i+1,j)k=2,3
        UR: (i,j)k=0,1  -- (i,j+1)k=3,4
        UL: (i,j)k=1,2  -- (i-1,j+1)k=4,5

    If interior_only=True (default), only keeps vertices where at least one
    participating void is strictly interior (i,j ∈ [1, n-2]), removing boundary effects.

    Returns a list of [s0, s1, s2, s3] site-index lists, one per vertex.
    """
    def flat(i, j, k):
        return (j * nx + i) * 6 + k

    def interior(i, j):
        return 1 <= i <= nx - 2 and 1 <= j <= ny - 2

    vertices = []
    for j in range(ny):
        for i in range(nx):
            # H: (i,j) -- (i+1,j)
            if i + 1 < nx:
                if not interior_only or (interior(i, j) or interior(i+1, j)):
                    vertices.append([flat(i,j,0), flat(i,j,5), flat(i+1,j,2), flat(i+1,j,3)])
            # UR: (i,j) -- (i,j+1)
            if j + 1 < ny:
                if not interior_only or (interior(i, j) or interior(i, j+1)):
                    vertices.append([flat(i,j,0), flat(i,j,1), flat(i,j+1,3), flat(i,j+1,4)])
            # UL: (i,j) -- (i-1,j+1)
            if i - 1 >= 0 and j + 1 < ny:
                if not interior_only or (interior(i, j) or interior(i-1, j+1)):
                    vertices.append([flat(i,j,1), flat(i,j,2), flat(i-1,j+1,4), flat(i-1,j+1,5)])
    return vertices


def _make_base_loop(s: int):
    """Generate base loop atoms for a size-s closed loop enclosing (s-1)^2 voids.

    The loop spans void indices i∈[0,s], j∈[0,s] and has 8s-4 atoms.
    Atom convention: (void_i, void_j, k) where k=0..5 at angles 30+k*60°.

    s=2 → 12 atoms,  s=3 → 20 atoms (the original),  s=4 → 28 atoms.
    Valid translations in nx×ny: di∈[0, nx-s-1], dj∈[0, ny-s-1]
    → (nx-s)(ny-s) copies.
    """
    if s < 2:
        raise ValueError("loop size s must be >= 2")
    atoms = []
    # Bottom edge (j=0): i from 1 to s
    atoms.append((1, 0, 0))
    for i in range(2, s):
        atoms.append((i, 0, 2))
        atoms.append((i, 0, 0))
    atoms.append((s, 0, 2))
    atoms.append((s, 0, 1))          # bottom-right corner
    # Right edge (i=s): j from 1 to s-1
    for j in range(1, s - 1):
        atoms.append((s, j, 3))
        atoms.append((s, j, 1))
    atoms.append((s, s - 1, 3))
    atoms.append((s, s - 1, 2))      # top-right corner
    # Top edge (j=s): i from s-1 down to 0
    atoms.append((s - 1, s, 4))
    atoms.append((s - 1, s, 3))
    for i in range(s - 2, 0, -1):
        atoms.append((i, s, 5))
        atoms.append((i, s, 3))
    atoms.append((0, s, 5))
    atoms.append((0, s, 4))          # top-left corner
    # Left edge (i=0): j from s-1 down to 1, then closing pair
    for j in range(s - 1, 1, -1):
        atoms.append((0, j, 0))
        atoms.append((0, j, 4))
    atoms.append((0, 1, 0))
    atoms.append((1, 0, 1))          # closing
    atoms.append((0, 1, 5))          # closing
    return atoms


def _make_base_string(s: int):
    """Generate base string atoms for a size-s open string.

    The string goes up s void-rows along i=0, then right s void-columns
    along the top row, spanning i∈[0,s], j∈[1,s+1]. Has 4s atoms.
    Valid translations in nx×ny:
        di∈[0, nx-s-1],  dj∈[-1, ny-s-2]
    → (nx-s)(ny-s) copies.

    s=1 → 4 atoms,  s=2 → 8 atoms (the original),  s=3 → 12 atoms.
    """
    if s < 1:
        raise ValueError("string size s must be >= 1")
    atoms = []
    # Left column: i=0, j from 1 to s+1
    atoms.append((0, 1, 0))
    for j in range(2, s + 1):
        atoms.append((0, j, 4))
        atoms.append((0, j, 0))
    atoms.append((0, s + 1, 4))
    atoms.append((0, s + 1, 5))      # top-left corner, turns right
    # Top row: j=s+1, i from 1 to s
    for i in range(1, s):
        atoms.append((i, s + 1, 3))
        atoms.append((i, s + 1, 5))
    atoms.append((s, s + 1, 3))      # end
    return atoms


def kagome_loop_string_translations(nx: int, ny: int,
                                    loop_size: int = 3,
                                    string_size: int = 2):
    """Enumerate all valid translated Z(l) loop and C_m(l) string site indices
    on an nx × ny kagome bond lattice.

    Loop (Z(l)): closed path of size `loop_size` (default 3).
        Spans void indices i,j ∈ [0, loop_size].  Has 8*loop_size-4 atoms.
        Valid translations: di∈[0, nx-loop_size-1], dj∈[0, ny-loop_size-1]
        → (nx-loop_size)(ny-loop_size) copies.
        Minimum loop_size=2 (12 atoms, 1 interior void).

    String (C_m(l)): open path of size `string_size` (default 2).
        Spans void indices i∈[0,string_size], j∈[1,string_size+1].
        Has 4*string_size atoms.
        Valid translations: di∈[0, nx-string_size-1], dj∈[-1, ny-string_size-2]
        → (nx-string_size)(ny-string_size) copies.
        Minimum string_size=1 (4 atoms).

    Copies for common lattice sizes:
        loop_size=3, nx=ny=6  → 9 loop copies   (original)
        loop_size=2, nx=ny=4  → 4 loop copies
        loop_size=2, nx=ny=6  → 16 loop copies
        string_size=2, nx=ny=6 → 16 string copies (original)
        string_size=1, nx=ny=4 → 9 string copies

    Returns
    -------
    loop_sets : list[list[int]]
    string_sets : list[list[int]]
    """
    def to_flat(i, j, k):
        return (j * nx + i) * 6 + k

    BASE_LOOP   = _make_base_loop(loop_size)
    BASE_STRING = _make_base_string(string_size)

    # Loop translations
    loop_sets = []
    for dj in range(ny - loop_size):
        for di in range(nx - loop_size):
            loop_sets.append(
                [to_flat(i + di, j + dj, k) for (i, j, k) in BASE_LOOP]
            )

    # String translations
    i_s = [e[0] for e in BASE_STRING]
    j_s = [e[1] for e in BASE_STRING]
    dj_min = -min(j_s)
    dj_max = ny - 1 - max(j_s)
    di_max = nx - 1 - max(i_s)

    string_sets = []
    for dj in range(dj_min, dj_max + 1):
        for di in range(0, di_max + 1):
            string_sets.append(
                [to_flat(i + di, j + dj, k) for (i, j, k) in BASE_STRING]
            )

    return loop_sets, string_sets


def kagome_multi_size_translations(nx: int, ny: int,
                                   loop_sizes=(3,), string_sizes=(2,)):
    """Concatenate loop/string sets across multiple sizes for simultaneous measurement.

    Returns
    -------
    all_loop_sets : list[list[int]]
        All loop copies from all sizes, concatenated.
    all_string_sets : list[list[int]]
        All string copies from all sizes, concatenated.
    loop_meta : list[dict]
        One entry per loop size: {'size': s, 'n_copies': n, 'offset': o}
        where all_loop_sets[o:o+n] are the copies for that size.
    string_meta : list[dict]
        Same structure for strings.

    Usage in post-processing
    ------------------------
    To get |<Z(l)>| for loop size s:
        m = next(m for m in loop_meta if m['size'] == s)
        sub = Z_l[:, :, m['offset']:m['offset']+m['n_copies']]  # (batches, pts, n)
        result = np.abs(sub.mean(axis=0)).mean(axis=-1)          # (n_points,)
    """
    all_loop_sets   = []
    all_string_sets = []
    loop_meta   = []
    string_meta = []

    for ls in loop_sizes:
        lsets, _ = kagome_loop_string_translations(nx, ny, loop_size=ls, string_size=1)
        loop_meta.append({'size': ls, 'n_copies': len(lsets), 'offset': len(all_loop_sets)})
        all_loop_sets.extend(lsets)

    for ss in string_sizes:
        _, ssets = kagome_loop_string_translations(nx, ny, loop_size=2, string_size=ss)
        string_meta.append({'size': ss, 'n_copies': len(ssets), 'offset': len(all_string_sets)})
        all_string_sets.extend(ssets)

    return all_loop_sets, all_string_sets, loop_meta, string_meta


def kagome_hex_centers(nx: int, ny: int, a: float = 1.0) -> np.ndarray:
    """Return the (nx*ny, 2) centres of the Kagome hexagonal voids."""
    v1 = np.array([a,   0.0])
    v2 = np.array([a/2, a * np.sqrt(3) / 2])
    centers = []
    for j in range(ny):
        for i in range(nx):
            centers.append(i * v1 + j * v2)
    return np.array(centers)


# ─── kagome_bond_triangle observable support ─────────────────────────────────
#
# The cropped triangle lattice places atoms on the SAME infinite family of
# points as generate_kagome_bond_lattice (kagome hexagonal-void edge midpoints,
# C(i,j) + offset_k), but (i,j) ranges over the oversized (nx+1)x(ny+1) void
# grid of kagome_edge_patch_graph and the crop polygon removes part of the
# boundary voids' atoms.  Two hexagonal voids never share an edge, so every
# atom carries a unique (i, j, k) label — all observable constructions written
# in void coordinates (bulk sites, A_v bowties, Z loops, C_m strings, VBS
# triangles, occ-SF cells) therefore carry over UNCHANGED; only the
# (i,j,k) -> flat-index map and the "does this copy fit inside the crop"
# filter differ.


def kagome_triangle_ijk_map(nx: int, ny: int, a: float = 1.0) -> dict:
    """Map (void_i, void_j, k) -> flat site index of the triangle lattice.

    (i, j) ranges over the oversized (nx+1) x (ny+1) void grid; keys are only
    present for atoms that survive the crop.  Every atom of
    ``generate_kagome_bond_triangle_lattice(nx, ny, a)`` receives exactly one
    label (asserted).
    """
    pos = generate_kagome_bond_triangle_lattice(nx, ny, a)
    def key(p):
        return (round(p[0] / a, 9), round(p[1] / a, 9))
    by_pos = {key(p): s for s, p in enumerate(pos)}

    v1 = np.array([a, 0.0])
    v2 = np.array([a / 2, a * np.sqrt(3) / 2])
    r = a * np.sqrt(3) / 4.0
    offsets = np.array([
        [r * np.cos(np.radians(30 + k * 60)),
         r * np.sin(np.radians(30 + k * 60))]
        for k in range(6)
    ])

    mapping = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            centre = i * v1 + j * v2
            for k in range(6):
                s = by_pos.get(key(centre + offsets[k]))
                if s is not None:
                    mapping[(i, j, k)] = s
    if len(mapping) != len(pos):
        raise RuntimeError(
            f"kagome_triangle_ijk_map: labelled {len(mapping)} of {len(pos)} atoms"
        )
    return mapping


def _triangle_complete_voids(ijk_map: dict, nx: int, ny: int) -> set:
    """Voids (i, j) of the oversized grid whose all 6 atoms survive the crop."""
    return {
        (i, j)
        for j in range(ny + 1)
        for i in range(nx + 1)
        if all((i, j, k) in ijk_map for k in range(6))
    }


_BULK_PARTNERS_CACHE: dict | None = None


def _bulk_partner_atoms() -> dict:
    """For each sublattice k: the specific partner atoms (dv_i, dv_j, k')
    within distance ~a that kagome_bulk_sites' partner-void requirement
    implies.  On the un-cropped bond lattice "these atoms exist" is exactly
    equivalent to "the two required partner voids are in the grid"; on the
    cropped triangle lattice it is the faithful (atom-level) generalisation.
    Derived geometrically once and cached.
    """
    global _BULK_PARTNERS_CACHE
    if _BULK_PARTNERS_CACHE is not None:
        return _BULK_PARTNERS_CACHE
    a = 1.0
    v1 = np.array([a, 0.0]); v2 = np.array([a / 2, a * np.sqrt(3) / 2])
    r = a * np.sqrt(3) / 4
    off = np.array([
        [r * np.cos(np.radians(30 + 60 * k)), r * np.sin(np.radians(30 + 60 * k))]
        for k in range(6)
    ])
    requirements = [
        [(+1,  0), ( 0, +1)],  # k=0
        [(-1, +1), ( 0, +1)],  # k=1
        [(-1,  0), (-1, +1)],  # k=2
        [(-1,  0), ( 0, -1)],  # k=3
        [( 0, -1), (+1, -1)],  # k=4
        [(+1,  0), (+1, -1)],  # k=5
    ]
    partners = {}
    for k in range(6):
        plist = []
        for (di, dj) in requirements[k]:
            centre = di * v1 + dj * v2
            for kp in range(6):
                if np.linalg.norm(centre + off[kp] - off[k]) < 1.001 * a:
                    plist.append((di, dj, kp))
        partners[k] = plist
    _BULK_PARTNERS_CACHE = partners
    return partners


def kagome_triangle_bulk_sites(nx: int, ny: int, a: float = 1.0,
                               ijk_map: dict | None = None) -> list:
    """Interior atoms of the cropped triangle lattice.

    Same physical criterion as kagome_bulk_sites — all inter-void partner
    atoms within distance ~a exist — applied at the atom level, so a partner
    void only partially removed by the crop still counts through its
    surviving near atoms.  (On the un-cropped bond lattice this reduces
    exactly to kagome_bulk_sites; verified in tests.)
    """
    if ijk_map is None:
        ijk_map = kagome_triangle_ijk_map(nx, ny, a)
    partners = _bulk_partner_atoms()
    sites = []
    for (i, j, k), s in ijk_map.items():
        if all((i + di, j + dj, kp) in ijk_map for di, dj, kp in partners[k]):
            sites.append(s)
    return sorted(sites)


def kagome_triangle_vertex_sites(nx: int, ny: int, a: float = 1.0,
                                 interior_only: bool = True,
                                 ijk_map: dict | None = None) -> list:
    """A_v bowtie quadruples on the cropped triangle lattice.

    Same three bond orientations as kagome_vertex_sites; a vertex is kept when
    all 4 atoms survive the crop, and (interior_only) when at least one of the
    two participating voids is complete — the cropped-lattice analogue of "at
    least one void strictly interior".
    """
    if ijk_map is None:
        ijk_map = kagome_triangle_ijk_map(nx, ny, a)
    complete = _triangle_complete_voids(ijk_map, nx, ny)

    def get4(quads):
        out = [ijk_map.get(q) for q in quads]
        return out if all(v is not None for v in out) else None

    vertices = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            candidates = [
                # H: (i,j) -- (i+1,j)
                ((i + 1, j), [(i, j, 0), (i, j, 5), (i + 1, j, 2), (i + 1, j, 3)]),
                # UR: (i,j) -- (i,j+1)
                ((i, j + 1), [(i, j, 0), (i, j, 1), (i, j + 1, 3), (i, j + 1, 4)]),
                # UL: (i,j) -- (i-1,j+1)
                ((i - 1, j + 1), [(i, j, 1), (i, j, 2), (i - 1, j + 1, 4), (i - 1, j + 1, 5)]),
            ]
            for partner, quads in candidates:
                flat = get4(quads)
                if flat is None:
                    continue
                if interior_only and (i, j) not in complete and partner not in complete:
                    continue
                vertices.append(flat)
    return vertices


def kagome_triangle_multi_size_translations(nx: int, ny: int,
                                            loop_sizes=(3,), string_sizes=(2,),
                                            a: float = 1.0,
                                            ijk_map: dict | None = None):
    """Triangle-lattice counterpart of kagome_multi_size_translations.

    The base loop/string shapes (void coordinates) are identical; translations
    are enumerated over the oversized (nx+1)x(ny+1) void grid and a copy is
    kept only when ALL of its atoms survive the crop.  Sizes that end up with
    zero copies are dropped from the meta (matching the driver's convention).
    """
    if ijk_map is None:
        ijk_map = kagome_triangle_ijk_map(nx, ny, a)
    grid_nx, grid_ny = nx + 1, ny + 1

    def translations(base_atoms):
        i_s = [e[0] for e in base_atoms]
        j_s = [e[1] for e in base_atoms]
        sets = []
        for dj in range(-min(j_s), grid_ny - max(j_s)):
            for di in range(-min(i_s), grid_nx - max(i_s)):
                flat = [ijk_map.get((i + di, j + dj, k)) for (i, j, k) in base_atoms]
                if all(v is not None for v in flat):
                    sets.append(flat)
        return sets

    all_loop_sets, all_string_sets = [], []
    loop_meta, string_meta = [], []
    for ls in loop_sizes:
        lsets = translations(_make_base_loop(ls))
        if lsets:
            loop_meta.append({'size': ls, 'n_copies': len(lsets),
                              'offset': len(all_loop_sets)})
            all_loop_sets.extend(lsets)
    for ss in string_sizes:
        ssets = translations(_make_base_string(ss))
        if ssets:
            string_meta.append({'size': ss, 'n_copies': len(ssets),
                                'offset': len(all_string_sets)})
            all_string_sets.extend(ssets)
    return all_loop_sets, all_string_sets, loop_meta, string_meta
