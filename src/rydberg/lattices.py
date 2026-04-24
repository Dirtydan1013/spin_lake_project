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

