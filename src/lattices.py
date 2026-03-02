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


def kagome_hex_centers(nx: int, ny: int, a: float = 1.0) -> np.ndarray:
    """Return the (nx*ny, 2) centres of the Kagome hexagonal voids."""
    v1 = np.array([a,   0.0])
    v2 = np.array([a/2, a * np.sqrt(3) / 2])
    centers = []
    for j in range(ny):
        for i in range(nx):
            centers.append(i * v1 + j * v2)
    return np.array(centers)



