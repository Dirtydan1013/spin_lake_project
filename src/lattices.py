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
