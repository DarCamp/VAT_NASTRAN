"""
laminates.py
============
Variable Angle Tow (VAT) laminate utilities.

Provides the ply-angle distribution over all mesh elements using a
Legendre-polynomial interpolation between the user-supplied control angles.
"""

import numpy as np
from scipy.special import legendre


def theta_vals(stacking, var_type, Lx, Ly, xy):
    """
    Compute the fibre-angle distribution for a VAT laminate.

    The angle field for each ply is described by a set of control angles
    [T0, T1, ...] and interpolated over the panel using Legendre polynomials.

    Parameters
    ----------
    stacking : np.ndarray, shape (N_layers, N_control_points)
        Control angles [T0, T1, ...] in degrees for each ply.
    var_type : str
        Spatial variation direction:
        - 'x'   : linear variation along x, mapped to [-1, +1]
        - '|x|' : symmetric variation along x, mapped to  [0,  1]
        - 'y'   : linear variation along y, mapped to [-1, +1]
        - '|y|' : symmetric variation along y, mapped to  [0,  1]
    Lx : float
        Panel dimension along x [m].
    Ly : float
        Panel dimension along y [m].
    xy : np.ndarray, shape (3, N_elements)
        (x, y, z) coordinates of element centroids.

    Returns
    -------
    theta : np.ndarray, shape (N_layers, N_elements)
        Fibre angles in radians at each element centroid for every ply.
    """
    n_layers, n_ctrl = stacking.shape
    n_elems = xy.shape[1]
    stacking_rad = np.deg2rad(stacking)

    theta = np.zeros((n_layers, n_elems))

    for layer in range(n_layers):
        ctrl_angles = stacking_rad[layer, :]

        # Map element centroids to the Legendre reference domain
        if var_type == 'x':
            raw      = xy[0, :]
            ctrl_pts = np.linspace(-1, 1, n_ctrl)
            pts      = raw * 2 / Lx - 1
        elif var_type == '|x|':
            raw      = xy[0, :]
            ctrl_pts = np.linspace(0, 1, n_ctrl)
            pts      = np.abs(raw * 2 / Lx - 1)
        elif var_type == 'y':
            raw      = xy[1, :]
            ctrl_pts = np.linspace(-1, 1, n_ctrl)
            pts      = raw * 2 / Ly - 1
        elif var_type == '|y|':
            raw      = xy[1, :]
            ctrl_pts = np.linspace(0, 1, n_ctrl)
            pts      = np.abs(raw * 2 / Ly - 1)
        else:
            raise ValueError(
                f"Unknown var_type '{var_type}'. "
                "Valid options: 'x', '|x|', 'y', '|y|'."
            )

        # Build Vandermonde-like matrix at control points and solve for coefficients
        V      = np.column_stack([legendre(j)(ctrl_pts) for j in range(n_ctrl)])
        coeffs = np.linalg.solve(V, ctrl_angles)

        # Evaluate the polynomial at all element centroids
        P              = np.column_stack([legendre(j)(pts) for j in range(n_ctrl)])
        theta[layer, :] = P @ coeffs

    return theta