"""
postprocess.py
==============
MSC Nastran .f06 parser and post-processing utilities.

Functions
---------
read_vibrations(filename)     Read real eigenvalues            (SOL 103)
print_vibrations(results)     Print eigenvalue table           (SOL 103)
read_divergence(filename)     Read divergence dynamic pres.    (SOL 144)
read_displacements(filename)  Read tip LE/TE displacements     (SOL 144)
plot_flutter(filename)        V-omega and V-sigma plots        (SOL 145)
export_vtk(filename)          Export .op2 results to .vtu      (all SOLs)
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# SOL 103 – Normal modes
# ---------------------------------------------------------------------------

def read_vibrations(filename):
    """
    Parse real eigenvalues from a Nastran .f06 file (SOL 103).

    Returns
    -------
    list of tuples (lambda, omega [rad/s], frequency [Hz])
    """
    file_path = filename + ".f06"
    results   = []
    in_section = False

    with open(file_path, "r") as fh:
        for line in fh:
            if "R E A L   E I G E N V A L U E S" in line:
                in_section = True
                continue
            if in_section:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        lam   = float(parts[2])
                        omega = float(parts[3])
                        freq  = float(parts[4])
                        results.append((lam, omega, freq))
                    except ValueError:
                        continue
                elif "MSC.NASTRAN" in line or line.strip() == "":
                    break
    return results


def print_vibrations(results):
    """Print the eigenvalue table in a formatted layout."""
    print(f"{'Mode':>4} | {'lambda':>12} | {'omega [rad/s]':>14} | {'freq [Hz]':>10}")
    print("-" * 50)
    for i, (lam, omega, freq) in enumerate(results, start=1):
        print(f"{i:4d} | {lam:12.4e} | {omega:14.4f} | {freq:10.4f}")


# ---------------------------------------------------------------------------
# SOL 144 – Static aeroelastic / divergence
# ---------------------------------------------------------------------------

def read_divergence(filename):
    """
    Extract divergence dynamic pressures from a Nastran .f06 file (SOL 144).

    Returns
    -------
    np.ndarray
        Array of divergence dynamic pressures [Pa].
    """
    file_path        = filename + ".f06"
    dynamic_pressures = []
    in_section        = False

    with open(file_path, "r") as fh:
        for line in fh:
            if "D I V E R G E N C E      S U M M A R Y" in line:
                in_section = True
                continue
            if in_section:
                if "*** USER INFORMATION MESSAGE" in line:
                    break
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        dynamic_pressures.append(float(parts[1]))
                    except ValueError:
                        continue

    return np.array(dynamic_pressures)


def read_displacements(filename, id_LE=4, id_TE=3):
    """
    Read the out-of-plane tip displacement at the leading edge (LE)
    and trailing edge (TE) from a Nastran .f06 file (SOL 144).

    Parameters
    ----------
    filename : str
        Full path without extension.
    id_LE : int
        Grid point ID at the leading edge tip.
    id_TE : int
        Grid point ID at the trailing edge tip.

    Returns
    -------
    uz_LE, uz_TE : float or None
        Out-of-plane (T3) displacements at LE and TE.
    """
    file_path  = filename + ".f06"
    in_section = False
    uz_LE      = None
    uz_TE      = None

    with open(file_path, "r") as fh:
        for line in fh:
            if "D I S P L A C E M E N T   V E C T O R" in line:
                in_section = True
                continue
            if in_section:
                if "POINT ID." in line and "T3" in line:
                    continue
                parts = line.strip().split()
                if len(parts) >= 6 and parts[0].isdigit():
                    try:
                        pid = int(parts[0])
                        uz  = float(parts[4])
                        if pid == id_TE and uz_TE is None:
                            uz_TE = uz
                        elif pid == id_LE and uz_LE is None:
                            uz_LE = uz
                    except ValueError:
                        continue
                elif "1    SSA OF COMPOSITE PANEL" in line:
                    break

    return uz_LE, uz_TE


# ---------------------------------------------------------------------------
# SOL 145 – Flutter
# ---------------------------------------------------------------------------

def _interpolate_linear(x1, y1, x2, y2, y_target=0.0):
    """
    Linear interpolation: find x such that f(x) = y_target
    given two points (x1, y1) and (x2, y2).
    """
    m = (y1 - y2) / (x1 - x2)
    q = y1 - m * x1
    return (y_target - q) / m


def _interpolate_y(x1, y1, x2, y2, x):
    """Evaluate the linear interpolant at x."""
    m = (y1 - y2) / (x1 - x2)
    return m * x + (y1 - m * x1)


def _parse_flutter_f06(filename, max_modes=4, omega_col=6):
    """
    Internal parser: extract the flutter tabular data from a .f06 file.

    Returns
    -------
    list of (mode_id, np.ndarray)
        Each array has columns:
        [KFREQ, 1/KFREQ, VELOCITY, DAMPING, FREQUENCY, REAL, IMAG]
        sorted by VELOCITY with duplicate velocities removed.
    """
    file_path    = filename + ".f06"
    mode_marker  = "POINT ="
    start_marker = (
        "KFREQ            1./KFREQ         VELOCITY            "
        "DAMPING         FREQUENCY            COMPLEX   EIGENVALUE"
    )
    end_marker = "MSC.NASTRAN"

    with open(file_path, "r") as fh:
        content = fh.readlines()

    mode_data    = {}
    current_mode = None
    capture      = False
    current_data = []

    for line in content:
        if mode_marker in line:
            m = re.search(r"POINT\s*=\s*(\d+)", line)
            if m:
                new_mode = int(m.group(1))
                if current_mode is not None and current_data:
                    mode_data.setdefault(current_mode, []).extend(current_data)
                current_mode = new_mode
                current_data = []
            capture = False

        if start_marker in line:
            capture = True
            continue

        if capture:
            if end_marker in line:
                capture = False
                if current_mode is not None and current_data:
                    mode_data.setdefault(current_mode, []).extend(current_data)
                current_data = []
            else:
                current_data.append(line.strip())

    # Flush last block
    if current_mode is not None and current_data:
        mode_data.setdefault(current_mode, []).extend(current_data)

    # Convert to sorted, deduplicated numpy arrays
    data_arrays = []
    for mode in sorted(mode_data)[:max_modes]:
        rows = []
        for line in mode_data[mode]:
            nums = [n for n in re.split(r"\s+", line) if n]
            if len(nums) == 7:
                try:
                    rows.append([float(n) for n in nums])
                except ValueError:
                    rows.append([np.nan] * 7)
        if rows:
            arr         = np.array(rows)
            arr         = arr[np.argsort(arr[:, 2])]        # sort by velocity
            _, idx      = np.unique(arr[:, 2], return_index=True)
            data_arrays.append((mode, arr[idx]))

    return data_arrays


def plot_flutter(filename, omega_idx=6, max_modes=4,
                 damping_threshold=1e-2, figures_dir=None):
    """
    Generate V-omega and V-sigma plots from a Nastran flutter .f06 file.

    Also prints a summary table of flutter onset velocities and
    an interpolated flutter onset table.

    Parameters
    ----------
    filename : str
        Full path to the analysis file without extension.
    omega_idx : int
        Column index for the frequency quantity: 6 = rad/s, 4 = Hz.
    max_modes : int
        Maximum number of modes to plot.
    damping_threshold : float
        Damping value above which flutter is considered to onset.
    figures_dir : str or None
        Directory where the plot PNG is saved.
        Defaults to a ``Figures/`` subfolder next to ``filename``.
    """
    omega_unit  = {6: " [rad/s]", 4: " [Hz]"}.get(omega_idx, "")
    data_arrays = _parse_flutter_f06(filename, max_modes, omega_idx)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    summary_rows = []
    interp_rows  = []

    for mode, arr in data_arrays:
        velocity = arr[:, 2]
        omega    = arr[:, omega_idx]
        sigma    = arr[:, 5]          # real part of eigenvalue

        axes[0].plot(velocity, omega, marker="o", linestyle="-", label=f"Mode {mode}")
        axes[1].plot(velocity, sigma, marker="o", linestyle="-", label=f"Mode {mode}")

        # Locate the first point where damping exceeds the threshold
        pos_idx = np.where(arr[:, 3] > damping_threshold)[0]
        if pos_idx.size > 0:
            fi = pos_idx[0]
            summary_rows.append((mode, arr[fi, 2], arr[fi, 3], arr[fi, omega_idx]))
            if fi > 0:
                v_fl = _interpolate_linear(
                    arr[fi-1, 2], arr[fi-1, 3],
                    arr[fi,   2], arr[fi,   3]
                )
                w_fl = _interpolate_y(
                    arr[fi-1, 2], arr[fi-1, omega_idx],
                    arr[fi,   2], arr[fi,   omega_idx], v_fl
                )
                interp_rows.append((mode, v_fl, w_fl))
        else:
            summary_rows.append((mode, np.nan, np.nan, np.nan))

    # Axis labels and legend
    axes[0].set_xlabel("Velocity [m/s]")
    axes[0].set_ylabel("Omega" + omega_unit)
    axes[0].set_title("V – Omega")
    axes[1].set_xlabel("Velocity [m/s]")
    axes[1].set_ylabel("σ  (real part of eigenvalue)")
    axes[1].set_title("V – σ")
    fig.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    # Print summary tables
    hdr = f"{'Mode':<10} {'V (damping>0)':<28} {'Damping':<14} {'Omega' + omega_unit:<14}"
    print(hdr)
    print("-" * len(hdr))
    for mode, v, d, w in summary_rows:
        if not np.isnan(v):
            print(f"{int(mode):<10} {v:<28.2f} {d:<14.4f} {w:<14.4f}")
        else:
            print(f"{int(mode):<10} {'—':<28} {'—':<14} {'—':<14}")

    if interp_rows:
        print("\nInterpolated flutter onset:")
        print(f"{'Mode':<10} {'V_flutter [m/s]':<28} {'Omega' + omega_unit:<16}")
        for mode, v, w in interp_rows:
            print(f"{int(mode):<10} {v:<28.4f} {w:<16.8f}")

    # Save figure
    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(os.path.abspath(filename)), "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, "Flt.png"), format="png", bbox_inches="tight")
    if os.name == "nt":
        plt.show()


# ---------------------------------------------------------------------------
# VTK export (all SOLs)
# ---------------------------------------------------------------------------

def export_vtk(filename, subcase=1, modes=None):
    """
    Convert a Nastran .op2 results file to VTK unstructured grid (.vtu).

    Reads the mesh from the companion .bdf file and attaches the following
    result fields to the VTK grid:
    - Nodal displacements  (SOL 144 – ``Displacement_SC<n>``)
    - Modal eigenvectors   (SOL 103/145 – ``Mode_<n>``)
    - Element von Mises stress (when available – ``vonMises_SC<n>``)

    The output file is readable by ParaView and VisIt.

    Parameters
    ----------
    filename : str
        Full path without extension. Both ``<filename>.bdf`` and
        ``<filename>.op2`` must exist in the same directory.
    subcase : int
        Nastran subcase ID to export (default: 1).
    modes : list of int or None
        Mode numbers to export as separate fields (1-based).
        If None, all available modes are exported.

    Output
    ------
    Writes ``<filename>.vtu``.
    """
    try:
        from pyNastran.op2.op2 import OP2
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk
    except ImportError as exc:
        raise ImportError(
            "export_vtk requires pyNastran and vtk.\n"
            "Install with:  pip install pyNastran vtk\n"
            f"Details: {exc}"
        ) from exc

    bdf_path = filename + ".bdf"
    op2_path = filename + ".op2"
    vtu_path = filename + ".vtu"

    if not os.path.exists(op2_path):
        raise FileNotFoundError(f".op2 file not found: {op2_path}")
    if not os.path.exists(bdf_path):
        raise FileNotFoundError(f".bdf file not found: {bdf_path}")

    # ------------------------------------------------------------------
    # 1. Read .op2
    # ------------------------------------------------------------------
    model = OP2()
    model.read_op2(op2_path, combine=True)

    # ------------------------------------------------------------------
    # 2. Read mesh from .bdf
    # ------------------------------------------------------------------
    from pyNastran.bdf.bdf import BDF
    bdf = BDF(debug=False)
    bdf.read_bdf(bdf_path)

    node_ids   = np.array(sorted(bdf.nodes.keys()), dtype=int)
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    coords     = np.array(
        [[bdf.nodes[n].xyz[0], bdf.nodes[n].xyz[1], bdf.nodes[n].xyz[2]]
         for n in node_ids]
    )

    # Collect CQUAD4 connectivity
    elem_ids     = []
    connectivity = []
    for eid, elem in sorted(bdf.elements.items()):
        if elem.type == "CQUAD4":
            elem_ids.append(eid)
            connectivity.append([node_index[n] for n in elem.node_ids])
    connectivity = np.array(connectivity, dtype=int)

    # ------------------------------------------------------------------
    # 3. Build VTK grid
    # ------------------------------------------------------------------
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(coords, deep=True))

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)

    # Map CQUAD4 → VTK_QUAD (cell type 9)
    cell_array = vtk.vtkCellArray()
    for conn in connectivity:
        cell = vtk.vtkQuad()
        for local_i, global_i in enumerate(conn):
            cell.GetPointIds().SetId(local_i, global_i)
        cell_array.InsertNextCell(cell)

    cell_types   = np.full(len(connectivity), vtk.VTK_QUAD, dtype=np.uint8)
    cell_offsets = np.arange(1, len(connectivity) + 1, dtype=int) * 4
    grid.SetCells(
        numpy_to_vtk(cell_types,   deep=True, array_type=vtk.VTK_UNSIGNED_CHAR),
        numpy_to_vtk(cell_offsets, deep=True, array_type=vtk.VTK_ID_TYPE),
        cell_array,
    )

    # ------------------------------------------------------------------
    # 4. Attach nodal fields (displacements / eigenvectors)
    # ------------------------------------------------------------------
    def _add_displacement_field(grid, disp_dict, node_index, n_nodes, label):
        """Attach a 3-component (tx, ty, tz) nodal vector field to the grid."""
        U = np.zeros((n_nodes, 3))
        for nid, vals in disp_dict.items():
            if nid in node_index:
                U[node_index[nid], :] = vals[:3]
        arr = numpy_to_vtk(U, deep=True)
        arr.SetName(label)
        arr.SetNumberOfComponents(3)
        grid.GetPointData().AddArray(arr)

    n_nodes = len(node_ids)

    # Static displacements (SOL 144)
    if hasattr(model, "displacements") and subcase in model.displacements:
        disp      = model.displacements[subcase].node_gridtype
        vals_data = model.displacements[subcase].data[0]
        disp_map  = {nid: vals_data[i] for i, nid in enumerate(disp[:, 0])}
        _add_displacement_field(grid, disp_map, node_index, n_nodes,
                                f"Displacement_SC{subcase}")

    # Modal eigenvectors (SOL 103 / 145)
    if hasattr(model, "eigenvectors") and subcase in model.eigenvectors:
        eig          = model.eigenvectors[subcase]
        avail_modes  = list(range(eig.data.shape[0]))
        export_modes = (
            avail_modes if modes is None
            else [m - 1 for m in modes if 0 <= m - 1 < len(avail_modes)]
        )
        for mi in export_modes:
            vals_m   = eig.data[mi]
            disp_map = {int(nid): vals_m[i]
                        for i, nid in enumerate(eig.node_gridtype[:, 0])}
            _add_displacement_field(grid, disp_map, node_index, n_nodes,
                                    f"Mode_{mi + 1}")

    # ------------------------------------------------------------------
    # 5. Attach element fields (von Mises stress, if available)
    # ------------------------------------------------------------------
    if hasattr(model, "cquad4_stress") and subcase in model.cquad4_stress:
        stress_obj = model.cquad4_stress[subcase]
        vm         = np.zeros(len(elem_ids))
        eid_index  = {eid: i for i, eid in enumerate(elem_ids)}
        for i, eid in enumerate(stress_obj.element):
            if int(eid) in eid_index:
                vm[eid_index[int(eid)]] = stress_obj.data[0, i, 7]  # von Mises index
        arr = numpy_to_vtk(vm, deep=True)
        arr.SetName(f"vonMises_SC{subcase}")
        grid.GetCellData().AddArray(arr)

    # ------------------------------------------------------------------
    # 6. Write .vtu file
    # ------------------------------------------------------------------
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(vtu_path)
    writer.SetInputData(grid)
    writer.Write()

    print(f"VTK exported: {vtu_path}")