"""
postprocess.py
==============
MSC Nastran .f06 parser and post-processing utilities.

Functions
---------
read_static(filename)         Read full nodal displacements    (SOL 101)
print_static_summary(disp)    Print max-displacement summary   (SOL 101)
read_vibrations(filename)     Read real eigenvalues            (SOL 103)
print_vibrations(results)     Print eigenvalue table           (SOL 103)
write_vibrations_txt(...)     Write Modo/Omega/Freq text file  (SOL 103)
read_divergence(filename)     Read divergence dynamic pres.    (SOL 144)
read_displacements(filename)  Read tip LE/TE displacements     (SOL 144)
plot_flutter(filename)        V-omega/V-sigma plots + text file (SOL 145)
export_vtk(filename)          Export .op2 results to .vtu      (all SOLs)
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# SOL 101 – Linear statics
# ---------------------------------------------------------------------------

def read_static(filename):
    """
    Parse the full nodal displacement/rotation field from a Nastran .f06
    file (SOL 101).

    Parameters
    ----------
    filename : str
        Full path without extension.

    Returns
    -------
    dict
        {node_id: (u1, u2, u3, r1, r2, r3)}  translations [m] and
        rotations [rad] at every output grid point.
    """
    file_path  = filename + ".f06"
    in_section = False
    disp       = {}

    with open(file_path, "r") as fh:
        for line in fh:
            if "D I S P L A C E M E N T   V E C T O R" in line:
                in_section = True
                continue
            if in_section:
                if "POINT ID." in line and "T3" in line:
                    continue
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0].isdigit():
                    try:
                        pid  = int(parts[0])
                        vals = tuple(float(v) for v in parts[2:8])
                        disp[pid] = vals
                    except ValueError:
                        continue
                elif "MSC.NASTRAN" in line or line.strip() == "":
                    break

    return disp


def print_static_summary(disp):
    """
    Print a short summary of a SOL 101 static analysis: the maximum
    out-of-plane (T3) displacement and the node at which it occurs.

    Parameters
    ----------
    disp : dict
        Output of read_static(), {node_id: (u1, u2, u3, r1, r2, r3)}.
    """
    if not disp:
        print("No displacement data found in the .f06 file.")
        return

    node_max = max(disp, key=lambda n: abs(disp[n][2]))
    uz_max   = disp[node_max][2]

    print(f"{'Node':>8} | {'U1 [m]':>12} | {'U2 [m]':>12} | {'U3 [m]':>12} | "
          f"{'R1 [rad]':>12} | {'R2 [rad]':>12} | {'R3 [rad]':>12}")
    print("-" * 100)
    print(f"{node_max:8d} | " + " | ".join(f"{v:12.4e}" for v in disp[node_max]))
    print(f"\nMax |Uz| = {uz_max:.6e} m at node {node_max}")


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


def write_vibrations_txt(results, txt_path):
    """
    Write the free-vibration results to a structured text file with
    columns: Modo, Omega [rad/s], Freq [Hz] (no velocity column, since
    SOL 103 has no velocity sweep).

    Parameters
    ----------
    results : list of tuples
        Output of read_vibrations(): (lambda, omega, freq) per mode.
    txt_path : str
        Full path (including filename) of the text file to write.
    """
    with open(txt_path, "w") as f:
        f.write(f"{'Modo':>6} {'Omega [rad/s]':>16} {'Freq [Hz]':>14}\n")
        for i, (lam, omega, freq) in enumerate(results, start=1):
            f.write(f"{i:>6} {omega:>16.6f} {freq:>14.6f}\n")
    print(f"Vibration data written: {txt_path}")


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


def _write_flutter_txt(data_arrays, txt_path, omega_idx):
    """
    Write the flutter V-omega/V-sigma data (the same values that get
    plotted) to a structured text file, sorted by velocity across all
    modes.

    Columns: Modo, Velocità [m/s], Autovalore (written as sigma+omega*j).

    Parameters
    ----------
    data_arrays : list of (mode_id, np.ndarray)
        Output of _parse_flutter_f06(); each array has columns
        [KFREQ, 1/KFREQ, VELOCITY, DAMPING, FREQUENCY, REAL, IMAG].
    txt_path : str
        Full path (including filename) of the text file to write.
    omega_idx : int
        Column index used for the imaginary (omega) part of the
        eigenvalue: 6 = rad/s, 4 = Hz.
    """
    rows = []
    for mode, arr in data_arrays:
        for row in arr:
            rows.append((row[2], mode, row[5], row[6]))  # (V, mode, sigma, omega)
    rows.sort(key=lambda r: r[0])

    with open(txt_path, "w") as f:
        f.write(f"{'Modo':>6} {'Velocità [m/s]':>16} {'Autovalore (sigma+omega*j)':>32}\n")
        for v, mode, sigma, omega in rows:
            eig_str = f"{sigma:+.6e}{omega:+.6e}j"
            f.write(f"{int(mode):>6} {v:>16.4f} {eig_str:>32}\n")
    print(f"Flutter data written: {txt_path}")


def _format_complex_array(values):
    """Format a complex array as a Python literal, e.g. numpy.array([-1.2e+00+3.4e+00j, ...])."""
    parts = [f"{v.real:+.6e}{v.imag:+.6e}j" for v in values]
    return "numpy.array([" + ", ".join(parts) + "])"


def _format_real_array(values):
    """Format a real array as a Python literal, e.g. numpy.array([1.0, 2.5, ...])."""
    return "numpy.array([" + ", ".join(f"{v:.6g}" for v in values) + "])"


def _write_flutter_fem_block(data_arrays, txt_path, omega_idx):
    """
    Append a FEM-style Python dict literal to the flutter text file, so
    the sigma/omega data can be copy-pasted directly into another script:

        FEM = { "v": numpy.linspace(...) (or numpy.array([...])),
                       1: numpy.array([sigma+omega*j, ...]),
                       2: numpy.array([...]),
                       ...}

    Velocities are taken from the first mode's velocity sweep; each
    mode's complex eigenvalues (sigma + omega*j) are re-ordered onto
    that same velocity vector if needed.

    Parameters
    ----------
    data_arrays : list of (mode_id, np.ndarray)
        Output of _parse_flutter_f06(); columns
        [KFREQ, 1/KFREQ, VELOCITY, DAMPING, FREQUENCY, REAL, IMAG].
    txt_path : str
        Text file to append the block to (created earlier by
        _write_flutter_txt).
    omega_idx : int
        Column index used for the imaginary (omega) part of the
        eigenvalue: 6 = rad/s, 4 = Hz.
    """
    if not data_arrays:
        return

    _, ref_arr = data_arrays[0]
    v_ref = ref_arr[:, 2]

    if len(v_ref) > 1 and np.allclose(np.diff(v_ref), v_ref[1] - v_ref[0]):
        v_str = f"numpy.linspace({v_ref[0]:g}, {v_ref[-1]:g}, num={len(v_ref)})"
    else:
        v_str = _format_real_array(v_ref)

    with open(txt_path, "a") as f:
        f.write("\n\n")
        f.write('FEM = { "v": ' + v_str + ',\n')
        for mode, arr in data_arrays:
            v_this = arr[:, 2]
            eig    = arr[:, 5] + 1j * arr[:, 6]

            if not np.array_equal(v_this, v_ref):
                # Re-order/match this mode's eigenvalues onto v_ref
                lookup = {round(v, 6): e for v, e in zip(v_this, eig)}
                eig = np.array([lookup.get(round(v, 6), complex(np.nan, np.nan))
                                for v in v_ref])

            f.write(f"               {int(mode)}: {_format_complex_array(eig)},\n")
        f.write("}\n")
    print(f"FEM data block appended: {txt_path}")


def _format_crossing_summary(mode_ids, interp_rows, omega_unit):
    """
    Build the final flutter-onset summary table in the format:

        Mode       V_crossing [m/s]     Omega [rad/s]
        ------------------------------------------------
        1          48.0721              2.71565687
        2          82.8353              24.15971780
        3          — (always stable)
        4          — (always stable)

    Parameters
    ----------
    mode_ids : list of int
        Mode numbers (in the order they should appear).
    interp_rows : list of (mode_id, v_flutter, omega_flutter)
        Interpolated flutter-onset points, one per mode that actually
        crosses the damping threshold (see plot_flutter()).
    omega_unit : str
        Unit string appended to the Omega column header, e.g. " [rad/s]".

    Returns
    -------
    str
        The formatted table, ready to print or write to file.
    """
    interp_map = {mode: (v, w) for mode, v, w in interp_rows}

    col1, col2, col3 = 10, 21, 16
    header = f"{'Mode':<{col1}}{'V_crossing [m/s]':<{col2}}{'Omega' + omega_unit:<{col3}}"
    lines = [header, "-" * len(header)]

    for mode in mode_ids:
        if mode in interp_map:
            v, w = interp_map[mode]
            lines.append(f"{mode:<{col1}}{v:<{col2}.4f}{w:<{col3}.8f}")
        else:
            lines.append(f"{mode:<{col1}}{'— (always stable)':<{col2}}")

    return "\n".join(lines)


def plot_flutter(filename, omega_idx=6, max_modes=4,
                 damping_threshold=1e-2, figures_dir=None):
    """
    Generate V-omega and V-sigma plots from a Nastran flutter .f06 file.

    Also prints a summary table of flutter onset velocities, an
    interpolated flutter onset table, and writes a structured text file
    (Modo, Velocità, Autovalore) with all the sigma/omega values used
    for the plots.

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
        Directory where the plot PNG and the text file are saved.
        Defaults to a ``Figures/`` subfolder next to ``filename``.
    """
    omega_unit  = {6: " [rad/s]", 4: " [Hz]"}.get(omega_idx, "")
    data_arrays = _parse_flutter_f06(filename, max_modes, omega_idx)

    if figures_dir is None:
        figures_dir = os.path.join(os.path.dirname(os.path.abspath(filename)), "Figures")
    os.makedirs(figures_dir, exist_ok=True)

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
        pos_idx = np.where(arr[:, 5] > damping_threshold)[0]
        if pos_idx.size > 0:
            fi = pos_idx[0]
            summary_rows.append((mode, arr[fi, 2], arr[fi, 3], arr[fi, omega_idx]))
            if fi > 0:
                v_fl = _interpolate_linear(
                    arr[fi-1, 2], arr[fi-1, 5],
                    arr[fi,   2], arr[fi,   5]
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

    # Final crossing summary (Mode / V_crossing / Omega, "always stable"
    # for modes that never cross the damping threshold)
    mode_ids = [mode for mode, _ in data_arrays]
    crossing_summary = _format_crossing_summary(mode_ids, interp_rows, omega_unit)
    print("\n" + crossing_summary)

    # Write the structured text file with all plotted sigma/omega values
    txt_path = os.path.join(figures_dir, "Flt.txt")
    _write_flutter_txt(data_arrays, txt_path, omega_idx)
    _write_flutter_fem_block(data_arrays, txt_path, omega_idx)

    # Append the final crossing summary at the end of the text file
    with open(txt_path, "a") as f:
        f.write("\n\n")
        f.write(crossing_summary)
        f.write("\n")

    # Save figure
    fig.savefig(os.path.join(figures_dir, "Flt.png"), format="png", bbox_inches="tight")
    if os.name == "nt":
        plt.show()

    return crossing_summary


# ---------------------------------------------------------------------------
# VTK export (all SOLs)
# ---------------------------------------------------------------------------

def export_vtk(filename, subcase=1, modes=None, output_dir=None, theta_deg=None):
    """
    Convert a Nastran .op2 results file to VTK unstructured grid (.vtu).

    Reads the mesh from the companion .bdf file and attaches results:
    - Static nodal displacements (SOL 101/144) + von Mises stress, when
      available, are written to a single file: ``<base>.vtu``.
    - Modal eigenvectors (SOL 103/145) are written to one file per mode,
      so each mode shape can be opened/animated independently in
      ParaView: ``<base>_1.vtu``, ``<base>_2.vtu``, etc. (the suffix is
      the actual Nastran mode number).
    - If ``theta_deg`` is given, the VAT fibre angle of every ply is
      attached to every exported file as an element field
      ``theta_ply<k>_deg`` (constant across load cases/modes, since the
      laminate lay-up doesn't change).

    Parameters
    ----------
    filename : str
        Full path without extension. Both ``<filename>.bdf`` and
        ``<filename>.op2`` must exist in the same directory.
    subcase : int
        Nastran subcase ID to export (default: 1).
    modes : list of int or None
        Mode numbers to export (1-based, actual Nastran mode numbers).
        If None, all available modes are exported.
    output_dir : str or None
        Directory where the .vtu file(s) are written. If None, they are
        written next to ``filename`` (same directory as the .bdf/.op2).
        If given, the directory is created if needed and the .vtu
        file(s) use the same base name as ``filename`` (e.g.
        ``<output_dir>/<basename>.vtu``).
    theta_deg : np.ndarray, shape (N_layers, N_elements), or None
        Fibre angles in degrees for every ply, indexed by element ID
        (i.e. ``theta_deg[:, eid - 1]``) — the same array passed to
        ``bdf_writer.write_bdf``. If None, no angle field is written.

    Output
    ------
    Writes ``<base>.vtu`` (static) and/or ``<base>_<mode>.vtu`` (modal)
    depending on which result types are present in the .op2.
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

    if not os.path.exists(op2_path):
        raise FileNotFoundError(f".op2 file not found: {op2_path}")
    if not os.path.exists(bdf_path):
        raise FileNotFoundError(f".bdf file not found: {bdf_path}")

    if output_dir is None:
        vtu_base = filename
    else:
        os.makedirs(output_dir, exist_ok=True)
        vtu_base = os.path.join(output_dir, os.path.basename(filename))

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
    # 3. Build the base VTK grid (points + cell topology), reused for
    #    every file written below via CopyStructure().
    # ------------------------------------------------------------------
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(coords, deep=True))

    base_grid = vtk.vtkUnstructuredGrid()
    base_grid.SetPoints(vtk_points)

    # Map CQUAD4 → VTK_QUAD (cell type 9)
    cell_array = vtk.vtkCellArray()
    for conn in connectivity:
        cell = vtk.vtkQuad()
        for local_i, global_i in enumerate(conn):
            cell.GetPointIds().SetId(local_i, global_i)
        cell_array.InsertNextCell(cell)

    cell_types   = np.full(len(connectivity), vtk.VTK_QUAD, dtype=np.uint8)
    cell_offsets = np.arange(1, len(connectivity) + 1, dtype=int) * 4
    base_grid.SetCells(
        numpy_to_vtk(cell_types,   deep=True, array_type=vtk.VTK_UNSIGNED_CHAR),
        numpy_to_vtk(cell_offsets, deep=True, array_type=vtk.VTK_ID_TYPE),
        cell_array,
    )

    # ------------------------------------------------------------------
    # 4. Helpers
    # ------------------------------------------------------------------
    def _add_field(grid, node_ids_vtk, data_rows, node_ids_result, label):
        """
        Attach a 3-component (tx, ty, tz) nodal vector field to the VTK grid.

        Parameters
        ----------
        node_ids_vtk    : np.ndarray  – node IDs in VTK grid order (from BDF)
        data_rows       : np.ndarray, shape (n_result_nodes, >=3)
                          displacement / eigenvector values from the .op2
        node_ids_result : np.ndarray  – node IDs matching data_rows rows
        label           : str         – field name shown in ParaView
        """
        # Build a fast lookup: result_node_id → row index in data_rows
        id_to_row = {int(nid): i for i, nid in enumerate(node_ids_result)}

        U = np.zeros((len(node_ids_vtk), 3))
        for vtk_i, nid in enumerate(node_ids_vtk):
            row = id_to_row.get(int(nid))
            if row is not None:
                U[vtk_i, :] = data_rows[row, :3]   # tx, ty, tz only

        arr = numpy_to_vtk(U, deep=True)
        arr.SetName(label)
        arr.SetNumberOfComponents(3)
        grid.GetPointData().AddArray(arr)

    def _write_grid(grid, path):
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(path)
        writer.SetInputData(grid)
        writer.Write()
        print(f"VTK exported: {path}")

    def _add_theta_fields(grid):
        """Attach one cell-data array per ply with the VAT fibre angle [deg]."""
        if theta_deg is None:
            return
        theta_arr = np.asarray(theta_deg)
        n_layers  = theta_arr.shape[0]
        for layer in range(n_layers):
            vals = np.array([theta_arr[layer, eid - 1] for eid in elem_ids])
            arr  = numpy_to_vtk(vals, deep=True)
            arr.SetName(f"theta_ply{layer + 1}_deg")
            grid.GetCellData().AddArray(arr)

    # ------------------------------------------------------------------
    # 5. Static displacements + stress (SOL 101 / 144) → single file
    # ------------------------------------------------------------------
    has_static = hasattr(model, "displacements") and subcase in model.displacements
    if has_static:
        static_grid     = vtk.vtkUnstructuredGrid()
        static_grid.CopyStructure(base_grid)

        disp_obj        = model.displacements[subcase]
        result_node_ids = disp_obj.node_gridtype[:, 0]
        # data shape: (n_load_steps, n_nodes, n_dof) — take step 0
        _add_field(static_grid, node_ids, disp_obj.data[0],
                   result_node_ids, f"Displacement_SC{subcase}")

        if hasattr(model, "cquad4_stress") and subcase in model.cquad4_stress:
            stress_obj = model.cquad4_stress[subcase]
            vm         = np.zeros(len(elem_ids))
            eid_index  = {eid: i for i, eid in enumerate(elem_ids)}
            for i, eid in enumerate(stress_obj.element):
                if int(eid) in eid_index:
                    vm[eid_index[int(eid)]] = stress_obj.data[0, i, 7]  # von Mises index
            arr = numpy_to_vtk(vm, deep=True)
            arr.SetName(f"vonMises_SC{subcase}")
            static_grid.GetCellData().AddArray(arr)

        _add_theta_fields(static_grid)
        _write_grid(static_grid, vtu_base + ".vtu")

    # ------------------------------------------------------------------
    # 6. Modal eigenvectors (SOL 103 / 145) → one file per mode
    # ------------------------------------------------------------------
    has_modal = hasattr(model, "eigenvectors") and subcase in model.eigenvectors
    if has_modal:
        eig             = model.eigenvectors[subcase]
        result_node_ids = eig.node_gridtype[:, 0]
        # eig.modes contains the actual Nastran mode numbers (1-based)
        nastran_modes   = list(eig.modes)           # e.g. [1, 2, 3, ...]

        if modes is None:
            export_indices = list(range(len(nastran_modes)))
        else:
            # Map requested Nastran mode numbers to array indices
            mode_to_idx = {int(m): i for i, m in enumerate(nastran_modes)}
            export_indices = [mode_to_idx[m] for m in modes if m in mode_to_idx]
            missing = [m for m in modes if m not in mode_to_idx]
            if missing:
                print(f"Warning: modes {missing} not found in .op2 "
                      f"(available: {nastran_modes})")

        for mi in export_indices:
            mode_num  = int(nastran_modes[mi])
            mode_grid = vtk.vtkUnstructuredGrid()
            mode_grid.CopyStructure(base_grid)

            # data shape: (n_modes, n_nodes, n_dof)
            _add_field(mode_grid, node_ids, eig.data[mi],
                       result_node_ids, "Mode_shape")

            _add_theta_fields(mode_grid)
            _write_grid(mode_grid, f"{vtu_base}_{mode_num}.vtu")

    if not has_static and not has_modal:
        print("Warning: no displacement or eigenvector results found "
              f"in {op2_path}; nothing exported.")