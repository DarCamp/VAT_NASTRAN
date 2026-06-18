"""
main.py
=======
Entry point for generating the BDF and running MSC Nastran.

Usage:
    python main.py FVIB
    python main.py DIV
    python main.py FLT

All parameters are defined in config.py.
The analysis files (BDF, f06, op2, vtu, figure) are written to config.WORKDIR.
"""

import sys
import os
import subprocess
import numpy as np
import gmsh

import config
from laminates import theta_vals
from bdf_writer import write_bdf
from postprocess import (
    read_vibrations, print_vibrations,
    read_divergence, read_displacements,
    plot_flutter,
    export_vtk,
)


# ---------------------------------------------------------------------------
# Solver map
# ---------------------------------------------------------------------------

SOLVERS = {
    "FVIB": (103, "panelFVIB"),
    "DIV":  (144, "panelDIV"),
    "FLT":  (145, "panelFLT"),
}


def parse_analysis():
    if len(sys.argv) < 2:
        if config.ANALYSIS in SOLVERS:
            return config.ANALYSIS, *SOLVERS[config.ANALYSIS]
        print("Usage: python main.py [FVIB|DIV|FLT]")
        sys.exit(1)
    key = sys.argv[1].upper()
    if key not in SOLVERS:
        print(f"Analysis '{key}' not recognized. Choose from: {', '.join(SOLVERS)}")
        sys.exit(1)
    return key, *SOLVERS[key]


# ---------------------------------------------------------------------------
# Working directory
# ---------------------------------------------------------------------------

def prepare_workdir(workdir):
    """
    Create the working directory if it does not exist.
    If it already exists, the files are silently overwritten.
    Returns the absolute path.
    """
    wd = os.path.abspath(workdir)
    os.makedirs(wd, exist_ok=True)
    os.makedirs(os.path.join(wd, "Figures"), exist_ok=True)
    return wd


# ---------------------------------------------------------------------------
# Mesh with gmsh
# ---------------------------------------------------------------------------

def build_mesh(Lx, Ly, Nx, Ny, sweep_deg):
    sweep = np.deg2rad(sweep_deg)
    gmsh.initialize()
    gmsh.model.add("panel_mesh")

    p1 = gmsh.model.geo.addPoint(0,                       0,  0, 0)
    p2 = gmsh.model.geo.addPoint(Lx,                      0,  0, 0)
    p3 = gmsh.model.geo.addPoint(Ly*np.tan(sweep) + Lx,  Ly, 0, 0)
    p4 = gmsh.model.geo.addPoint(Ly*np.tan(sweep),        Ly, 0, 0)

    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)

    cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    ps = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.setTransfiniteSurface(ps)
    gmsh.model.mesh.setTransfiniteCurve(l1, Nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(l3, Nx + 1)
    gmsh.model.mesh.setTransfiniteCurve(l2, Ny + 1)
    gmsh.model.mesh.setTransfiniteCurve(l4, Ny + 1)
    gmsh.model.mesh.setRecombine(2, ps)
    gmsh.model.mesh.generate(2)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)

    # Rinumerazione elementi da 1
    gmsh.model.mesh.renumberElements(
        elem_tags[0], [i + 1 for i in range(len(elem_tags[0]))]
    )
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)

    return node_tags, node_coords, elem_types, elem_tags, elem_node_tags, p1, p2, p3, p4


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    analysis, sol, basename = parse_analysis()
    Nelems = config.Nx * config.Ny

    # --- Working directory ---
    wd = prepare_workdir(config.WORKDIR)
    # Percorso completo del file di analisi (senza estensione)
    filepath = os.path.join(wd, basename)

    print()
    print("=" * 50)
    print(f"  JOB      : {basename}  (SOL {sol})")
    print(f"  WORKDIR  : {wd}")
    print(f"  DOF str  : {(config.Nx+1)*(config.Ny+1)*6}")
    print(f"  DOF aero : {Nelems}")
    print(f"  DOF tot  : {Nelems + (config.Nx+1)*(config.Ny+1)*6}")
    print("=" * 50)
    print()

    # --- Mesh ---
    (node_tags, node_coords,
     elem_types, elem_tags, elem_node_tags,
     p1, p2, p3, p4) = build_mesh(config.Lx, config.Ly,
                                    config.Nx, config.Ny,
                                    config.sweep_deg)

    if config.save_mesh:
        gmsh.write(filepath + ".msh")

    # --- VAT angles ---
    centers = gmsh.model.mesh.getBarycenters(3, 1, 0, 0).reshape(Nelems, 3)
    x_c, y_c, z_c = centers.T
    xy = np.vstack((x_c, y_c, z_c))

    theta = theta_vals(
        stacking=config.stacking,
        var_type=config.var_type,
        Lx=config.Lx, Ly=config.Ly,
        xy=xy,
    )
    theta_deg = np.rad2deg(theta)

    # --- Material properties ---
    props = [config.E1, config.E2, config.nu12,
             config.G12, config.G23, config.G32, config.rho]

    # --- Trim: dynamic pressure ---
    q_trim = 0.5 * config.rho_fluid * config.V_trim**2

    # --- BDF writing ---
    write_bdf(
        filename=filepath, sol=sol,
        node_tags=node_tags, node_coords=node_coords,
        elem_types=elem_types, elem_tags=elem_tags,
        elem_node_tags=elem_node_tags,
        theta_deg=theta_deg, t_ply=config.t_ply,
        props=props,
        boundary_conditions=config.boundary_conditions,
        Lx=config.Lx, Ly=config.Ly,
        Nx=config.Nx, Ny=config.Ny,
        Nelems=Nelems,
        output_format=config.output_format,
        gmsh=gmsh, p1=p1, p2=p2, p3=p3, p4=p4,
        rho_fluid=config.rho_fluid,
        q=q_trim,
        alpha_deg=config.alpha_deg,
        N_roots=config.N_roots,
        k_freq=config.k_freq,
        V_range=config.V_range,
        V_plot=config.V_plot,
    )

    gmsh.finalize()
    print(f"BDF written : {filepath}.bdf")

    # --- Execution of NASTRAN ---
    if config.run:
        print(f"Starting NASTRAN SOL {sol} …")
        # NASTRAN writes the output files in the directory from which it is launched.
        # We pass cwd=wd and the base name of the BDF: all results
        # (f06, op2, etc.) land directly in the WORKDIR.
        bdf_name = basename + ".bdf"
        if os.name == "nt":
            nastran_exe = r"C:\MSC.Software\MSC_Nastran\20190\bin\nastran.exe"
            cmd = f"{nastran_exe} {bdf_name} scr=yes old=no"
        else:
            cmd = f"nast20234 {bdf_name} scr=yes old=no"

        result = subprocess.run(cmd, capture_output=True, text=True,
                                shell=True, cwd=wd)
        print(result.stdout)
        if result.returncode != 0:
            print("NASTRAN stderr:", result.stderr)

    # --- VTK export ---
    if config.save_vtk:
        if config.output_format != "op2":
            print("Warning: save_vtk requires output_format = 'op2' in config.py")
        else:
            export_vtk(filepath)

    # --- Post-processing ---
    if config.plot:
        figures_dir = os.path.join(wd, "Figures")

        if sol == 103:
            res = read_vibrations(filepath)
            print_vibrations(res)

        elif sol == 144:
            q_div = read_divergence(filepath)
            v_div = np.sqrt(2 * q_div / config.rho_fluid)
            print("Velocità di divergenza [m/s]:", v_div)

            uz_LE, uz_TE = read_displacements(filepath)
            print(f"uz LE = {uz_LE},  uz TE = {uz_TE},  twist = {uz_LE - uz_TE:.6f}")

        elif sol == 145:
            plot_flutter(filepath, omega_idx=config.omega_idx,
                         figures_dir=figures_dir)


if __name__ == "__main__":
    main()