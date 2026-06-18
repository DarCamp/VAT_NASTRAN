"""
config.py
=========
Single configuration file for the composite panel aeroelastic analysis.
Edit ONLY this file to change analysis type, geometry, material, mesh, BCs,
aerodynamics, solver settings, and output options.
"""

import numpy as np

# ---------------------------------------------------------------------------
# WORKING DIRECTORY
# ---------------------------------------------------------------------------
# Folder where all Nastran files (BDF, f06, op2, vtu) and figures are written.
# Accepts relative or absolute paths. Created automatically if it does not exist.
# Existing files are silently overwritten.
WORKDIR = "results/case_01"

# ---------------------------------------------------------------------------
# NASTRAN EXECUTABLE
# ---------------------------------------------------------------------------
# Full path to the Nastran executable on your system.
NASTRAN_EXE_WINDOWS = r"C:\MSC.Software\MSC_Nastran\20190\bin\nastran.exe"
NASTRAN_EXE_LINUX   = "nast20234"   # command available on PATH, or full path

# ---------------------------------------------------------------------------
# ANALYSIS TYPE
# ---------------------------------------------------------------------------
# Choose one of: "FVIB" (free vibration, SOL 103)
#                "DIV"  (static aeroelastic + divergence, SOL 144)
#                "FLT"  (flutter, SOL 145)
#            or set the analysis type from the command line:
#                python main.py FVIB
ANALYSIS = "FVIB"

# ---------------------------------------------------------------------------
# GEOMETRY
# ---------------------------------------------------------------------------
Lx        = 1.0   # panel length along x  [m]
Ly        = 1.0   # panel length along y  [m]
sweep_deg = 0.0   # leading-edge sweep angle [deg]

# ---------------------------------------------------------------------------
# MESH
# ---------------------------------------------------------------------------
Nx = 64   # number of elements along x
Ny = 64   # number of elements along y

# ---------------------------------------------------------------------------
# MATERIAL  (unidirectional composite – MAT8)
# ---------------------------------------------------------------------------
E1   = 173.0e9   # fibre Young's modulus      [Pa]
E2   =   7.2e9   # matrix Young's modulus     [Pa]
G12  =   3.76e9  # in-plane shear modulus     [Pa]
G23  =   3.76e9  # out-of-plane shear modulus [Pa]
G32  =   3.76e9  # out-of-plane shear modulus [Pa]
nu12 =   0.29    # major Poisson's ratio      [-]
rho  = 1540.0    # material density           [kg/m³]

# ---------------------------------------------------------------------------
# VAT LAMINATE
# ---------------------------------------------------------------------------
# Each row defines one ply as [T0, T1] control angles in degrees.
# var_type controls the spatial variation direction:
#   'x'   → T0 at x = -Lx/2,  T1 at x = +Lx/2   (linear in x, range [-1,+1])
#   '|x|' → T0 at x =  0,     T1 at x =  Lx      (symmetric about x = 0)
#   'y'   → T0 at y = -Ly/2,  T1 at y = +Ly/2
#   '|y|' → T0 at y =  0,     T1 at y =  Ly
stacking = np.array([[0, 90]])
var_type  = '|x|'
t_ply     = 0.01   # thickness of a single ply [m]

# ---------------------------------------------------------------------------
# BOUNDARY CONDITIONS
# ---------------------------------------------------------------------------
# disp : [u1, u2, u3]  –  translational DOFs
# rot  : [r1, r2, r3]  –  rotational DOFs
# 1 = constrained, 0 = free
boundary_conditions = {
    "bottom": {"disp": [1, 1, 1], "rot": [1, 1, 1]},
    "top":    {"disp": [1, 1, 1], "rot": [1, 1, 1]},
    "left":   {"disp": [1, 1, 1], "rot": [1, 1, 1]},
    "right":  {"disp": [1, 1, 1], "rot": [1, 1, 1]},
}

# ---------------------------------------------------------------------------
# AERODYNAMICS
# ---------------------------------------------------------------------------
rho_fluid = 1.226   # fluid density [kg/m³]

# Static aeroelastic / divergence (SOL 144)
alpha_deg = 1.0    # trim angle of attack [deg]
V_trim    = 10.0   # trim velocity        [m/s]
N_roots   = 10     # number of divergence roots requested

# Flutter sweep (SOL 145)
V_range = np.linspace(1, 100, 10)    # velocity sweep range [m/s]
V_plot  = -10                         # velocity for eigenvector output (negative)
k_freq  = np.linspace(0.001, 5, 20)  # reduced frequencies for MKAERO2

# ---------------------------------------------------------------------------
# RUN CONTROLS
# ---------------------------------------------------------------------------
run       = True    # execute Nastran after writing the BDF
plot      = True    # generate post-processing plots after the run
save_mesh = False   # save the gmsh .msh file
save_vtk  = True   # export results to VTK (.vtu) — requires output_format = "op2"

# Nastran output format – choose exactly one
output_format = "op2"   # "op2" | "xdb" | "hdf5"

# Omega column index used in flutter plots
# 6 → rad/s,   4 → Hz
omega_idx = 4