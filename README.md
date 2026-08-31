# VAT_NASTRAN

A Python framework for generating, running, and post-processing **MSC/NX Nastran** finite element analyses of Variable Angle Tow (VAT) composite panels, covering both structural and aeroelastic analyses.

## Features

- Automatic generation of Nastran input decks (`.bdf`) from a single Python configuration file
- Support for multiple Nastran solution sequences:
  - **SOL 101** (`STAT`) — Linear statics
  - **SOL 103** (`FVIB`) — Normal modes / free vibration
  - **SOL 144** (`DIV`) — Static aeroelasticity / divergence
  - **SOL 145** (`FLT`) — Flutter (PK method)
- Structural mesh generation via **gmsh**
- Variable Angle Tow (VAT) laminate support: fibre-angle distribution over the panel computed with Legendre-polynomial interpolation (`laminates.py`)
- Unified boundary-condition convention per panel edge:
  - `None` → degree of freedom free
  - `0` → degree of freedom fixed
  - numeric value → enforced displacement (m) or rotation (deg)
- Robust SPC card formatting (`_real_fmt`) that always writes a decimal point in real fields, avoiding Nastran fatal errors caused by integer values in real fields
- Post-processing for every analysis type:
  - Static analysis: displacement summary at the node of maximum out-of-plane deflection
  - Normal modes: eigenvalue table plus a structured `Vib.txt` (Mode / Omega [rad/s] / Freq [Hz])
  - Divergence: divergence velocity and leading/trailing edge tip displacements
  - Flutter: V–ω and V–σ plots, a structured `Flt.txt` sorted by velocity, and automatic flutter-onset detection via sign changes of the real part of the eigenvalue (zero-crossing detection), with modes lacking a crossing classified as "always stable" or "always unstable"
- VTK (`.vtu`) export of results for visualization in ParaView:
  - Static results (SOL 101/144) in a single file
  - Modal results (SOL 103/145) split into one file per mode
  - Ply fibre angles (`theta_deg`) included as per-element cell data
- Automatic, analysis-specific output folder structure (`Figures/` and `data/` subfolders per analysis type)

## Project structure

```
.
├── config.py         # Single configuration file: geometry, mesh, material, laminate,
│                      # boundary conditions, aerodynamics, solver and output settings
├── bdf_writer.py      # Nastran input deck (.bdf) writer for all supported SOLs
├── laminates.py       # VAT laminate fibre-angle distribution (Legendre interpolation)
├── postprocess.py     # .f06 parsing, VTK export, plots, and text report generation
└── main.py            # Entry point: builds the mesh, writes the BDF, runs Nastran,
                        # and triggers post-processing
```

### Output layout

Running an analysis creates a dedicated subfolder under `config.WORKDIR`, named after the analysis key (`STAT`, `FVIB`, `DIV`, `FLT`):

```
<WORKDIR>/<ANALYSIS>/
├── <basename>.bdf         # Nastran input deck
├── <basename>.f06/.op2    # Nastran output files (written by Nastran)
├── Figures/               # Plots and structured text reports (Vib.txt, Flt.txt, Flt.png, ...)
└── data/                  # VTK (.vtu) exports for ParaView
```

e.g. for `WORKDIR = "results/case_01"` and analysis `FVIB`:

```
results/case_01/FVIB/panelFVIB.bdf
results/case_01/FVIB/Figures/Vib.txt
results/case_01/FVIB/data/panelFVIB_1.vtu
```

## Requirements

- **Python** ≥ 3.9
- **MSC Nastran** or **NX Nastran**, installed and reachable from the command line (only required to actually run the solver; generating the `.bdf` deck alone does not require Nastran)
  - On Windows, the executable path is set in `main.py` / `config.py` (default: `C:\MSC.Software\MSC_Nastran\20190\bin\nastran.exe`)
  - On Linux, the executable name is expected on `PATH` (default: `nast20234`)
- Python packages:
  - `numpy`
  - `scipy` (Legendre-polynomial interpolation in `laminates.py`)
  - `gmsh` (structural mesh generation)
  - `matplotlib` (flutter plots)
  - `pyNastran` and `vtk` (only required for VTK export, `export_vtk` / `config.save_vtk = True`)

Install the dependencies with:

```bash
pip install numpy scipy gmsh matplotlib pyNastran vtk
```

## Usage

1. **Configure the analysis**
   Edit `config.py` to set:
   - working directory (`WORKDIR`)
   - Nastran executable paths
   - analysis type (`ANALYSIS = "STAT" | "FVIB" | "DIV" | "FLT"`)
   - panel geometry, mesh density, material properties
   - VAT laminate stacking and variation type
   - boundary conditions per panel edge
   - aerodynamic parameters (fluid density, trim angle, velocity sweep, reduced frequencies)
   - run controls (`run`, `plot`, `save_mesh`, `save_vtk`, `output_format`)

2. **Run an analysis**
   From the command line, either pass the analysis type as an argument:

   ```bash
   python main.py STAT   # linear statics
   python main.py FVIB   # normal modes
   python main.py DIV    # static aeroelasticity / divergence
   python main.py FLT    # flutter
   ```

   or leave it unset and let `main.py` use `config.ANALYSIS`:

   ```bash
   python main.py
   ```

   `main.py` will:
   - build the structural mesh with gmsh
   - compute the VAT fibre-angle distribution
   - write the Nastran input deck (`.bdf`)
   - launch Nastran on the model (if `config.run = True`)
   - export results to VTK (if `config.save_vtk = True`, requires `output_format = "op2"`)
   - run the appropriate post-processing routine and print/save a summary (if `config.plot = True`)

3. **Inspect the results**
   - Open the `.vtu` files in `data/` with **ParaView** to visualize mode shapes, static deformation, stress, or the VAT fibre-angle field
   - `Figures/Vib.txt` lists mode number and natural frequency (rad/s and Hz) for normal-modes analyses
   - `Figures/Flt.txt` lists, for each mode and velocity, the flutter eigenvalue (σ + jω), plus a ready-to-paste Python `FEM = {...}` dictionary block
   - `Figures/Flt.png` shows the V–ω and V–σ flutter plots, together with the console-printed flutter-onset summary

## Notes

- Real-valued Nastran fields (e.g. SPC enforced values) are always formatted with an explicit decimal point to avoid Nastran fatal errors caused by passing an integer where a real number is expected.
- Flutter onset is detected from sign changes (`np.diff(np.sign(...))`) of the real part of the eigenvalue (σ) along the velocity sweep, filtering out numerical artifacts at the lowest swept velocity when σ is already positive at the start of the sweep.

---

