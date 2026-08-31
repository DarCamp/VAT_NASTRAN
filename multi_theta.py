"""
multi_theta.py
==================
Batch runner: launches the flutter analysis (SOL 145) for every
combination of the VAT control angles [T0, T1]

For each (T0, T1) pair:
    - config.stacking is set to numpy.array([[T0, T1]])
    - config.WORKDIR is set to a dedicated sub-folder, so the BDF/f06/op2,
      the flutter plot (Flt.png) and the flutter text file (Flt.txt) of
      each combination are kept separate and never overwritten.
    - main.main() is called exactly as if you had run
      `python main.py FLT` with that stacking in config.py.

The final "Mode / V_crossing / Omega" summary that main.py normally only
prints to screen is also appended automatically to each Flt.txt file
(this is handled inside postprocess.plot_flutter, already updated to do
so).

Usage
-----
    python multi_theta.py

Edit T0_vals / T1_vals / BASE_WORKDIR below to change the sweep.
Everything else (geometry, mesh, material, BCs, aero, run/plot flags)
is taken from config.py as usual.
"""

import os
import sys
import traceback

import numpy as np

import config
import main as main_module


# ---------------------------------------------------------------------------
# SWEEP DEFINITION
# ---------------------------------------------------------------------------
T0_vals = np.linspace(90, -90, num=13)
T1_vals = np.linspace(90, -90, num=13)

# Root folder collecting every combination's sub-folder
BASE_WORKDIR = "results/multi_theta"

# Analysis to run for every combination (SOL 145 = flutter)
ANALYSIS_KEY = "FLT"


def _tag(T0, T1):
    """Build a filesystem-safe folder name for a given (T0, T1) pair."""
    def fmt(v):
        sign = "p" if v >= 0 else "m"
        return f"{sign}{abs(v):05.1f}".replace(".", "_")
    return f"T0_{fmt(T0)}__T1_{fmt(T1)}"


def run_multi_theta():
    n_total = len(T0_vals) * len(T1_vals)
    n_done  = 0
    failed  = []

    # main.py reads sys.argv[1] to select the analysis type
    sys.argv = ["main.py", ANALYSIS_KEY]

    os.makedirs(BASE_WORKDIR, exist_ok=True)
    summary_path = os.path.join(BASE_WORKDIR, "multi_theta_summary.txt")
    # Start from a clean file at the beginning of every sweep run
    with open(summary_path, "w") as f:
        f.write("Flutter summary for every (T0, T1) stacking combination\n")
        f.write("=" * 70 + "\n")

    for T0 in T0_vals:
        for T1 in T1_vals:
            n_done += 1
            tag = _tag(T0, T1)
            workdir = os.path.join(BASE_WORKDIR, tag)

            # Mutate the shared config module in place: main.py, laminates.py
            # and bdf_writer.py all read from the same imported instance.
            config.stacking = np.array([[T0, T1]])
            config.WORKDIR  = workdir

            print("\n" + "=" * 70)
            print(f"[{n_done}/{n_total}]  T0 = {T0:+.2f} deg   T1 = {T1:+.2f} deg")
            print(f"          -> {workdir}")
            print("=" * 70)

            try:
                result = main_module.main()
                crossing_summary = result.get("flutter_summary") if result else None
            except Exception as exc:
                print(f"*** FAILED for T0={T0:.2f}, T1={T1:.2f}: {exc}")
                traceback.print_exc()
                failed.append((T0, T1))
                crossing_summary = None

            # Append this combination's block to the single master summary
            # file, even on failure (so it's clear which pairs did not run).
            with open(summary_path, "a") as f:
                f.write(f"\nT0 = {T0:+.2f} deg   T1 = {T1:+.2f} deg\n")
                if crossing_summary is not None:
                    f.write(crossing_summary + "\n")
                else:
                    f.write("(analysis failed — see console log)\n")

    print("\n" + "#" * 70)
    print(f"Multi theta completed: {n_total - len(failed)}/{n_total} combinations succeeded.")
    if failed:
        print("Failed combinations:")
        for T0, T1 in failed:
            print(f"  T0={T0:+.2f}, T1={T1:+.2f}")
    print(f"Master summary written to: {summary_path}")
    print("#" * 70)


if __name__ == "__main__":
    run_multi_theta()