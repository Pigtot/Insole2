"""Fusion 360 bridge for Visole geometry.  *** NOT YET EXECUTED ***

Fusion has no headless mode on macOS: this file is a **script for Fusion to run**,
not something the Visole pipeline can call. Install and run it by hand:

    cp adapters/fusion/visole_import.py \
      ~/Library/Application\\ Support/Autodesk/Autodesk\\ Fusion\\ 360/API/Scripts/

then in Fusion: Utilities -> ADD-INS -> Scripts and Add-Ins -> visole_import -> Run.

Honesty note: this script has been written against the Fusion API stubs shipped
with the local install (v2704.1.53) but **has not been run**, because running it
requires the Fusion GUI open and an Autodesk sign-in. Treat it as a starting
point to debug interactively, not as verified code.

Division of labour, and why Fusion is not the workhorse
-------------------------------------------------------
Fusion's simulation API is real -- ``adsk.sim`` exposes Study, Loads,
Constraints, Contacts, MeshSettings, StudyMaterial. But:

* it solves in the **cloud** and needs an entitlement many Fusion licences lack;
* it cannot run headless, so it cannot sit inside a reproducible pipeline;
* meshing a full gyroid at strut resolution would be prohibitive -- our lattice
  coupon already reaches ~18,000 elements for a 3x3x3-cell block.

So `visole.lattice.stiffness` (scikit-fem) stays the workhorse for the many
unit-cell runs, and Fusion is useful as an **independent cross-check on one
representative case**: import a coarse insole solid, assign the *homogenised*
modulus measured by our own solver, and compare deflection under the same load.
Two independent solvers agreeing is worth far more than one solver repeated.
"""

import traceback

try:  # only importable inside Fusion
    import adsk.core
    import adsk.fusion
except ImportError:  # pragma: no cover - documents intent outside Fusion
    adsk = None

# Measured by scripts/calibrate_lattice_stiffness.py; see
# experiments/insole_physics/ for the calibration this came from.
DEFAULT_HOMOGENISED_MODULUS_MPA = 6.6
DEFAULT_POISSON = 0.45


def run(context):
    """Entry point Fusion calls."""
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        dlg = ui.createFileDialog()
        dlg.title = "Select a Visole insole mesh or solid"
        dlg.filter = "Geometry (*.stl;*.step;*.stp);;All (*.*)"
        if dlg.showOpen() != adsk.core.DialogResults.DialogOK:
            return
        path = dlg.filename

        design = adsk.fusion.Design.cast(app.activeProduct)
        if design is None:
            ui.messageBox("Open a Design document first.")
            return
        root = design.rootComponent
        imp = app.importManager

        low = path.lower()
        if low.endswith((".step", ".stp")):
            # STEP goes through ImportManager...
            imp.importToTarget(imp.createSTEPImportOptions(path), root)
        elif low.endswith((".stl", ".obj", ".3mf")):
            # ...but meshes do NOT. ImportManager has no mesh option type at all
            # (only FusionArchive/IGES/SAT/SMT/STEP/SVG); STL arrives via
            # MeshBodies.add. Checked against the local API stubs, because the
            # obvious-looking createMeshImportOptions does not exist.
            root.meshBodies.add(path, adsk.fusion.MeshUnits.MillimeterMeshUnit)
        else:
            ui.messageBox(f"Unsupported file type: {path}")
            return

        lines = [f"Imported: {path}", ""]
        for i, body in enumerate(root.bRepBodies):
            props = body.physicalProperties
            lines.append(f"solid {i}: volume {props.volume:.2f} cm^3, "
                         f"area {props.area:.2f} cm^2")
        for i, body in enumerate(root.meshBodies):
            lines.append(f"mesh {i}: {body.name}")

        lines += [
            "",
            "To cross-check Visole's own FEA:",
            "  1. Switch to the SIMULATION workspace and create a Static Stress study.",
            f"  2. Assign a custom material with E = {DEFAULT_HOMOGENISED_MODULUS_MPA} MPa, "
            f"nu = {DEFAULT_POISSON}.",
            "     (This is the HOMOGENISED lattice modulus, so use a solid insole",
            "      body -- do not try to mesh the lattice struts.)",
            "  3. Fix the base face, apply the same load Visole used (default 700 N).",
            "  4. Compare peak deflection with the Winkler model's indentation.",
        ]
        ui.messageBox("\n".join(lines))

    except Exception:  # pragma: no cover
        if ui:
            ui.messageBox("Failed:\n{}".format(traceback.format_exc()))
