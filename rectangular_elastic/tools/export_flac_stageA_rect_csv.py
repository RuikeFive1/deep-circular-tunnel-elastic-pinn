# -*- coding: ascii -*-
"""
FLAC3D elastic Stage-A exporter for PINN comparison.

Run inside FLAC3D Python:
    python-call "export_flac_stageA_rect_csv.py"

Output CSV columns:
    x,y,z,ux,uy,uz,sxx,syy,szz,sxy,sxz,syz,zone_id

Units:
    displacement: m
    stress: MPa
"""

import csv
import os
import sys

import itasca as it


def _it_module():
    """
    Return the FLAC itasca module.
    This guards against cases where alias `it` is missing in FLAC Python state.
    """
    global it
    try:
        _ = it  # noqa: F841
    except Exception:
        import itasca as _it
        it = _it
    return it


# ------------------------------------------------------------------
# User settings
# ------------------------------------------------------------------
RESTORE_SAVE = True
SAVE_NAME = "elastic-rect-stageA-full.sav"
OUT_CSV = "flac_stageA_rect_fields.csv"

# Keep only one layer around y=0.1 (plane-strain middle layer)
Y_MID = 0.1
Y_TOL = 0.11

# Expected rows for your current model (50x1x50 with 10x1x10 null tunnel):
# 2500 total - 100 null = 2400
EXPECTED_ROWS_HINT = 2400


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
def _maybe_call(v):
    if callable(v):
        return v()
    return v


def _obj_get(obj, names):
    """
    Read object attribute by trying several possible names.
    """
    for n in names:
        if hasattr(obj, n):
            return _maybe_call(getattr(obj, n))
    raise RuntimeError("Cannot read object attribute. names={0}".format(names))


def _try_restore():
    """
    Try several save-file paths for FLAC3D 7 project layouts.
    """
    cands = [
        SAVE_NAME,
        os.path.join("test", SAVE_NAME),
        os.path.join(".", SAVE_NAME),
        os.path.join(".\\test", SAVE_NAME),
    ]

    last_err = None
    _it = _it_module()
    for s in cands:
        try:
            _it.command("model restore '{0}'".format(s))
            print("[OK] model restored from: {0}".format(s))
            return
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError("model restore failed for all candidates: {0}\nlast error: {1}".format(cands, last_err))


def _vec_get(v, idx, names):
    """
    Read vector component by name first, then by index.
    """
    for n in names:
        if hasattr(v, n):
            return float(_maybe_call(getattr(v, n)))
    try:
        return float(v[idx])
    except Exception:
        raise RuntimeError("Cannot read vector component. idx={0}, names={1}".format(idx, names))


def _gp_disp(gp):
    """
    Read gridpoint displacement (m).
    Compatible with both scalar and vector style APIs.
    """
    try:
        ux = float(_obj_get(gp, ("disp_x", "displacement_x")))
        uy = float(_obj_get(gp, ("disp_y", "displacement_y")))
        uz = float(_obj_get(gp, ("disp_z", "displacement_z")))
        return ux, uy, uz
    except Exception:
        pass

    d = _obj_get(gp, ("disp", "displacement", "displacement_vector"))
    ux = _vec_get(d, 0, ("x",))
    uy = _vec_get(d, 1, ("y",))
    uz = _vec_get(d, 2, ("z",))
    return ux, uy, uz


def _zone_stress_mpa(z):
    """
    Read zone stress and convert Pa -> MPa.
    Return order: sxx, syy, szz, sxy, sxz, syz
    """
    s = _obj_get(z, ("stress", "stress_tensor"))
    sxx = _vec_get(s, 0, ("xx",)) / 1.0e6
    syy = _vec_get(s, 1, ("yy",)) / 1.0e6
    szz = _vec_get(s, 2, ("zz",)) / 1.0e6
    sxy = _vec_get(s, 3, ("xy",)) / 1.0e6
    sxz = _vec_get(s, 4, ("xz",)) / 1.0e6
    syz = _vec_get(s, 5, ("yz",)) / 1.0e6
    return sxx, syy, szz, sxy, sxz, syz


def _csv_write_rows(path, header, rows):
    """
    Python2/3 compatible CSV writer.
    """
    if sys.version_info[0] < 3:
        f = open(path, "wb")
        try:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        finally:
            f.close()
    else:
        f = open(path, "w", newline="", encoding="utf-8")
        try:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        finally:
            f.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    script_path = os.path.abspath(__file__) if "__file__" in globals() else "(no __file__)"
    print("[INFO] running:", script_path)
    print("[INFO] cwd:", os.getcwd())
    _it = _it_module()

    # 1) Restore elastic Stage-A result
    if RESTORE_SAVE:
        _try_restore()

    rows = []

    # 2) Loop zones
    for z in _it.zone.list():
        # Skip tunnel null zones if possible
        try:
            model_name = str(_obj_get(z, ("model",))).lower()
            if "null" in model_name:
                continue
        except Exception:
            pass

        # Zone center position
        p = _obj_get(z, ("pos", "position", "centroid"))
        x = _vec_get(p, 0, ("x",))
        y = _vec_get(p, 1, ("y",))
        zz = _vec_get(p, 2, ("z",))

        # Keep only one layer around y=0.1
        if abs(y - Y_MID) > Y_TOL:
            continue

        # Average displacement from zone gridpoints
        gps = list(_obj_get(z, ("gridpoints", "gridpointlist", "gp_list")))
        if len(gps) == 0:
            continue

        ux_sum = 0.0
        uy_sum = 0.0
        uz_sum = 0.0
        for gp in gps:
            ux_gp, uy_gp, uz_gp = _gp_disp(gp)
            ux_sum += ux_gp
            uy_sum += uy_gp
            uz_sum += uz_gp

        n_gp = float(len(gps))
        ux = ux_sum / n_gp
        uy = uy_sum / n_gp
        uz = uz_sum / n_gp

        # Stress in MPa
        sxx, syy, szz, sxy, sxz, syz = _zone_stress_mpa(z)

        # Zone id
        zid = int(_obj_get(z, ("id", "zone_id")))

        rows.append([x, y, zz, ux, uy, uz, sxx, syy, szz, sxy, sxz, syz, zid])

    # 3) Sort rows for stable downstream plotting
    rows.sort(key=lambda r: (r[2], r[0], r[1]))

    # 4) Write CSV
    # Export to current FLAC working directory for reproducibility.
    out_path = os.path.abspath(OUT_CSV)
    header = ["x", "y", "z", "ux", "uy", "uz", "sxx", "syy", "szz", "sxy", "sxz", "syz", "zone_id"]
    _csv_write_rows(out_path, header, rows)

    # 5) Print summary
    print("[OK] Exported rows: {0}".format(len(rows)))
    print("[OK] CSV: {0}".format(out_path))
    print("[OK] Units: displacement=m, stress=MPa")
    if EXPECTED_ROWS_HINT > 0:
        err_pct = abs(len(rows) - EXPECTED_ROWS_HINT) * 100.0 / float(EXPECTED_ROWS_HINT)
        print("[CHECK] expected rows ~ {0}, got {1}, diff = {2:.2f}%".format(EXPECTED_ROWS_HINT, len(rows), err_pct))
        if err_pct > 10.0:
            print("[WARN] Row count differs a lot from expected. Please check y layer filter or save file.")


# FLAC3D `pythonfile` may not set __name__ == "__main__".
# Execute directly so the exporter always runs.
main()
