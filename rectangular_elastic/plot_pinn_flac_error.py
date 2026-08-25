# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Plot PINN-vs-FLAC3D error fields in the same 2x3 Fig. 11 style.

Default difference:
    diff = PINN - FLAC

Both PINN and FLAC stresses use the PINN display convention:
compression positive, MPa.
Displacement differences are displayed in m.
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from plot_flac_fields import (
    DISP_SCALE,
    DISP_UNIT,
    HX,
    HY,
    Lx,
    Ly,
    STRESS_UNIT,
    load_flac_fields,
)
from plot_pinn_fields import (
    DELTA_SIGMA_REF_DEFAULT,
    TAU_REF_DEFAULT,
    eval_quarter_fields,
    infer_stress_param_mode,
    load_model,
    p0,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


def _eval_pinn_on_grid(
    model,
    XX: np.ndarray,
    ZZ: np.ndarray,
    hard_u_bc: bool,
    stress_param: str,
    p0_ref: float,
    delta_sigma_ref: float,
    tau_ref: float,
) -> dict[str, np.ndarray]:
    x_abs = np.abs(XX).reshape(-1)
    z_abs = np.abs(ZZ).reshape(-1)
    q = eval_quarter_fields(
        model=model,
        x_abs=x_abs,
        z_abs=z_abs,
        hard_u_bc=hard_u_bc,
        stress_param=stress_param,
        p0_ref=p0_ref,
        delta_sigma_ref=delta_sigma_ref,
        tau_ref=tau_ref,
    )

    sx = np.sign(XX).reshape(-1)
    sz = np.sign(ZZ).reshape(-1)
    sx[sx == 0.0] = 1.0
    sz[sz == 0.0] = 1.0

    fields = {
        "ux": (sx * q["ux"]).reshape(XX.shape) * DISP_SCALE,
        "uz": (sz * q["uz"]).reshape(XX.shape) * DISP_SCALE,
        "sxx": q["sxx"].reshape(XX.shape),
        "szz": q["szz"].reshape(XX.shape),
        "syy": q["syy"].reshape(XX.shape),
        "sxz": (sx * sz * q["sxz"]).reshape(XX.shape),
    }
    return fields


def _compute_diff(pinn: dict[str, np.ndarray], flac: dict[str, np.ndarray], mode: str) -> dict[str, np.ndarray]:
    if mode not in ("pinn_minus_flac", "flac_minus_pinn"):
        raise ValueError("mode must be pinn_minus_flac or flac_minus_pinn")
    diff: dict[str, np.ndarray] = {}
    for k in ("ux", "uz", "sxx", "szz", "syy", "sxz"):
        if mode == "pinn_minus_flac":
            diff[k] = pinn[k] - flac[k]
        else:
            diff[k] = flac[k] - pinn[k]
    return diff


def _symmetric_limits(vals: np.ndarray, percentile: float = 100.0) -> tuple[float, float]:
    a = np.abs(vals[np.isfinite(vals)])
    if a.size == 0:
        return -1.0, 1.0
    if percentile >= 100.0:
        vmax = float(np.max(a))
    else:
        vmax = float(np.percentile(a, percentile))
    vmax = max(vmax, 1e-12)
    return -vmax, vmax


def plot_error_fields(
    XX: np.ndarray,
    ZZ: np.ndarray,
    diff: dict[str, np.ndarray],
    hole_mask: np.ndarray,
    out_path: Path,
    mode: str,
    levels: int = 61,
    dpi: int = 300,
    limit_percentile: float = 99.0,
) -> None:
    cmap = plt.get_cmap("coolwarm")
    names = [
        ("ux", r"$u_x$" + f"-Error ({DISP_UNIT})"),
        ("uz", r"$u_z$" + f"-Error ({DISP_UNIT})"),
        ("sxx", r"$\sigma_{xx}$" + f"-Error ({STRESS_UNIT})"),
        ("szz", r"$\sigma_{zz}$" + f"-Error ({STRESS_UNIT})"),
        ("syy", r"$\sigma_{yy}$" + f"-Error ({STRESS_UNIT})"),
        ("sxz", r"$\sigma_{xz}$" + f"-Error ({STRESS_UNIT})"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.6), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.065, top=0.965, wspace=0.36, hspace=0.32)
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, names):
        Z = np.ma.array(diff[key], mask=hole_mask)
        vals = Z.compressed()
        vmin, vmax = _symmetric_limits(vals, percentile=limit_percentile)
        levels_arr = np.linspace(vmin, vmax, levels)
        cf = ax.contourf(XX, ZZ, Z, levels=levels_arr, cmap=cmap, extend="both")
        ax.add_patch(Rectangle((-HX, -HY), 2.0 * HX, 2.0 * HY, facecolor="white", edgecolor="black", linewidth=0.9, zorder=10))
        ax.set_aspect("equal")
        ax.set_xlim(-Lx, Lx)
        ax.set_ylim(-Ly, Ly)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=9)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.6%", pad=0.08)
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_ticks(np.linspace(vmin, vmax, 9))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_error_report(XX: np.ndarray, ZZ: np.ndarray, diff: dict[str, np.ndarray], hole_mask: np.ndarray, mode: str) -> None:
    valid = ~hole_mask
    mode_text = "PINN - FLAC3D" if mode == "pinn_minus_flac" else "FLAC3D - PINN"
    print("\n误差统计：" + mode_text)
    for key, name, unit in [
        ("ux", "u_x", DISP_UNIT),
        ("uz", "u_z", DISP_UNIT),
        ("sxx", "sigma_xx", STRESS_UNIT),
        ("szz", "sigma_zz", STRESS_UNIT),
        ("syy", "sigma_yy", STRESS_UNIT),
        ("sxz", "sigma_xz", STRESS_UNIT),
    ]:
        vals = diff[key][valid]
        vals = vals[np.isfinite(vals)]
        rmse = float(np.sqrt(np.mean(vals * vals)))
        max_abs = float(np.max(np.abs(vals)))
        abs_field = np.where(valid, np.abs(diff[key]), np.nan)
        iy, ix = np.unravel_index(np.nanargmax(abs_field), abs_field.shape)
        print(
            f"  {name:<22}: min={vals.min():+.6f} {unit}, max={vals.max():+.6f} {unit}, "
            f"mean={vals.mean():+.6f} {unit}, MAE={np.mean(np.abs(vals)):.6f} {unit}, "
            f"RMSE={rmse:.6f} {unit}, max|.|={max_abs:.6f} {unit} "
            f"@({XX[iy, ix]:+.3f},{ZZ[iy, ix]:+.3f})"
        )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Plot PINN-FLAC3D error fields in v42 Fig. 11 style.")
    p.add_argument("--csv", type=str, default=str(root / "data" / "flac_stageA_rect_fields.csv"), help="FLAC3D exported CSV path")
    p.add_argument("--ckpt", type=str, default=str(root / "checkpoints" / "pinn_model_stageA_rectangular_gpt_v42.pth"), help="PINN checkpoint path")
    p.add_argument("--out", type=str, default=str(root / "figures" / "pinn_minus_flac_error.png"), help="Output image path")
    p.add_argument("--diff-mode", type=str, default="pinn_minus_flac", choices=["pinn_minus_flac", "flac_minus_pinn"])
    p.add_argument("--x-center", type=float, default=5.0)
    p.add_argument("--z-center", type=float, default=5.0)
    p.add_argument("--levels", type=int, default=61)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--limit-percentile", type=float, default=99.0, help="Color limit percentile of |error|; use 100 for full range")
    p.add_argument("--hard-u-bc", action="store_true", help="Force hard displacement BC during PINN inference")
    p.add_argument("--stress-param", type=str, default="auto", choices=["auto", "absolute", "delta"])
    p.add_argument("--p0-ref", type=float, default=p0)
    p.add_argument("--delta-sigma-ref", type=float, default=DELTA_SIGMA_REF_DEFAULT)
    p.add_argument("--tau-ref", type=float, default=TAU_REF_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    ckpt_path = Path(args.ckpt).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    XX, ZZ, flac_fields, hole_mask = load_flac_fields(csv_path, x_center=args.x_center, z_center=args.z_center)
    model = load_model(ckpt_path)

    stem = ckpt_path.stem.lower()
    m = re.search(r"_v(\d+)", stem)
    ver = int(m.group(1)) if m else -1
    use_hard_u_bc = bool(args.hard_u_bc or ver >= 21)
    stress_param = args.stress_param
    if stress_param == "auto":
        stress_param = "delta" if ver >= 21 else infer_stress_param_mode(model)

    pinn_fields = _eval_pinn_on_grid(
        model=model,
        XX=XX,
        ZZ=ZZ,
        hard_u_bc=use_hard_u_bc,
        stress_param=stress_param,
        p0_ref=args.p0_ref,
        delta_sigma_ref=args.delta_sigma_ref,
        tau_ref=args.tau_ref,
    )
    diff = _compute_diff(pinn_fields, flac_fields, mode=args.diff_mode)
    plot_error_fields(
        XX=XX,
        ZZ=ZZ,
        diff=diff,
        hole_mask=hole_mask,
        out_path=out_path,
        mode=args.diff_mode,
        levels=args.levels,
        dpi=args.dpi,
        limit_percentile=args.limit_percentile,
    )
    print_error_report(XX, ZZ, diff, hole_mask=hole_mask, mode=args.diff_mode)

    print("\n出图完成")
    print(f"  csv        = {csv_path}")
    print(f"  checkpoint = {ckpt_path}")
    print(f"  hard_u_bc  = {use_hard_u_bc}")
    print(f"  stress_dec = {stress_param}")
    print(f"  diff_mode  = {args.diff_mode}")
    print(f"  color_pct  = {args.limit_percentile:g}")
    print(f"  output     = {out_path}")


if __name__ == "__main__":
    main()
