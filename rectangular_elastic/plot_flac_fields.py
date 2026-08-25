# -*- coding: utf-8 -*-
from __future__ import annotations

"""Redraw Stage A FLAC3D CSV six fields with the shared clean 2x3 style."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


TUNNEL_W = 2.0
TUNNEL_H = 2.0
HX = 0.5 * TUNNEL_W
HY = 0.5 * TUNNEL_H
Lx = 5.0
Ly = 5.0

DISP_SCALE = 1.0
DISP_UNIT = "m"
STRESS_UNIT = "MPa"


def _load_csv(csv_path: Path) -> dict[str, np.ndarray]:
    arr = np.genfromtxt(str(csv_path), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.size == 0:
        raise RuntimeError(f"Empty CSV: {csv_path}")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)

    required = {"x", "z", "ux", "uz", "sxx", "syy", "szz", "sxz"}
    missing = sorted(required - set(arr.dtype.names or []))
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}")
    return {key: np.asarray(arr[key], dtype=float) for key in required}


def _center_coordinates(x: np.ndarray, z: np.ndarray, x_center: float, z_center: float) -> tuple[np.ndarray, np.ndarray]:
    return x - x_center, z - z_center


def _build_grid(
    xc: np.ndarray,
    zc: np.ndarray,
    values: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    xs = np.sort(np.unique(np.round(xc, 10)))
    zs = np.sort(np.unique(np.round(zc, 10)))
    x_to_i = {v: i for i, v in enumerate(xs)}
    z_to_j = {v: j for j, v in enumerate(zs)}

    fields = {key: np.full((zs.size, xs.size), np.nan, dtype=float) for key in values}
    xk = np.round(xc, 10)
    zk = np.round(zc, 10)
    for n in range(xc.size):
        i = x_to_i[xk[n]]
        j = z_to_j[zk[n]]
        for key, value in values.items():
            fields[key][j, i] = value[n]

    XX, ZZ = np.meshgrid(xs, zs)
    return XX, ZZ, fields


def load_flac_fields(
    csv_path: Path,
    x_center: float = 5.0,
    z_center: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    raw = _load_csv(csv_path)
    xc, zc = _center_coordinates(raw["x"], raw["z"], x_center=x_center, z_center=z_center)
    values = {
        "ux": raw["ux"] * DISP_SCALE,
        "uz": raw["uz"] * DISP_SCALE,
        "sxx": -raw["sxx"],
        "szz": -raw["szz"],
        "syy": -raw["syy"],
        "sxz": -raw["sxz"],
    }
    XX, ZZ, fields = _build_grid(xc, zc, values)
    hole_mask = (np.abs(XX) < HX) & (np.abs(ZZ) < HY)
    for field in fields.values():
        hole_mask |= ~np.isfinite(field)
    return XX, ZZ, fields, hole_mask


def symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    vals = np.abs(values[np.isfinite(values)])
    vmax = max(float(np.max(vals)) if vals.size else 1.0, 1e-12)
    return -vmax, vmax


def build_plot_limits(fields: dict[str, np.ndarray], hole_mask: np.ndarray) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    valid = ~hole_mask
    for key in ("ux", "uz", "sxx", "szz", "syy", "sxz"):
        vals = fields[key][valid]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise RuntimeError(f"No valid data for {key}")
        if key in ("ux", "uz", "sxz"):
            limits[key] = symmetric_limits(vals)
        else:
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6
            limits[key] = (vmin, vmax)
    return limits


def plot_six_fields(
    XX: np.ndarray,
    ZZ: np.ndarray,
    fields: dict[str, np.ndarray],
    hole_mask: np.ndarray,
    out_path: Path,
    plot_limits: dict[str, tuple[float, float]] | None = None,
    levels: int = 61,
    dpi: int = 300,
) -> None:
    names = [
        ("ux", rf"$u_x$-FLAC3D ({DISP_UNIT})"),
        ("uz", rf"$u_z$-FLAC3D ({DISP_UNIT})"),
        ("sxx", rf"$\sigma_{{xx}}$-FLAC3D ({STRESS_UNIT})"),
        ("szz", rf"$\sigma_{{zz}}$-FLAC3D ({STRESS_UNIT})"),
        ("syy", rf"$\sigma_{{yy}}$-FLAC3D ({STRESS_UNIT})"),
        ("sxz", rf"$\sigma_{{xz}}$-FLAC3D ({STRESS_UNIT})"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.6), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.065, top=0.965, wspace=0.36, hspace=0.32)
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, names):
        values = np.ma.array(fields[key], mask=hole_mask)
        if plot_limits is not None and key in plot_limits:
            vmin, vmax = plot_limits[key]
        elif key in ("ux", "uz", "sxz"):
            vmin, vmax = symmetric_limits(values.compressed())
        else:
            vals = values.compressed()
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1e-6

        contour = ax.contourf(XX, ZZ, values, levels=np.linspace(vmin, vmax, levels), cmap="coolwarm", extend="both")
        ax.add_patch(
            Rectangle((-HX, -HY), 2.0 * HX, 2.0 * HY, facecolor="white", edgecolor="black", linewidth=0.9, zorder=10)
        )
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
        cbar = fig.colorbar(contour, cax=cax)
        cbar.set_ticks(np.linspace(vmin, vmax, 9))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_range_report(fields: dict[str, np.ndarray], hole_mask: np.ndarray) -> None:
    valid = ~hole_mask
    print("\nStage A FLAC CSV plot-grid ranges")
    print("Display convention: displacement=m, stress=MPa and compression positive.")
    for key, name in [
        ("ux", "XDisplacement"),
        ("uz", "ZDisplacement"),
        ("sxx", "XX Stress"),
        ("szz", "ZZ Stress"),
        ("syy", "YY Stress"),
        ("sxz", "XZ Stress"),
    ]:
        vals = fields[key][valid]
        if key in ("ux", "uz"):
            print(f"  {name:<18} (m)  : [{vals.min():+.6f}, {vals.max():+.6f}]")
        else:
            print(
                f"  {name:<18} (MPa): display=[{vals.min():+.6f}, {vals.max():+.6f}], "
                f"FLAC=[{-vals.max():+.6f}, {-vals.min():+.6f}]"
            )


def print_limit_report(plot_limits: dict[str, tuple[float, float]]) -> None:
    print("\nCommon FLAC-reference colorbar limits")
    for key in ("ux", "uz", "sxx", "szz", "syy", "sxz"):
        vmin, vmax = plot_limits[key]
        print(f"  {key:<4}: [{vmin:+.6f}, {vmax:+.6f}]")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot Stage A FLAC3D CSV fields in clean Fig. 11 style.")
    parser.add_argument("--csv", type=str, default=str(root / "data" / "flac_stageA_rect_fields.csv"))
    parser.add_argument("--out", type=str, default=str(root / "figures" / "flac_fields.png"))
    parser.add_argument("--x-center", type=float, default=5.0)
    parser.add_argument("--z-center", type=float, default=5.0)
    parser.add_argument("--levels", type=int, default=61)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    XX, ZZ, fields, hole_mask = load_flac_fields(csv_path, x_center=args.x_center, z_center=args.z_center)
    plot_limits = build_plot_limits(fields, hole_mask)
    plot_six_fields(XX, ZZ, fields, hole_mask, out_path, plot_limits=plot_limits, levels=args.levels, dpi=args.dpi)
    print_range_report(fields, hole_mask)
    print_limit_report(plot_limits)
    print("\nPlot completed")
    print(f"  csv    = {csv_path}")
    print(f"  output = {out_path}")


if __name__ == "__main__":
    main()
