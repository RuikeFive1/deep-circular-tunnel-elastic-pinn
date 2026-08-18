# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Deep-buried circular tunnel, Section 5.1 pure elastic stage.

Create a 3x3 Fig.10-style comparison:
    row 1: sigma_r PINN, analytical, signed error
    row 2: sigma_theta PINN, analytical, signed error
    row 3: u_r PINN, analytical, signed error

Stress is plotted in MPa with compression positive. Displacement is plotted in
cm, matching the paper subsection and the existing rectangular-tunnel plot
style where displacement is displayed in cm.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from apinn.models import MLP_APINN  # noqa: E402

torch.set_default_dtype(torch.float64)


def cart_to_polar(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return r, theta


def stress_cart_to_polar(
    sigma_xx: np.ndarray,
    sigma_yy: np.ndarray,
    sigma_xy: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.cos(theta)
    s = np.sin(theta)
    c2 = c * c
    s2 = s * s
    cs = c * s
    sigma_r = sigma_xx * c2 + sigma_yy * s2 + 2.0 * sigma_xy * cs
    sigma_t = sigma_xx * s2 + sigma_yy * c2 - 2.0 * sigma_xy * cs
    tau_rt = (sigma_yy - sigma_xx) * cs + sigma_xy * (c2 - s2)
    return sigma_r, sigma_t, tau_rt


def radial_displacement(ux: np.ndarray, uy: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return ux * np.cos(theta) + uy * np.sin(theta)


def draw_annulus(ax: plt.Axes, a: float, r_outer: float) -> None:
    theta = np.linspace(0.0, 0.5 * np.pi, 361)
    ax.plot(a * np.cos(theta), a * np.sin(theta), color="black", linewidth=0.9, zorder=10)
    ax.plot(r_outer * np.cos(theta), r_outer * np.sin(theta), color="black", linewidth=0.9, zorder=10)
    ax.plot([a, r_outer], [0.0, 0.0], color="black", linewidth=0.8, zorder=10)
    ax.plot([0.0, 0.0], [a, r_outer], color="black", linewidth=0.8, zorder=10)


def load_model(weights: Path, device: torch.device) -> MLP_APINN:
    model = MLP_APINN(hidden=40, depth=6).to(device=device, dtype=torch.float64)
    state = torch.load(str(weights), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def eval_pinn(
    model: MLP_APINN,
    device: torch.device,
    a: float,
    r_outer: float,
    nxy: int,
    stress_convention: str,
    anchor_ur: str,
) -> dict[str, np.ndarray]:
    xs = np.linspace(0.0, r_outer, nxy)
    ys = np.linspace(0.0, r_outer, nxy)
    xx, yy = np.meshgrid(xs, ys)
    rr, theta = cart_to_polar(xx, yy)
    valid = (rr >= a) & (rr <= r_outer)

    points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    with torch.no_grad():
        out = model(torch.tensor(points, dtype=torch.float64, device=device)).detach().cpu().numpy()
    out = out.reshape(xx.shape[0], xx.shape[1], -1)

    ux = out[..., 0]
    uy = out[..., 1]
    sigma_xx = out[..., 2]
    sigma_yy = out[..., 3]
    sigma_xy = out[..., 5]

    sigma_r, sigma_t, _tau_rt = stress_cart_to_polar(sigma_xx, sigma_yy, sigma_xy, theta)
    ur = radial_displacement(ux, uy, theta)

    sign = 1.0
    if stress_convention == "tension_positive":
        sign = -1.0
    elif stress_convention == "auto":
        shell = valid & (np.abs(rr - r_outer) <= max((r_outer - a) / nxy, 1e-6))
        if np.any(shell) and float(np.nanmean(sigma_r[shell])) < 0.0:
            sign = -1.0
    elif stress_convention != "compression_positive":
        raise ValueError("stress_convention must be auto, compression_positive, or tension_positive")
    sigma_r = sign * sigma_r
    sigma_t = sign * sigma_t

    if anchor_ur == "outer_mean":
        shell = valid & (np.abs(rr - r_outer) <= max((r_outer - a) / nxy, 1e-6))
        base = float(np.nanmean(ur[shell])) if np.any(shell) else 0.0
        ur = ur - base
    elif anchor_ur == "none":
        pass
    else:
        raise ValueError("anchor_ur must be outer_mean or none")

    def masked(arr: np.ndarray) -> np.ndarray:
        out_arr = np.array(arr, dtype=float, copy=True)
        out_arr[~valid] = np.nan
        return out_arr

    return {
        "x": xx,
        "y": yy,
        "r": rr,
        "theta": theta,
        "valid": valid,
        "sigma_r": masked(sigma_r),
        "sigma_t": masked(sigma_t),
        "ur_cm": masked(100.0 * ur),
        "stress_sign": sign,
    }


def analytical_elastic(
    xx: np.ndarray,
    yy: np.ndarray,
    a: float,
    r_outer: float,
    p0: float,
    p_inner: float,
    young: float,
    nu: float,
) -> dict[str, np.ndarray]:
    rr, _theta = cart_to_polar(xx, yy)
    valid = (rr >= a) & (rr <= r_outer)
    r_safe = np.where(valid, rr, 1.0)

    # Compression-positive Lamé solution for a finite annulus.
    b = (p0 - p_inner) / (1.0 / (a * a) - 1.0 / (r_outer * r_outer))
    a0 = p_inner + b / (a * a)
    sigma_r = a0 - b / (r_safe * r_safe)
    sigma_t = a0 + b / (r_safe * r_safe)

    # Displacement is the incremental excavation displacement relative to the
    # initial hydrostatic p0 state. The outer boundary is anchored to u_r=0,
    # matching the existing APINN visualization.
    delta_a = a0 - p0
    ur = (1.0 + nu) / young * (
        -(1.0 - 2.0 * nu) * delta_a * r_safe
        - b / r_safe
        + (1.0 - 2.0 * nu) * delta_a * r_outer
        + b / r_outer
    )

    def masked(arr: np.ndarray) -> np.ndarray:
        out = np.array(arr, dtype=float, copy=True)
        out[~valid] = np.nan
        return out

    return {
        "sigma_r": masked(sigma_r),
        "sigma_t": masked(sigma_t),
        "ur_cm": masked(100.0 * ur),
    }


def finite_values(arrays: list[np.ndarray]) -> np.ndarray:
    vals = []
    for arr in arrays:
        a = np.asarray(arr)
        vals.append(a[np.isfinite(a)])
    if not vals:
        return np.array([0.0])
    out = np.concatenate(vals)
    return out if out.size else np.array([0.0])


def field_limits(pinn: np.ndarray, exact: np.ndarray) -> tuple[float, float]:
    vals = finite_values([pinn, exact])
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if np.isclose(vmin, vmax):
        eps = max(abs(vmin), 1.0) * 1e-6
        vmin -= eps
        vmax += eps
    return vmin, vmax


def signed_error_limits(error: np.ndarray, percentile: float) -> tuple[float, float]:
    vals = np.abs(error[np.isfinite(error)])
    if vals.size == 0:
        vmax = 1.0
    elif percentile >= 100.0:
        vmax = float(np.max(vals))
    else:
        vmax = float(np.quantile(vals, percentile / 100.0))
    vmax = max(vmax, 1e-12)
    return -vmax, vmax


def plot_panel(
    ax: plt.Axes,
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    a: float,
    r_outer: float,
    title: str,
    cmap,
    vmin: float,
    vmax: float,
    levels: int,
    extend: str,
):
    data = np.ma.masked_invalid(zz)
    level_values = np.linspace(vmin, vmax, levels)
    cf = ax.contourf(xx, yy, data, levels=level_values, cmap=cmap, extend=extend, antialiased=True)
    draw_annulus(ax, a, r_outer)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, r_outer)
    ax.set_ylim(0.0, r_outer)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=9)
    return cf


def make_figure(
    pinn: dict[str, np.ndarray],
    exact: dict[str, np.ndarray],
    out_path: Path,
    a: float,
    r_outer: float,
    levels: int,
    error_percentile: float,
    dpi: int,
) -> dict[str, dict[str, float]]:
    xx = pinn["x"]
    yy = pinn["y"]
    rows = [
        ("sigma_r", r"$\sigma_r$", "MPa"),
        ("sigma_t", r"$\sigma_\theta$", "MPa"),
        ("ur_cm", r"$u_r$", "cm"),
    ]
    cmap = plt.get_cmap("coolwarm")

    fig, axes = plt.subplots(3, 3, figsize=(14.5, 13.0), constrained_layout=True)
    stats: dict[str, dict[str, float]] = {}

    for i, (key, label, _unit) in enumerate(rows):
        z_pinn = pinn[key]
        z_exact = exact[key]
        z_error = z_pinn - z_exact
        vals = z_error[np.isfinite(z_error)]
        stats[key] = {
            "mean": float(np.mean(vals)),
            "mean_abs": float(np.mean(np.abs(vals))),
            "rmse": float(np.sqrt(np.mean(vals * vals))),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "max_abs": float(np.max(np.abs(vals))),
            "q99_abs": float(np.quantile(np.abs(vals), 0.99)),
        }

        vmin, vmax = field_limits(z_pinn, z_exact)
        cf0 = plot_panel(
            axes[i, 0], xx, yy, z_pinn, a, r_outer,
            f"{label} - PINN", cmap, vmin, vmax, levels, "both",
        )
        cf1 = plot_panel(
            axes[i, 1], xx, yy, z_exact, a, r_outer,
            f"{label} - Analytical", cmap, vmin, vmax, levels, "both",
        )
        emin, emax = signed_error_limits(z_error, error_percentile)
        cf2 = plot_panel(
            axes[i, 2], xx, yy, z_error, a, r_outer,
            f"{label} - Error", cmap, emin, emax, levels, "both",
        )
        for ax, cf in [(axes[i, 0], cf0), (axes[i, 1], cf1), (axes[i, 2], cf2)]:
            cbar = fig.colorbar(cf, ax=ax, shrink=0.92, pad=0.02)
            cbar.ax.tick_params(labelsize=8)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return stats


def print_stats(stats: dict[str, dict[str, float]]) -> None:
    name_map = {
        "sigma_r": "sigma_r (MPa)",
        "sigma_t": "sigma_theta (MPa)",
        "ur_cm": "u_r (cm)",
    }
    print("\nDeep elastic PINN - analytical signed error statistics")
    print("-" * 108)
    print(
        f"{'field':<18} | {'mean':>12} | {'mean|err|':>12} | {'RMSE':>12} | "
        f"{'min':>12} | {'max':>12} | {'max|err|':>12} | {'q99|err|':>12}"
    )
    print("-" * 108)
    for key in ("sigma_r", "sigma_t", "ur_cm"):
        s = stats[key]
        print(
            f"{name_map[key]:<18} | {s['mean']:12.6g} | {s['mean_abs']:12.6g} | {s['rmse']:12.6g} | "
            f"{s['min']:12.6g} | {s['max']:12.6g} | {s['max_abs']:12.6g} | {s['q99_abs']:12.6g}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot APINN deep elastic PINN/exact/error 3x3 figure.")
    parser.add_argument("--weights", type=str, default=str(THIS_DIR / "checkpoints" / "apinn_deep_elastic_paper_exact.pt"))
    parser.add_argument("--out", type=str, default=str(THIS_DIR / "figures" / "comparison" / "deep_elastic_pinn_analytical_error_fig10_style.png"))
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--R", type=float, default=3.0)
    parser.add_argument("--p0", type=float, default=10.0)
    parser.add_argument("--P", type=float, default=8.0)
    parser.add_argument("--E", type=float, default=10.0)
    parser.add_argument("--nu", type=float, default=0.2)
    parser.add_argument("--nxy", type=int, default=401)
    parser.add_argument("--levels", type=int, default=61)
    parser.add_argument("--error-percentile", type=float, default=100.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--stress-convention", type=str, default="auto", choices=["auto", "compression_positive", "tension_positive"])
    parser.add_argument("--anchor-ur", type=str, default="outer_mean", choices=["outer_mean", "none"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(weights, device)
    pinn = eval_pinn(
        model=model,
        device=device,
        a=args.a,
        r_outer=args.R,
        nxy=args.nxy,
        stress_convention=args.stress_convention,
        anchor_ur=args.anchor_ur,
    )
    exact = analytical_elastic(
        xx=pinn["x"],
        yy=pinn["y"],
        a=args.a,
        r_outer=args.R,
        p0=args.p0,
        p_inner=args.P,
        young=args.E,
        nu=args.nu,
    )
    stats = make_figure(
        pinn=pinn,
        exact=exact,
        out_path=out_path,
        a=args.a,
        r_outer=args.R,
        levels=args.levels,
        error_percentile=args.error_percentile,
        dpi=args.dpi,
    )
    print_stats(stats)
    print("\nPlot completed")
    print(f"  weights      = {weights}")
    print(f"  output       = {out_path}")
    print(f"  device       = {device}")
    print(f"  stress_sign  = {pinn['stress_sign']:+.0f}")
    print(f"  error scale  = q{args.error_percentile:g}")


if __name__ == "__main__":
    main()
