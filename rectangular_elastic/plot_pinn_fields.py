# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Stage A 正方形/矩形巷道 PINN 出图脚本（Fig.11 风格，2x3 布局）

功能:
1. 读取训练得到的 .pth
2. 在全域 [-Lx, Lx] x [-Ly, Ly] 上做 1/4 结果镜像重建
3. 输出 6 个场量: ux, uz, sxx, szz, syy, sxz

兼容:
- 混合输出网络（u + sigma）
- 仅位移输出网络（应力由位移梯度反推）
- v21 的硬位移边界推理
- v21 的应力增量参数化 sigma = p0 + scale * head
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from plot_flac_fields import build_plot_limits, load_flac_fields

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass


# =========================
# 基础参数（与训练脚本一致）
# =========================
DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TUNNEL_W = 2.0
TUNNEL_H = 2.0
HX = 0.5 * TUNNEL_W
HY = 0.5 * TUNNEL_H

FAR_FIELD_FACTOR = 5.0
Lx = FAR_FIELD_FACTOR * HX
Ly = FAR_FIELD_FACTOR * HY

p0 = 10.0
P_STAGE_A = 8.0
E = 10.0
nu = 0.2
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))

DELTA_SIGMA_REF_DEFAULT = max(abs(p0 - P_STAGE_A), 1.0)
TAU_REF_DEFAULT = DELTA_SIGMA_REF_DEFAULT
FLAC_GRID_H = 0.2  # 与 FLAC3D 区域网格一致的单元边长（m）

CORNER_RBF_SIGMA = 1.35 * FLAC_GRID_H
DISP_SCALE = 1.0
DISP_UNIT = "m"
STRESS_UNIT = "MPa"


# =========================
# 网络与特征
# =========================
def scale_xy(xy: torch.Tensor, in_dim: int = 7) -> torch.Tensor:
    """
    与训练脚本一致的输入特征映射。
    - in_dim=7: 基础特征
    - in_dim>=10: 追加洞壁与角点距离特征
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    x_s = 2.0 * x / Lx - 1.0
    y_s = 2.0 * y / Ly - 1.0
    dx_h = 2.0 * (x - HX) / max(Lx - HX, 1e-12) - 1.0
    dy_h = 2.0 * (y - HY) / max(Ly - HY, 1e-12) - 1.0

    x2 = x_s ** 2
    y2 = y_s ** 2
    xy_cross = x_s * y_s
    base7 = torch.cat([x_s, y_s, dx_h, dy_h, x2, y2, xy_cross], dim=1)
    if in_dim <= 7:
        return base7

    d_fx = torch.sqrt((x - HX) ** 2 + 1e-12)
    d_fy = torch.sqrt((y - HY) ** 2 + 1e-12)
    d_corner = torch.sqrt((x - HX) ** 2 + (y - HY) ** 2 + 1e-12)
    s_h = min(HX, HY)
    phi_fx = HX / (HX + d_fx)
    phi_fy = HY / (HY + d_fy)
    phi_corner = s_h / (s_h + d_corner)
    base10 = torch.cat([base7, phi_fx, phi_fy, phi_corner], dim=1)
    if in_dim <= 10:
        return base10

    sig2 = max(CORNER_RBF_SIGMA * CORNER_RBF_SIGMA, 1e-12)
    h2 = 0.5 * FLAC_GRID_H
    rbf_top_peak = torch.exp(-(((x - (HX - h2)) ** 2 + (y - (HY + h2)) ** 2) / (2.0 * sig2)))
    rbf_side_peak = torch.exp(-(((x - (HX + h2)) ** 2 + (y - (HY - h2)) ** 2) / (2.0 * sig2)))
    return torch.cat([base10, rbf_top_peak, rbf_side_peak], dim=1)


class MLP(nn.Module):
    """与训练脚本一致的 MLP 结构。"""

    def __init__(self, in_dim: int = 10, out_dim: int = 6, width: int = 60, depth: int = 6):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(ckpt_path: Path) -> nn.Module:
    """
    从 .pth 加载模型。
    支持纯 state_dict，或包装字典内的 state_dict/model_state_dict。
    """
    obj = torch.load(str(ckpt_path), map_location=DEVICE)
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    else:
        state_dict = obj

    linear_weights: list[tuple[int, str]] = []
    for k, v in state_dict.items():
        if k.startswith("net.") and k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2:
            try:
                idx = int(k.split(".")[1])
                linear_weights.append((idx, k))
            except (ValueError, IndexError):
                pass
    if not linear_weights:
        raise RuntimeError("无法从 checkpoint 推断网络输入输出维度。")

    # 由 checkpoint 自动推断网络结构:
    # MLP 构造为: [in->width] + (depth-1)*[width->width] + [width->out]
    # 因而线性层总数 = depth + 1
    linear_weights = sorted(linear_weights, key=lambda t: t[0])
    first_key = linear_weights[0][1]
    last_key = linear_weights[-1][1]
    n_linear = len(linear_weights)

    in_dim = int(state_dict[first_key].shape[1])
    width = int(state_dict[first_key].shape[0])
    out_dim = int(state_dict[last_key].shape[0])
    depth = max(1, n_linear - 1)

    model = MLP(in_dim=in_dim, out_dim=out_dim, width=width, depth=depth).to(DEVICE).to(DTYPE)
    model.input_dim = in_dim
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =========================
# 推理辅助
# =========================
def grad_scalar(f: torch.Tensor, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对标量场 f(x,z) 求梯度。"""
    g = torch.autograd.grad(
        f,
        xy,
        grad_outputs=torch.ones_like(f),
        create_graph=False,
        retain_graph=True,
    )[0]
    return g[:, 0:1], g[:, 1:2]


def apply_hard_u_bc(
    xy: torch.Tensor,
    ux: torch.Tensor,
    uz: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    v21 硬位移边界:
    - ux 在 x=0 与 x=Lx 强制为 0
    - uz 在 z=0 与 z=Lz 强制为 0（此脚本里 Lz=Ly）
    """
    if not enabled:
        return ux, uz
    x = xy[:, 0:1]
    z = xy[:, 1:2]
    w_ux = 4.0 * x * (Lx - x) / max(Lx * Lx, 1e-12)
    w_uz = 4.0 * z * (Ly - z) / max(Ly * Ly, 1e-12)
    return w_ux * ux, w_uz * uz


def infer_stress_param_mode(model: nn.Module, probe_n: int = 256, threshold: float = 4.0) -> str:
    """
    自动判断混合输出应力解码模式:
    - absolute: 网络直接输出应力（MPa）
    - delta: 需要用 p0 + scale*head 解码
    """
    in_dim = getattr(model, "input_dim", 7)
    x = torch.rand((probe_n, 1), dtype=DTYPE, device=DEVICE) * Lx
    y = torch.rand((probe_n, 1), dtype=DTYPE, device=DEVICE) * Ly
    xy = torch.cat([x, y], dim=1)
    with torch.no_grad():
        out = model(scale_xy(xy, in_dim=in_dim))
    if out.shape[1] < 5:
        return "absolute"
    med_abs = float(torch.median(torch.abs(out[:, 2:5])).item())
    return "delta" if med_abs < threshold else "absolute"


def eval_quarter_fields(
    model: nn.Module,
    x_abs: np.ndarray,
    z_abs: np.ndarray,
    hard_u_bc: bool = False,
    stress_param: str = "auto",
    p0_ref: float = p0,
    delta_sigma_ref: float = DELTA_SIGMA_REF_DEFAULT,
    tau_ref: float = TAU_REF_DEFAULT,
) -> dict[str, np.ndarray]:
    """
    在 1/4 域 (|x|, |z|) 上评估场量。
    返回字段: ux, uz, sxx, szz, syy, sxz
    单位: 位移 m，应力 MPa
    """
    xy = np.stack([x_abs, z_abs], axis=1)
    xy_t = torch.tensor(xy, dtype=DTYPE, device=DEVICE)
    in_dim = getattr(model, "input_dim", 7)

    with torch.no_grad():
        out_dim = int(model(scale_xy(xy_t[:1], in_dim=in_dim)).shape[1])

    # 混合输出（5或6维）
    if out_dim >= 5:
        mode = stress_param
        if mode == "auto":
            mode = infer_stress_param_mode(model)
        if mode not in ("absolute", "delta"):
            raise ValueError("stress_param 仅支持 absolute/delta/auto")

        with torch.no_grad():
            out_t = model(scale_xy(xy_t, in_dim=in_dim))
            ux_t = out_t[:, 0:1]
            uz_t = out_t[:, 1:2]
            ux_t, uz_t = apply_hard_u_bc(xy_t, ux_t, uz_t, enabled=hard_u_bc)

        out = out_t.cpu().numpy()
        out[:, 0:1] = ux_t.cpu().numpy()
        out[:, 1:2] = uz_t.cpu().numpy()

        if mode == "delta":
            sxx = p0_ref + delta_sigma_ref * out[:, 2]
            if out_dim >= 6:
                # v21: [ux, uz, sxx_head, szz_head, syy_head, sxz_head]
                szz = p0_ref + delta_sigma_ref * out[:, 3]
                syy = p0_ref + delta_sigma_ref * out[:, 4]
                sxz = tau_ref * out[:, 5]
            else:
                # 兼容 5 维老模型: [ux, uz, sxx_head, szz_head, sxz_head]
                szz = p0_ref + delta_sigma_ref * out[:, 3]
                sxz = tau_ref * out[:, 4]
                syy = p0_ref + nu * ((sxx - p0_ref) + (szz - p0_ref))
        else:
            sxx = out[:, 2]
            if out_dim >= 6:
                szz = out[:, 3]
                syy = out[:, 4]
                sxz = out[:, 5]
            else:
                szz = out[:, 3]
                sxz = out[:, 4]
                syy = p0_ref + nu * ((sxx - p0_ref) + (szz - p0_ref))

        return {
            "ux": out[:, 0],
            "uz": out[:, 1],
            "sxx": sxx,
            "szz": szz,
            "syy": syy,
            "sxz": sxz,
        }

    # 仅位移输出（2维）
    if out_dim != 2:
        raise RuntimeError(f"不支持的输出维度 out_dim={out_dim}")

    n = xy_t.shape[0]
    chunk = 4096
    ux_all: list[np.ndarray] = []
    uz_all: list[np.ndarray] = []
    sxx_all: list[np.ndarray] = []
    szz_all: list[np.ndarray] = []
    syy_all: list[np.ndarray] = []
    sxz_all: list[np.ndarray] = []

    for i in range(0, n, chunk):
        xyc = xy_t[i:i + chunk].detach().clone().requires_grad_(True)
        out = model(scale_xy(xyc, in_dim=in_dim))
        ux = out[:, 0:1]
        uz = out[:, 1:2]
        ux, uz = apply_hard_u_bc(xyc, ux, uz, enabled=hard_u_bc)

        dux_dx, dux_dz = grad_scalar(ux, xyc)
        duz_dx, duz_dz = grad_scalar(uz, xyc)

        # 压缩为正约定
        eps_xx = -dux_dx
        eps_zz = -duz_dz
        eps_yy = torch.zeros_like(eps_xx)
        eps_xz = -0.5 * (dux_dz + duz_dx)
        tr = eps_xx + eps_zz + eps_yy

        sxx = p0 + lam * tr + 2.0 * mu * eps_xx
        szz = p0 + lam * tr + 2.0 * mu * eps_zz
        syy = p0 + lam * tr + 2.0 * mu * eps_yy
        sxz = 2.0 * mu * eps_xz

        ux_all.append(ux.detach().cpu().numpy().reshape(-1))
        uz_all.append(uz.detach().cpu().numpy().reshape(-1))
        sxx_all.append(sxx.detach().cpu().numpy().reshape(-1))
        szz_all.append(szz.detach().cpu().numpy().reshape(-1))
        syy_all.append(syy.detach().cpu().numpy().reshape(-1))
        sxz_all.append(sxz.detach().cpu().numpy().reshape(-1))

    return {
        "ux": np.concatenate(ux_all, axis=0),
        "uz": np.concatenate(uz_all, axis=0),
        "sxx": np.concatenate(sxx_all, axis=0),
        "szz": np.concatenate(szz_all, axis=0),
        "syy": np.concatenate(syy_all, axis=0),
        "sxz": np.concatenate(sxz_all, axis=0),
    }


# =========================
# 1/4 -> 全域镜像
# =========================
def mirror_to_full_domain(
    model: nn.Module,
    nxy: int = 401,
    grid_mode: str = "flac",
    grid_h: float = FLAC_GRID_H,
    hard_u_bc: bool = False,
    stress_param: str = "auto",
    p0_ref: float = p0,
    delta_sigma_ref: float = DELTA_SIGMA_REF_DEFAULT,
    tau_ref: float = TAU_REF_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """
    将 1/4 域镜像到全域 [-Lx, Lx] x [-Lz, Lz]（这里 Lz=Ly）。
    """
    if grid_mode == "flac":
        # 按 FLAC3D 单元中心点采样（若区间为 [-5,5]、h=0.2，则为 -4.9, -4.7, ..., 4.9）
        h = float(grid_h)
        if h <= 0.0:
            raise ValueError("grid_h 必须为正数。")
        xs = np.arange(-Lx + 0.5 * h, Lx, h)
        zs = np.arange(-Ly + 0.5 * h, Ly, h)
    elif grid_mode == "dense":
        xs = np.linspace(-Lx, Lx, nxy)
        zs = np.linspace(-Ly, Ly, nxy)
    else:
        raise ValueError("grid_mode 仅支持 'flac' 或 'dense'。")

    XX, ZZ = np.meshgrid(xs, zs)

    hole_mask = (np.abs(XX) < HX) & (np.abs(ZZ) < HY)

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
        "ux": (sx * q["ux"]).reshape(XX.shape),
        "uz": (sz * q["uz"]).reshape(XX.shape),
        "sxx": q["sxx"].reshape(XX.shape),
        "szz": q["szz"].reshape(XX.shape),
        "syy": q["syy"].reshape(XX.shape),
        "sxz": (sx * sz * q["sxz"]).reshape(XX.shape),
    }

    # 位移改为 cm 显示
    fields["ux"] = fields["ux"] * DISP_SCALE
    fields["uz"] = fields["uz"] * DISP_SCALE
    return XX, ZZ, fields, hole_mask


# =========================
# 绘图
# =========================
def symmetric_limits(a: np.ndarray) -> tuple[float, float]:
    """对位移/剪应力设置对称色标（使用全量绝对最大值，不做分位截断）。"""
    vals = np.abs(a[np.isfinite(a)])
    vmax = np.max(vals) if vals.size > 0 else 1.0
    vmax = max(float(vmax), 1e-12)
    return -vmax, vmax


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
    """按 2x3 布局绘制 6 个场量图。"""
    cmap = plt.get_cmap("coolwarm")
    names = [
        ("ux", r"$u_x$" + f"-PINN ({DISP_UNIT})"),
        ("uz", r"$u_z$" + f"-PINN ({DISP_UNIT})"),
        ("sxx", r"$\sigma_{xx}$" + f"-PINN ({STRESS_UNIT})"),
        ("szz", r"$\sigma_{zz}$" + f"-PINN ({STRESS_UNIT})"),
        ("syy", r"$\sigma_{yy}$" + f"-PINN ({STRESS_UNIT})"),
        ("sxz", r"$\sigma_{xz}$" + f"-PINN ({STRESS_UNIT})"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.6), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.965, bottom=0.065, top=0.965, wspace=0.36, hspace=0.32)
    axes = axes.ravel()

    for ax, (key, title) in zip(axes, names):
        Z = np.ma.array(fields[key], mask=hole_mask)

        if plot_limits is not None and key in plot_limits:
            vmin, vmax = plot_limits[key]
        else:
            if key in ("ux", "uz", "sxz"):
                vmin, vmax = symmetric_limits(Z.compressed())
            else:
                vals = Z.compressed()
                vmin = float(np.min(vals))
                vmax = float(np.max(vals))
                if np.isclose(vmin, vmax):
                    vmax = vmin + 1e-6

        levels_arr = np.linspace(vmin, vmax, levels)
        cf = ax.contourf(XX, ZZ, Z, levels=levels_arr, cmap=cmap, extend="both")

        # 洞室留白
        hole_patch = Rectangle(
            (-HX, -HY),
            2.0 * HX,
            2.0 * HY,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=10,
        )
        ax.add_patch(hole_patch)

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


def print_plot_grid_range_report(fields: dict[str, np.ndarray], hole_mask: np.ndarray) -> None:
    """
    打印“当前出图网格”上的范围统计（与训练脚本 FLAC 对照口径一致）：
    - 位移按绝对值对称范围（单位 m）
    - 应力按 FLAC 符号（压缩为负）
    """
    valid = ~hole_mask
    ux_cm = fields["ux"][valid]
    uz_cm = fields["uz"][valid]
    sxx = fields["sxx"][valid]
    szz = fields["szz"][valid]
    syy = fields["syy"][valid]
    sxz = fields["sxz"][valid]

    ux_absmax_m = float(np.max(np.abs(ux_cm))) / DISP_SCALE
    uz_absmax_m = float(np.max(np.abs(uz_cm))) / DISP_SCALE
    sxz_absmax = float(np.max(np.abs(sxz)))

    pred_flac = {
        "u_x": (-ux_absmax_m, ux_absmax_m),
        "u_z": (-uz_absmax_m, uz_absmax_m),
        "xx": (-float(np.max(sxx)), -float(np.min(sxx))),
        "zz": (-float(np.max(szz)), -float(np.min(szz))),
        "yy": (-float(np.max(syy)), -float(np.min(syy))),
        "xz": (-sxz_absmax, sxz_absmax),
    }
    name_map = {
        "u_x": "XDisplacement (m)",
        "u_z": "ZDisplacement (m)",
        "xx": "XX Stress (MPa, FLAC sign)",
        "zz": "ZZ Stress (MPa, FLAC sign)",
        "yy": "YY Stress (MPa, FLAC sign)",
        "xz": "XZ Stress (MPa, FLAC sign)",
    }
    print("\n出图网格范围对照（与训练日志同口径）")
    for k in ("u_x", "u_z", "xx", "zz", "yy", "xz"):
        pmin, pmax = pred_flac[k]
        print(f"  {name_map[k]:<34} | PINN_plot_grid:[{pmin:+.6f}, {pmax:+.6f}]")


# =========================
# 命令行
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制矩形巷道 PINN 的 6 场量图（Fig.11 风格）")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(Path(__file__).resolve().parent / "checkpoints" / "pinn_model_stageA_rectangular_gpt_v42.pth"),
        help="模型权重 .pth 路径",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "figures" / "pinn_fields.png"),
        help="输出图片路径",
    )
    parser.add_argument(
        "--grid-mode",
        type=str,
        default="flac",
        choices=["flac", "dense"],
        help="出图网格模式：flac=按FLAC网格中心(默认)，dense=高密度均匀网格",
    )
    parser.add_argument(
        "--grid-h",
        type=float,
        default=FLAC_GRID_H,
        help="当 grid-mode=flac 时的网格步长（m）",
    )
    parser.add_argument("--nxy", type=int, default=401, help="当 grid-mode=dense 时每方向点数")
    parser.add_argument("--levels", type=int, default=61, help="等值线层数")
    parser.add_argument("--dpi", type=int, default=300, help="输出 DPI")

    parser.add_argument(
        "--hard-u-bc",
        action="store_true",
        help="推理时启用硬位移边界（v21 默认自动启用）",
    )
    parser.add_argument(
        "--stress-param",
        type=str,
        default="auto",
        choices=["auto", "absolute", "delta"],
        help="混合输出应力解码模式",
    )
    parser.add_argument("--p0-ref", type=float, default=p0, help="增量解码参考 p0 (MPa)")
    parser.add_argument(
        "--delta-sigma-ref",
        type=float,
        default=DELTA_SIGMA_REF_DEFAULT,
        help="增量正应力尺度 DELTA_SIGMA_REF (MPa)",
    )
    parser.add_argument(
        "--tau-ref",
        type=float,
        default=TAU_REF_DEFAULT,
        help="剪应力尺度 TAU_REF (MPa)",
    )
    parser.add_argument(
        "--ref-csv",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "flac_stageA_rect_fields.csv"),
        help="FLAC3D CSV used only to fix common colorbar limits; pass empty string to disable",
    )
    parser.add_argument("--ref-x-center", type=float, default=5.0)
    parser.add_argument("--ref-z-center", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {ckpt_path}\n请通过 --ckpt 指定正确的 .pth 路径。"
        )

    model = load_model(ckpt_path)

    # v21+ 默认启用 HARD_U_BC（训练脚本该系列均开启）
    stem = ckpt_path.stem.lower()
    m = re.search(r"_v(\d+)", stem)
    ver = int(m.group(1)) if m else -1
    auto_hard = (ver >= 21)
    use_hard_u_bc = bool(args.hard_u_bc or auto_hard)

    stress_param = args.stress_param
    if stress_param == "auto":
        # v21+ 训练脚本采用增量应力参数化，推理默认强制按 delta 解码，避免自动判别误差
        if ver >= 21:
            stress_param = "delta"
        else:
            stress_param = infer_stress_param_mode(model)

    XX, ZZ, fields, hole_mask = mirror_to_full_domain(
        model=model,
        nxy=args.nxy,
        grid_mode=args.grid_mode,
        grid_h=args.grid_h,
        hard_u_bc=use_hard_u_bc,
        stress_param=stress_param,
        p0_ref=args.p0_ref,
        delta_sigma_ref=args.delta_sigma_ref,
        tau_ref=args.tau_ref,
    )
    plot_limits = None
    if args.ref_csv.strip():
        ref_csv = Path(args.ref_csv).resolve()
        if not ref_csv.exists():
            raise FileNotFoundError(f"Reference CSV not found: {ref_csv}")
        _, _, ref_fields, ref_hole_mask = load_flac_fields(ref_csv, x_center=args.ref_x_center, z_center=args.ref_z_center)
        plot_limits = build_plot_limits(ref_fields, ref_hole_mask)

    plot_six_fields(
        XX=XX,
        ZZ=ZZ,
        fields=fields,
        hole_mask=hole_mask,
        out_path=out_path,
        plot_limits=plot_limits,
        levels=args.levels,
        dpi=args.dpi,
    )
    print_plot_grid_range_report(fields=fields, hole_mask=hole_mask)

    print("出图完成")
    print(f"  checkpoint = {ckpt_path}")
    print(f"  hard_u_bc  = {use_hard_u_bc}")
    print(f"  stress_dec = {stress_param}")
    print(f"  grid_mode  = {args.grid_mode}")
    if args.grid_mode == "flac":
        print(f"  grid_h     = {args.grid_h} m（FLAC单元中心网格）")
    else:
        print(f"  nxy        = {args.nxy}")
    print(f"  output     = {out_path}")
    if plot_limits is not None:
        print(f"  colorbar   = FLAC reference CSV: {ref_csv}")
        for key in ("ux", "uz", "sxx", "szz", "syy", "sxz"):
            vmin, vmax = plot_limits[key]
            print(f"    {key:<4}: [{vmin:+.6f}, {vmax:+.6f}]")
    print("  FLAC映射   = ux↔XDisplacement, uz↔ZDisplacement, sxx↔XX, szz↔ZZ, syy↔YY, sxz↔XZ")


if __name__ == "__main__":
    main()
