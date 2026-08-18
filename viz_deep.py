# viz_deep.py — Reproduce deep-buried APINN fields in Fig. 11 style
# Units per paper: stress in MPa, displacement in cm.  (深埋弹塑工况，图11风格)

import argparse, glob, math, os, sys, pathlib
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle'] = 'solid'  # 负等值线实线（论文风格）
# 允许从脚本目录导入 apinn 包
THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from apinn.models import MLP_APINN
# 下面两个不用于出图，但保留以便你扩展塑性区可视化
from apinn.physics import stress_tensor_from_components, mohr_coulomb_F

torch.set_default_dtype(torch.float64)

# -------------------------- 工具 --------------------------
def cart_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y, x)
    return r, th

def stress_cart_to_polar(s_xx, s_yy, s_xy, th):
    """(sigma_rr, sigma_tt, sigma_rt)，以数学极角 θ 从 +x 逆时针到 +y。"""
    c, s = np.cos(th), np.sin(th)
    c2, s2, cs = c*c, s*s, c*s
    s_rr = s_xx*c2 + s_yy*s2 + 2.0*s_xy*cs
    s_tt = s_xx*s2 + s_yy*c2 - 2.0*s_xy*cs
    s_rt = (s_yy - s_xx)*cs + s_xy*(c2 - s2)
    return s_rr, s_tt, s_rt

def displacement_to_ur(ux, uy, th):
    return ux*np.cos(th) + uy*np.sin(th)

def auto_pick_weights(search_dir: str) -> str:
    pats = [os.path.join(search_dir, "apinn_deep_*.pt"),
            os.path.join(search_dir, "*.pt")]
    cand = []
    for pat in pats:
        cand += glob.glob(pat)
    cand = [c for c in cand if os.path.isfile(c)]
    if not cand:
        raise FileNotFoundError("No *.pt found. Pass --weights <path-to-pt>.")
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]

def draw_annulus_outline(ax, a, R):
    th = np.linspace(0.0, 0.5*np.pi, 361)
    ax.plot(a*np.cos(th), a*np.sin(th), linewidth=1.2)
    ax.plot(R*np.cos(th), R*np.sin(th), linewidth=1.2)

def robust_minmax(arr, q=(0.01, 0.99)):
    a = np.asarray(arr)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(a, q[0]))
    hi = float(np.quantile(a, q[1]))
    if lo == hi:
        hi = lo + 1.0
    return lo, hi

def ensure_range(lo, hi):
    """确保 lo ≤ hi 且区间非退化。"""
    if lo is None or hi is None:
        return lo, hi
    if hi < lo:
        lo, hi = hi, lo
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if hi == lo:
        eps = 1e-6
        lo, hi = lo - eps, hi + eps
    return lo, hi

def nice_levels(vmin, vmax, nlev=21, step=None):
    """生成“好看”的等值层，保证层数≥2，不会出现负层数。"""
    vmin, vmax = ensure_range(vmin, vmax)
    span = vmax - vmin
    if not np.isfinite(span) or span <= 0:
        vmin, vmax = vmin - 0.5, vmax + 0.5
        span = vmax - vmin

    if step is None or not np.isfinite(step) or step <= 0:
        raw = span / max(nlev - 1, 1)
        base = 10 ** math.floor(math.log10(max(raw, 1e-12)))
        step = None
        for m in [1, 2, 2.5, 5, 10]:
            if raw <= m * base:
                step = m * base; break
        if step is None:
            step = 10 * base
    step = abs(step)

    count = max(int(math.floor(span / step)) + 1, nlev, 2)
    levels = np.linspace(vmin, vmax, count)
    return levels, vmin, vmax

def mask_or_fill(zz, fill=False, inside_mask=None):
    """把 NaN 屏蔽（mask）以免出现白斜杠；若 fill=True 且 inside_mask 给出，只对域内极少量 NaN 用中位数填充。"""
    if not fill:
        return np.ma.masked_invalid(zz)
    z = zz.copy()
    if inside_mask is None:
        inside_mask = np.isfinite(zz)
    bad = np.isnan(z) & inside_mask
    if np.any(bad):
        med = np.nanmedian(z[inside_mask])
        z[bad] = med
    return z

# -------------------------- 评估 --------------------------
def eval_model_on_grid(model, device, a=1.0, R=3.0, n=401):
    x = np.linspace(0.0, R, n)
    y = np.linspace(0.0, R, n)
    xx, yy = np.meshgrid(x, y)   # (ny, nx)
    rr = np.sqrt(xx**2 + yy**2)
    mask = (rr >= a) & (rr <= R)

    XY = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    X_t = torch.from_numpy(XY).to(device=device, dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        out = model(X_t)  # (..., 6):(ux,uy,s_xx,s_yy,s_zz,s_xy)
    out_np = out.detach().cpu().numpy().reshape(y.size, x.size, -1)

    ux, uy = out_np[..., 0], out_np[..., 1]
    s_xx, s_yy, s_zz, s_xy = out_np[..., 2], out_np[..., 3], out_np[..., 4], out_np[..., 5]

    r, th = cart_to_polar(xx, yy)
    ur = displacement_to_ur(ux, uy, th)
    s_rr, s_tt, s_rt = stress_cart_to_polar(s_xx, s_yy, s_xy, th)

    # ========== 新增：在这里打印一次原始网络输出的量级 ==========
    try:
        print("\n=== Debug: raw network outputs on grid (SI units) ===")
        print("ux (m)   min/max:", float(np.nanmin(ux)), float(np.nanmax(ux)))
        print("uy (m)   min/max:", float(np.nanmin(uy)), float(np.nanmax(uy)))
        print("s_xx (MPa) min/max:", float(np.nanmin(s_xx)), float(np.nanmax(s_xx)))
        print("s_yy (MPa) min/max:", float(np.nanmin(s_yy)), float(np.nanmax(s_yy)))
        print("s_xy (MPa) min/max:", float(np.nanmin(s_xy)), float(np.nanmax(s_xy)))
        print("s_rr (MPa) min/max:", float(np.nanmin(s_rr)), float(np.nanmax(s_rr)))
        print("s_tt (MPa) min/max:", float(np.nanmin(s_tt)), float(np.nanmax(s_tt)))
        print("u_r (m)   min/max:", float(np.nanmin(ur)), float(np.nanmax(ur)))
    except Exception as e:
        print("[Debug] raw output min/max print failed:", e)
    # ========== 调试打印结束 ==========

    def m(a_):
        out = np.array(a_, dtype=np.float64, copy=True)
        out[~mask] = np.nan
        return out

    fields = dict(
        x=xx, y=yy, r=r, th=th, mask=mask,
        ux=m(ux), uy=m(uy), ur=m(ur),
        s_xx=m(s_xx), s_yy=m(s_yy), s_zz=m(s_zz), s_xy=m(s_xy),
        s_rr=m(s_rr), s_tt=m(s_tt), s_rt=m(s_rt),
    )
    return fields

# -------------------------- 关键：号制统一 + 位移锚定 --------------------------
def apply_stress_convention(fields, a, R, convention="auto"):
    """
    把应力统一为“压缩为正”：
      - convention='auto'：看外圆 r≈R 的 σ_rr 平均，如果为负则整体乘 -1
      - convention='compression_pos'：不变
      - convention='tension_pos'：整体乘 -1
    """
    sgn = 1.0
    if convention == "tension_pos":
        sgn = -1.0
    elif convention == "auto":
        # 取一圈邻近外圆的壳层
        delta = max((R - a) / 200.0, 1e-4)
        m = np.isfinite(fields["s_rr"]) & (np.abs(fields["r"] - R) < delta)
        if np.any(m):
            mean_outer = np.nanmean(fields["s_rr"][m])
            if mean_outer < 0:
                sgn = -1.0
    # 整体翻转应力
    for k in ["s_xx","s_yy","s_zz","s_xy","s_rr","s_tt","s_rt"]:
        fields[k] = sgn * fields[k]
    return fields, sgn

def anchor_ur(fields, a, R, mode="outer_mean"):
    """
    给位移一个“零基准”，避免刚体漂移：
      - outer_mean：以 r≈R 外圆平均 u_r 为 0
      - outer_mid ：以 (x=R/√2, y=R/√2) 的 u_r 为 0
      - none      ：不锚定
    """
    ur = fields["ur"].copy()
    if mode == "none":
        return ur
    if mode == "outer_mid":
        xm, ym = R/np.sqrt(2), R/np.sqrt(2)
        # 找最近网格点
        i = np.argmin((fields["x"]-xm)**2 + (fields["y"]-ym)**2)
        base = ur.reshape(-1)[i]
    else:  # outer_mean
        delta = max((R - a) / 200.0, 1e-4)
        m = np.isfinite(ur) & (np.abs(fields["r"] - R) < delta)
        base = np.nanmean(ur[m]) if np.any(m) else 0.0
    return ur - base

# -------------------------- 出图（Fig. 11 风格） --------------------------
def contour_fig11(xx, yy, zz, out_path, title, cbar_label, a, R, nlev=21, vmin=None, vmax=None, fill_nan_inside=False, inside_mask=None):
    plt.figure(figsize=(6.0, 5.3))
    if vmin is None or vmax is None:
        lo, hi = robust_minmax(zz, (0.01, 0.99))
    else:
        lo, hi = vmin, vmax
    levels, lo, hi = nice_levels(lo, hi, nlev=nlev)

    Z = mask_or_fill(zz, fill=fill_nan_inside, inside_mask=inside_mask)
    cs = plt.contourf(xx, yy, Z, levels=levels, cmap='jet')
    cbar = plt.colorbar(cs); cbar.set_label(cbar_label)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, R); ax.set_ylim(0, R)
    draw_annulus_outline(ax, a, R)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

# -------------------------- 主程序 --------------------------
def main():
    # ==================== 这里针对 Fig.10 做的配置 ====================
    # 1. 使用 train_deep.py 训练好的“弹性阶段 Fig.10”权重
    weights_path = THIS_DIR / "checkpoints" / "apinn_deep_elastic_paper_exact.pt"

    # 2. 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. 几何参数：必须和 train_deep.py 完全一致
    a = 1.0
    R = 3.0                          # ← 由原来的 50 改为 3（论文和训练脚本都用 R=3 m）

    # 4. 网格密度和输出目录
    n_grid = 401                     # 401 已经很细，601 也可以；按你需要
    outdir = THIS_DIR / "figures" / "fields"

    # ========================================================

    # 严格匹配论文 Fig.10 的色条范围（已经按解析解调过）
    vmin_rr, vmax_rr = None,None      # σ_rr  8–10 MPa
    vmin_tt, vmax_tt = None,None    # σ_θθ  10–12 MPa
    vmin_ur, vmax_ur = None,None   # u_r   -0.70–0.00 cm

    os.makedirs(outdir, exist_ok=True)
    device = torch.device(device)

    # 加载模型结构（要和 train_deep.py 里的 MLP_APINN 配置一致）
    model = MLP_APINN(hidden=40, depth=6).to(device)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd)
    print(f"[Info] Loaded weights: {weights_path}")

    # 在 [0,R]×[0,R] 网格上评估场量，内部只保留 a ≤ r ≤ R 的结果
    fields = eval_model_on_grid(model, device, a=a, R=R, n=n_grid)

    # 应力号制处理（这里假设训练时已经是“压缩为正”的号制，如需自动翻转可改成 convention='auto'）
    fields, sgn = apply_stress_convention(fields, a, R, convention="auto")
    print(f"[Info] stress sign factor sgn = {sgn}")

    # 位移锚定：外边界平均 u_r = 0
    ur_anchored = anchor_ur(fields, a, R, mode="outer_mean")

    # 单位换算
    s_rr_MPa = fields["s_rr"]  # 现在网络输出数值就视作 MPa
    s_tt_MPa = fields["s_tt"]
    ur_cm = ur_anchored * 100.0  # 位移仍然乘 100 变成 “cm 数值”
    inside_mask = np.isfinite(fields["s_rr"])

    # ========== 新增：单位转换之后再打印一次量级（MPa / cm） ==========
    try:
        print("\n=== Debug: after unit conversion & anchoring ===")
        print("sigma_rr (MPa) min/max:", float(np.nanmin(s_rr_MPa)), float(np.nanmax(s_rr_MPa)))
        print("sigma_tt (MPa) min/max:", float(np.nanmin(s_tt_MPa)), float(np.nanmax(s_tt_MPa)))
        print("u_r (cm)       min/max:", float(np.nanmin(ur_cm)),    float(np.nanmax(ur_cm)))
    except Exception as e:
        print("[Debug] unit-converted min/max print failed:", e)
    # ========== 调试打印结束 ==========

    # 三张图：σ_rr、σ_θθ、u_r
    contour_fig11(fields["x"], fields["y"], s_rr_MPa,
                  os.path.join(outdir, "Fig10_sigma_rr.png"),
                  title=r"$\sigma_{rr}$ (MPa)", cbar_label="[MPa]",
                  a=a, R=R, nlev=25, vmin=vmin_rr, vmax=vmax_rr,
                  fill_nan_inside=True, inside_mask=inside_mask)

    contour_fig11(fields["x"], fields["y"], s_tt_MPa,
                  os.path.join(outdir, "Fig10_sigma_tt.png"),
                  title=r"$\sigma_{\theta\theta}$ (MPa)", cbar_label="[MPa]",
                  a=a, R=R, nlev=25, vmin=vmin_tt, vmax=vmax_tt,
                  fill_nan_inside=True, inside_mask=inside_mask)

    contour_fig11(fields["x"], fields["y"], ur_cm,
                  os.path.join(outdir, "Fig10_ur.png"),
                  title=r"$u_{r}$ (cm)", cbar_label="[cm]",
                  a=a, R=R, nlev=25, vmin=vmin_ur, vmax=vmax_ur,
                  fill_nan_inside=True, inside_mask=inside_mask)

    print(f"\nFig.10 风格三张图已生成在文件夹: {outdir}/")
    print("   Fig10_sigma_rr.png")
    print("   Fig10_sigma_tt.png")
    print("   Fig10_ur.png")


if __name__ == "__main__":
    main()
