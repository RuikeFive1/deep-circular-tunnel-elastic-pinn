from __future__ import annotations

"""Deep square/rectangular tunnel Stage A elastic mixed-output PINN.

The network predicts ``u_x, u_z, sigma_xx, sigma_zz, sigma_yy, sigma_xz``
on a quarter domain. Training uses equilibrium, boundary-condition, and
plane-strain elastic constitutive residuals. The fixed interior sample combines
global points with near-corner enrichment while excluding only the exact sharp
corner coordinate.

This file is the publication-oriented copy of the validated local v42 model.
The numerical formulation and training hyperparameters are unchanged.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# =========================
# 运行开关：--smoke 用于快速自检
# =========================
SMOKE = ("--smoke" in sys.argv)

# =========================
# 随机种子与设备设置
# =========================
SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 一、几何参数：1/4 域 + 中心方形/矩形洞室
# =========================
TUNNEL_W = 2.0  # 巷道总宽度 m
TUNNEL_H = 2.0  # 巷道总高度 m

HX = 0.5 * TUNNEL_W  # 1/4 域中的内边界半宽
HY = 0.5 * TUNNEL_H  # 1/4 域中的内边界半高

FAR_FIELD_FACTOR = 5.0
Lx = FAR_FIELD_FACTOR * HX  # 1/4 域外边界 x 尺寸
Ly = FAR_FIELD_FACTOR * HY  # 1/4 域外边界 y 尺寸

# 与 FLAC3D 的 X-Z 平面语义对齐（数值上 Lz=Ly, HZ=HY）
HZ = HY
Lz = Ly

# =========================
# 二、物理参数：与 Stage A 纯弹性问题一致
# =========================
p0 = 10.0        # 原位应力 MPa，压缩为正
P_STAGE_A = 8.0  # Stage A 洞壁支护压力 MPa，压缩为正

E = 10.0         # 杨氏模量 MPa
nu = 0.2         # 泊松比

# 拉梅常数
lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))

# =========================
# 三、训练与采样参数：固定采样，不做自适应
# =========================
N_F_FIXED = 2500  # 固定内部残差点数（回退到 2500，优先节省训练时间）

N_B_OUT_RIGHT = 100
N_B_OUT_TOP = 100
N_B_IN_RIGHT = 100
N_B_IN_TOP = 100
N_B_SYM_X0 = 100
N_B_SYM_Y0 = 100
N_B_TOTAL = (
    N_B_OUT_RIGHT + N_B_OUT_TOP + N_B_IN_RIGHT + N_B_IN_TOP + N_B_SYM_X0 + N_B_SYM_Y0
)

LR = 1e-3
EPOCHS_ADAM = 10000 if not SMOKE else 20
PRINT_EVERY = 1000 if not SMOKE else 5

USE_LBFGS = (not SMOKE)
LBFGS_STEPS = 1000 if not SMOKE else 20
LBFGS_LR = 0.8

# 三个损失项权重：保持 v3 原值不变
W_E = 1.6  # 平衡方程（温和增强）
W_B = 1.0  # 边界条件
W_C = 1.1  # 本构一致性（轻度增强）
W_FAR = 0.02   # 弱远场应力锚定
W_SWAP = 0.25  # x<->z 交换对称约束（小幅增强）
W_SIGN = 0.02  # 洞壁入洞方向轻约束（再降，避免过约束）
W_FAR_ZZ = 0.03  # out-of-plane 锚定（温和）
W_OUTER_SXY = 0.015  # 外边界弱剪应力约束（小幅下调，避免剪应力幅值被压缩）
W_B_INNER_SIGMA = 3.0  # 内边界法向应力权重
W_B_INNER_TAU = 1.0    # 内边界剪应力权重：低于法向项，但保持总量不变
W_B_OUTER = 1.0  # 外边界段权重
W_B_SYM = 1.0    # 对称边段权重
W_IN_MEAN = 0.6  # 洞壁均值附加项：保留稳定作用，但避免与点值内壁约束重复
W_PS = 1.6       # 略加强平面应变闭合，使 sigma_yy 跟随 sigma_xx/sigma_zz 的首层峰值跨度
W_TAN_MEAN = 0.01  # 切向均值附加项：保留少量外边界稳态约束，避免外角伪模态
W_CORNER_FIX = 0.0  # 在 HARD_U_BC 下该项理论上恒为 0，保留配置名但关闭

# 本构通道加权（x-z语义）：
# W_C_SZZ -> 作用于 sigma_yy 通道（对应 FLAC:YY）
# W_C_SXY -> 作用于 sigma_xz 通道（对应 FLAC:XZ）
W_C_SZZ = 1.25
W_C_SXY = 2.05

# 对称成对采样：用成对训练点减少由随机采样引入的 x-z 非对称偏置
PAIR_SWAP_SAMPLING = True

# 参考 pinn_tunnel.py 思路：位移采用 distance-based 硬边界约束
HARD_U_BC = True  # x=0/Lx 上 ux=0，z=0/Lz 上 uz=0

# 归一化参考尺度：只修改“应力型残差”的尺度，不改位移和平衡项尺度
SIGMA_REF_TOTAL = 10.0                       # 总应力量级，仅用于保留原来的平衡项尺度
DELTA_SIGMA_REF = max(abs(p0 - P_STAGE_A), 1.0)  # Stage A 主导驱动力：10→8 MPa，所以这里是 2 MPa
TAU_REF = DELTA_SIGMA_REF                    # 剪应力也按同一驱动力尺度归一化
U_REF = 0.1                                  # 位移参考 m（保持原值不变）
EQ_REF = SIGMA_REF_TOTAL / 5.0               # 平衡残差参考 MPa/m（保持原值不变）
EPS_CORNER_POINT = 1e-6                      # 只避开精确角点单点，不删掉角点邻域
CORNER_BAND = 0.12                           # 近角点分层加密带宽（相对 HX/HY 的比例）
CORNER_RELIEF_R = 0.22 * min(HX, HY)         # 小幅收窄缓释半径，减少对角点峰值的过度压平
CORNER_RELIEF_WMIN = 0.08                    # 小幅抬高最小权重，保留奇异性但不过度削峰
INNER_CORNER_SOFT_DIST = 0.22 * min(HX, HY) # 小幅收窄洞壁角点软化区，提升 XX/ZZ/XZ 振幅
INNER_CORNER_SOFT_WMIN = 0.08                # 小幅抬高洞壁角点最小边界权重
SIGN_EXCLUDE_DIST = 0.12 * min(HX, HY)      # 位移方向约束忽略角点邻域，避免奇异区冲突
FLAC_GRID_H = 0.2                            # 当前 FLAC3D 模型单元边长
CORNER_RBF_SIGMA = 1.35 * FLAC_GRID_H        # 轻微放宽首层单元中心特征，补足 v40 中 ZZ/XZ 的少量低估

# 固定采样：关闭周期重采样，避免 Adam 阶段目标分布不断漂移
RESAMPLE_EVERY = 0

# 边界长度（1/4 域），用于“边界积分”加权，避免短边界被过度放大
LEN_OUT_RIGHT = Ly
LEN_OUT_TOP = Lx
LEN_IN_RIGHT = HY
LEN_IN_TOP = HX
LEN_SYM_X0 = Ly - HY
LEN_SYM_Y0 = Lx - HX
LEN_B_SUM = LEN_OUT_RIGHT + LEN_OUT_TOP + LEN_IN_RIGHT + LEN_IN_TOP + LEN_SYM_X0 + LEN_SYM_Y0

# FLAC 图例读数（来自你给的6张图），用于训练后直接对比
FLAC_REF_RANGES = {
    "u_x": (-3.3644e-1, 3.3626e-1),   # XDisplacement (m)
    "u_y": (-3.3615e-1, 3.3658e-1),   # ZDisplacement (m)
    "xx": (-12.017, -8.0213),         # XX Stress (MPa, FLAC压缩为负)
    "zz": (-12.017, -8.0213),         # ZZ Stress (MPa, FLAC压缩为负)
    "yy": (-10.388, -9.5669),         # YY Stress (MPa, FLAC压缩为负)
    "xz": (-1.6970, 1.6981),          # XZ Stress (MPa)
}


# =========================
# 四、辅助函数：格式化、张量转换、洞室判定
# =========================
def fmt_num(x: float, digits: int = 6) -> str:
    """把数值格式化成普通小数。"""
    return f"{float(x):.{digits}f}"


def to_torch(xy_np: np.ndarray, requires_grad: bool = True) -> torch.Tensor:
    """把 numpy 点集转成 torch 张量。"""
    return torch.tensor(xy_np, dtype=DTYPE, device=DEVICE, requires_grad=requires_grad)


def inside_hole_np(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """判断点是否落在 1/4 域空洞 [0,HX]×[0,HY] 内。"""
    return (x < HX) & (y < HY)


def is_exact_corner_point_np(x: np.ndarray, y: np.ndarray, eps: float = EPS_CORNER_POINT) -> np.ndarray:
    """判断点是否落在精确角点单点 (HX, HY) 的极小容差邻域内。"""
    return np.isclose(x, HX, atol=eps) & np.isclose(y, HY, atol=eps)


# =========================
# 五、采样函数：内部点 + 6 段边界点
# =========================
def sample_interior_rect(n: int) -> np.ndarray:
    """
    在 1/4 岩体域内采样内部残差点。
    采用“全域均匀 + 洞角附近加密”的固定采样策略。
    """
    # 降低角点过采样比例，避免奇异角点主导整体训练
    n_uniform = int(0.85 * n)
    n_corner = n - n_uniform

    pts: list[np.ndarray] = []

    # 全域均匀采样，剔除空洞内点
    batch = max(4 * n_uniform, 2000)
    while len(pts) < n_uniform:
        x = np.random.rand(batch) * Lx
        y = np.random.rand(batch) * Ly
        mask = ~inside_hole_np(x, y)
        xy = np.stack([x[mask], y[mask]], axis=1)
        pts.extend(list(xy))
    pts = pts[:n_uniform]

    # 洞角附近额外加密，帮助捕捉角点应力集中
    corner_pts: list[np.ndarray] = []
    wx = min(1.4 * HX, max(0.8, 0.40 * Lx))
    wy = min(1.4 * HY, max(0.8, 0.40 * Ly))
    x0 = max(0.0, HX - 0.20 * wx)
    x1 = min(Lx, HX + wx)
    y0 = max(0.0, HY - 0.20 * wy)
    y1 = min(Ly, HY + wy)
    batch_corner = max(4 * max(n_corner, 1), 1200)
    while len(corner_pts) < n_corner:
        x = x0 + np.random.rand(batch_corner) * (x1 - x0)
        y = y0 + np.random.rand(batch_corner) * (y1 - y0)
        mask = ~inside_hole_np(x, y)
        xy = np.stack([x[mask], y[mask]], axis=1)
        corner_pts.extend(list(xy))
    pts.extend(corner_pts[:n_corner])

    return np.asarray(pts, dtype=np.float64)


def sample_outer_right(n: int) -> np.ndarray:
    """右外边界 x=Lx。"""
    if n <= 2:
        y = np.linspace(0.0, Ly, n)
    else:
        y_rand = np.random.rand(n - 2) * Ly
        y = np.concatenate([y_rand, np.array([0.0, Ly])])
        np.random.shuffle(y)
    x = Lx * np.ones_like(y)
    return np.stack([x, y], axis=1).astype(np.float64)


def sample_outer_top(n: int) -> np.ndarray:
    """上外边界 y=Ly。"""
    if n <= 2:
        x = np.linspace(0.0, Lx, n)
    else:
        x_rand = np.random.rand(n - 2) * Lx
        x = np.concatenate([x_rand, np.array([0.0, Lx])])
        np.random.shuffle(x)
    y = Ly * np.ones_like(x)
    return np.stack([x, y], axis=1).astype(np.float64)


def sample_inner_right(n: int) -> np.ndarray:
    """右内边界 x=HX：不取精确角点单点，并对靠近角点的小段做更强分层加密。"""
    # 边界仍保留角点加密，但进一步压低“极近角点”占比
    n_uniform = int(0.70 * n)
    n_mid = int(0.25 * n)
    n_near = n - n_uniform - n_mid

    # 常规段：覆盖整个右壁，但上端用极小 eps 避开精确角点
    y_u = np.random.rand(n_uniform) * max(HY - EPS_CORNER_POINT, 1e-12)

    # 中近角点段：继续向角点方向加密
    y_m = HY * np.random.beta(a=4.0, b=1.0, size=n_mid)
    y_m = np.minimum(y_m, HY - EPS_CORNER_POINT)

    # 最靠近角点的小段：再加强一层，但仍不取精确角点单点
    y_n = HY - CORNER_BAND * HY * np.random.beta(a=3.0, b=0.7, size=n_near)
    y_n = np.minimum(y_n, HY - EPS_CORNER_POINT)

    y = np.concatenate([y_u, y_m, y_n])
    # 显式加入端点（右壁底端与近角点端），防止边界端点长期“采不到”
    if n >= 2:
        y[:2] = np.array([0.0, HY - EPS_CORNER_POINT])
    np.random.shuffle(y)
    x = HX * np.ones_like(y)
    return np.stack([x, y], axis=1).astype(np.float64)


def sample_inner_top(n: int) -> np.ndarray:
    """上内边界 y=HY：不取精确角点单点，并对靠近角点的小段做更强分层加密。"""
    # 边界仍保留角点加密，但进一步压低“极近角点”占比
    n_uniform = int(0.70 * n)
    n_mid = int(0.25 * n)
    n_near = n - n_uniform - n_mid

    # 常规段：覆盖整个顶壁，但右端用极小 eps 避开精确角点
    x_u = np.random.rand(n_uniform) * max(HX - EPS_CORNER_POINT, 1e-12)

    # 中近角点段：继续向角点方向加密
    x_m = HX * np.random.beta(a=4.0, b=1.0, size=n_mid)
    x_m = np.minimum(x_m, HX - EPS_CORNER_POINT)

    # 最靠近角点的小段：再加强一层，但仍不取精确角点单点
    x_n = HX - CORNER_BAND * HX * np.random.beta(a=3.0, b=0.7, size=n_near)
    x_n = np.minimum(x_n, HX - EPS_CORNER_POINT)

    x = np.concatenate([x_u, x_m, x_n])
    # 显式加入端点（顶壁左端与近角点端），防止边界端点长期“采不到”
    if n >= 2:
        x[:2] = np.array([0.0, HX - EPS_CORNER_POINT])
    np.random.shuffle(x)
    y = HY * np.ones_like(x)
    return np.stack([x, y], axis=1).astype(np.float64)


def sample_sym_x0(n: int) -> np.ndarray:
    """竖直对称边界 x=0，仅在岩体区域取点。"""
    if n <= 2:
        y = np.linspace(HY, Ly, n)
    else:
        y_rand = HY + np.random.rand(n - 2) * (Ly - HY)
        y = np.concatenate([y_rand, np.array([HY, Ly])])
        np.random.shuffle(y)
    x = np.zeros_like(y)
    return np.stack([x, y], axis=1).astype(np.float64)


def sample_sym_y0(n: int) -> np.ndarray:
    """水平对称边界 y=0，仅在岩体区域取点。"""
    if n <= 2:
        x = np.linspace(HX, Lx, n)
    else:
        x_rand = HX + np.random.rand(n - 2) * (Lx - HX)
        x = np.concatenate([x_rand, np.array([HX, Lx])])
        np.random.shuffle(x)
    y = np.zeros_like(x)
    return np.stack([x, y], axis=1).astype(np.float64)


# =========================
# 六、输入特征映射：把坐标映射到更适合 MLP 的尺度
# =========================
def scale_xy(xy: torch.Tensor) -> torch.Tensor:
    x = xy[:, 0:1]
    y = xy[:, 1:2]

    # 全域坐标缩放到 [-1,1]
    x_s = 2.0 * x / Lx - 1.0
    y_s = 2.0 * y / Ly - 1.0

    # 相对洞壁位置特征，帮助网络感知“离洞室有多近”
    dx_h = 2.0 * (x - HX) / max(Lx - HX, 1e-12) - 1.0
    dy_h = 2.0 * (y - HY) / max(Ly - HY, 1e-12) - 1.0

    # 简单多项式特征
    x2 = x_s ** 2
    y2 = y_s ** 2
    xy_cross = x_s * y_s
    # 矩形洞室专用几何特征（避免圆形问题的径向特征偏置）：
    # 到右洞壁、顶洞壁、内角点(HX,HY)的平滑距离映射，均为有界特征
    d_fx = torch.sqrt((x - HX) ** 2 + 1e-12)                    # 到 x=HX
    d_fy = torch.sqrt((y - HY) ** 2 + 1e-12)                    # 到 y=HY
    d_corner = torch.sqrt((x - HX) ** 2 + (y - HY) ** 2 + 1e-12)  # 到内角点
    s_h = min(HX, HY)
    phi_fx = HX / (HX + d_fx)
    phi_fy = HY / (HY + d_fy)
    phi_corner = s_h / (s_h + d_corner)
    sig2 = max(CORNER_RBF_SIGMA * CORNER_RBF_SIGMA, 1e-12)
    h2 = 0.5 * FLAC_GRID_H
    rbf_top_peak = torch.exp(-(((x - (HX - h2)) ** 2 + (y - (HY + h2)) ** 2) / (2.0 * sig2)))
    rbf_side_peak = torch.exp(-(((x - (HX + h2)) ** 2 + (y - (HY - h2)) ** 2) / (2.0 * sig2)))

    return torch.cat(
        [
            x_s, y_s, dx_h, dy_h, x2, y2, xy_cross,
            phi_fx, phi_fy, phi_corner,
            rbf_top_peak, rbf_side_peak,
        ],
        dim=1,
    )


# =========================
# 七、神经网络：混合输出 u 与 sigma
# =========================
class MLP(nn.Module):
    """标准多层感知机，输出 2 个位移分量 + 4 个应力分量。"""

    def __init__(self, in_dim: int = 12, out_dim: int = 6, width: int = 80, depth: int = 8):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        # Xavier 初始化，保持与 v3 一致
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        # 物理先验初始化：本版采用应力“增量头”参数化，头部偏置初始化为 0
        last = self.net[-1]
        if isinstance(last, nn.Linear) and last.out_features >= 6:
            with torch.no_grad():
                last.bias[2] = 0.0
                last.bias[3] = 0.0
                last.bias[4] = 0.0
                last.bias[5] = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = MLP().to(DEVICE).to(DTYPE)


def net_fields_from_model(xy: torch.Tensor, net: nn.Module):
    """将网络输出拆成 X-Z 语义：ux, uz, sxx, szz, syy, sxz。"""
    out = net(scale_xy(xy))

    # 位移硬约束（distance-based）
    u_x_raw = out[:, 0:1]
    u_z_raw = out[:, 1:2]
    if HARD_U_BC:
        x = xy[:, 0:1]
        z = xy[:, 1:2]
        w_ux = 4.0 * x * (Lx - x) / max(Lx * Lx, 1e-12)
        w_uz = 4.0 * z * (Lz - z) / max(Lz * Lz, 1e-12)
        u_x = w_ux * u_x_raw
        u_z = w_uz * u_z_raw
    else:
        u_x = u_x_raw
        u_z = u_z_raw

    # 应力增量参数化（与 FLAC X-Z 面对齐）
    sigma_xx = p0 + DELTA_SIGMA_REF * out[:, 2:3]
    sigma_zz = p0 + DELTA_SIGMA_REF * out[:, 3:4]  # 面内
    sigma_yy = p0 + DELTA_SIGMA_REF * out[:, 4:5]  # 面外
    sigma_xz = TAU_REF * out[:, 5:6]
    return u_x, u_z, sigma_xx, sigma_zz, sigma_yy, sigma_xz


def net_fields(xy: torch.Tensor):
    """使用当前全局模型取场量。"""
    return net_fields_from_model(xy, model)


# =========================
# 八、微分算子：求梯度、求应变、求平衡残差
# =========================
def grad_scalar(f: torch.Tensor, xy: torch.Tensor):
    """对标量场 f 对 (x,y) 求一阶导。"""
    ones = torch.ones_like(f)
    g = torch.autograd.grad(f, xy, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    return g[:, 0:1], g[:, 1:2]


def strain_from_u(u_x: torch.Tensor, u_z: torch.Tensor, xy: torch.Tensor):
    """
    X-Z 平面小应变（压缩为正）：
    eps_xx = -du_x/dx
    eps_zz = -du_z/dz
    eps_xz = -0.5*(du_x/dz + du_z/dx)
    eps_yy = 0（平面应变）
    """
    dux_dx, dux_dz = grad_scalar(u_x, xy)
    duz_dx, duz_dz = grad_scalar(u_z, xy)

    eps_xx = -dux_dx
    eps_zz = -duz_dz
    eps_yy = torch.zeros_like(eps_xx)
    eps_xz = -0.5 * (dux_dz + duz_dx)
    return eps_xx, eps_zz, eps_yy, eps_xz


def equilibrium_cartesian(sigma_xx: torch.Tensor, sigma_zz: torch.Tensor, sigma_xz: torch.Tensor, xy: torch.Tensor):
    """X-Z 平面二维平衡方程残差。"""
    dsxx_dx, _ = grad_scalar(sigma_xx, xy)
    dsxz_dx, dsxz_dz = grad_scalar(sigma_xz, xy)
    _, dszz_dz = grad_scalar(sigma_zz, xy)

    eq1 = dsxx_dx + dsxz_dz
    eq2 = dsxz_dx + dszz_dz
    return eq1, eq2


def corner_relief_weight(xy: torch.Tensor) -> torch.Tensor:
    """
    对角点奇异区域做平滑降权（不删点），减轻“单角点不可解析奇异”对全场解的主导。
    """
    dx = xy[:, 0:1] - HX
    dy = xy[:, 1:2] - HY
    r = torch.sqrt(dx * dx + dy * dy + 1e-18)
    w = r / max(CORNER_RELIEF_R, 1e-12)
    return torch.clamp(w, min=CORNER_RELIEF_WMIN, max=1.0)


def inner_right_corner_weight(xy: torch.Tensor) -> torch.Tensor:
    """
    右洞壁靠近角点 y->HY 处边界损失降权，弱化角点应力奇异对整体训练的主导。
    """
    d = torch.clamp(HY - xy[:, 1:2], min=0.0)
    # 用二次权重让“极近角点”衰减更强，远离角点快速恢复到 1
    w = torch.clamp(d / max(INNER_CORNER_SOFT_DIST, 1e-12), min=0.0, max=1.0) ** 2
    return INNER_CORNER_SOFT_WMIN + (1.0 - INNER_CORNER_SOFT_WMIN) * w


def inner_top_corner_weight(xy: torch.Tensor) -> torch.Tensor:
    """
    顶洞壁靠近角点 x->HX 处边界损失降权，弱化角点应力奇异对整体训练的主导。
    """
    d = torch.clamp(HX - xy[:, 0:1], min=0.0)
    # 用二次权重让“极近角点”衰减更强，远离角点快速恢复到 1
    w = torch.clamp(d / max(INNER_CORNER_SOFT_DIST, 1e-12), min=0.0, max=1.0) ** 2
    return INNER_CORNER_SOFT_WMIN + (1.0 - INNER_CORNER_SOFT_WMIN) * w


# =========================
# 九、纯弹性本构：由位移导出目标应力
# =========================
def elastic_stress_target(eps_xx, eps_zz, eps_yy, eps_xz):
    """X-Z 平面应变弹性本构，输出总应力目标。"""
    tr_eps = eps_xx + eps_zz + eps_yy
    sig_xx = p0 + lam * tr_eps + 2.0 * mu * eps_xx
    sig_zz = p0 + lam * tr_eps + 2.0 * mu * eps_zz
    sig_yy = p0 + lam * tr_eps + 2.0 * mu * eps_yy
    sig_xz = 2.0 * mu * eps_xz
    return sig_xx, sig_zz, sig_yy, sig_xz


def sigma_yy_plane_strain_from_stress(sigma_xx: torch.Tensor, sigma_zz: torch.Tensor) -> torch.Tensor:
    """
    X-Z 平面应变（eps_yy=0）闭合关系（总应力）：
    sigma_yy = p0 + nu * [(sigma_xx - p0) + (sigma_zz - p0)]
    """
    return p0 + nu * ((sigma_xx - p0) + (sigma_zz - p0))


def constitutive_residual_on_points(xy: torch.Tensor):
    """混合输出一致性残差：网络应力 vs 位移导出的弹性应力。"""
    u_x, u_z, sigma_xx, sigma_zz, sigma_yy, sigma_xz = net_fields(xy)
    eps_xx, eps_zz, eps_yy, eps_xz = strain_from_u(u_x, u_z, xy)
    sig_tgt_xx, sig_tgt_zz, sig_tgt_yy, sig_tgt_xz = elastic_stress_target(eps_xx, eps_zz, eps_yy, eps_xz)

    r_xx = ((sigma_xx - sig_tgt_xx) / DELTA_SIGMA_REF) ** 2
    r_zz = ((sigma_zz - sig_tgt_zz) / DELTA_SIGMA_REF) ** 2
    r_yy = ((sigma_yy - sig_tgt_yy) / DELTA_SIGMA_REF) ** 2
    r_xz = ((sigma_xz - sig_tgt_xz) / TAU_REF) ** 2
    # 注意：x-z 语义下，sigma_yy 才是 out-of-plane（对应 FLAC:YY）
    res = r_xx + r_zz + W_C_SZZ * r_yy + W_C_SXY * r_xz
    return res, (u_x, u_z, sigma_xx, sigma_zz, sigma_yy, sigma_xz), (sig_tgt_xx, sig_tgt_zz, sig_tgt_yy, sig_tgt_xz)


# =========================
# 十、固定采样点初始化：内部点 + 各条边界点
# =========================
xy_f: torch.Tensor
xy_out_right: torch.Tensor
xy_out_top: torch.Tensor
xy_in_right: torch.Tensor
xy_in_top: torch.Tensor
xy_sym_x0: torch.Tensor
xy_sym_y0: torch.Tensor
xy_corner_outer: torch.Tensor


def resample_training_points() -> None:
    """
    非自适应重采样：策略不变，仅刷新随机点，降低固定点过拟合。
    """
    global xy_f, xy_out_right, xy_out_top, xy_in_right, xy_in_top, xy_sym_x0, xy_sym_y0, xy_corner_outer
    if PAIR_SWAP_SAMPLING:
        # 1) 内部点：一半随机 + 一半 x<->z 镜像，保证域内训练分布交换对称
        n_half = N_F_FIXED // 2
        xy_half = sample_interior_rect(n_half)
        xy_half_sw = xy_half[:, [1, 0]]
        xy_f_np = np.vstack([xy_half, xy_half_sw])
        if xy_f_np.shape[0] < N_F_FIXED:
            xy_f_np = np.vstack([xy_f_np, sample_interior_rect(1)])
        xy_f = to_torch(xy_f_np[:N_F_FIXED], requires_grad=True)

        # 2) 外边界：右外边与上外边成对
        xy_or_np = sample_outer_right(N_B_OUT_RIGHT)
        xy_ot_np = np.stack([xy_or_np[:, 1], np.full_like(xy_or_np[:, 1], Lz)], axis=1)
        xy_out_right = to_torch(xy_or_np, requires_grad=True)
        xy_out_top = to_torch(xy_ot_np, requires_grad=True)

        # 3) 内边界：右内边与上内边成对
        xy_ir_np = sample_inner_right(N_B_IN_RIGHT)
        xy_it_np = np.stack([xy_ir_np[:, 1], np.full_like(xy_ir_np[:, 1], HZ)], axis=1)
        xy_in_right = to_torch(xy_ir_np, requires_grad=True)
        xy_in_top = to_torch(xy_it_np, requires_grad=True)

        # 4) 对称边界：x=0 与 z=0 成对
        xy_sx_np = sample_sym_x0(N_B_SYM_X0)
        xy_sz_np = np.stack([xy_sx_np[:, 1], np.zeros_like(xy_sx_np[:, 1])], axis=1)
        xy_sym_x0 = to_torch(xy_sx_np, requires_grad=True)
        xy_sym_y0 = to_torch(xy_sz_np, requires_grad=True)
    else:
        xy_f = to_torch(sample_interior_rect(N_F_FIXED), requires_grad=True)
        xy_out_right = to_torch(sample_outer_right(N_B_OUT_RIGHT), requires_grad=True)
        xy_out_top = to_torch(sample_outer_top(N_B_OUT_TOP), requires_grad=True)
        xy_in_right = to_torch(sample_inner_right(N_B_IN_RIGHT), requires_grad=True)
        xy_in_top = to_torch(sample_inner_top(N_B_IN_TOP), requires_grad=True)
        xy_sym_x0 = to_torch(sample_sym_x0(N_B_SYM_X0), requires_grad=True)
        xy_sym_y0 = to_torch(sample_sym_y0(N_B_SYM_Y0), requires_grad=True)
    xy_corner_outer = to_torch(np.array([[Lx, Ly]], dtype=np.float64), requires_grad=True)


resample_training_points()


# =========================
# 十一、总损失：平衡 + 边界 + 本构
# =========================
def build_total_loss(P_tunnel: float):
    """构造总损失函数，统一为 X-Z 平面语义。"""

    def loss_terms():
        # 1) 平衡方程损失
        u_x_f, u_z_f, sigma_xx_f, sigma_zz_f, sigma_yy_f, sigma_xz_f = net_fields(xy_f)
        eq1, eq2 = equilibrium_cartesian(sigma_xx_f, sigma_zz_f, sigma_xz_f, xy_f)
        w_eq = corner_relief_weight(xy_f)
        L_E = torch.mean(w_eq * ((eq1 / EQ_REF) ** 2 + (eq2 / EQ_REF) ** 2))

        # 2) 本构损失：所有训练点
        xy_all = torch.cat(
            [xy_f, xy_out_right, xy_out_top, xy_in_right, xy_in_top, xy_sym_x0, xy_sym_y0],
            dim=0,
        )
        res_c_all, _, _ = constitutive_residual_on_points(xy_all)
        w_c_all = corner_relief_weight(xy_all)
        L_C_phys = torch.mean(w_c_all * res_c_all)

        # 正方形交换对称：x<->z
        xy_swap = torch.cat([xy_f[:, 1:2], xy_f[:, 0:1]], dim=1)
        ux_sw, uz_sw, sxx_sw, szz_sw, syy_sw, sxz_sw = net_fields(xy_swap)
        L_SWAP = torch.mean(
            ((u_x_f - uz_sw) / U_REF) ** 2
            + ((u_z_f - ux_sw) / U_REF) ** 2
            + ((sigma_xx_f - szz_sw) / DELTA_SIGMA_REF) ** 2
            + ((sigma_zz_f - sxx_sw) / DELTA_SIGMA_REF) ** 2
            + ((sigma_yy_f - syy_sw) / DELTA_SIGMA_REF) ** 2
            + ((sigma_xz_f - sxz_sw) / TAU_REF) ** 2
        )

        # 平面应变应力闭合：约束面外 sigma_yy
        syy_ps = sigma_yy_plane_strain_from_stress(sigma_xx_f, sigma_zz_f)
        L_PS = torch.mean(w_eq * ((sigma_yy_f - syy_ps) / DELTA_SIGMA_REF) ** 2)

        L_C = L_C_phys + W_SWAP * L_SWAP + W_PS * L_PS

        # 3) 边界损失
        ux_or, uz_or, sxx_or, szz_or, syy_or, sxz_or = net_fields(xy_out_right)
        ux_ot, uz_ot, sxx_ot, szz_ot, syy_ot, sxz_ot = net_fields(xy_out_top)
        ux_ir, uz_ir, sxx_ir, szz_ir, syy_ir, sxz_ir = net_fields(xy_in_right)
        ux_it, uz_it, sxx_it, szz_it, syy_it, sxz_it = net_fields(xy_in_top)
        ux_sx, uz_sx, sxx_sx, szz_sx, syy_sx, sxz_sx = net_fields(xy_sym_x0)
        ux_sz, uz_sz, sxx_sz, szz_sz, syy_sz, sxz_sz = net_fields(xy_sym_y0)

        # 外边界：法向位移 + 弱剪应力
        L_out_right = torch.mean((ux_or / U_REF) ** 2 + W_OUTER_SXY * (sxz_or / TAU_REF) ** 2)
        L_out_top = torch.mean((uz_ot / U_REF) ** 2 + W_OUTER_SXY * (sxz_ot / TAU_REF) ** 2)

        # 对称边界：法向位移 + 剪应力
        L_sym_x0 = torch.mean((ux_sx / U_REF) ** 2 + (sxz_sx / TAU_REF) ** 2)
        L_sym_y0 = torch.mean((uz_sz / U_REF) ** 2 + (sxz_sz / TAU_REF) ** 2)

        # 洞壁边界：法向应力 + 剪应力
        w_ir = inner_right_corner_weight(xy_in_right)
        w_it = inner_top_corner_weight(xy_in_top)
        L_in_right_sigma = torch.mean(w_ir * (((sxx_ir - P_tunnel) / DELTA_SIGMA_REF) ** 2))
        L_in_right_tau = torch.mean(w_ir * ((sxz_ir / TAU_REF) ** 2))
        L_in_top_sigma = torch.mean(w_it * (((szz_it - P_tunnel) / DELTA_SIGMA_REF) ** 2))
        L_in_top_tau = torch.mean(w_it * ((sxz_it / TAU_REF) ** 2))
        w_inner_sum = max(W_B_INNER_SIGMA + W_B_INNER_TAU, 1e-12)
        # 保持内壁总权重与旧版一致，只调整法向/剪应力在内壁内部的相对占比
        L_in_right = (W_B_INNER_SIGMA * L_in_right_sigma + W_B_INNER_TAU * L_in_right_tau) / w_inner_sum
        L_in_top = (W_B_INNER_SIGMA * L_in_top_sigma + W_B_INNER_TAU * L_in_top_tau) / w_inner_sum

        # 洞壁均值附加项
        L_in_mean = ((torch.mean(sxx_ir) - P_tunnel) / DELTA_SIGMA_REF) ** 2 + ((torch.mean(szz_it) - P_tunnel) / DELTA_SIGMA_REF) ** 2

        # 远场锚定
        L_far_right = torch.mean(((sxx_or - p0) / DELTA_SIGMA_REF) ** 2)
        L_far_top = torch.mean(((szz_ot - p0) / DELTA_SIGMA_REF) ** 2)
        L_far_yy = 0.5 * (
            torch.mean(((syy_or - p0) / DELTA_SIGMA_REF) ** 2)
            + torch.mean(((syy_ot - p0) / DELTA_SIGMA_REF) ** 2)
        )

        # 入洞方向约束（中段，避开角点）
        mask_ir_mid = (HZ - xy_in_right[:, 1:2]) >= SIGN_EXCLUDE_DIST
        mask_it_mid = (HX - xy_in_top[:, 0:1]) >= SIGN_EXCLUDE_DIST
        pen_ir = torch.relu(ux_ir / U_REF) ** 2
        pen_it = torch.relu(uz_it / U_REF) ** 2
        L_sign_ir = torch.mean(pen_ir[mask_ir_mid]) if torch.any(mask_ir_mid) else torch.mean(pen_ir)
        L_sign_it = torch.mean(pen_it[mask_it_mid]) if torch.any(mask_it_mid) else torch.mean(pen_it)
        L_sign = L_sign_ir + L_sign_it

        # 外边界切向位移均值约束
        L_tan_mean = (torch.mean(uz_or) / U_REF) ** 2 + (torch.mean(ux_ot) / U_REF) ** 2

        # 外角点规约
        ux_c, uz_c, _, _, _, _ = net_fields(xy_corner_outer)
        L_corner_fix = torch.mean((ux_c / U_REF) ** 2 + (uz_c / U_REF) ** 2)

        # 边界长度加权
        len_out_right_eff = W_B_OUTER * LEN_OUT_RIGHT
        len_out_top_eff = W_B_OUTER * LEN_OUT_TOP
        len_sym_x0_eff = W_B_SYM * LEN_SYM_X0
        len_sym_y0_eff = W_B_SYM * LEN_SYM_Y0
        len_in_right_eff = (W_B_INNER_SIGMA + W_B_INNER_TAU) * LEN_IN_RIGHT
        len_in_top_eff = (W_B_INNER_SIGMA + W_B_INNER_TAU) * LEN_IN_TOP
        len_main_eff_sum = (
            len_out_right_eff + len_out_top_eff
            + len_sym_x0_eff + len_sym_y0_eff
            + len_in_right_eff + len_in_top_eff
        )

        L_B_main = (
            len_out_right_eff * L_out_right
            + len_out_top_eff * L_out_top
            + len_sym_x0_eff * L_sym_x0
            + len_sym_y0_eff * L_sym_y0
            + len_in_right_eff * L_in_right
            + len_in_top_eff * L_in_top
        ) / max(len_main_eff_sum, 1e-12)

        L_B_far = (
            LEN_OUT_RIGHT * L_far_right
            + LEN_OUT_TOP * L_far_top
        ) / max(LEN_OUT_RIGHT + LEN_OUT_TOP, 1e-12)

        L_B = (
            L_B_main
            + W_FAR * L_B_far
            + W_FAR_ZZ * L_far_yy
            + W_SIGN * L_sign
            + W_IN_MEAN * L_in_mean
            + W_TAN_MEAN * L_tan_mean
            + W_CORNER_FIX * L_corner_fix
        )

        with torch.no_grad():
            max_eq = torch.sqrt(torch.max(eq1 ** 2 + eq2 ** 2))
            max_sig = torch.max(torch.abs(torch.cat([sigma_xx_f, sigma_zz_f, sigma_yy_f, sigma_xz_f], dim=1)))

        return L_E, L_B, L_C, max_eq, max_sig

    return loss_terms



# =========================
# 十二、训练函数：Adam + L-BFGS
# =========================
def print_header(tag: str):
    """打印训练日志表头。"""
    print(f"\n[{tag}] 设备 = {DEVICE}, 数据类型 = {DTYPE}")
    print("-" * 120)
    print(f"{'迭代步':<8} | {'总损失':<14} | {'平衡损失':<14} | {'边界损失':<14} | {'本构损失':<14} | {'最大平衡残差':<16} | {'最大应力绝对值(MPa)':<20}")
    print("-" * 120)


def run_adam(total_loss_fn, epochs: int, tag: str, weights: tuple[float, float, float]):
    """先用 Adam 做长步训练。"""
    w_e, w_b, w_c = weights
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print_header(tag)

    for it in range(epochs + 1):
        if RESAMPLE_EVERY > 0 and it > 0 and (it % RESAMPLE_EVERY == 0):
            resample_training_points()
            if it % PRINT_EVERY == 0:
                print(f"[Adam] 已执行周期重采样：it = {it}")

        optimizer.zero_grad(set_to_none=True)
        L_E, L_B, L_C, max_eq, max_sig = total_loss_fn()
        L = w_e * L_E + w_b * L_B + w_c * L_C
        L.backward()
        optimizer.step()

        if it % PRINT_EVERY == 0:
            print(
                f"{it:<8d} | {fmt_num(L.item(), 6):<14} | {fmt_num(L_E.item(), 6):<14} | {fmt_num(L_B.item(), 6):<14} | "
                f"{fmt_num(L_C.item(), 6):<14} | {fmt_num(max_eq.item(), 6):<16} | {fmt_num(max_sig.item(), 6):<20}"
            )


def run_lbfgs(total_loss_fn, steps: int, tag: str, weights: tuple[float, float, float]):
    """再用 L-BFGS 做精细优化。"""
    w_e, w_b, w_c = weights
    print(f"\n[{tag}] 开始执行 L-BFGS，步数 = {steps}")

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=steps,
        max_eval=steps,
        history_size=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        L_E, L_B, L_C, _, _ = total_loss_fn()
        L = w_e * L_E + w_b * L_B + w_c * L_C
        L.backward()
        return L

    optimizer.step(closure)

    L_E, L_B, L_C, max_eq, max_sig = total_loss_fn()
    L = w_e * L_E + w_b * L_B + w_c * L_C
    print(
        f"[{tag}] L-BFGS 结束后："
        f"总损失 = {fmt_num(L.item(), 6)}，"
        f"平衡损失 = {fmt_num(L_E.item(), 6)}，"
        f"边界损失 = {fmt_num(L_B.item(), 6)}，"
        f"本构损失 = {fmt_num(L_C.item(), 6)}，"
        f"最大平衡残差 = {fmt_num(max_eq.item(), 6)}，"
        f"最大应力绝对值 = {fmt_num(max_sig.item(), 6)} MPa"
    )


# =========================
# 十三、结果诊断：边界、极值、本构一致性、自检
# =========================
@torch.no_grad()
def evaluate_boundary_report():
    """检查各条边界是否满足 X-Z 语义边界条件。"""
    print("\n" + "=" * 120)
    print("边界条件诊断（应力单位：MPa；位移单位：m）")
    print("=" * 120)

    _, _, sxx_or, _, _, sxz_or = net_fields(xy_out_right)
    _, _, _, szz_ot, _, sxz_ot = net_fields(xy_out_top)
    ux_sx, _, _, _, _, sxz_sx = net_fields(xy_sym_x0)
    _, uz_sz, _, _, _, sxz_sz = net_fields(xy_sym_y0)
    _, _, sxx_ir, _, _, sxz_ir = net_fields(xy_in_right)
    _, _, _, szz_it, _, sxz_it = net_fields(xy_in_top)

    print(f"外边界右侧  x=Lx : 平均|u_x|         = {fmt_num(torch.mean(torch.abs(net_fields(xy_out_right)[0])).item())} m，平均|sigma_xz| = {fmt_num(torch.mean(torch.abs(sxz_or)).item())} MPa")
    print(f"外边界顶部  z=Lz : 平均|u_z|         = {fmt_num(torch.mean(torch.abs(net_fields(xy_out_top)[1])).item())} m，平均|sigma_xz| = {fmt_num(torch.mean(torch.abs(sxz_ot)).item())} MPa")
    print(f"外边界右侧  x=Lx : 平均|sigma_xx-p0| = {fmt_num(torch.mean(torch.abs(sxx_or - p0)).item())} MPa")
    print(f"外边界顶部  z=Lz : 平均|sigma_zz-p0| = {fmt_num(torch.mean(torch.abs(szz_ot - p0)).item())} MPa")
    print(f"对称边界    x=0  : 平均|u_x|         = {fmt_num(torch.mean(torch.abs(ux_sx)).item())} m，平均|sigma_xz| = {fmt_num(torch.mean(torch.abs(sxz_sx)).item())} MPa")
    print(f"对称边界    z=0  : 平均|u_z|         = {fmt_num(torch.mean(torch.abs(uz_sz)).item())} m，平均|sigma_xz| = {fmt_num(torch.mean(torch.abs(sxz_sz)).item())} MPa")
    print(f"巷道右壁    x=HX : 平均|sigma_xx-P| = {fmt_num(torch.mean(torch.abs(sxx_ir - P_STAGE_A)).item())} MPa，平均|sigma_xz| = {fmt_num(torch.mean(torch.abs(sxz_ir)).item())} MPa")
    print(f"巷道顶壁    z=HZ : 平均|sigma_zz-P| = {fmt_num(torch.mean(torch.abs(szz_it - P_STAGE_A)).item())} MPa，平均|sigma_xz| = {fmt_num(torch.mean(torch.abs(sxz_it)).item())} MPa")


def _dense_domain_points(nx: int = 181, ny: int = 181) -> torch.Tensor:
    """生成稠密评估网格，用于看全场极值；只排除精确角点单点。"""
    xs = np.linspace(0.0, Lx, nx)
    ys = np.linspace(0.0, Ly, ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    mask = (~inside_hole_np(pts[:, 0], pts[:, 1])) & (~is_exact_corner_point_np(pts[:, 0], pts[:, 1]))
    pts = pts[mask]
    return torch.tensor(pts, dtype=DTYPE, device=DEVICE)


@torch.no_grad()
def evaluate_field_minmax():
    """输出稠密网格上的位移与应力极值（X-Z 语义）。"""
    xz_t = _dense_domain_points()
    ux, uz, sxx, szz, syy, sxz = net_fields(xz_t)

    ux_min = ux.min().item()
    ux_max = ux.max().item()
    uz_min = uz.min().item()
    uz_max = uz.max().item()

    print("\n" + "=" * 120)
    print("稠密网格场量极值（以下为网络直接输出的混合场量）")
    print("=" * 120)
    print(f"u_x      : 最小值 = {fmt_num(ux_min)} m  ({fmt_num(100.0 * ux_min)} cm)，最大值 = {fmt_num(ux_max)} m  ({fmt_num(100.0 * ux_max)} cm)")
    print(f"u_z      : 最小值 = {fmt_num(uz_min)} m  ({fmt_num(100.0 * uz_min)} cm)，最大值 = {fmt_num(uz_max)} m  ({fmt_num(100.0 * uz_max)} cm)")
    print(f"sigma_xx : 最小值 = {fmt_num(sxx.min().item())} MPa，最大值 = {fmt_num(sxx.max().item())} MPa")
    print(f"sigma_zz : 最小值 = {fmt_num(szz.min().item())} MPa，最大值 = {fmt_num(szz.max().item())} MPa")
    print(f"sigma_yy : 最小值 = {fmt_num(syy.min().item())} MPa，最大值 = {fmt_num(syy.max().item())} MPa")
    print(f"sigma_xz : 最小值 = {fmt_num(sxz.min().item())} MPa，最大值 = {fmt_num(sxz.max().item())} MPa")


@torch.no_grad()
def evaluate_flac_reference_compare():
    """按 FLAC 网格中心点（0.2 m）统计范围并直接对照。"""
    print("\n" + "=" * 120)
    print("FLAC 图例范围对照（基于 0.2m 网格中心点采样）")
    print("=" * 120)

    h = 0.2
    xs = np.arange(0.5 * h, Lx, h)
    zs = np.arange(0.5 * h, Lz, h)
    XX, ZZ = np.meshgrid(xs, zs)
    pts = np.stack([XX.reshape(-1), ZZ.reshape(-1)], axis=1)
    mask = ~inside_hole_np(pts[:, 0], pts[:, 1])
    pts = pts[mask]
    xz_t = torch.tensor(pts, dtype=DTYPE, device=DEVICE)

    ux, uz, sxx, szz, syy, sxz = net_fields(xz_t)
    ux_np = ux.detach().cpu().numpy().reshape(-1)
    uz_np = uz.detach().cpu().numpy().reshape(-1)
    sxx_np = sxx.detach().cpu().numpy().reshape(-1)
    szz_np = szz.detach().cpu().numpy().reshape(-1)
    syy_np = syy.detach().cpu().numpy().reshape(-1)
    sxz_np = sxz.detach().cpu().numpy().reshape(-1)

    ux_absmax = float(np.max(np.abs(ux_np)))
    uz_absmax = float(np.max(np.abs(uz_np)))
    sxz_absmax = float(np.max(np.abs(sxz_np)))

    pred_flac = {
        "u_x": (-ux_absmax, ux_absmax),
        "u_y": (-uz_absmax, uz_absmax),
        "xx": (-float(np.max(sxx_np)), -float(np.min(sxx_np))),
        "zz": (-float(np.max(szz_np)), -float(np.min(szz_np))),
        "yy": (-float(np.max(syy_np)), -float(np.min(syy_np))),
        "xz": (-sxz_absmax, sxz_absmax),
    }

    name_map = {
        "u_x": "XDisplacement (m)",
        "u_y": "ZDisplacement (m)",
        "xx": "XX Stress (MPa, FLAC sign)",
        "zz": "ZZ Stress (MPa, FLAC sign)",
        "yy": "YY Stress (MPa, FLAC sign)",
        "xz": "XZ Stress (MPa, FLAC sign)",
    }

    for k in ("u_x", "u_y", "xx", "zz", "yy", "xz"):
        ref_min, ref_max = FLAC_REF_RANGES[k]
        pred_min, pred_max = pred_flac[k]
        ref_span = max(ref_max - ref_min, 1e-12)
        pred_span = pred_max - pred_min
        span_err = abs(pred_span - ref_span) / ref_span * 100.0
        print(
            f"{name_map[k]:<34} | FLAC:[{ref_min:+.6f}, {ref_max:+.6f}] "
            f"| PINN:[{pred_min:+.6f}, {pred_max:+.6f}] | 跨度误差={span_err:.2f}%"
        )


def evaluate_constitutive_consistency():
    """检查混合输出中的应力-位移自一致性（X-Z 语义）。"""
    xz_t = _dense_domain_points()
    xz_t = xz_t.clone().detach().requires_grad_(True)
    u_x, u_z, sigma_xx, sigma_zz, sigma_yy, sigma_xz = net_fields(xz_t)
    eps_xx, eps_zz, eps_yy, eps_xz = strain_from_u(u_x, u_z, xz_t)
    sig_tgt_xx, sig_tgt_zz, sig_tgt_yy, sig_tgt_xz = elastic_stress_target(eps_xx, eps_zz, eps_yy, eps_xz)

    with torch.no_grad():
        e_xx = torch.abs(sigma_xx - sig_tgt_xx)
        e_zz = torch.abs(sigma_zz - sig_tgt_zz)
        e_yy = torch.abs(sigma_yy - sig_tgt_yy)
        e_xz = torch.abs(sigma_xz - sig_tgt_xz)
        e_all = torch.cat([e_xx, e_zz, e_yy, e_xz], dim=1)

        mean_sxx = torch.mean(e_xx).item()
        mean_szz = torch.mean(e_zz).item()
        mean_syy = torch.mean(e_yy).item()
        mean_sxz = torch.mean(e_xz).item()
        max_all = torch.max(e_all).item()

        dx = xz_t[:, 0:1] - HX
        dz = xz_t[:, 1:2] - HZ
        r = torch.sqrt(dx * dx + dz * dz + 1e-18)
        far_mask = (r >= CORNER_RELIEF_R).reshape(-1)
        e_far = e_all[far_mask, :].reshape(-1)
        q999_far = torch.quantile(e_far, 0.999).item()
        max_far = torch.max(e_far).item()

    print("\n" + "=" * 120)
    print("混合输出自一致性诊断（网络输出应力 vs 由位移导出的弹性应力）")
    print("=" * 120)
    print(f"mean|sigma_xx - sigma_xx(u)| = {fmt_num(mean_sxx)} MPa")
    print(f"mean|sigma_zz - sigma_zz(u)| = {fmt_num(mean_szz)} MPa")
    print(f"mean|sigma_yy - sigma_yy(u)| = {fmt_num(mean_syy)} MPa")
    print(f"mean|sigma_xz - sigma_xz(u)| = {fmt_num(mean_sxz)} MPa")
    print(f"max  constitutive mismatch    = {fmt_num(max_all)} MPa")
    print(f"max  mismatch (r>=reliefR)    = {fmt_num(max_far)} MPa")
    print(f"q99.9 mismatch (r>=reliefR)   = {fmt_num(q999_far)} MPa")


@torch.no_grad()
def evaluate_self_consistency():
    """做最基础物理自检（不依赖 FLAC 数据）。"""
    print("\n" + "=" * 120)
    print("结果自检（不依赖 FLAC3D）")
    print("=" * 120)

    ux_or, uz_or, sxx_or, szz_or, syy_or, sxz_or = net_fields(xy_out_right)
    ux_ot, uz_ot, sxx_ot, szz_ot, syy_ot, sxz_ot = net_fields(xy_out_top)
    ux_ir, uz_ir, sxx_ir, szz_ir, syy_ir, sxz_ir = net_fields(xy_in_right)
    ux_it, uz_it, sxx_it, szz_it, syy_it, sxz_it = net_fields(xy_in_top)
    ux_sx, uz_sx, sxx_sx, szz_sx, syy_sx, sxz_sx = net_fields(xy_sym_x0)
    ux_sz, uz_sz, sxx_sz, szz_sz, syy_sz, sxz_sz = net_fields(xy_sym_y0)

    frac_right_inward = torch.mean((ux_ir <= 0.0).to(DTYPE)).item()
    frac_top_inward = torch.mean((uz_it <= 0.0).to(DTYPE)).item()

    mean_outer_ux = torch.mean(torch.abs(ux_or)).item()
    mean_outer_uz = torch.mean(torch.abs(uz_ot)).item()

    rel_inner_right = torch.mean(torch.abs(sxx_ir - P_STAGE_A)).item() / max(abs(P_STAGE_A), 1e-12)
    rel_inner_top = torch.mean(torch.abs(szz_it - P_STAGE_A)).item() / max(abs(P_STAGE_A), 1e-12)
    rel_sym_sxz_x = torch.mean(torch.abs(sxz_sx)).item() / max(abs(p0), 1e-12)
    rel_sym_sxz_z = torch.mean(torch.abs(sxz_sz)).item() / max(abs(p0), 1e-12)

    print(f"右内边界相对误差  mean|sigma_xx-P|/P    = {fmt_num(100.0 * rel_inner_right)} %")
    print(f"上内边界相对误差  mean|sigma_zz-P|/P    = {fmt_num(100.0 * rel_inner_top)} %")
    print(f"x=0 对称边界剪应力相对量 mean|sigma_xz|/p0 = {fmt_num(100.0 * rel_sym_sxz_x)} %")
    print(f"z=0 对称边界剪应力相对量 mean|sigma_xz|/p0 = {fmt_num(100.0 * rel_sym_sxz_z)} %")
    print(f"右洞壁上 u_x<=0 的点占比 = {fmt_num(100.0 * frac_right_inward)} %")
    print(f"上洞壁上 u_z<=0 的点占比 = {fmt_num(100.0 * frac_top_inward)} %")
    print(f"右外边界平均 |u_x| = {fmt_num(mean_outer_ux)} m  ({fmt_num(100.0 * mean_outer_ux)} cm)")
    print(f"上外边界平均 |u_z| = {fmt_num(mean_outer_uz)} m  ({fmt_num(100.0 * mean_outer_uz)} cm)")


if __name__ == "__main__":
    print("=" * 120)
    print("深埋正方形/矩形巷道 PINN - Stage A 纯弹性阶段（笛卡尔坐标，混合输出一致框架，无自适应稳定版，单点避让+近角点分层加密版）")
    print(f"计算设备：{DEVICE}")
    print(f"数据类型：{DTYPE}")
    print(f"巷道宽度 TUNNEL_W = {fmt_num(TUNNEL_W)} m")
    print(f"巷道高度 TUNNEL_H = {fmt_num(TUNNEL_H)} m")
    print(f"1/4 内边界尺寸：HX = {fmt_num(HX)} m，HZ = {fmt_num(HZ)} m")
    print(f"1/4 外边界尺寸：Lx = {fmt_num(Lx)} m，Lz = {fmt_num(Lz)} m")
    print(f"原位应力 p0 = {fmt_num(p0)} MPa")
    print(f"Stage A 洞壁压力 P = {fmt_num(P_STAGE_A)} MPa")
    print(f"弹性参数：E = {fmt_num(E)} MPa，nu = {fmt_num(nu)}")
    print("FLAC 对应关系：u_x↔XDisplacement，u_z↔ZDisplacement，sigma_xx↔XX，sigma_zz↔ZZ，sigma_yy↔YY，sigma_xz↔XZ。")
    print("符号说明：本脚本压缩为正；FLAC3D 压缩为负。对比应力时请取相反号。")
    print(f"内部残差点：固定 {N_F_FIXED} 个（不做自适应加点）")
    print(f"边界点总数：{N_B_TOTAL} 个（右外边{N_B_OUT_RIGHT}，上外边{N_B_OUT_TOP}，右内边{N_B_IN_RIGHT}，上内边{N_B_IN_TOP}，x=0对称边{N_B_SYM_X0}，z=0对称边{N_B_SYM_Y0}）")
    if W_OUTER_SXY > 0:
        print("边界逻辑：外边界约束法向位移+弱剪应力；对称边约束法向位移+剪应力；洞壁约束法向应力+剪应力。")
    else:
        print("边界逻辑：外边界仅约束法向位移；对称边约束法向位移+剪应力；洞壁约束法向应力+剪应力。")
    print(f"位移硬约束：HARD_U_BC = {HARD_U_BC}（x=0/Lx 上 ux=0，z=0/Lz 上 uz=0）")
    print("应力参数化：sigma = p0 + scale * sigma_head（混合输出保持不变，仅改为增量形式）")
    print(
        f"边界损失：有效边界长度加权积分（W_OUTER_SXY = {fmt_num(W_OUTER_SXY)}，W_B_INNER_SIGMA = {fmt_num(W_B_INNER_SIGMA)}，"
        f"W_B_INNER_TAU = {fmt_num(W_B_INNER_TAU)}，"
        f"W_B_OUTER = {fmt_num(W_B_OUTER)}，W_B_SYM = {fmt_num(W_B_SYM)}，W_IN_MEAN = {fmt_num(W_IN_MEAN)}，"
        f"W_TAN_MEAN = {fmt_num(W_TAN_MEAN)}，W_CORNER_FIX = {fmt_num(W_CORNER_FIX)}，W_FAR = {fmt_num(W_FAR)}，W_FAR_ZZ = {fmt_num(W_FAR_ZZ)}）。"
    )
    print(f"对称增强：x<->z 交换对称约束（W_SWAP = {fmt_num(W_SWAP)}），平面应变应力闭合（W_PS = {fmt_num(W_PS)}）。")
    print(f"方向约束：洞壁入洞位移约束（W_SIGN = {fmt_num(W_SIGN)}）。")
    print(f"本构通道加权：W_C_SZZ(作用于sigma_yy[FLAC:YY]) = {fmt_num(W_C_SZZ)}，W_C_SXY(作用于sigma_xz[FLAC:XZ]) = {fmt_num(W_C_SXY)}。")
    print(f"L-BFGS 参数：步数 = {LBFGS_STEPS}，学习率 = {fmt_num(LBFGS_LR)}")
    if RESAMPLE_EVERY > 0:
        print(f"训练采样：非自适应周期重采样（RESAMPLE_EVERY = {RESAMPLE_EVERY}）。")
    else:
        print("训练采样：固定采样（关闭周期重采样）。")
    print(f"采样对称性：PAIR_SWAP_SAMPLING = {PAIR_SWAP_SAMPLING}（x<->z 成对采样）")
    print(f"归一化尺度：SIGMA_REF_TOTAL = {fmt_num(SIGMA_REF_TOTAL)} MPa，DELTA_SIGMA_REF = {fmt_num(DELTA_SIGMA_REF)} MPa，TAU_REF = {fmt_num(TAU_REF)} MPa，U_REF = {fmt_num(U_REF)} m")
    print(f"单点避让容差：EPS_CORNER_POINT = {EPS_CORNER_POINT:.1e} m，近角点加密带宽比例：CORNER_BAND = {fmt_num(CORNER_BAND)}")
    print(f"角点奇异缓释：CORNER_RELIEF_R = {fmt_num(CORNER_RELIEF_R)} m，CORNER_RELIEF_WMIN = {fmt_num(CORNER_RELIEF_WMIN)}")
    print(f"洞壁角点软化：INNER_CORNER_SOFT_DIST = {fmt_num(INNER_CORNER_SOFT_DIST)} m，INNER_CORNER_SOFT_WMIN = {fmt_num(INNER_CORNER_SOFT_WMIN)}，SIGN_EXCLUDE_DIST = {fmt_num(SIGN_EXCLUDE_DIST)} m")
    print(f"近角点 RBF 特征：FLAC_GRID_H = {fmt_num(FLAC_GRID_H)} m，CORNER_RBF_SIGMA = {fmt_num(CORNER_RBF_SIGMA)} m，输入维度 = 12")
    print("FLAC图例范围（用于对照，单位同图例）:")
    print(f"  XDisp=[{FLAC_REF_RANGES['u_x'][0]:+.6f}, {FLAC_REF_RANGES['u_x'][1]:+.6f}] m, ZDisp=[{FLAC_REF_RANGES['u_y'][0]:+.6f}, {FLAC_REF_RANGES['u_y'][1]:+.6f}] m")
    print(f"  XX=[{FLAC_REF_RANGES['xx'][0]:+.4f}, {FLAC_REF_RANGES['xx'][1]:+.4f}] MPa, ZZ=[{FLAC_REF_RANGES['zz'][0]:+.4f}, {FLAC_REF_RANGES['zz'][1]:+.4f}] MPa")
    print(f"  YY=[{FLAC_REF_RANGES['yy'][0]:+.4f}, {FLAC_REF_RANGES['yy'][1]:+.4f}] MPa, XZ=[{FLAC_REF_RANGES['xz'][0]:+.4f}, {FLAC_REF_RANGES['xz'][1]:+.4f}] MPa")
    print("说明：本版在 v41 基础上只把平面应变 sigma_yy 闭合权重从 1.2 调到 1.6，用于补足 YY 的少量低估。")
    print("=" * 120)

    loss_fn = build_total_loss(P_STAGE_A)
    weights = (W_E, W_B, W_C)

    # 先用 Adam 粗训练，再用 L-BFGS 精修
    run_adam(loss_fn, epochs=EPOCHS_ADAM, tag=f"Stage A 固定采样训练（{N_F_FIXED}个内部点）", weights=weights)
    if USE_LBFGS:
        run_lbfgs(loss_fn, steps=LBFGS_STEPS, tag="Stage A 固定采样训练 L-BFGS", weights=weights)

    # 输出若干诊断量，方便和 FLAC3D 及后续版本比较
    evaluate_boundary_report()
    evaluate_flac_reference_compare()
    evaluate_constitutive_consistency()

    # 保存模型参数
    save_dir = Path(__file__).resolve().parent / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    if SMOKE:
        out_name = save_dir / "pinn_model_stageA_rectangular_smoke.pth"
    else:
        out_name = save_dir / "pinn_model_stageA_rectangular_gpt_v42.pth"
    torch.save(model.state_dict(), str(out_name))
    print("\n模型参数已保存：", str(out_name))
