#物理、PDE、本构和边界残差
#物理核心，里面的函数可以分为 4 类：
#材料常数 / 应变 / 应力张量；
#应力不变量 & Mohr–Coulomb 屈服函数；
#弹塑性本构增量（APINN 的“物理损失”核心）；
#PDE & 边界条件的残差。
import torch
from torch import autograd
#输入 杨氏模量 / 泊松比  输出：拉梅常数 用于后面的线弹性 / 弹塑性刚度矩阵
def lame_parameters(E, nu):
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return lam, mu
#自动微分从位移场  计算小应变分量
def strain_components(u, X):
    ux = u[..., 0:1]
    uy = u[..., 1:2]
    ones = torch.ones_like(ux, dtype=ux.dtype, device=ux.device)
    dux = autograd.grad(ux, X, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    duy = autograd.grad(uy, X, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    exx = dux[..., 0:1]
    eyy = duy[..., 1:2]
    exy = 0.5 * (dux[..., 1:2] + duy[..., 0:1])
    return exx, eyy, exy
#把网络输出的 4 个主分量 组装成 3×3 的应力张量
def stress_tensor_from_components(s):
    s_xx = s[..., 0]
    s_yy = s[..., 1]
    s_zz = s[..., 2]
    s_xy = s[..., 3]
    shape = s_xx.shape
    S = torch.zeros((*shape, 3, 3), dtype=s.dtype, device=s.device)
    S[..., 0, 0] = s_xx
    S[..., 1, 1] = s_yy
    S[..., 2, 2] = s_zz
    S[..., 0, 1] = s_xy
    S[..., 1, 0] = s_xy
    return S

def invariants_and_lode_angle(S):
    I = torch.eye(3, dtype=S.dtype, device=S.device)
    I1 = torch.einsum("...ii->...", S)
    mean = I1 / 3.0
    meanI = mean[..., None, None] * I
    dev = S - meanI
    J2 = 0.5 * torch.einsum("...ij,...ij->...", dev, dev)
    J3 = torch.det(dev)
    eps = torch.finfo(S.dtype).eps
    denom = torch.clamp(J2, min=eps)**1.5
    xi = torch.clamp( (3.0*torch.sqrt(torch.tensor(3.0, dtype=S.dtype, device=S.device))/2.0) * (J3/denom), min=-0.999999, max=0.999999 )
    theta = (1.0/3.0) * torch.asin(xi)
    return I1, J2, J3, theta

#计算 Mohr–Coulomb 屈服函数
def mohr_coulomb_F(S, c, phi):
    I1, J2, J3, theta = invariants_and_lode_angle(S)
    sqrtJ2 = torch.sqrt(torch.clamp(J2, min=1e-18))
    term1 = sqrtJ2 * (torch.cos(theta) - (torch.sin(phi)/torch.sqrt(torch.tensor(3.0, dtype=S.dtype, device=S.device))) * torch.sin(theta))
    term2 = (1.0/3.0) * I1 * torch.sin(phi) - c * torch.cos(phi)
    return term1 + term2
#在增量塑性理论里就是“塑性流动方向”相关的向量，用来构造塑性增量、求解塑性乘子
def dF_dsigma(F, s_components):
    grads = []
    for i in range(4):
        grad = autograd.grad(F, s_components, grad_outputs=torch.ones_like(F), retain_graph=True, create_graph=True)[0][..., i:i+1]
        grads.append(grad)
    return torch.cat(grads, dim=-1)

#弹塑本构残差（APINN 的核心）
#从位移场 计算应变
#用线弹性 （λμ）得到“假定全弹性”的应力增量
#用 Mohr–Coulomb看是否越过屈服面
#修正应力 → 得到弹塑应力
#和APINN预测的σ 做差 这就是“本构残差” C，训练时让它尽量趋近 0。
def constitutive_increment(u, s_pred, X, E, nu, c, phi, r0):
    lam, mu = lame_parameters(E, nu)
    exx, eyy, exy = strain_components(u, X)

    S = stress_tensor_from_components(s_pred)
    c_t = torch.tensor(c, dtype=S.dtype, device=S.device)
    phi_t = torch.tensor(phi, dtype=S.dtype, device=S.device)
    F = mohr_coulomb_F(S, c_t, phi_t)

    dF = dF_dsigma(F, s_pred)
    dF_xx = dF[..., 0:1]; dF_yy = dF[..., 1:2]; dF_zz = dF[..., 2:3]; dF_xy = dF[..., 3:4]

    num = lam*(exx + eyy)*(dF_xx + dF_yy + dF_zz) + 2*mu*(dF_xx*exx + dF_yy*eyy + 2*dF_xy*exy)
    den = lam*torch.square(dF_xx + dF_yy + dF_zz) + 2*mu*(torch.square(dF_xx) + torch.square(dF_yy) + torch.square(dF_zz) + 2*torch.square(dF_xy))
    dk_raw = num / (torch.clamp(den, min=1e-18))
    dk = torch.where(F.view(-1,1) > 0, dk_raw, torch.zeros_like(dk_raw))

    common = (dF_xx + dF_yy + dF_zz)
    dr_xx = lam*(exx + eyy - dk*common) + 2*mu*(exx - dk*dF_xx)
    dr_yy = lam*(exx + eyy - dk*common) + 2*mu*(eyy - dk*dF_yy)
    dr_zz = lam*(exx + eyy - dk*common) - 2*mu*(dk*dF_zz)
    dr_xy = 2*mu*(exy - dk*dF_xy)

    r0_xx, r0_yy, r0_zz = r0
    C_xx = dr_xx + r0_xx - s_pred[..., 0:1]
    C_yy = dr_yy + r0_yy - s_pred[..., 1:2]
    C_zz = dr_zz + r0_zz - s_pred[..., 2:3]
    C_xy = dr_xy             - s_pred[..., 3:4]
    C = torch.cat([C_xx, C_yy, C_zz, C_xy], dim=-1)
    return C, F, dk
#动量平衡方程残差
def equilibrium_residuals(s_pred, X):
    s_xx = s_pred[..., 0:1]
    s_yy = s_pred[..., 1:2]
    s_xy = s_pred[..., 3:4]
    ones = torch.ones_like(s_xx, dtype=s_xx.dtype, device=s_xx.device)
    dsxx = autograd.grad(s_xx, X, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    dsyy = autograd.grad(s_yy, X, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    dsxy = autograd.grad(s_xy, X, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    r1 = dsxx[..., 0:1] + dsxy[..., 1:2]
    r2 = dsyy[..., 1:2] + dsxy[..., 0:1]
    return torch.cat([r1, r2], dim=-1)
#圆边界上的牵引残差
def traction_residual_on_circle(s_pred, X, radius, target_pressure):
    x = X[..., 0:1]; y = X[..., 1:2]
    r = torch.sqrt(x**2 + y**2) + 1e-18
    nx = x / r; ny = y / r
    s_xx = s_pred[..., 0:1]; s_yy = s_pred[..., 1:2]; s_xy = s_pred[..., 3:4]
    tx = s_xx*nx + s_xy*ny
    ty = s_xy*nx + s_yy*ny
    px = target_pressure * nx
    py = target_pressure * ny
    tx_hat = -ny; ty_hat = nx
    rn = (tx - px)*nx + (ty - py)*ny
    rt =  tx*tx_hat + ty*ty_hat
    return torch.cat([rn, rt], dim=-1)
#对称边界条件
def symmetry_residuals(s_pred, u_pred, X, axis='x'):
    s_xy = s_pred[..., 3:4]
    ux = u_pred[..., 0:1]; uy = u_pred[..., 1:2]
    if axis == 'x':
        return torch.cat([ux, s_xy], dim=-1)
    else:
        return torch.cat([uy, s_xy], dim=-1)
