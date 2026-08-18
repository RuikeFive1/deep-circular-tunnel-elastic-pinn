#采样点的生成与自适应重采样  决定了“在什么位置”给网络看物理方程。
import torch, math
#在四分之一环域 内均匀采样 N 个点 PDE / 本构残差 的主战场
def sample_quarter_annulus(N, a, R, device, dtype):
    u = torch.rand((N,1), device=device, dtype=dtype)
    r = torch.sqrt(a*a + (R*R - a*a) * u)
    theta = (torch.rand((N,1), device=device, dtype=dtype)) * (math.pi/2.0)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    X = torch.cat([x,y], dim=-1)
    X.requires_grad_(True)
    return X
#在指定半径 radius 的四分之一圆 上取 N 个点 用来施加边界条件
def sample_on_circle(N, radius, device, dtype):
    theta = torch.rand((N,1), device=device, dtype=dtype) * (math.pi/2.0)
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    X = torch.cat([x,y], dim=-1)
    X.requires_grad_(True)
    return X
#在 x 轴或 y 轴上均匀采样，用于对称边界条件：
def sample_on_axis(N, axis, limit, device, dtype):
    t = torch.rand((N,1), device=device, dtype=dtype) * limit
    if axis == 'x':
        x = torch.zeros_like(t); y = t
    else:
        x = t; y = torch.zeros_like(t)
    X = torch.cat([x,y], dim=-1)
    X.requires_grad_(True)
    return X
#全局自适应采样的关键步骤   残差驱动的全局自适应采样
def importance_resample(X_eval, res_values, k=1.0, m=0.05, n_new=20):
    r = res_values.reshape(-1).abs()
    weights = torch.pow(r, torch.tensor(k, dtype=r.dtype, device=r.device)) + m
    prob = (weights / torch.sum(weights)).detach().cpu().numpy()
    import numpy as np
    idx = np.random.choice(len(prob), size=n_new, replace=False, p=prob)
    return X_eval[idx]
