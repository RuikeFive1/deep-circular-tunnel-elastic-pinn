# 矩形巷道弹性 PINN 方法

## 1. 计算域与未知场

训练域为四分之一正方形区域 `[0,5] x [0,5] m`，去除洞内 `[0,1] x [0,1] m`。利用关于 `x=0`、`z=0` 的对称性重建完整 `10 m x 10 m` 模型。

网络混合输出六个总量场：

```text
u_x, u_z, sigma_xx, sigma_zz, sigma_yy, sigma_xz.
```

总应力参数化为 `sigma_normal = p0 + Delta_sigma * head`，`sigma_xz = Delta_sigma * head`，其中 `Delta_sigma = |p0-P| = 2 MPa`。

## 2. 平衡与本构

无体力静力平衡为

```text
d(sigma_xx)/dx + d(sigma_xz)/dz = 0,
d(sigma_xz)/dx + d(sigma_zz)/dz = 0.
```

小应变采用压缩为正：

```text
epsilon_xx = -du_x/dx,
epsilon_zz = -du_z/dz,
epsilon_xz = -0.5(du_x/dz + du_z/dx),
epsilon_yy = 0.
```

由三维各向同性线弹性和平面应变条件得到应力目标，`L_C` 约束网络应力与位移导出应力一致。`sigma_yy` 同时使用平面应变闭合关系约束。

## 3. 边界条件

- `x=5 m`：`u_x=0`；保留弱 `sigma_xz=0` 稳定项。
- `z=5 m`：`u_z=0`；保留弱 `sigma_xz=0` 稳定项。
- `x=0`：`u_x=0, sigma_xz=0`。
- `z=0`：`u_z=0, sigma_xz=0`。
- 右洞壁 `x=1 m`：`sigma_xx=P, sigma_xz=0`。
- 顶洞壁 `z=1 m`：`sigma_zz=P, sigma_xz=0`。

法向位移通过距离函数硬编码为零。边界损失按各边长度组合，并包含弱远场应力、洞壁均值和交换对称稳定项；全部系数保留在训练脚本顶部，未在发布整理时改动。

## 4. 采样

内部固定 2,500 点：85% 在岩体域内均匀采样，15% 在洞角邻域加密。采样排除精确尖角坐标，但不删除角点邻域。正方形工况使用 `x<->z` 成对采样降低随机非对称误差。

六段边界各取 100 点。洞壁点包含均匀段、中近角点段和极近角点段，但以 `1e-6 m` 避开精确尖角。

## 5. 优化

首先使用 Adam，学习率 `1e-3`，执行 10,000 步；随后执行一次 PyTorch L-BFGS，`lr=0.8`、`max_iter=max_eval=1000`、`history_size=50`、强 Wolfe 线搜索。

L-BFGS 的 closure 可能被重复调用，因此 `max_iter=1000` 不应解释为恰好 1000 次网络前向计算。
