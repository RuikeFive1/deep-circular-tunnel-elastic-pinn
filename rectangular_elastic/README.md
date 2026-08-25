# Rectangular Tunnel Elastic PINN

本目录是深埋正方形/矩形巷道 Stage A 纯弹性 PINN 算例。它在笛卡尔坐标中使用位移与应力混合输出，通过静力平衡、边界条件和平面应变弹性本构残差训练，并使用 FLAC3D 导出的最终场做独立对照。

## 工况

- 全域：`10 m x 10 m`；训练使用四分之一域 `0 <= x,z <= 5 m`。
- 中心巷道：`2 m x 2 m`。
- 初始静水应力：`p0 = 10 MPa`，压缩为正。
- Stage A 洞壁压力：`P = 8 MPa`。
- 材料：`E = 10 MPa`，`nu = 0.2`。
- 假设：小变形、平面应变、无体力、纯弹性。

## 网络与训练

- 输入：12 个坐标/几何特征。
- 输出：`u_x, u_z, sigma_xx, sigma_zz, sigma_yy, sigma_xz`。
- MLP：8 个隐藏层，每层 80 个神经元，`Tanh`，`float64`。
- 内部点：固定 2,500 个，其中 85% 全域采样、15% 洞角附近加密。
- 边界点：6 段各 100 个，共 600 个。
- 优化：Adam 10,000 步，随后一次 L-BFGS，`max_iter=1000`。
- 总损失：`L = 1.6 L_E + 1.0 L_B + 1.1 L_C`。

FLAC3D CSV 不参与训练损失。脚本内的 `FLAC_REF_RANGES` 只在训练后输出范围诊断；绘图脚本使用 CSV 统一色标并计算误差。

## 文件结构

```text
rectangular_elastic/
|-- checkpoints/                 # 已训练 v42 权重
|-- data/                        # FLAC3D Stage A CSV
|-- docs/                        # 方法、数据、验证和来源说明
|-- figures/                     # PINN、FLAC3D 和误差图
|-- tests/smoke_test.py
|-- tools/export_flac_stageA_rect_csv.py
|-- train_rectangular_elastic.py
|-- plot_pinn_fields.py
|-- plot_flac_fields.py
`-- plot_pinn_flac_error.py
```

## 快速验证和绘图

从本目录运行：

```bash
python tests/smoke_test.py
python plot_pinn_fields.py
python plot_flac_fields.py
python plot_pinn_flac_error.py
```

图片输出到 `figures/`。重新训练：

```bash
python train_rectangular_elastic.py
```

快速检查训练链条可使用：

```bash
python train_rectangular_elastic.py --smoke
```

`--smoke` 只执行 20 步 Adam、关闭 L-BFGS，并保存为单独的 smoke 权重，不覆盖正式 checkpoint。

## 对照口径

PINN 内部应力采用压缩为正，FLAC3D CSV 应力采用压缩为负。对照脚本会把 FLAC3D 正应力统一转换为压缩正；`sigma_xz` 只按坐标镜像规则变号。位移单位为 m，应力单位为 MPa。

CSV 坐标是 `0.2 m` 网格的 zone centroid。PINN 在相同坐标上直接求值后逐点比较。误差图默认显示 `PINN - FLAC3D`，颜色范围使用绝对误差的 99% 分位数，但终端报告始终给出完整 MAE、RMSE 和最大绝对误差。

详见 [`docs/METHOD.md`](docs/METHOD.md)、[`docs/DATA.md`](docs/DATA.md) 和 [`docs/VALIDATION.md`](docs/VALIDATION.md)。
