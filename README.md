# Deep Circular Tunnel Elastic PINN

这是深埋圆形巷道纯弹性阶段的 PyTorch PINN/APINN 复现整理版。仓库保留最初模型的网络结构、物理残差、采样与优化流程，并附带已训练权重、场图脚本以及 PINN 与有限圆环解析解的误差对比脚本。

## 计算工况

- 四分之一圆环计算域：洞半径 `a = 1 m`，外边界半径 `R = 3 m`。
- 初始静水应力：`p0 = 10 MPa`。
- 弹性阶段洞壁压力：`P = 8 MPa`。
- 弹性参数：`E = 10 MPa`，`nu = 0.2`。
- 网络输入：`x, y`。
- 混合输出：`ux, uy, sigma_xx, sigma_yy, sigma_zz, sigma_xy`。
- 网络结构：6 个隐藏层，每层 40 个神经元，`Tanh` 激活，`float64`。

内部采用拉应力为正；绘图脚本统一将应力转换为压应力为正。位移图按原复现脚本以外边界平均径向位移为零点。

## 文件结构

```text
.
|-- apinn/
|   |-- models.py                 # 混合输出 MLP
|   |-- physics.py                # 平衡、本构、屈服与边界残差
|   `-- sampling.py               # 圆环域与自适应采样
|-- checkpoints/
|   `-- apinn_deep_elastic_paper_exact.pt
|-- figures/
|   |-- fields/                   # sigma_rr, sigma_tt, u_r
|   `-- comparison/               # PINN/解析解/误差九宫格
|-- docs/
|   |-- METHOD.md
|   |-- PROVENANCE.md
|   `-- VALIDATION.md
|-- tests/smoke_test.py
|-- train_deep.py
|-- viz_deep.py
|-- plot_deep_elastic_fig10_comparison.py
`-- requirements.txt
```

## 环境安装

建议使用 Python 3.10 或更高版本，并根据 CUDA 版本先从 PyTorch 官方渠道安装对应的 `torch`。随后执行：

```bash
python -m pip install -r requirements.txt
```

## 快速验证

```bash
python tests/smoke_test.py
```

该命令只检查网络输出、自动微分物理残差和权重加载，不执行完整训练。

## 使用已有权重绘图

生成三张弹性场图：

```bash
python viz_deep.py
```

生成 PINN、解析解及有符号误差的 3x3 对比图：

```bash
python plot_deep_elastic_fig10_comparison.py
```

对比脚本支持参数覆盖，例如：

```bash
python plot_deep_elastic_fig10_comparison.py --nxy 301 --dpi 300
```

## 重新训练

```bash
python train_deep.py
```

训练流程为：初始 10,000 步 Adam；随后 10 轮残差自适应采样，每轮新增 20 个内部点，并执行 1,000 步 Adam 和一次 `max_iter=1000` 的 L-BFGS。完整训练计算量较大，L-BFGS 的 closure 评估次数不等同于日志中的外层步数。

原脚本设置 `n_b = 50`，但四类边界均使用 `n_b // 4`，因此实际边界点数为 `12 x 4 = 48`。本整理版没有擅自改变这一历史训练行为，以保证所附权重与源代码口径一致。

## 结果与适用范围

本仓库只对应深埋圆形巷道的 `P = 8 MPa` 纯弹性阶段。它不包含矩形巷道、Stage B 弹塑性、FLAC3D 数据监督或 Zone State 历史塑性判断。解析解仅用于绘图后的独立误差诊断，不进入训练损失。

详细物理链条见 [docs/METHOD.md](docs/METHOD.md)，文件来源与整理改动见 [docs/PROVENANCE.md](docs/PROVENANCE.md)，当前权重的可复核误差见 [docs/VALIDATION.md](docs/VALIDATION.md)。

## 发布说明

参考论文 PDF 未包含在仓库中。公开发布前请补充作者信息、论文完整引文，并选择合适的开源许可证；当前目录未代替作者授权任何许可证。

完成上述信息后，可在本目录执行：

```bash
git init
git add .
git commit -m "Initial deep circular tunnel elastic PINN reproduction"
git branch -M main
git remote add origin <your-repository-url>
git push -u origin main
```
