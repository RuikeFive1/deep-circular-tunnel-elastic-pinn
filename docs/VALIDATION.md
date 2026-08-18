# 权重验证记录

验证环境：Windows、Python 3.10、PyTorch、CUDA、`torch.float64`。对比网格为 `401 x 401`，只统计四分之一圆环域内点。应力单位为 MPa，径向位移单位为 cm。

| 物理量 | Mean error | MAE | RMSE | Max absolute error |
|---|---:|---:|---:|---:|
| `sigma_r` | 0.000470526 | 0.000742824 | 0.00116122 | 0.00689359 |
| `sigma_theta` | -0.00164405 | 0.00186722 | 0.00237061 | 0.0330508 |
| `u_r` | 0.00814212 | 0.0170382 | 0.0201056 | 0.0711463 |

复核命令：

```bash
python tests/smoke_test.py
python plot_deep_elastic_fig10_comparison.py
```

这些误差来自随仓库提供的 checkpoint 与脚本内有限圆环解析解的逐点比较。解析解没有参与训练。

