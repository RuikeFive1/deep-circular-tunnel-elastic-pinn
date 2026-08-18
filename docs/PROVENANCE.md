# 文件来源与整理记录

本目录从工作区 `apinn_deep/` 的深埋圆形巷道弹性复现文件复制整理而来，原目录未移动、未删除。

| 新文件 | 原文件 |
|---|---|
| `train_deep.py` | `apinn_deep/train_deep.py` |
| `viz_deep.py` | `apinn_deep/viz_deep.py` |
| `plot_deep_elastic_fig10_comparison.py` | `apinn_deep/plot_deep_elastic_fig10_comparison.py` |
| `apinn/*.py` | `apinn_deep/apinn/*.py` |
| `checkpoints/apinn_deep_elastic_paper_exact.pt` | `apinn_deep/apinn_deep_elastic_paper_exact.pt` |
| `figures/fields/*.png` | `apinn_deep/figs_fig10/*.png` |
| `figures/comparison/*.png` | `apinn_deep/figs_fig10_compare/*.png` |

为了使目录可独立运行，只进行了以下工程化改动：

1. 权重保存和加载改为相对于脚本目录的 `checkpoints/`。
2. 图片输出改为相对于脚本目录的 `figures/`。
3. 新增 `apinn/__init__.py`、README、依赖清单、忽略规则和 smoke test。
4. 未修改网络结构、物理残差、材料参数、采样算法或训练轮次。

