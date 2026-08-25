# FLAC3D 对照数据

`data/flac_stageA_rect_fields.csv` 来自 Stage A 弹性模型的 zone centroid 导出，共包含坐标、位移、应力和 zone id。

主要字段：

| CSV 字段 | PINN 对应量 | 单位 |
|---|---|---|
| `x, z` | 空间坐标 | m |
| `ux, uz` | `u_x, u_z` | m |
| `sxx, szz` | `sigma_xx, sigma_zz` | MPa |
| `syy` | `sigma_yy` | MPa |
| `sxz` | `sigma_xz` | MPa |
| `zone_id` | FLAC3D zone id | - |

CSV 使用 FLAC3D 的压缩负号约定；绘图与误差脚本将正应力转换为压缩正后再比较。CSV 仅用于训练后验证，不进入 `train_rectangular_elastic.py` 的损失函数。

`tools/export_flac_stageA_rect_csv.py` 是 FLAC3D 内置 Python 环境使用的导出脚本。仓库不包含 `.sav` 文件；使用者需要自行恢复相同模型状态后导出。
