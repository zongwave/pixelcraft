# 3A 算法笔记仓库

> 本仓库汇总相机成像三大核心算法：AE（自动曝光）、AWB（自动白平衡）、AF（自动对焦）的详细设计、调优与实测记录。

## 快速导航

| 文件 | 内容摘要 | 直达 |
|---|---|---|
| [ae_v2.md](./ae_v2.md) | AE 算法总结（18 % 灰准则、直方图策略、曝光-增益联合调优等） | [👉 阅读](./ae_v2.md) |
| [awb.md](./awb.md) | AWB 算法综述（灰世界 / 白点 / EP3149936B1 专利、CIE 色度坐标、ΔE₀₀ 色差、AWB 调优 Checklist） | [👉 阅读](./awb.md) |

## 目录结构

```text
3a/
├── README.md          # 本文件
├── ae_v2.md           # Auto Exposure
├── awb.md             # Auto White Balance
├── code/              # 示例脚本 & 调优工具
└── diagram/           # 示意图与曲线