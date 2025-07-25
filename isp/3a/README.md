# 3A 算法总结

> 相机成像三大核心算法：AE（自动曝光）、AWB（自动白平衡）、AF（自动对焦）的详细设计、调优与实测记录。

## 阅读导引

| 文件 | 内容摘要 | 直达 |
|---|---|---|
| [ae_v2.md](./ae_v2.md) | AE 算法总结（18 % 灰准则、直方图策略、曝光-增益联合调优等） | [👉 阅读](./ae_v2.md) |
| [awb.md](./awb.md) | AWB 算法综述（灰世界 / 白点 / EP3149936B1 专利、CIE 色度坐标、ΔE₀₀ 色差、AWB 调优 Checklist） | [👉 阅读](./awb.md) |
| [af.md](./af.md) | AF 算法简介（主流方案 / 评价函数 / 控制策略 / 性能指标 / 调优全流程） | [👉 阅读](./af.md) |

## 目录结构

```text
3a/
├── README.md          # 本文件
├── ae_v2.md           # Auto Exposure
├── awb.md             # Auto White Balance
├── code/              # 示例脚本 & 调优工具
└── diagram/           # 示意图与曲线