**English | [中文](./README.md)**

# 3A Algorithms — Technical Summary

> The three core algorithms of camera imaging: AE (Auto Exposure), AWB (Auto White Balance), AF (Auto Focus) — detailed design, tuning, and field-test records.

## Reading Guide

| File | Summary | Quick Access |
|---|---|---|
| [ae_v2.en.md](./ae_v2.en.md) | AE algorithm summary (18% gray target, histogram strategy, joint exposure-gain tuning, etc.) | [👉 Read](./ae_v2.en.md) |
| [awb.en.md](./awb.en.md) | AWB survey (gray-world / white-point / EP3149936B1 patent, CIE chromaticity coordinates, ΔE₀₀ color difference, AWB tuning checklist) | [👉 Read](./awb.en.md) |
| [af.en.md](./af.en.md) | AF primer (mainstream approaches / focus measures / control strategies / performance metrics / full tuning flow) | [👉 Read](./af.en.md) |

## Directory Structure

```text
3a/
├── README.md          # this file (Chinese)
├── README.en.md       # English version
├── ae_v2.md           # Auto Exposure
├── awb.md             # Auto White Balance
├── code/              # sample scripts & tuning tools
└── diagram/           # diagrams & curves
```
