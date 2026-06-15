# Decoder-Only Transformer 极简示例

## 概述

本目录包含一个从零实现的 Decoder-Only Transformer 示例代码，用于理解 GPT/LLaMA 系列模型的核心流程和原理。

## 文件说明

| 文件 | 说明 |
|------|------|
| `decoder_only_demo.py` | 主文件：完整实现 + 训练 + 推理演示 |

## 代码组件

| 组件 | 对应文档章节 | 说明 |
|------|-------------|------|
| `RMSNorm` | Ch1.1 | Pre-Norm 归一化（LLaMA 风格） |
| `CausalSelfAttention` | Ch1.3 | 单头因果自注意力 + KV Cache |
| `SwiGLU_FFN` | Ch1.2 | 门控前馈网络 |
| `DecoderBlock` | - | Pre-Norm 结构：Attn + FFN + 残差 |
| `DecoderOnlyTransformer` | - | 完整模型：Embed → N×Decoder → LM Head |
| `make_arithmetic_dataset` | - | 等差数列玩具数据集 |
| `demo()` | - | 训练 + 推理演示入口 |

## 运行方式

```bash
# 确保已安装 PyTorch (>= 2.0)
pip install torch

# 运行演示
python decoder_only_demo.py
```

## 演示内容

1. **训练**：在等差数列数据集上做 Next Token Prediction
2. **推理对比**：有/无 KV Cache 的速度差异
3. **采样多样性**：不同 temperature 下的生成结果
4. **速度基准**：KV Cache 加速比量化

## 与完整文档的对应关系

本示例代码与 `ai/transformer/llm-inference-systems.md` 文档配套：
- 文档提供理论深度和数学推导
- 代码提供可运行的工程实现
- 两者结合理解效果最佳