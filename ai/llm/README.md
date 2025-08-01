
---

## 📑 目录

* [大模型（LLM）技术总结](#大模型llm技术总结)


  * [📌 架构演进与 LLaMA 模型结构](#架构演进与-llama-模型结构)
    * [🔹 Transformer 基础结构回顾](#transformer-基础结构回顾)
    * [🔹 LLaMA 的架构特色](#llama-的架构特色)
    * [🔹 LLaMA 2 / LLaMA 3 的更新点](#llama-2--llama-3-的更新点)

  * [🧠 预训练技术](#预训练技术)
    * [🔹 数据集与清洗策略](#数据集与清洗策略)
    * [🔹 分词算法（BPE vs SentencePiece）](#分词算法bpe-vs-sentencepiece)
    * [🔹 训练目标（Causal LM）](#训练目标causal-lm)

  * [💬 指令微调与对齐技术](#指令微调与对齐技术)
    * [🔹 SFT（监督微调）](#sft监督微调)
    * [🔹 RLHF（强化学习人类反馈）](#rlhf强化学习人类反馈)
    * [🔹 DPO / ORPO 等对齐新范式](#dpo--orpo-等对齐新范式)

  * [🧪 模型压缩与加速](#模型压缩与加速)
    * [🔹 量化（Int4 / GPTQ / AWQ）](#量化int4--gptq--awq)
    * [🔹 蒸馏与剪枝](#蒸馏与剪枝)
    * [🔹 KV Cache 优化](#kv-cache-优化)

  * [⚙️ 模型部署与推理框架](#模型部署与推理框架)
    * [🔹 Transformers & vLLM](#transformers--vllm)
    * [🔹 GGUF / GGML / llama.cpp](#gguf--ggml--llamacpp)
    * [🔹 分布式推理与张量并行](#分布式推理与张量并行)

  * [📈 性能分析与调优建议](#性能分析与调优建议)
    * [🔹 Profiling 工具链](#profiling-工具链)
    * [🔹 内存与计算瓶颈](#内存与计算瓶颈)
    * [🔹 推理延迟优化策略](#推理延迟优化策略)

  * [📚 总结与未来趋势](#总结与未来趋势)
    * [🔹 模型尺寸 vs 能力](#模型尺寸-vs-能力)
    * [🔹 多模态扩展](#多模态扩展)
    * [🔹 MoE / Sparse 架构发展](#moe--sparse-架构发展)

---


---


---