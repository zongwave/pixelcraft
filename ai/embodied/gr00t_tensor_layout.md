
## GR00T 模型的数据流总结：

---

## 1️⃣ 原始输入 (Normalized Input)

| Tensor | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `state` | [1, 1, 64] | float64 | 机器人状态（关节角度、速度等）|
| `state_mask` | [1, 1, 64] | bool | 状态掩码（前52有效）|
| `eagle_input_ids` | [1, 296] | int64 | 文本 token IDs |
| `eagle_attention_mask` | [1, 296] | int64 | 注意力掩码（全1）|
| `eagle_pixel_values` | [1, 3, 224, 224] | float32 | 图像像素值 |
| `eagle_image_sizes` | [1, 2] | int64 | 图像尺寸 [224,224] |
| `embodiment_id` | [1] | int64 | 机器人类型 ID (24) |

---

## 2️⃣ EagleBackbone 输入 (转换后)

| Tensor | Shape | dtype | 变化 |
|--------|-------|-------|------|
| `state` | [1, 1, 64] | **bfloat16** | float64 → bfloat16 |
| `eagle_pixel_values` | [1, 3, 224, 224] | **bfloat16** | float32 → bfloat16 |
| 其他同上 | - | - | 不变 |

---

## 3️⃣ EagleBackbone 输出

| Tensor | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `logits` | - | - | 语言模型输出（未使用）|
| `hidden_states` | - | - | 中间层状态 |
| **`backbone_features`** | **[1, 296, 2048]** | **bfloat16** | 实际使用的条件特征 |
| `backbone_attention_mask` | [1, 296] | int64 | 注意力掩码（全1）|

**输出 keys**: `odict_keys(['logits', 'hidden_states'])`  
**注意**：没有 `past_key_values` ✅

---

## 4️⃣ Flowmatching Action Head 内部

### 输入 (来自 Backbone)
| Tensor | Shape | dtype |
|--------|-------|-------|
| `vl_embs` | [1, 296, 2048] | bfloat16 |
| `state` | [1, 1, 64] | bfloat16 |
| `embodiment_id` | [1] | int64 |

### DiT 输入 (`sa_embs`)
| 组成 | 数量 | Shape |
|------|------|-------|
| state_features | 1 | [1, 1, 1536] |
| future_tokens | 32 | [1, 32, 1536] |
| action_features | 16 | [1, 16, 1536] |
| **总计** | **49** | **[1, 49, 1536]** |

### DiT 输出
| Tensor | Shape | 说明 |
|--------|-------|------|
| `pred_velocity` | **[1, 16, 32]** | 速度场（用于更新动作）|

### 迭代更新
```
for t in range(4):  # 4 步固定迭代
    pred_velocity = model(...)  # [1, 16, 32]
    actions = actions + dt * pred_velocity  # dt = 0.25
```

---

## 5️⃣ 最终输出

| Tensor | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `action_pred` | **[1, 16, 32]** | float32 | 预测的未来16步动作 |

---

## 📈 数据流图

```
原始输入 (float64/float32)
    │
    ▼ 类型转换
Backbone 输入 (bfloat16)
    │
    ├──► EagleBackbone (1次)
    │        │
    │        ▼
    │   backbone_features [1,296,2048] (bfloat16)
    │        │
    │        ▼
    └──► Flowmatching Action Head
              │
              ├── 初始化噪声 [1,16,32]
              │
              ├── 迭代 0: pred_velocity [1,16,32] ──► actions 更新
              ├── 迭代 1: pred_velocity [1,16,32] ──► actions 更新
              ├── 迭代 2: pred_velocity [1,16,32] ──► actions 更新
              └── 迭代 3: pred_velocity [1,16,32] ──► actions 更新
                    │
                    ▼
              action_pred [1,16,32] (float32)
```

---

## 🔑 关键发现总结

| 模块 | 输入 Shape | 输出 Shape | KV Cache |
|------|-----------|-----------|----------|
| **EagleBackbone** | [1,296,?] + [1,3,224,224] | [1,296,2048] | ❌ 无 |
| **DiT** | [1,49,1536] + [1,296,2048] | [1,49,1536] | ❌ 无 |
| **Action Decoder** | [1,49,1536] | [1,16,32] | - |
| **最终输出** | - | [1,16,32] | - |

**所有张量形状完全静态** ✅