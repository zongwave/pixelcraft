"""N1.7 on aarch64 host 的 patch_embed 灾难性慢速修复（实测 L20 + GB200-class ARM 服务器，torch 2.9+cu128）。

现象：transformers==4.57.3 的 Qwen3VLVisionPatchEmbed 把输入 reshape 成
[N=1024, 3, 2, 16, 16]（batch=1024、空间 1×1 的退化 Conv3d）再走 nn.Conv3d。
该形状组合在本机 cuDNN/原生 conv 路径上均触发病态实现：**每次调用 ~25 s**
（同形状独立微基准可复现；喂 [1,3,180,320] 视频形输入却只要 0.05 s；
torch.backends.cudnn.enabled=False 也不改善）。

修复：kernel==stride 的卷积等价于"每 patch 展平后的一次矩阵乘"。
输入按 (c,t,h,w) 自然展平、权重同序 reshape，F.linear 与 Conv3d 数值等价
（fp32 对照 max err ~9e-4，纯约化顺序差；端到端 MSE 复跑一致）。
效果：get_action 35 s/步 → 1.5 s/步（24×）。

用法（在 n1.7 仓库目录）：
    PYTHONPATH=$PWD:<本文件所在目录> python -c "
    import n17_arm_patch; n17_arm_patch.apply()
    import runpy; runpy.run_path('scripts/deployment/standalone_inference_script.py', run_name='__main__')" \
      --model-path ... --dataset-path demo_data/droid_sample ...

自检：python n17_arm_patch.py --self-test
"""

import sys


def apply() -> int:
    """Monkeypatch Qwen3VL VisionPatchEmbed.forward -> 等价 F.linear。返回打补丁的类数(0/1)。"""
    import torch.nn.functional as F
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed as PE
    except ImportError:
        return 0

    def forward(self, hidden_states):
        target_dtype = self.proj.weight.dtype
        x = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        x = x.view(
            -1,
            self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size,
        )
        w = self.proj.weight.reshape(self.embed_dim, -1)  # 与 x 同为 (c,t,h,w) 展平序
        return F.linear(x.to(dtype=target_dtype), w, self.proj.bias)

    PE.forward = forward
    return 1


def self_test() -> bool:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C, Tp, P, O = 3, 2, 16, 64
    conv = nn.Conv3d(C, O, kernel_size=(Tp, P, P), stride=(Tp, P, P), bias=True).to(dev)
    x = torch.randn(512, C, Tp, P, P, device=dev)
    ref = conv(x).view(-1, O)
    out = F.linear(x.view(-1, C * Tp * P * P), conv.weight.reshape(O, -1), conv.bias)
    err = (ref - out).abs().max().item()
    print(f"self-test max abs err = {err:.2e} (fp32, 期望 <1e-3)")
    return err < 1e-3


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    n = apply()
    print(f"patched {n} class(es); 用 runpy 方式启动 n1.7 脚本后生效")
