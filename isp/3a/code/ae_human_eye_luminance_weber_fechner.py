import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(figsize=(10, 6))

# 生成数据
brightness = np.logspace(-2, 6, 500)  # 10^-2到10^6 cd/m²
perception = np.log10(brightness + 0.01)  # 对数响应模拟

# 绘制主曲线
ax.semilogx(brightness, perception, 'b-', linewidth=3, label='人眼感知响应')

# 标记关键区域
ax.axvspan(1e-2, 1e0, color='navy', alpha=0.2, label='暗视觉 (k≈0.01)')
ax.axvspan(1e0, 1e2, color='purple', alpha=0.2, label='过渡区 (18%灰)')
ax.axvspan(1e2, 1e6, color='red', alpha=0.2, label='明视觉 (k≈0.1)')

# 标记18%灰位置
ax.axvline(18, color='black', linestyle='--', linewidth=1)
ax.annotate('18%灰标准', xy=(18, 1.2), xytext=(30, 1.5),
            arrowprops=dict(arrowstyle="->"))

# 设置坐标轴
ax.set_xlabel('亮度 (cd/m², 对数坐标)', fontsize=12)
ax.set_ylabel('感知响应 (线性化)', fontsize=12)
ax.set_title('人眼亮度感知对数响应曲线 (Weber-Fechner定律)', fontsize=14)
ax.grid(True, which="both", ls="--")
ax.legend(loc='upper left')

# 特殊刻度标注
ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
ax.set_xticks([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
ax.set_xticklabels(['10⁻²', '10⁻¹', '10⁰', '10¹', '10²', '10³', '10⁴', '10⁵', '10⁶'])

plt.tight_layout()
plt.savefig('brightness_perception.png', dpi=300, bbox_inches='tight')
plt.show()