import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import re
from pathlib import Path

# 设置命令行参数
parser = argparse.ArgumentParser(description='Compare performance data from multiple CSV files using bar charts.')
parser.add_argument('--log', type=str, action='append', required=True, help='Paths to the CSV log files (multiple --log options allowed)')
parser.add_argument('--enable-qps', action='store_true', default=False, help='Include QPS data in the plot, default is False')
args = parser.parse_args()
print(f"Enable QPS: {args.enable_qps}")  # Debug print

# 定义关键字提取函数
def extract_keywords(filename):
    filename = Path(filename).stem.lower()  # 提取文件名（不含路径和扩展名），转为小写
    keywords = []

    # 匹配 dynamic 或 static
    if 'dynamic' in filename:
        keywords.append('dynamic')
    elif 'static' in filename:
        keywords.append('static')

    # 仅当包含 "sync_true" 时添加 "sync" 关键字
    if 'sync_true' in filename:
        keywords.append('sync')

    # 匹配 mem_thres_XX
    mem_thres_match = re.search(r'mem_thres_(\d+)', filename)
    if mem_thres_match:
        mem_thres_value = int(mem_thres_match.group(1))
        if mem_thres_value != 100:  # 100 表示关闭功能，不显示
            keywords.append(f'mem_thres_{mem_thres_value}')

    # 匹配 queue_size_XX
    queue_size_match = re.search(r'queue_size_(-?\d+)', filename)
    if queue_size_match:
        queue_size_value = int(queue_size_match.group(1))
        if queue_size_value != -1:  # -1 表示关闭功能，不显示
            keywords.append(f'queue_size_{queue_size_value}')

    # 匹配日期 YYYY-MM-DD
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if date_match:
        date_value = date_match.group(1)
        keywords.append(date_value)

    # 返回连接的关键字，若无则使用文件名
    return '_'.join(keywords) if keywords else filename

# 读取所有 CSV 文件并提取关键字
dfs = []
labels = []
for log_file in args.log:
    print(f"读取文件 '{log_file}'...")
    df = pd.read_csv(log_file)
    df['IPS'] = pd.to_numeric(df['IPS'], errors='coerce')
    df['QPS'] = pd.to_numeric(df['QPS'], errors='coerce')
    dfs.append(df)
    labels.append(extract_keywords(log_file))

# 定义分组标签
configs = [
    (128, 128, '128/128'),
    (128, 1024, '128/1024'),
    (1024, 128, '1024/128'),
    (1024, 1024, '1024/1024')
]

# 设置全局参数
batch_sizes = [1, 4, 8, 16, 32, 64, 128]
num_files = len(dfs)
bar_width = 0.8 / (num_files if not args.enable_qps else 2 * num_files)  # Adjust width based on QPS inclusion
opacity = 0.8
colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'purple', 'cyan', 'yellow', 'pink']

# 创建 2x2 子图布局
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle(f'Performance Comparison: {" vs ".join(labels)}', fontsize=16)

# 遍历每个配置并绘制柱状图
for idx, (input_len, output_len, label) in enumerate(configs):
    ax = axes[idx // 2, idx % 2]  # 确定子图位置

    # X 轴位置
    x = np.arange(len(batch_sizes))

    # 为每个文件绘制 IPS (和 QPS，如果启用)
    for i, (df, file_label) in enumerate(zip(dfs, labels)):
        subset = df[(df['input_len'] == input_len) & (df['output_len'] == output_len)]
        subset = subset.set_index('batch_size').reindex(batch_sizes).reset_index()

        # IPS 数据 (始终绘制)
        ips = subset['IPS'].fillna(0)

        # 计算柱状图位置
        if args.enable_qps:
            # 绘制 IPS 和 QPS，QPS 在 IPS 右侧
            qps = subset['QPS'].fillna(0)
            offset = (i - (num_files - 1) / 2) * bar_width
            ax.bar(x + offset, ips, bar_width, alpha=opacity, color=colors[i % len(colors)], label=f'{file_label} IPS')
            ax.bar(x + offset + num_files * bar_width, qps, bar_width, alpha=opacity, color=colors[i % len(colors)], hatch='//', label=f'{file_label} QPS')
        else:
            # 只绘制 IPS
            offset = (i - (num_files - 1) / 2) * bar_width
            ax.bar(x + offset, ips, bar_width, alpha=opacity, color=colors[i % len(colors)], label=f'{file_label} IPS')

    # 设置标题和标签
    ax.set_title(f'Input/Output: {label}')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('IPS (Tokens/s)' if not args.enable_qps else 'IPS (Tokens/s) / QPS (Requests/s)', color='black')

    # 设置 X 轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes, rotation=45)

    # 添加图例
    ax.legend(loc='upper left', fontsize=8)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.95])  # 留出空间给主标题

# 保存图片，使用提取的关键字作为前缀
output_prefix = '_vs_'.join(labels)
output_file = f'{output_prefix}_performance_comparison_bar{"_with_qps" if args.enable_qps else ""}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"图表已保存为 '{output_file}'")

# 显示图表（可选）
# plt.show()