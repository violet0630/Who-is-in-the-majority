"""Plot frozen design checks, not LLM experimental outcomes."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', type=Path, required=True)
    args = ap.parse_args()
    root = args.prepared
    rows = json.loads((root/'preparation_diagnostics.json').read_text(encoding='utf-8'))
    labels = [f"G{d['graph']+1} / R{d['repeat']+1}" for d in rows]
    colors = ['#2376a5', '#d18a23', '#3a9475']
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 10), sharey=True)
    y = np.arange(len(rows))
    for ax, key, title in zip(axes, ['IG_gap', 'max_match_gap'],
                              ['初始结构幻觉差 TH − TL（百分点）', '三项资源的最大配对差异（%）']):
        ax.barh(y, [d[key]*100 for d in rows], color=[colors[d['graph']] for d in rows], height=.7)
        ax.set_title(title, fontsize=12, pad=14)
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        for i, d in enumerate(rows):
            ax.text(d[key]*100+.1, i, f"{d[key]*100:.2f}", va='center', fontsize=9)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 16)
    axes[1].set_xlim(0, 6.2)
    axes[1].axvline(5, color='#a72c37', linestyle='--', label='5%匹配上限')
    axes[1].legend(loc='lower right', frameon=False)
    fig.suptitle('主实验一：21个新增配对组的离线准备检查', fontsize=17, y=.98)
    fig.text(.08, .035, '沿用3张网络；每图新增7组；每组TH/TL/U0。G、R从1编号。\n资源：倡议者粉丝总数、基准发言概率之和、初始期望发言人数。\n此图仅核验输入；结构IG是位置选择目标，不是LLM感知结果或独立机制证据。', fontsize=10)
    fig.tight_layout(rect=[0, .105, 1, .95])
    fig.savefig(root/'07_preparation_checks.png', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
