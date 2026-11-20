"""Offline audit and exploratory structure analysis. No API calls; archive is read-only.
Run: python analyze_exp1_positions.py --repo PATH --output NEW_DIRECTORY
"""
import argparse
import gzip
import hashlib
import itertools
import json
import subprocess
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.repo / 'experiments/2026-11-16'
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    archived = read(root/'03_positions/results.json')
    lookup = {(r['graph'], r['repeat'], r['branch']): r for r in archived}
    manifest = read(root/'archive_manifest.json')
    for item in manifest['files']:
        p = root/item['path']
        require(p.stat().st_size == item['bytes'], f'Size mismatch: {p}')
        require(hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], f'Hash mismatch: {p}')
    branches = ['TH', 'TL', 'U0']
    rows, node_rows, hashes = [], [], {}
    for g, rep, branch in itertools.product(range(3), range(3), branches):
        path = root/f'03_positions/g{g}-r{rep}-{branch}/trajectory.jsonl.gz'
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            entries = [json.loads(line) for line in stream]
        setup = next(x['data'] for x in entries if x['kind'] == 'manifest')
        rounds = [x['data'] for x in entries if x['kind'] == 'round']
        require([r['round'] for r in rounds] == list(range(1, 11)), 'Incomplete rounds')
        network = setup['network']
        observers, advocates = set(network['observers']), set(setup['advocates'])
        follow = [set(x) for x in network['following']]
        fans = [set(x) for x in network['fans']]
        ordinary = set(range(len(follow))) - advocates
        require(len(follow) == 1000 and len(observers) == 50 and len(advocates) == 200, 'Design mismatch')
        require(not advocates & observers, 'Advocate overlaps observer')
        prepared = read(root/f'03_positions/g{g}-r{rep}_positions.json')
        require(advocates == set(prepared['positions'][branch]), 'Seeds mismatch')
        require(setup['profiles'] == prepared['profiles'], 'Profiles mismatch')
        require(network == read(root/f'03_positions/network-{g}.json')['data']['network'], 'Network mismatch')
        phi = [len(f & advocates)/len(f) for f in follow]
        union = set().union(*(fans[a] for a in advocates))
        covered = union & ordinary
        external_edges = sum(len(follow[i] & advocates) for i in ordinary)
        baseline = [p['baseline'] for p in setup['profiles']]
        rho = setup['config']['rho']
        speak = [rho*x/(1-x+rho*x) for x in baseline]
        structural = dict(
            IG=float(np.mean([phi[i] > .5 for i in ordinary])),
            observer_IG=float(np.mean([phi[i] > .5 for i in observers])),
            ordinary_coverage=len(covered)/len(ordinary),
            external_links_per_covered=external_edges/len(covered),
            covered_following_mean=float(np.mean([len(follow[i]) for i in covered])),
            cross_community_fraction=sum(network['communities'][i] != network['communities'][a]
                for i in ordinary for a in follow[i] & advocates)/external_edges,
            fan_total=sum(len(fans[a]) for a in advocates),
            baseline_total=sum(baseline[a] for a in advocates),
            initial_speakers=sum(speak[a] for a in advocates),
            activity_weighted_fans=sum(speak[a]*len(fans[a]) for a in advocates))
        require(abs(structural['IG']-lookup[g,rep,branch]['seed_diagnostics']['illusion']) < 1e-10, 'IG mismatch')
        messages, metrics = {}, []
        for r in rounds:
            messages.update({m['id']:m for m in r['published']})
            by_author = {m['author']:m for m in r['published']}
            selected = [n for n in r['nodes'] if n['node'] in observers]
            require(len(selected) == 50 and len(r['nodes']) == 1000, 'Node count mismatch')
            require(all(n['before']['stance'] == n['after']['stance'] == (-1 if n['node'] in advocates else 1)
                        for n in r['nodes']), 'Private stance mismatch')
            q = np.mean([n['after']['perceived_support'] < .5 for n in selected])
            j = np.mean([sum(messages[m]['stance']==-1 for m in n['reading']['read']) > len(n['reading']['read'])/2 for n in selected])
            d = np.mean([n['node'] in by_author and by_author[n['node']]['stance']==-1 for n in selected])
            s = np.mean([n['node'] not in by_author for n in selected])
            record = dict(round=r['round'], Q=float(q), J=float(j), D=float(d), S=float(s))
            original = lookup[g,rep,branch]['metrics'][r['round']-1]
            for key, old in [('Q','perceived_majority'),('J','feed_majority'),('D','public_conformity'),('S','opponent_silence')]:
                require(abs(record[key]-original[old]) < 1e-10, f'Metric mismatch {g}/{rep}/{branch}/{r["round"]}/{key}')
            metrics.append(record)
            for n in selected:
                node_rows.append(dict(graph=g, repeat=rep, branch=branch, round=r['round'], node=n['node'],
                    phi=phi[n['node']], following=len(follow[n['node']]),
                    perceived_support=n['after']['perceived_support'],
                    read_count=len(n['reading']['read']),
                    advocate_messages=sum(messages[m]['stance']==-1 for m in n['reading']['read'])))
        rows.append(dict(graph=g, repeat=rep, branch=branch, **structural,
            **{k:float(np.mean([m[k] for m in metrics[-3:]])) for k in ['Q','J','D','S']}, metrics=metrics))
        print(f'Validated g{g}-r{rep}-{branch}', flush=True)
    index = {(r['graph'],r['repeat'],r['branch']):r for r in rows}
    pairs = [('TH','TL'),('TH','U0'),('TL','U0')]
    contrasts = np.array([[[index[g,r,a]['Q']-index[g,r,b]['Q'] for a,b in pairs] for r in range(3)] for g in range(3)])
    rng = np.random.default_rng(20261120)
    draws = np.array([np.array([contrasts[g,rng.integers(0,3,3)] for g in range(3)]).mean(axis=(0,1)) for _ in range(10000)])
    signs = np.array(list(itertools.product([-1,1],repeat=9)))
    estimates = contrasts.mean(axis=(0,1))
    null = signs @ contrasts.reshape(9,3)/9
    pvals = np.mean(np.abs(null) >= np.abs(estimates)-1e-12,axis=0)
    adjusted = np.zeros(3)
    previous = 0
    for rank, i in enumerate(np.argsort(pvals)):
        previous = max(previous,min(1,float(pvals[i])*(3-rank)))
        adjusted[i] = previous
    comparisons = [dict(comparison=f'{a}-{b}', difference=float(estimates[i]),
        ci95=np.quantile(draws[:,i],[.025,.975]).tolist(),p_raw=float(pvals[i]),p_holm=float(adjusted[i])) for i,(a,b) in enumerate(pairs)]
    group_diffs = [dict(graph=g,repeat=r,TH=index[g,r,'TH']['Q'],TL=index[g,r,'TL']['Q'],difference=index[g,r,'TH']['Q']-index[g,r,'TL']['Q']) for g,r in itertools.product(range(3),repeat=2)]
    # Resource matching independently checked for all three pairs, within each group.
    matching = []
    for g,r in itertools.product(range(3),repeat=2):
        prepared = read(root/f'03_positions/g{g}-r{r}_positions.json')
        net = read(root/f'03_positions/network-{g}.json')['data']['network']
        for a,b in pairs:
            for attr in ['fan_total','baseline_total','initial_speakers']:
                x,y=index[g,r,a][attr],index[g,r,b][attr]
                gap=2*abs(x-y)/(x+y)
                require(gap <= .05+1e-12, 'Resource mismatch')
                matching.append(gap)
            quotas=[Counter((net['communities'][n],prepared['profiles'][n]['strength']) for n in prepared['positions'][v]) for v in [a,b]]
            require(quotas[0]==quotas[1], 'Community/strength mismatch')
    summary = {b:{k:float(np.mean([r[k] for r in rows if r['branch']==b])) for k in ['Q','J','D','S',*structural]} for b in branches}
    save(out/'run_features.json',rows)
    save(out/'observer_rounds.json',node_rows)
    save(out/'paired_differences.json',group_diffs)
    save(out/'statistics.json',dict(summary=summary,comparisons=comparisons,bootstrap_seed=20261120,repeats=10000,
        inference_scope='Conditional on three fixed networks; complete groups sampled within networks. Exact paired sign-flip assumes exchangeable signs under null. Exploratory structural comparisons are descriptive.'))
    save(out/'audit.json',dict(archive_files=len(manifest['files']),runs=27,rounds=270,observer_rounds=len(node_rows),
        independently_recomputed_metrics=['Q','J','D','S'],max_resource_gap=max(matching),source_hashes=hashes,
        base_commit='fb537a65216b816f2039b73282dd95afcc82b8ac',api_calls_made=0))
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
        'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors={'TH':'#2166ac','TL':'#d97732','U0':'#34856c'}
    def finish(fig,name):
        fig.savefig(out/name,dpi=200,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.7),layout='constrained')
    vals=[x['difference']*100 for x in group_diffs]
    ax.barh(range(9),vals,color=[colors['TH'] if v>=0 else colors['TL'] for v in vals])
    ax.set_yticks(range(9),[f'网络{x["graph"]+1} / 参数组{x["repeat"]+1}' for x in group_diffs]);ax.invert_yaxis()
    ax.axvline(0,color='gray',lw=1)
    ax.axvline(estimates[0]*100,color='black',ls='--',label=f'配对均值 {estimates[0]*100:.2f} 个百分点')
    for i,v in enumerate(vals):ax.text(v+(.3 if v>=0 else -.3),i,f'{v:+.2f}',ha='left' if v>=0 else 'right',va='center')
    ax.set_xlim(min(vals)-4,max(vals)+5);ax.set_xlabel('第8—10轮 Q 的 TH−TL 差值（百分点）')
    ax.set_title(f'连接位置效应并非每组一致：{sum(v>0 for v in vals)}/9 组为正\n精确配对检验 p={pvals[0]:.4f}；三项 Holm 校正 p={adjusted[0]:.4f}')
    ax.legend(loc='lower right');finish(fig,'01_paired_differences.png')
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for ax,key,label in zip(axes,['Q','J'],['多数误判比例 Q（%）','信息流多数比例 J（%）']):
        for b in branches:
            data=np.array([[m[key] for m in r['metrics']] for r in rows if r['branch']==b])
            ax.plot(range(1,11),data.mean(axis=0)*100,marker='o',color=colors[b],label=b)
        ax.axvspan(8,10,color='gray',alpha=.12);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].set_title('已有27条轨迹的逐轮均值｜每种位置9次运行，三张网络等权\n阴影标出主要汇总窗口，不表示置信区间')
    axes[0].legend(ncol=3);axes[1].set_xticks(range(1,11));axes[1].set_xlabel('轮次');finish(fig,'02_time_courses.png')
    features=[('IG','普通节点静态 IG（%）',100),('observer_IG','观察者静态覆盖（%）',100),
        ('ordinary_coverage','普通节点去重覆盖（%）',100),('external_links_per_covered','每名被覆盖普通节点关注的倡议者数',1),
        ('covered_following_mean','被覆盖普通节点的平均关注人数',1),('cross_community_fraction','跨社团外部传播连接比例（%）',100)]
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,(key,label,scale) in zip(axes.flat,features):
        for g,r in itertools.product(range(3),repeat=2):ax.plot(range(3),[index[g,r,b][key]*scale for b in branches],color='gray',alpha=.3,lw=.8)
        for i,b in enumerate(branches):
            values=[r[key]*scale for r in rows if r['branch']==b]
            ax.scatter([i]*9,values,color=colors[b],s=25,zorder=3)
            ax.scatter(i,np.mean(values),marker='_',s=300,color='black',zorder=4)
        ax.set_xticks(range(3),branches);ax.set_title(label);ax.grid(axis='y',alpha=.15)
    fig.suptitle('位置特征的描述性比较｜灰线连接同一配对组，黑横线表示均值\nIG 是搜索目标，不能将其组间差异作为独立机制证据',fontsize=13)
    finish(fig,'03_structural_features.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for ax,key,label in zip(axes,['IG','observer_IG'],['普通节点静态 IG（%）','观察者静态覆盖（%）']):
        for b in branches:
            rr=[r for r in rows if r['branch']==b]
            ax.scatter([r[key]*100 for r in rr],[r['Q']*100 for r in rr],color=colors[b],label=b,alpha=.8)
        ax.set_xlabel(label);ax.set_ylabel('末3轮多数误判 Q（%）');ax.legend();ax.grid(alpha=.15)
    fig.suptitle('结构暴露与动态误判｜每点一条运行，27点并非27个独立样本\n仅作探索性展示，不将跨组相关解释为因果效应',fontsize=12)
    finish(fig,'04_structure_and_perception.png')
    lines=['2026.11.20 邹佳怡 主实验一离线复核与位置特征分析','',
        '本次完成：建立个人分支；校验归档；从27条原始轨迹独立重算270轮的Q/J/公开顺从/沉默；生成4张图。未调用API，未新增实验。',
        f'203个归档文件哈希通过，9组匹配约束通过，资源最大相对差{max(matching)*100:.2f}%。',
        '主要汇总窗口为第8—10轮；N=1000，k=200，50名观察者。观察者均非倡议者。', '', '结果复核：']
    for b in branches:lines.append(f'{b}：Q={summary[b]["Q"]*100:.2f}%，J={summary[b]["J"]*100:.2f}%，公开顺从={summary[b]["D"]*100:.2f}%，沉默={summary[b]["S"]*100:.2f}%。')
    for c in comparisons:lines.append(f'{c["comparison"]}：{c["difference"]*100:+.2f}个百分点；95%配对bootstrap区间[{c["ci95"][0]*100:+.2f},{c["ci95"][1]*100:+.2f}]；原始p={c["p_raw"]:.4f}，Holm p={c["p_holm"]:.4f}。')
    lines += ['',f'TH−TL有{sum(v>0 for v in vals)}/9组为正。图01中的网络和组编号从1开始，对应原始g/r编号加1。',
        '结构特征均值（描述性，不作独立因果解释）：']
    for key,label,scale in features:lines.append(label+'：'+'；'.join(f'{b}={summary[b][key]*scale:.3f}' for b in branches))
    lines += ['按初始发言概率加权的粉丝总数（匹配未直接约束该项）：'+'；'.join(f'{b}={summary[b]["activity_weighted_fans"]:.2f}' for b in branches),
        '', '图表说明：',
        '01：9组配对差值，展示异质性；不能把3轮或50个观察者当成独立重复。',
        '02：各分支逐轮Q与J；灰色窗口为第8—10轮，不是置信区间。',
        '03：静态结构特征。普通节点覆盖排除倡议者；重叠指标定义为外部连接数/去重覆盖人数；跨社团比例按普通节点关注倡议者的边计算。',
        '04：全体普通节点IG、观察者覆盖与Q；TL的静态IG为零仍可能产生动态误判，不能将静态与动态多数等同。',
        '', '解释边界与下一步：',
        'TH按IG搜索，因此IG差异属操作检验。9组来自3张固定图，结论首先适用于这些图；结构特征差异是探索性描述。',
        '本次统计使用固定新bootstrap种子20261120，区间末位可能与原归档不同；精确符号翻转检验需要配对差符号在零假设下可交换。',
        '先针对TH不优于TL和差异接近零的组检查观察者暴露与实际阅读，再固定新增组数和搜索预算开展补充。保持50观察者，80观察者另设稳健性条件。',
        '待定方案：每张图增加至10组并保留TH/TL/U0，共新增63条10轮轨迹；实际运行前确认API配置、技术试跑和预算。此计划尚未执行。',
        '', '复现：python scripts/analyze_exp1_positions.py --repo . --output 新输出目录',
        '附带audit.json、statistics.json、run_features.json、paired_differences.json、observer_rounds.json。']
    (out/'飞书阶段记录.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'summary':summary,'comparisons':comparisons},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
