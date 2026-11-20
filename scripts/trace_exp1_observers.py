"""Read-only exploratory follow-up of paired observer exposure. No model calls."""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=a.repo/'experiments/2026-11-16/03_positions'
    source=read(root/'results.json')
    refs={(r['graph'],r['repeat'],r['branch']):r for r in source}
    data=[];details={};hashes={};observers={}
    for g in range(3):
        for rep in range(3):
            for branch in ['TH','TL']:
                p=root/f'g{g}-r{rep}-{branch}/trajectory.jsonl.gz'
                hashes[p.parent.name]=hashlib.sha256(p.read_bytes()).hexdigest()
                messages={};history=defaultdict(list);seen=defaultdict(set)
                with gzip.open(p,'rt',encoding='utf-8') as stream:
                    for line in stream:
                        item=json.loads(line);r=item['data']
                        if item['kind']=='manifest':
                            obs=set(r['network']['observers']); advocates=set(r['advocates']);network=r['network']
                            observers[g,rep,branch]=obs
                        if item['kind']!='round':continue
                        messages.update({m['id']:m for m in r['published']})
                        for n in r['nodes']:
                            i=n['node']
                            if i not in obs:continue
                            ids=n['reading']['read'];context=n['reading']['context']
                            # Unique per exposure: original shown as read AND context counted once here.
                            # This is an analyst's text-exposure diagnostic, not the LLM's estimator.
                            full=list(dict.fromkeys(ids+context));history[i].extend(full);seen[i].update(full)
                            frac=lambda keys:float(np.mean([messages[k]['stance']==-1 for k in keys])) if keys else None
                            q=n['after']['perceived_support']<.5
                            row=dict(graph=g,repeat=rep,branch=branch,round=r['round'],node=i,
                                phi=len(set(network['following'][i])&advocates)/len(network['following'][i]),
                                Q=int(q),support=n['after']['perceived_support'],
                                J=int(sum(messages[k]['stance']==-1 for k in ids)>len(ids)/2),
                                read_n=len(ids),read_opposition=frac(ids),
                                current_with_context_opposition=frac(full),
                                historical_exposure_opposition=frac(history[i]),
                                historical_unique_opposition=frac(sorted(seen[i])),
                                historical_unique_messages=len(seen[i]),
                                historical_unique_authors=len({messages[k]['author'] for k in seen[i]}),
                                carried_forward=not ids,near_half=abs(n['after']['perceived_support']-.5)<=.0500000001)
                            data.append(row)
                            if r['round']>=8:
                                details[g,rep,branch,i,r['round']]=dict(**row,
                                    reason=(n.get('perception') or {}).get('reason',''),
                                    evidence_ids=(n.get('perception') or {}).get('evidence_ids',[]),
                                    read_messages=[{k:messages[mid][k] for k in ['id','author','stance','text']} for mid in ids],
                                    context_messages=[{k:messages[mid][k] for k in ['id','author','stance','text']} for mid in context])
                print('Read',p.parent.name,flush=True)
    group=defaultdict(list);node=defaultdict(list);time=defaultdict(list)
    for r in data:
        time[r['graph'],r['repeat'],r['branch'],r['round']].append(r)
        if r['round']>=8:
            group[r['graph'],r['repeat'],r['branch']].append(r)
            node[r['graph'],r['repeat'],r['branch'],r['node']].append(r)
    summary=[];paired=[]
    fields=['Q','J','read_n','read_opposition','current_with_context_opposition','historical_exposure_opposition',
            'historical_unique_opposition','historical_unique_messages','historical_unique_authors','support','carried_forward','near_half']
    def means(rows):
        return {k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in fields}
    for key,rows in sorted(group.items()):
        g,rep,b=key
        assert len(rows)==150
        v=means(rows)
        assert abs(v['Q']-refs[key]['summary']['perceived_majority'])<1e-10
        summary.append(dict(graph=g,repeat=rep,branch=b,**v))
    for g in range(3):
        for rep in range(3):
            assert observers[g,rep,'TH']==observers[g,rep,'TL']
            for i in sorted(observers[g,rep,'TH']):
                h,l=means(node[g,rep,'TH',i]),means(node[g,rep,'TL',i])
                paired.append(dict(graph=g,repeat=rep,node=i,TH=h,TL=l,
                    Q_difference=h['Q']-l['Q'],support_difference=h['support']-l['support']))
    lookup={(r['graph'],r['repeat'],r['branch']):r for r in summary}
    diffs=[(lookup[g,r,'TH']['Q']-lookup[g,r,'TL']['Q'],g,r) for g in range(3) for r in range(3)]
    # Predeclared descriptive selection: negative, all absolute differences <= 1pp, largest positive.
    selected=sorted({(g,r) for d,g,r in diffs if d<0 or abs(d)<=.01}|{max(diffs)[1:]})
    thresholds=[]
    for g,rep in selected:
        for t in [.4,.45,.5,.55,.6]:
            h=np.mean([r['support']<t for r in group[g,rep,'TH']]);l=np.mean([r['support']<t for r in group[g,rep,'TL']])
            thresholds.append(dict(graph=g,repeat=rep,threshold=t,TH=float(h),TL=float(l),difference=float(h-l)))
    cases=[]
    for g,rep in selected:
        p=[r for r in paired if (r['graph'],r['repeat'])==(g,rep)]
        chosen={min(p,key=lambda x:(x['Q_difference'],x['node']))['node'],max(p,key=lambda x:(x['Q_difference'],-x['node']))['node']}
        for i in sorted(chosen):
            cases.append(dict(graph=g,repeat=rep,node=i,selection='minimum/maximum tail paired Q difference; ties smallest node',
                trajectory=[details[g,rep,b,i,t] for b in ['TH','TL'] for t in [8,9,10]]))
    save(a.output/'group_exposure_summary.json',summary)
    save(a.output/'paired_observer_summary.json',paired)
    save(a.output/'observer_round_exposures.json',data)
    save(a.output/'threshold_diagnostic.json',thresholds)
    save(a.output/'selected_observer_cases.json',cases)
    save(a.output/'audit.json',dict(runs=18,rounds=180,observer_rounds=len(data),paired_observers=len(paired),
         summary_Q_matches_archive=True,paired_observers_match=True,api_calls=0,source_hashes=hashes,
         selected_groups=selected,selection_rule='negative TH-TL; abs difference <=1pp; maximum positive',
         limitations='Post-hoc selected cases, descriptive only. Repeated rounds/observers not independent samples. LLM reasons not causal explanations.'))
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'TH':'#2166ac','TL':'#d97732'}
    def finish(fig,name):fig.savefig(a.output/name,dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(len(selected),3,figsize=(14,3*len(selected)),layout='constrained')
    for rr,(g,rep) in enumerate(selected):
        for col,(field,label) in enumerate([('Q','多数误判 Q'),('J','信息流多数 J'),('historical_unique_opposition','历史去重文本反对占比')]):
            ax=axes[rr,col]
            for b in ['TH','TL']:
                vals=[means(time[g,rep,b,t])[field] for t in range(1,11)]
                ax.plot(range(1,11),np.array(vals)*100,label=b,color=colors[b],marker='o',markersize=3)
            ax.axvspan(8,10,color='gray',alpha=.1);ax.set_ylim(0,100);ax.grid(alpha=.15)
            ax.set_title(f'网络{g+1}/组{rep+1} · {label}');ax.set_ylabel('%');ax.set_xticks([1,3,5,8,10])
            if rr==0:ax.legend(ncol=2)
    fig.suptitle('事后选取的反向、接近零及最大正向组｜灰区为末3轮\n历史文本包含引用原帖并按消息编号去重；该比例不是LLM的判断公式',fontsize=13)
    finish(fig,'05_selected_group_exposure.png')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    labels=[];counts=[]
    for g,rep in selected:
        pp=[r for r in paired if (r['graph'],r['repeat'])==(g,rep)]
        counts.append([sum(r['Q_difference']< -1e-10 for r in pp),sum(abs(r['Q_difference'])<=1e-10 for r in pp),sum(r['Q_difference']>1e-10 for r in pp)])
        labels.append(f'网络{g+1}/组{rep+1}')
    bottom=np.zeros(len(selected))
    for j,(lab,color) in enumerate([('TH下误判更少','#d97732'),('两分支相同','#aab2bb'),('TH下误判更多','#2166ac')]):
        v=np.array(counts)[:,j];axes[0].barh(labels,v,left=bottom,label=lab,color=color)
        for i,n in enumerate(v):
            if n:axes[0].text(bottom[i]+n/2,i,str(n),ha='center',va='center')
        bottom+=v
    axes[0].set_xlabel('观察者人数（每组同一批50人）');axes[0].set_title('末3轮个人误判次数的配对比较');axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=1)
    for g,rep in selected:
        rr=[r for r in thresholds if (r['graph'],r['repeat'])==(g,rep)]
        axes[1].plot([r['threshold'] for r in rr],[r['difference']*100 for r in rr],marker='o',label=f'网络{g+1}/组{rep+1}')
    axes[1].axvline(.5,color='black',ls='--');axes[1].axhline(0,color='gray',lw=1)
    axes[1].set_xlabel('诊断阈值：估计支持率 < 此值');axes[1].set_ylabel('TH−TL（百分点）');axes[1].set_title('仅0.5是原定义的多数误判阈值');axes[1].legend(fontsize=9);axes[1].grid(alpha=.15)
    fig.suptitle('整体差值可能掩盖观察者之间的相反变化｜探索性诊断，不新增显著性检验',fontsize=12)
    finish(fig,'06_observer_pairs_and_thresholds.png')
    lines=['2026.11.20 邹佳怡 主实验一观察者追踪（第二阶段）','',
        '本次读取TH/TL的18条完整轨迹，共180轮、9000条观察者轮次记录；同组观察者名单一致，末3轮Q复算与归档一致。未调用API。',
        '事后选择规则：TH−TL为负、绝对差不超过1个百分点，以及最大正向差。用于寻找线索，不是预注册验证。',
        '表中所有比例都是末3轮观察者平均值。当前阅读反对占比对空阅读记录取缺失后求均值；Q和J保留空阅读。历史比例是截至当轮的累计文本诊断，包含引用原帖；按消息编号去重不等于按作者或候选文本去重。','']
    for (g,rep),cc in zip(selected,counts):
        h,l=lookup[g,rep,'TH'],lookup[g,rep,'TL']
        lines.append(f'网络{g+1}/参数组{rep+1}（原始g{g}-r{rep}）：')
        for k,label in [('Q','多数误判Q'),('J','信息流多数J'),('read_opposition','当轮阅读反对占比'),('historical_unique_opposition','历史去重文本反对占比'),('support','平均估计支持率'),('near_half','估计支持率处于[0.45,0.55]的比例'),('carried_forward','空阅读沿用比例')]:
            lines.append(f'  {label}：TH={h[k]*100:.2f}%，TL={l[k]*100:.2f}%，差={100*(h[k]-l[k]):+.2f}个百分点。')
        lines.append(f'  同一批50人：TH下误判更少{cc[0]}人，相同{cc[1]}人，更多{cc[2]}人。')
    lines += ['', '结论边界：',
        '末3轮总体Q差接近零不表示每个观察者都没有受到影响；需要查看个体变化的相互抵消。',
        '局部网络结构、消息数量、历史文本与感知不是同一个量。少数作者重复发言、评论受众、抽样和记忆都可能造成差异，但当前描述性比较不能区分因果贡献。',
        'LLM提供的reason仅作为原始回答保存，不应视作真实因果机制证明；案例按预先写明的规则选取，不能代表全部观察者。',
        '不修改多数误判定义：阈值扫描只是检验分类结果对临界判断的敏感性，不挑选最显著阈值。',
        '', '下一步建议：',
        '先保持N=1000/k=200/50观察者及原模型条件不变，以固定预算新增独立参数/随机过程组；每个组在运行前冻结TH/TL/U0配置。',
        '记录观察者静态暴露、按活跃度加权的传播资源以及个人历史曝光；原始组和新增组分别报告后再合并。',
        '若要区分位置选择与随机过程，另设计固定同一套位置和个人参数、只改变过程随机种子的重复；其统计层级不能冒充新增独立网络。',
        '80观察者涉及LLM驱动节点数量变化，应作为单独的稳健性条件。',
        '', '文件：05选中组轨迹；06个体配对与阈值诊断；JSON保存组汇总、逐观察者记录及可追溯原文案例。',
        '复现：python scripts/trace_exp1_observers.py --repo . --output 新目录']
    (a.output/'飞书观察者追踪记录.txt').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'completed':True,'runs':18,'observer_rounds':len(data),'selected_groups':selected}),flush=True)


if __name__=='__main__':main()
