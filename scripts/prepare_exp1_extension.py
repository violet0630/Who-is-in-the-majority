"""Offline preparation: frozen networks, original search function, no LLM imports/calls.
The AST loader executes ONLY the reviewed select_positions function, preserving its
exact algorithm while avoiding the original runner's optional OpenAI imports.
"""
import argparse
import ast
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import itertools
import json
from math import exp
from pathlib import Path
from random import Random
import sys
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();repo=args.repo.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(repo))
    from majority.artifacts import load_network
    from majority.preparation import draw_profiles,seed_diagnostics
    source=repo/'scripts/majority_study.py'
    fn=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='select_positions')
    scope=dict(np=np,Random=Random,exp=exp,asdict=asdict,seed_diagnostics=seed_diagnostics)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
    search=scope['select_positions']
    archive=repo/'experiments/2026-11-16'
    config=read(repo/'configs/initialized_20261116.json')
    inputs=out/'inputs';inputs.mkdir()
    protocol=dict(status='offline_prepared_not_run',hash_policy='SHA256 of text bytes with CRLF normalized to LF',base_commit='fb537a65216b816f2039b73282dd95afcc82b8ac',
        objective='Confirm position effect using 21 new matched groups on the SAME three fixed networks',
        size=1000,k=200,observers=50,new_groups_per_network=7,new_repeat_indices=list(range(3,10)),
        branches=['TH','TL','U0'],new_trajectories=63,existing_trajectories=27,settings=config['settings'],
        search=dict(steps=20000,restarts=4,rho=6,algorithm='unchanged select_positions from scripts/majority_study.py'),
        seeds='profiles=16000+graph*10+repeat; search=17000+graph*10+repeat; process=18000+graph*10+repeat',
        primary='mean observer Q rounds 8-10; report new-only first, then old/new and pooled',
        comparisons=[['TH','TL'],['TH','U0'],['TL','U0']],bootstrap_repeats=10000,analysis_seed=20261120,
        inference='Graph-stratified whole-group paired bootstrap; paired sign-flip Monte Carlo for 21 or 30 groups, with documented exchangeability assumption and Holm within 3 comparisons. Do not enumerate 2^30 signs.',
        stopping='Fixed preparation budget; no selection or extension based on Q or statistical significance. Pairing failure is reported; do not silently redraw.',
        observer_count_80='Not part of this batch',technical_trial='First new group on each network, all 3 branches: 9 full-length trajectories; remaining 54 held until technical/cost review',
        trial_acceptance='10 rounds, immutable inputs verified, metrics reproducible, only observers call model, no unresolved response errors; no condition on Q size',
        technical_trials_inclusion='Include in final batch only with unchanged inputs/code/model/settings and complete successful trajectories',
        api_authorized=False)
    source_paths=list((repo/'majority').rglob('*.py'))+list((repo/'majority/prompts').glob('*.md'))+[source,repo/'configs/study.json',repo/'configs/initialized_20261116.json',archive/'01_texts_50/bank.json']+[archive/f'03_positions/network-{g}.json' for g in range(3)]
    source_hashes={str(p.relative_to(repo)).replace('\\','/'):sha(p) for p in source_paths}
    save(out/'protocol.json',protocol);save(out/'source_hashes.json',source_hashes)
    tasks=[];diagnostics=[]
    for g in range(3):
        network=load_network(archive/f'03_positions/network-{g}.json')
        for r in range(3,10):
            profiles=draw_profiles(1000,Random(16000+g*10+r))
            positions,diag=search(network,profiles,200,17000+g*10+r,steps=20000,restarts=4,rho=6)
            gaps={}
            for b,s in positions.items():
                if len(s)!=200 or s & network.observers:raise ValueError('Invalid seeds')
                boosted={i:6*profiles[i].baseline/(1+5*profiles[i].baseline) for i in s}
                diag[b]['activity_weighted_fans']=sum(boosted[i]*len(network.fans[i]) for i in s)
            for a,b in itertools.combinations(['TH','TL','U0'],2):
                qa=Counter((network.communities[i],profiles[i].strength) for i in positions[a]);qb=Counter((network.communities[i],profiles[i].strength) for i in positions[b])
                if qa!=qb:raise ValueError('Quotas mismatch')
                for field in ['fan_total','baseline_total','initial_expected_advocate_speakers']:
                    x,y=diag[a][field],diag[b][field];gap=2*abs(x-y)/(x+y)
                    if gap>.05+1e-12:raise ValueError('Resource mismatch')
                    gaps[f'{a}-{b}:{field}']=gap
            valid=diag['TH']['illusion']>diag['TL']['illusion']
            payload=dict(graph=g,repeat=r,profiles=[asdict(p) for p in profiles],positions={b:sorted(s) for b,s in positions.items()},diagnostics=diag,
                seeds=dict(profiles=16000+g*10+r,search=17000+g*10+r,process=18000+g*10+r),matching=gaps,paired_position_valid=valid)
            p=inputs/f'g{g}-r{r}_positions.json';save(p,payload)
            diagnostics.append(dict(graph=g,repeat=r,valid=valid,IG_gap=diag['TH']['illusion']-diag['TL']['illusion'],max_match_gap=max(gaps.values())))
            order=['TH','TL','U0'];offset=r%3;order=order[offset:]+order[:offset]
            for b in order:
                tasks.append(dict(task_id=f'g{g}-r{r}-{b}',graph=g,repeat=r,branch=b,group_seed=18000+g*10+r,
                    network=f'experiments/2026-11-16/03_positions/network-{g}.json',prepared=f'inputs/{p.name}',prepared_sha256=sha(p),
                    phase='technical_trial' if r==3 else 'remaining',paired_position_valid=valid))
            print(json.dumps(diagnostics[-1]),flush=True)
    save(out/'tasks.json',tasks);save(out/'preparation_diagnostics.json',diagnostics)
    costs=[]
    for p in sorted((archive/'03_positions').glob('g*-r*-*/calls.jsonl.gz')):
        c=dict(run=p.parent.name,actual_calls=0,reused_rows=0,known_input_tokens=0,known_output_tokens=0,known_cached_input_tokens=0,missing_usage=0,statuses={})
        with gzip.open(p,'rt',encoding='utf-8') as f:
            for line in f:
                d=json.loads(line)['data']
                if d.get('reused'):c['reused_rows']+=1;continue
                c['actual_calls']+=1;status=d.get('status','unknown');c['statuses'][status]=c['statuses'].get(status,0)+1
                resp=d.get('response') or {};u=resp.get('usage') if isinstance(resp,dict) else None
                if not u:c['missing_usage']+=1;continue
                c['known_input_tokens']+=u.get('prompt_tokens') or 0;c['known_output_tokens']+=u.get('completion_tokens') or 0
                c['known_cached_input_tokens']+=u.get('prompt_cache_hit_tokens') or (u.get('prompt_tokens_details') or {}).get('cached_tokens') or 0
        costs.append(c)
    estimates={}
    for n in [9,54,63]:
        estimates[str(n)]={k:round(sum(c[k] for c in costs)/len(costs)*n) for k in ['actual_calls','known_input_tokens','known_output_tokens','known_cached_input_tokens','missing_usage']}
    save(out/'historical_usage_and_projection.json',dict(historical_runs=costs,projections=estimates,
        caveat='Historical actual calls exclude reused rows but include failed/invalid attempts with reported usage. Missing usage not zero. Historical retries inflate estimates; full-history token growth, model and cache changes may alter cost. No currency estimate without gateway prices.',
        nominal_first_attempt_request_ceiling_per_trajectory=500))
    lines=['主实验一补充验证 离线准备完成','',
        '状态：新增实验尚未运行，未调用API。准备产物用于固定条件和随机种子，不依据新实验结果筛选。',
        f'共21个新增配对组、63条任务，结构有效配对{sum(d["valid"] for d in diagnostics)}/21。',
        f'IG高低差范围{min(d["IG_gap"] for d in diagnostics):.4f}—{max(d["IG_gap"] for d in diagnostics):.4f}，匹配最大相对差{max(d["max_match_gap"] for d in diagnostics)*100:.2f}%。',
        '沿用3张原网络，N=1000/k=200/50观察者、冻结文本库和原行为参数；每图新增repeat=3..9，不与已有0..2混淆。',
        '技术试跑固定为每张图第1个新增组的TH/TL/U0，共9条10轮轨迹。检查完整性、模型标识、恢复、实际费用后再运行其余54条。',
        '原有结果、独立新增结果分别报告，再报告合并；不以显著与否决定停止。bootstrap以完整配对组为单位，在图内重采样。',
        '', '历史调用用量的线性外推（仅供预算，非计费承诺）：']
    for n,e in estimates.items():lines.append(f'{n}条：实际尝试约{e["actual_calls"]}次；已知输入约{e["known_input_tokens"]/1e6:.2f}百万tokens，已知输出约{e["known_output_tokens"]/1e6:.2f}百万tokens。')
    lines += ['历史重试和网关差异会影响以上估算；缺失usage不等于零；收费还取决于网关输入/输出/缓存价格。',
        '', '前置条件：当前解释器缺少openai依赖，需要独立环境；需在本地配置有效API，不能把密钥提交。',
        'tasks.json与inputs/*.json保存任务及冻结名单；source_hashes.json保存源码/提示/文本/网络哈希。',
        '个人分支尚未推送。正式运行需使用受控执行入口核验哈希、指定任务并避免覆盖结果。']
    (out/'补充实验准备说明.txt').write_text('\n'.join(lines),encoding='utf-8')
    save(out/'preparation_manifest.json',{str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*.json'))})
    print(json.dumps(dict(prepared_groups=len(diagnostics),tasks=len(tasks),cost_projection=estimates)),flush=True)


if __name__=='__main__':main()
