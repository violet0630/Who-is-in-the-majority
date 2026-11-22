"""V2 position supplement: prepare inputs, offline smoke test, or one explicit API run."""
import argparse
import ast
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def save(p, data):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def check(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--batch', type=Path, required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--execute', action='store_true')
    ap.add_argument('--task')
    args = ap.parse_args()
    repo, batch = args.repo.resolve(), args.batch.resolve()
    sys.path.insert(0, str(repo))
    from majority.artifacts import load_network, load_bank
    from majority.models import Profile, RunConfig, Coefficients
    from majority.simulation import Simulation
    from majority.randomness import RandomStreams
    from majority.analysis import measure
    from majority.perception import PerceptionResult
    old = repo/'experiments/2026-11-20-zjy-extension-prepared'
    if args.prepare:
        check(not batch.exists(), 'Batch already exists')
        tasks = read(old/'tasks.json')
        check(len(tasks) == 63 and len({t['task_id'] for t in tasks}) == 63, 'Task coverage mismatch')
        settings = dict(read(old/'protocol.json')['settings'], rounds=15, tail_rounds=5, read_limit=11)
        inputs = sorted({t['prepared'] for t in tasks})
        for name in inputs:
            data = read(old/name)
            network = load_network(repo/tasks[next(i for i,t in enumerate(tasks) if t['prepared']==name)]['network'])
            check(len(data['profiles']) == 1000 and len(network.observers) == 50, 'Population mismatch')
            for branch in ('TH','TL','U0'):
                nodes = data['positions'][branch]
                check(len(nodes) == len(set(nodes)) == 200 and not set(nodes)&network.observers, 'Invalid advocates')
            for task in [t for t in tasks if t['prepared']==name]:
                check(sha(old/name) == task['prepared_sha256'], 'Frozen input changed')
        sources = list((repo/'majority').rglob('*.py')) + list((repo/'majority/prompts').glob('*.md'))
        sources += [repo/'scripts/majority_study.py', repo/'configs/study.json', repo/'configs/rules_v2.json',
                    repo/'experiments/2026-11-16/01_texts_50/bank.json']
        sources += [repo/t['network'] for t in tasks]
        sources += [old/name for name in inputs]
        batch.mkdir(parents=True)
        save(batch/'settings.json', dict(settings=settings, tasks=tasks,
             comments=dict(max_comments=3, comment_alpha=1., comment_beta=9.),
             public_stance='last_published_message_per_author', perception='local_proxy',
             primary_rounds=[11,12,13,14,15], status='prepared_no_API_runs',
             runner_sha256=sha(Path(__file__)),
             sources={p.relative_to(repo).as_posix():sha(p) for p in sources}))
        print(json.dumps(dict(status='prepared', groups=21, tasks=63, rounds=15, tail_rounds=5)))
        return
    frozen = read(batch/'settings.json')
    check(sha(Path(__file__)) == frozen['runner_sha256'], 'Runner changed')
    for name, expected in frozen['sources'].items():
        check(sha(repo/name)==expected, f'Frozen source changed: {name}')
    config = RunConfig(rounds=15, rho=6, read_limit=11, **frozen['comments'])
    check((config.rounds,config.window,config.read_limit)==(15,2,11), 'V2 config mismatch')
    if args.smoke:
        # Load the exact committed observer wrapper without importing optional API libraries.
        source = repo/'scripts/majority_study.py'
        node = next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
                    if isinstance(n,ast.ClassDef) and n.name=='ObserverPerception')
        scope = dict(json=json, PerceptionResult=PerceptionResult)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        from examples.demo import demo_inputs
        from majority.models import Network
        network, profiles, bank = demo_inputs()
        network = Network(network.communities, tuple(frozenset(set(range(8))-{i}) for i in range(8)), network.observers)
        profiles = tuple(replace(p,baseline=.95) for p in profiles)
        class FakeObserver:
            def estimate(self, request):
                return PerceptionResult(.7, '{}', model='offline_stub_not_research')
        observed_multi = observed_11 = local_updates = 0
        for seed in range(4):
            sim = Simulation(network, profiles, frozenset({0}), bank, Coefficients((1,0,-1,0),.5,1),
                config, scope['ObserverPerception'](network.observers,bank,FakeObserver()), RandomStreams(seed), 'SMOKE')
            metrics=[]
            for record in sim.run():
                counts=Counter(m.author for m in record.published)
                observed_multi += sum(c>1 for c in counts.values())
                check(max(counts.values(),default=0)<=3, 'Comment bound failed')
                metrics.append(asdict(measure(record,network,sim.messages)))
                for row in record.nodes:
                    check(len(row.reading.read)<=11, 'Read cap failed')
                    observed_11 += len(row.reading.read)==11
                    if row.node not in network.observers:
                        authors={}
                        for mid in row.reading.read:
                            msg=sim.messages[mid]; authors[msg.author]=int(msg.stance)
                        expected=sum(s==1 for s in authors.values())/len(authors) if authors else row.before.perceived_support
                        check(abs(row.after.perceived_support-expected)<1e-12,'Local perception mismatch')
                        local_updates += row.after.perceived_support != row.before.perceived_support
            check([m['round'] for m in metrics[-5:]]==[11,12,13,14,15], 'Tail mismatch')
        check(observed_multi>0 and observed_11>0 and local_updates>0, 'Smoke failed to exercise rules')
        report=dict(status='offline_smoke_passed', API_calls=0, simulated_runs=4,
            rounds_per_run=15, tail_rounds=[11,12,13,14,15], multi_message_author_rounds=observed_multi,
            reads_of_11=observed_11, local_state_changes=local_updates, research_results=False)
        save(batch/'offline_check.json', report)
        print(json.dumps(report))
        return
    check((batch/'offline_check.json').is_file(), 'Offline check required')
    task=next((t for t in frozen['tasks'] if t['task_id']==args.task),None)
    check(task is not None, 'Specify a valid --task')
    output=batch/'runs'/task['task_id']
    check(not output.exists(), 'Existing task: review recovery separately')
    from scripts.majority_study import run_one
    from majority.llm import load_settings, canonical_model
    llm, _ = load_settings(repo/'.env')
    check(canonical_model(llm.model)=='deepseek-flash' and llm.max_tokens==4096,'Model configuration differs')
    runtime=dict(base_url=llm.base_url,model=llm.model,max_tokens=llm.max_tokens,thinking='disabled',temperature=0)
    if (batch/'runtime.json').exists():
        check(read(batch/'runtime.json')==runtime,'Batch model configuration changed')
    else:
        save(batch/'runtime.json',runtime)
    data=read(old/task['prepared'])
    network=load_network(repo/task['network'])
    _,bank=load_bank(repo/'experiments/2026-11-16/01_texts_50/bank.json')
    # The existing run_one uses these RunConfig defaults; verify before allowing API calls.
    check(all(getattr(RunConfig(),k)==v for k,v in frozen['comments'].items()),'Comment defaults changed')
    output.mkdir(parents=True)
    save(output/'input_identity.json',task)
    result=run_one(network,tuple(Profile(**p) for p in data['profiles']),frozenset(data['positions'][task['branch']]),
        bank,frozen['settings'],task['group_seed'],task['branch'],output,True)
    check(len(result['metrics'])==15 and result['tail_rounds']==5,'Incomplete result')
    print(json.dumps(dict(status='finished_pending_audit',task=task['task_id'])))


if __name__=='__main__':
    main()
