"""Portable HTML index over generated research artifacts; no external resources."""
from pathlib import Path
import html
import json
import pandas as pd


def build_overview(output='results'):
    root=Path(output); sections=[]
    for mode in ['reproduction','strict']:
        directory=root/mode
        if not (directory/'metadata.json').exists(): continue
        m=json.loads((directory/'metadata.json').read_text()); d=pd.read_csv(directory/'comparison.csv')
        columns=['method','mean_hours','backlog_total','mean_coverage']
        table=d[columns].rename(columns=dict(method='Method',mean_hours='Service proxy hours',backlog_total='Unassigned tickets',mean_coverage='Coverage')).copy()
        table['Coverage']=table.Coverage.map(lambda x:f'{100*x:.2f}%')
        images=[]
        captions={
            'comparison':'Assigned-ticket service proxy. Read together with coverage and backlog.',
            'optimizers_curse':'Planning proxies are optimistic; held-out scoring changes the comparison.',
            'validation':'Validation-only tuning. At z=0, the selected policy is shrinkage-only.',
            'fairness':'Lower workload inequality can come with additional unassigned tickets.',
            'capacity_duals':'Marginal value per ticket slot; strict-mode duals hold backlog fixed.',
            'cross_training':'Central new-pair cost assumptions can yield no benefit from training.',
            'cell_stability':'Observed employee-category associations are not causal employee effects.',
            'employee_spread':'Shrunk training costs vary among eligible employee-category pairs.',
            'capacity_sensitivity':'Service proxies change with capacity; consult sensitivity.csv for backlog.',
            'workflow_map':'Descriptive observed status transitions; not scheduling precedence constraints.'}
        for name,caption in captions.items():
            if (directory/(name+'.png')).exists():
                images.append(f'<figure><a href="{mode}/{name}.svg"><img loading="lazy" src="{mode}/{name}.png" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>')
        explanation='Completed-ticket cohort; hindsight weekly roster; creation-date fit.' if mode=='reproduction' else 'All category arrivals; prior observed roster; resolution cutoff; complete weeks.'
        sections.append(f'''<section id="{mode}"><h2>{mode.capitalize()}</h2>
<p>{explanation} {m['test_weeks']} test weeks. Validation selected k={m['selected_k']}, z={m['selected_z']}.</p>
<p><a href="{mode}/RESULTS.md">Full interpretation</a> · <a href="{mode}/comparison.csv">Comparison CSV</a> · <a href="{mode}/metadata.json">Audit and provenance</a> · <a href="{mode}/cross_training.csv">Training scenarios</a></p>
{table.to_html(index=False,float_format=lambda x:f'{x:.2f}',border=0)}<div class="figures">{''.join(images)}</div></section>''')
    if not sections: raise FileNotFoundError('Run the experiments before building the overview')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robust ticket routing — research results</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#17252b;background:#fafaf8;max-width:1180px;margin:40px auto;padding:0 24px}h1{font-size:32px;line-height:1.15}h2{font-size:24px;margin-top:48px}a{color:#245d7b}p{max-width:85ch}.note{border-left:3px solid #245d7b;padding:8px 18px;background:#eef3f4}.figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}figure{margin:24px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#46545c}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 12px;border-bottom:1px solid #d8dedd;text-align:right}th:first-child,td:first-child{text-align:left}section{scroll-margin-top:24px}@media(max-width:600px){body{padding:0 12px}.figures{display:block}table{font-size:12px}td,th{padding:5px}}
</style><header><p>6.C571 · Sam Chen, Christina Ding, Leo Sun</p><h1>Robust ticket routing</h1>
<p>A complete real-data implementation of routing LPs, uncertainty penalties, fairness and cross-training.</p>
<p class="note">These are retrospective observational proxies, not measured outcomes under alternative assignments. Compare backlog first. Historical unknown assignments are missing labels, not verified operational backlog. Shifted geometric hours are not arithmetic mean resolution times.</p>
<nav><a href="#reproduction">Plan reproduction</a> · <a href="#strict">Stricter scenario</a></nav></header>'''+''.join(sections)+'''<footer><p>Source: Mendeley Help Desk Tickets v3, DOI 10.17632/btm76zndnt.3. Source hashes and exact settings are saved with each scenario. Generated with OpenAI Codex assistance; independently verify course policy before submission.</p></footer></html>'''
    target=root/'index.html'; target.write_text(page); return target
