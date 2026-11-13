"""E1–E9, validation and audit scenarios. No preliminary results are embedded."""
from pathlib import Path
from dataclasses import asdict
import json
import math
import platform
from time import perf_counter
import numpy as np
import pandas as pd
import scipy
from .data import load_data, split_weeks, workflow, stamp, CATEGORIES
from .estimates import fit, oracle
from .evaluate import score, route_week, summarize
from .baselines import baseline
from .models import cross_train_mip, route_lp


def select_validation(grid):
    best=grid.sort_values(['backlog','penalized_log_score','k','z']).iloc[0]
    return int(best.k),float(best.z)


def evaluation_cohort(outcomes,start,end,strict=False):
    d=outcomes[(outcomes.created>=stamp(start))&(outcomes.created<stamp(end))]
    return d[d.resolved<stamp(end)] if strict else d


def roster_for(arrivals,tickets,est,week,hindsight):
    if hindsight:
        ids=set(tickets.assignee.dropna())
    else:
        week=stamp(week)
        prior=arrivals[(arrivals.created<week)&(arrivals.created>=week-pd.Timedelta(weeks=8))]
        if 'assigned_at' in prior: prior=prior[prior.assigned_at<week]
        ids=set(prior.assignee.dropna())
    return sorted(i for i in ids if i in est.capacity and est.eligible.loc[i].any())


def run(raw='data/raw',output='results',mode='reproduction',seeds=20,min_tickets=50):
    start_time=perf_counter(); out=Path(output)/mode; out.mkdir(parents=True,exist_ok=True)
    arrivals,outcomes,audit=load_data(raw)
    strict=mode=='strict'
    if mode not in ['reproduction','strict']: raise ValueError(mode)
    cohort=arrivals if strict else outcomes
    fit_arrivals=arrivals if strict else outcomes
    capacity_mode='calendar' if strict else 'nonzero'
    settings=dict(strict=strict,capacity_mode=capacity_mode)
    complete=strict
    train=outcomes[outcomes.created<stamp('2022-07-01')]
    test=outcomes[outcomes.created>=stamp('2022-07-01')]
    validation_outcomes=evaluation_cohort(outcomes,'2022-01-01','2022-07-01',strict)
    val_oracle=oracle(validation_outcomes)
    eval_oracle=oracle(test)
    val_weeks=list(split_weeks(cohort,'2022-01-01','2022-07-01',min_tickets,complete))
    test_weeks=list(split_weeks(cohort,'2022-07-01','2023-03-01',min_tickets,complete))
    if not val_weeks or not test_weeks: raise ValueError('No evaluation weeks at requested minimum volume')
    print(f'{mode}: {len(val_weeks)} validation weeks, {len(test_weeks)} test weeks',flush=True)
    grid=[]
    for k in [5,20,50]:
        e=fit(outcomes,fit_arrivals,'2021-01-01','2022-01-01',k=k,**settings)
        for z in [0,.5,1,1.5,2]:
            metrics=[]
            for week,tickets in val_weeks:
                roster=roster_for(arrivals,tickets,e,week,not strict)
                r,cap=route_week(tickets,e,roster,z=z,backlog_first=strict)
                metrics.append(score(r,val_oracle,cap,len(tickets)))
            grid.append(dict(k=k,z=z,**pd.DataFrame(metrics).mean(numeric_only=True).to_dict()))
    grid=pd.DataFrame(grid); grid.to_csv(out/'validation.csv',index=False)
    k,z=select_validation(grid)
    print(f'{mode}: selected k={k}, z={z} using validation only',flush=True)
    est=fit(outcomes,fit_arrivals,'2021-01-01','2022-07-01',k=k,**settings)
    naive=fit(outcomes,fit_arrivals,'2021-01-01','2022-07-01',k=5,**settings)
    training_oracle=oracle(train)
    rows=[]; duals=[]; mappings=[]; allocation=[]; curse=[]
    for week,tickets in test_weeks:
        roster=roster_for(arrivals,tickets,est,week,not strict)
        cap={i:int(est.capacity[i]) for i in roster}
        runs=[]
        for method,label in [('historical','Historical'),('round_robin','Round-robin'),('greedy','Greedy')]:
            runs.append((label,571,baseline(tickets,naive,roster,method,capacity=cap),cap))
        for seed in range(seeds):
            runs.append(('Historical proportions',571+seed,baseline(tickets,naive,roster,'proportions',571+seed,cap),cap))
        for label,e,zz in [('M1 naive',naive,0),('M1 tuned shrinkage',est,0),('M2 robust',est,z)]:
            r,cc=route_week(tickets,e,roster,z=zz,backlog_first=strict)
            runs.append((label,571,r,cc))
            if label=='M2 robust':
                for i in roster: duals.append(dict(week=str(week.date()),employee=i,dual=r.duals.get(i,0),load=sum(n for (ii,c),n in r.x.items() if ii==i),capacity=cc[i]))
                mappings.extend(dict(week=str(week.date()),ticket_id=t,employee=i,category=c) for t,i,c in r.assignments)
                allocation.extend(dict(week=str(week.date()),employee=i,category=c,tickets=n) for (i,c),n in r.x.items())
        for alpha in [1.25,1.5,2,3]:
            r,cc=route_week(tickets,est,roster,z=z,alpha=alpha,backlog_first=strict)
            runs.append((f'Fairness {alpha:g}',571,r,cc))
        for label,seed,r,cc in runs:
            m=score(r,eval_oracle,cc,len(tickets),eligible=est.eligible)
            rows.append(dict(method=label,seed=seed,week=str(week.date()),**m))
            if label in ['M1 naive','Historical']:
                for period,oo in [('Planning proxy', (dict(((i,c),float(naive.cost.loc[i,c])) for i in naive.cost.index for c in CATEGORIES),naive.means)),('Held-out proxy',eval_oracle)]:
                    curse.append(dict(week=str(week.date()),method=label,scoring=period,**score(r,oo,cc,len(tickets))))
    rows=pd.DataFrame(rows); rows.to_csv(out/'weekly_metrics.csv',index=False)
    summary=summarize(rows); summary.to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(mappings).to_csv(out/'ticket_assignments.csv',index=False)
    pd.DataFrame(allocation).to_csv(out/'routing_counts.csv',index=False)
    pd.DataFrame(curse).to_csv(out/'optimizers_curse.csv',index=False)
    duals=pd.DataFrame(duals); duals.to_csv(out/'capacity_duals_weekly.csv',index=False)
    ranking=duals.groupby('employee').agg(mean_shadow_value=('dual',lambda s:-s.mean()),active_weeks=('week','count'),mean_load=('load','mean'),capacity=('capacity','first')).sort_values('mean_shadow_value',ascending=False)
    ranking['mean_shadow_over_all_test_weeks']=ranking.mean_shadow_value*ranking.active_weeks/len(test_weeks)
    ranking=ranking.sort_values('mean_shadow_over_all_test_weeks',ascending=False)
    ranking.to_csv(out/'capacity_duals.csv')
    # E6: one-slot finite difference on the week with the largest conditional shadow value.
    best=duals.sort_values('dual').iloc[0]; check_week=next((w,t) for w,t in test_weeks if str(w.date())==best.week)
    w,t=check_week; roster=roster_for(arrivals,t,est,w,not strict)
    r,cap=route_week(t,est,roster,z=z,backlog_first=strict)
    costs={(i,c):float(est.robust(z).loc[i,c]) for i in cap for c in t.category.unique() if est.eligible.loc[i,c]}
    cap2=dict(cap); cap2[best.employee]+=1
    rr=route_lp(costs,t.category.value_counts().to_dict(),cap2,backlog_first=strict)
    fd=dict(employee=best.employee,week=best.week,negative_dual=-float(best.dual),one_slot_objective_drop=r.objective-rr.objective,
            backlog_before=sum(r.backlog.values()),backlog_after=sum(rr.backlog.values()),conditional_dual=strict)
    # E2: cell-mean stability, with transparent common support.
    cells=train.groupby(['assignee','category']).y.agg(train_mean='mean',train_n='size').join(test.groupby(['assignee','category']).y.agg(test_mean='mean',test_n='size'),how='inner')
    cells=cells[(cells.train_n>=5)&(cells.test_n>=5)].copy()
    cells['train_residual']=cells.train_mean-cells.index.get_level_values('category').map(train.groupby('category').y.mean())
    cells['test_residual']=cells.test_mean-cells.index.get_level_values('category').map(test.groupby('category').y.mean())
    cells.to_csv(out/'cell_stability.csv')
    estimates=est.cost.stack().rename('cost').to_frame().join(est.se.stack().rename('se')).join(est.count.stack().rename('n')).join(est.eligible.stack().rename('eligible'))
    estimates.index.names=['employee','category']; estimates.to_csv(out/'estimates.csv')
    # E7: capacity and eligibility sensitivity, same fixed validated k,z.
    sensitivity=[]
    for n_min in [3,5,10]:
        ee=fit(outcomes,fit_arrivals,'2021-01-01','2022-07-01',k=k,n_min=n_min,**settings)
        for scale in [.5,.75,1,1.25,1.5]:
            for week,t in test_weeks:
                roster=roster_for(arrivals,t,ee,week,not strict)
                r,cc=route_week(t,ee,roster,z=z,scale=scale,backlog_first=strict)
                sensitivity.append(dict(n_min=n_min,scale=scale,week=str(week.date()),**score(r,eval_oracle,cc,len(t))))
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity_weekly.csv',index=False)
    sens=pd.DataFrame(sensitivity).groupby(['n_min','scale']).agg(mean_hours=('hours','mean'),backlog=('backlog','sum'),coverage=('coverage','mean'),penalized_log_score=('penalized_log_score','mean')).reset_index()
    sens.to_csv(out/'sensitivity.csv',index=False)
    transitions=workflow(raw,arrivals.id); transitions.to_csv(out/'workflow_transitions.csv',index=False)
    cross=cross_training(outcomes,fit_arrivals,est,k,z,settings,min_tickets,strict)
    cross.to_csv(out/'cross_training.csv',index=False)
    metadata=dict(mode=mode,selected_k=k,selected_z=z,seed=571,random_seeds=seeds,validation_weeks=len(val_weeks),test_weeks=len(test_weeks),
        validation_oracle_resolved_by_cutoff=strict,validation_oracle_tickets=len(validation_outcomes),
        mean_tickets=float(np.mean([len(t) for w,t in test_weeks])),audit=audit,capacity_mode=capacity_mode,
        strict_resolution_cutoff=strict,hindsight_roster=not strict,complete_weeks=complete,
        missing_assignment_times_excluded_from_prior_roster=strict,
        cohort='all category arrivals' if strict else 'Done outcomes with first assignee',
        evaluation='Retrospective test-cell shrinkage proxy; not causal or measured counterfactual outcomes',
        unverified_arrival_features='Type and priority are final-record fields; histories contain no type/priority changes.',
        capacity_dual_check=fd,common_support_cells=len(cells),cell_correlation=float(cells.train_mean.corr(cells.test_mean)),
        residual_cell_correlation=float(cells.train_residual.corr(cells.test_residual)),
        python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,
        elapsed_seconds=perf_counter()-start_time)
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    plot_results(out,summary,grid,cells,est,ranking,pd.DataFrame(curse),cross,sens,transitions)
    write_report(out,metadata,summary,cross,fd)
    print(summary[['method','mean_hours','backlog_total','mean_coverage']].to_string(index=False),flush=True)
    return metadata,summary


def cross_training(outcomes,arrivals,est,k,z,settings,min_tickets,strict):
    weeks=list(split_weeks(arrivals,'2021-01-01','2022-07-01',min_tickets,strict))
    if not weeks: raise ValueError('No cross-training planning weeks')
    indices=np.unique(np.linspace(0,len(weeks)-1,min(8,len(weeks))).round().astype(int))
    selected=[weeks[i] for i in indices]
    # Candidate filtering uses TRAINING-period routing only, never test-period duals.
    records=[]; demands=[]; capacities=[]
    for w,t in selected:
        roster=roster_for(arrivals,t,est,w,True)
        r,cap=route_week(t,est,roster,z=z,backlog_first=False)
        demands.append(t.category.value_counts().to_dict()); capacities.append(cap)
        for i,kk in cap.items():
            records.append(dict(employee=i,spare=kk-sum(v for (ii,c),v in r.x.items() if ii==i),shadow=-r.duals[i]))
    stats=pd.DataFrame(records).groupby('employee').mean()
    pool=stats[stats.spare>0].sort_values(['shadow','spare'],ascending=[True,False]).head(12).index
    candidate={}
    for i in pool:
        handled=est.eligible.loc[i]
        delta=float((est.cost.loc[i,handled]-est.means[handled]).mean())
        for c in CATEGORIES:
            if not handled[c]: candidate[i,c]=max(0.,float(est.means[c]+delta+z*est.sigma))
    existing={(i,c):float(est.robust(z).loc[i,c]) for i in est.cost.index for c in CATEGORIES if est.eligible.loc[i,c]}
    records=[]
    for multiplier in [.75,1.,1.25]:
        costs={**existing,**{p:v*multiplier for p,v in candidate.items()}}
        for budget in [0,1,2,3,5]:
            result=cross_train_mip(costs,demands,capacities,set(candidate),budget)
            records.append(dict(budget=budget,candidate_cost_multiplier=multiplier,**asdict(result),candidate_count=len(candidate),planning_weeks=len(selected),
                chosen_json=json.dumps(result.chosen),planning_week_starts=json.dumps([str(w.date()) for w,t in selected])))
    result=pd.DataFrame(records).drop(columns='chosen')
    result['objective_saving']=result.groupby('candidate_cost_multiplier').objective.transform('first')-result.objective
    return result


def plot_results(out,summary,grid,cells,est,ranking,curse,cross,sens,transitions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
    mode=out.name
    def save(name,title,xlabel,ylabel):
        plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
        plt.gcf().text(.01,.01,f'Mendeley Help Desk v3 · {mode} · observational proxies',fontsize=7)
        plt.tight_layout(rect=[0,.035,1,1]); plt.savefig(out/(name+'.png')); plt.savefig(out/(name+'.svg')); plt.close()
    plt.figure(figsize=(8,5))
    selected=summary[summary.method.isin(['Historical','Round-robin','Greedy','Historical proportions','M1 naive','M2 robust'])]
    pos=np.arange(len(selected)); plt.barh(pos,selected.mean_hours,color='#31688e')
    plt.yticks(pos,selected.method)
    save('comparison','Held-out service proxy; compare coverage separately','Mean weekly shifted geometric hours (assigned tickets)','Method')
    plt.figure(figsize=(7,4))
    for k,d in grid.groupby('k'): plt.plot(d.z,d.hours,'o-',label=f'k={k}')
    plt.legend(); save('validation','Validation service proxy (selection also prioritizes backlog)','Uncertainty penalty z','Mean weekly shifted geometric hours')
    plt.figure(figsize=(6,5)); plt.scatter(cells.train_mean,cells.test_mean,s=20,alpha=.65)
    save('cell_stability','Employee-category associations across periods','Training cell mean log(1 + hours)','Test cell mean log(1 + hours)')
    plt.figure(figsize=(7,4))
    eligible=[est.cost.loc[est.eligible[c],c].to_numpy() for c in CATEGORIES]
    plt.boxplot(eligible); plt.xticks(range(1,5),CATEGORIES)
    save('employee_spread','Shrunk training costs among eligible employees','Ticket category','Estimated mean log(1 + hours)')
    plt.figure(figsize=(7,4)); grouped=curse.groupby(['method','scoring']).hours.mean().unstack()
    grouped.plot.bar(ax=plt.gca(),rot=0); plt.legend(title='Scoring proxy')
    save('optimizers_curse','Planning-score gains need not survive held-out scoring','Method','Mean weekly shifted geometric hours')
    plt.figure(figsize=(7,4)); top=ranking.head(10).iloc[::-1]
    plt.barh(top.index,top.mean_shadow_over_all_test_weeks,color='#31688e')
    save('capacity_duals','Marginal routing-cost value of a ticket slot','Mean negative capacity dual across test weeks','Masked employee ID')
    plt.figure(figsize=(8,5)); fair=summary[summary.method.str.startswith('Fairness') | summary.method.eq('M2 robust')]
    for index,row in enumerate(fair.itertuples()):
        plt.scatter(row.mean_gini,row.mean_hours,s=45,marker=['o','s','^','D','x'][index],label=f'{row.method}: {row.backlog_total:.0f} unassigned')
    plt.margins(.12); plt.legend(fontsize=8,loc='best')
    save('fairness','Workload fairness versus conditional service proxy','Mean employee-load Gini (zeros included)','Mean weekly shifted geometric hours')
    plt.figure(figsize=(7,4))
    for n,d in sens.groupby('n_min'): plt.plot(d.scale,d.mean_hours,'o-',label=f'n_min={n}')
    plt.legend(); save('capacity_sensitivity','Capacity sensitivity; see CSV for backlog','Capacity multiplier (integer floor)','Mean weekly shifted geometric hours')
    plt.figure(figsize=(7,4))
    for mult,d in cross.groupby('candidate_cost_multiplier'): plt.plot(d.budget,d.objective_saving,'o-',label=f'New-pair cost × {mult:g}')
    plt.legend(); save('cross_training','Assumed cross-training value on training-period weeks','Maximum trained employee-category pairs','Penalized log-objective reduction (not hours)')
    plt.figure(figsize=(9,6)); top=transitions.head(18)
    labels=[f'{r.previous} → {r.value}' for r in top.itertuples()]
    plt.barh(range(len(top)),top['count'],color='#31688e'); plt.yticks(range(len(top)),labels,fontsize=8); plt.gca().invert_yaxis()
    save('workflow','Most frequent observed workflow transitions','Transition count (2021–February 2023 cohort)','Status transition')
    # Directed descriptive map of the six busiest status nodes.
    totals=pd.concat([transitions[['previous','count']].rename(columns={'previous':'node'}),transitions[['value','count']].rename(columns={'value':'node'})]).groupby('node')['count'].sum()
    nodes=list(totals.nlargest(6).index); positions={node:(math.cos(2*math.pi*j/len(nodes)),math.sin(2*math.pi*j/len(nodes))) for j,node in enumerate(nodes)}
    edges=transitions[transitions.previous.isin(nodes)&transitions.value.isin(nodes)].head(12)
    fig,ax=plt.subplots(figsize=(9,7)); ax.set_xlim(-1.6,1.6); ax.set_ylim(-1.4,1.4)
    for edge in edges.itertuples():
        a,b=positions[edge.previous],positions[edge.value]
        ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.16',lw=.6+2*edge.count/edges['count'].max(),color='gray',shrinkA=35,shrinkB=35))
    for node,(x,y) in positions.items(): ax.text(x,y,node.replace('_','\n'),ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.5',fc='white',ec='#31688e'))
    ax.axis('off'); save('workflow_map','Observed status map; arrow thickness represents frequency','','')


def write_report(out,metadata,summary,cross,fd):
    rows=summary.set_index('method'); m1=rows.loc['M1 naive']; m2=rows.loc['M2 robust']; hist=rows.loc['Historical']; rr=rows.loc['Round-robin']
    lines=[f'# Results: {out.name}', '',
        'Generated from the local real Mendeley v3 CSVs. These are retrospective model-based proxies, not causal treatment effects or measured counterfactual improvements.', '',
        f"Usable Done outcomes: {metadata['audit']['D5_usable']:,}; train/test outcomes: {metadata['audit']['train_outcomes']:,}/{metadata['audit']['test_outcomes']:,}. Evaluated {metadata['test_weeks']} test weeks. Validated k={metadata['selected_k']}, z={metadata['selected_z']}.", '',
        f"Historical: {hist.mean_hours:.2f}; round-robin: {rr.mean_hours:.2f}; naive: {m1.mean_hours:.2f}; robust: {m2.mean_hours:.2f} mean weekly shifted geometric hours among routed tickets.", '',
        f"Backlogged/unassigned counts: historical {hist.backlog_total:.0f}, round-robin {rr.backlog_total:.0f}, naive {m1.backlog_total:.0f}, robust {m2.backlog_total:.0f}. Historical missing-assignee counts are unknown routing labels, not evidence of operational backlog.", '',
        f"At equal weekly coverage, robust descriptive change versus historical: {m2.get('equal_coverage_pct_vs_historical',np.nan):.2f}% across {m2.get('equal_coverage_weeks_vs_historical',0):.0f} weeks; versus round-robin: {m2.get('equal_coverage_pct_vs_round_robin',np.nan):.2f}%. Positive means lower proxy hours. Comparisons at different coverage are not service improvements.", '',
        f"Maximum LP fractionality: {summary.max_fractionality.max():.3g}. Common-support cell correlation: {metadata['cell_correlation']:.3f}; category-adjusted correlation: {metadata['residual_cell_correlation']:.3f}, {metadata['common_support_cells']} cells.", '',
        f"Capacity finite-difference check: employee {fd['employee']}, week {fd['week']}; negative dual {fd['negative_dual']:.3f}; one-slot objective drop {fd['one_slot_objective_drop']:.3f}. In backlog-first mode duals are conditional on fixed minimum backlog; a capacity change may change that minimum.", '',
        '## Interpretation and limits', '',
        '- Training costs use historical assignment associations and cannot separate employee effects from dispatcher selection.',
        '- Test-cell estimates are an offline evaluation oracle and use outcomes across the test period. They never enter planning but are not prospective causal validation.',
        '- Even strict mode uses final-record type/priority and known batch demand, so it is a stronger audit scenario, not a fully deployable online policy.',
        '- Reproduction mode uses hindsight weekly staffing, completed-ticket demand, and creation-date fitting. Strict mode uses all category arrivals, prior-eight-week observed staffing, calendar-zero capacity and outcomes resolved before cutoff.',
        '- Strict validation scoring uses only outcomes resolved before the test-period start. Missing assignment timestamps are excluded from prior rosters. Historical unknown assignees are not actual operational backlog and prevent fair coverage comparisons.',
        '- Calendar durations include waiting, nights and weekends. Shifted geometric hours are not mean expected hours or labor hours.',
        '- Fairness can add backlog. Compare backlog/coverage first; assigned-only service proxies alone can mislead.',
        '- Cross-training uses assumed new-pair costs and a restricted candidate set, planned on eight representative training-period weeks. Its savings are penalized log-objective units, not hours.',
        f"- Cross-training with the stipulated central new-pair costs saves {cross.loc[cross.candidate_cost_multiplier.eq(1),'objective_saving'].max():.3f} objective units at the tested budgets. Zero value is a valid result, not evidence that training never helps.",
        f"- The validated uncertainty penalty is z={metadata['selected_z']}. If z=0, M2 is exactly the tuned-shrinkage ablation; any improvement over naive M1 must not be attributed to robustness.",
        '- Paired bootstrap intervals resample weeks independently and hold cost estimates fixed. They do not account for time dependence, estimation uncertainty or confounding.', '',
        '## Artifacts', '',
        'comparison.csv: all method metrics; weekly_metrics.csv: weeks and seeds; validation.csv: complete tuning grid; metadata.json: settings, source hashes and audit; routing_counts.csv: robust weekly allocation; ticket_assignments.csv: local ticket mapping (gitignored); estimates.csv: shrunk costs/support; cell_stability.csv; capacity_duals.csv; sensitivity.csv; cross_training.csv; workflow_transitions.csv. PNG/SVG figures share descriptive names.', '',
        '## AI disclosure', '',
        'This implementation and accompanying documentation were generated with assistance from OpenAI Codex. Team members must independently understand and verify the code, results and assumptions, and check the complete course policy before submission. This is a project implementation, not an assertion that it satisfies an unprovided academic AI policy.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines))
