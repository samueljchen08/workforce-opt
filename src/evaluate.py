"""Coverage-aware, explicitly observational out-of-sample evaluation."""
import math
from collections import Counter
import numpy as np
import pandas as pd
from .models import route_lp, fairness_capacity, assign_tickets


def route_week(tickets,est,roster,z=0,alpha=math.inf,scale=1.,backlog_first=True):
    capacity={i:max(0,int(math.floor(est.capacity[i]*scale))) for i in roster if i in est.capacity}
    capacity=fairness_capacity(capacity,len(tickets),alpha)
    demand=tickets.category.value_counts().to_dict()
    robust=est.robust(z)
    costs={(i,c):float(robust.loc[i,c]) for i in capacity for c in demand if est.eligible.loc[i,c]}
    result=route_lp(costs,demand,capacity,backlog_first=backlog_first)
    return assign_tickets(result,tickets),capacity


def score(routing,oracle,capacity,total,penalty=100.,eligible=None):
    values,means=oracle
    assigned=sum(routing.x.values()); backlog=sum(routing.backlog.values())
    if assigned+backlog!=total: raise ValueError('Routing does not conserve arriving tickets')
    cost=sum(v*values.get((i,c),float(means[c])) for (i,c),v in routing.x.items())
    fallback=sum(v for p,v in routing.x.items() if p not in values)
    loads=Counter()
    for (i,c),n in routing.x.items(): loads[i]+=n
    ids=sorted(set(capacity)|set(loads)); counts=np.array([loads[i] for i in ids],dtype=float)
    n=len(counts)
    if assigned and n:
        sorted_counts=np.sort(counts)
        gini=float(np.sum((2*np.arange(1,n+1)-n-1)*sorted_counts)/(n*assigned))
    else: gini=0.
    unsupported=0
    if eligible is not None:
        unsupported=sum(v for (i,c),v in routing.x.items() if i not in eligible.index or not eligible.loc[i,c])
    excess=sum(max(0,loads[i]-capacity.get(i,0)) for i in loads)
    log_score=cost/assigned if assigned else np.nan
    return dict(tickets=total,assigned=assigned,backlog=backlog,coverage=assigned/total if total else 1.,
        log_score=log_score,hours=float(np.expm1(log_score)) if assigned else np.nan,
        penalized_log_score=(cost+penalty*backlog)/total if total else 0.,
        max_share=max(loads.values(),default=0)/assigned if assigned else 0.,gini=gini,
        utilization=assigned/sum(capacity.values()) if sum(capacity.values()) else np.nan,
        fallback_tickets=fallback,unsupported_tickets=unsupported,capacity_excess=excess,
        solve_seconds=routing.seconds,fractionality=routing.fractionality)


def summarize(rows):
    """First average random seeds within week; then weight each week equally."""
    numeric=rows.select_dtypes(include='number').columns.difference(['seed'])
    weekly=rows.groupby(['method','week'],as_index=False)[list(numeric)].mean()
    records=[]
    historical=weekly[weekly.method.eq('Historical')].set_index('week')
    rr=weekly[weekly.method.eq('Round-robin')].set_index('week')
    for method,d in weekly.groupby('method',sort=False):
        row={'method':method,'weeks':len(d),'mean_hours':d.hours.mean(),'p10_hours':d.hours.quantile(.1),
             'p90_hours':d.hours.quantile(.9),'backlog_total':d.backlog.sum(),
             'mean_coverage':d.coverage.mean(),'mean_penalized_log_score':d.penalized_log_score.mean(),
             'mean_gini':d.gini.mean(),'mean_max_share':d.max_share.mean(),
             'mean_utilization':d.utilization.mean(),'mean_solve_ms':1000*d.solve_seconds.mean(),
             'fallback_tickets':d.fallback_tickets.sum(),'capacity_excess':d.capacity_excess.sum(),
             'unsupported_tickets':d.unsupported_tickets.sum(),'max_fractionality':d.fractionality.max()}
        indexed=d.set_index('week')
        for name,ref in [('historical',historical),('round_robin',rr)]:
            joined=indexed[['hours','backlog','penalized_log_score']].join(ref[['hours','backlog','penalized_log_score']],rsuffix='_ref',how='inner')
            if joined.empty: continue
            row['descriptive_pct_vs_'+name]=100*(1-joined.hours.mean()/joined.hours_ref.mean())
            row['weeks_better_backlog_first_vs_'+name]=int(((joined.backlog<joined.backlog_ref) | ((joined.backlog==joined.backlog_ref)&(joined.penalized_log_score<joined.penalized_log_score_ref-1e-9))).sum())
            equal=joined[joined.backlog==joined.backlog_ref]
            row['equal_coverage_weeks_vs_'+name]=len(equal)
            row['equal_coverage_pct_vs_'+name]=100*(1-equal.hours.mean()/equal.hours_ref.mean()) if len(equal) else np.nan
            differences=(joined.penalized_log_score_ref-joined.penalized_log_score).to_numpy()
            rng=np.random.default_rng(571)
            boots=rng.choice(differences,size=(2000,len(differences)),replace=True).mean(axis=1)
            row['paired_log_gain_vs_'+name]=differences.mean()
            row['paired_log_gain_ci_low_vs_'+name]=float(np.quantile(boots,.025))
            row['paired_log_gain_ci_high_vs_'+name]=float(np.quantile(boots,.975))
        records.append(row)
    return pd.DataFrame(records)
