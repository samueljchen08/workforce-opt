"""Capacity-aware automatic rules and an unconstrained historical reference."""
from collections import Counter
from time import perf_counter
import numpy as np
from .models import Routing


def baseline(tickets,est,roster,method,seed=571,capacity=None):
    start=perf_counter(); rng=np.random.default_rng(seed)
    cap=dict(est.capacity.reindex(roster).dropna().astype(int)) if capacity is None else dict(capacity)
    remaining=dict(cap); x=Counter(); backlog=Counter(); assignments=[]; cursor=Counter()
    if method not in ['historical','round_robin','greedy','proportions']: raise ValueError(method)
    for row in tickets.sort_values(['created','id']).itertuples():
        c=row.category; i=None
        if method=='historical':
            i=row.assignee if isinstance(row.assignee,str) and row.assignee else None
        else:
            eligible=sorted(i for i in cap if i in est.eligible.index and est.eligible.loc[i,c])
            available=[i for i in eligible if remaining[i]>0]
            if available:
                if method=='greedy': i=min(available,key=lambda i:(est.cost.loc[i,c],i))
                elif method=='proportions':
                    weights=est.count.loc[available,c].to_numpy(dtype=float)
                    i=str(rng.choice(available,p=weights/weights.sum()))
                else:
                    for step in range(len(eligible)):
                        index=(cursor[c]+step)%len(eligible)
                        if remaining[eligible[index]]>0:
                            i=eligible[index]; cursor[c]=(index+1)%len(eligible); break
                remaining[i]-=1
        if i is None: backlog[c]+=1
        else: x[i,c]+=1
        assignments.append((int(row.id),i,c))
    return Routing(dict(x),dict(backlog),0.,seconds=perf_counter()-start,assignments=assignments)
