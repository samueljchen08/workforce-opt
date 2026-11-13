"""Transportation LPs and shared-decision cross-training MILP (SciPy HiGHS)."""
from dataclasses import dataclass, field
from time import perf_counter
import math
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix, vstack, csr_matrix


@dataclass
class Routing:
    x: dict
    backlog: dict
    objective: float
    duals: dict = field(default_factory=dict)
    seconds: float = 0.
    fractionality: float = 0.
    assignments: list = field(default_factory=list)


def _validate_counts(values):
    if any(not np.isfinite(v) or v<0 or abs(v-round(v))>1e-7 for v in values):
        raise ValueError('Demand and capacity must be nonnegative finite integers')


def fairness_capacity(capacity,total,alpha=math.inf):
    if alpha<=0: raise ValueError('alpha must be positive')
    if not capacity or math.isinf(alpha): return dict(capacity)
    ceiling=math.ceil(alpha*total/len(capacity))
    return {i:min(k,ceiling) for i,k in capacity.items()}


def route_lp(cost,demand,capacity,penalty=100.,backlog_first=True):
    start=perf_counter()
    _validate_counts(list(demand.values())+list(capacity.values()))
    if penalty<=0 or not np.isfinite(penalty): raise ValueError('Invalid penalty')
    cats=list(demand); people=list(capacity)
    pairs=sorted(p for p in cost if p[0] in capacity and p[1] in demand)
    if any(not np.isfinite(cost[p]) for p in pairs): raise ValueError('Nonfinite cost')
    if not cats: return Routing({}, {},0.)
    n=len(pairs); m=n+len(cats)
    eq=lil_matrix((len(cats),m)); ub=lil_matrix((len(people),m))
    ci={c:j for j,c in enumerate(cats)}; ii={i:j for j,i in enumerate(people)}
    for j,(i,c) in enumerate(pairs): eq[ci[c],j]=1; ub[ii[i],j]=1
    for j,c in enumerate(cats): eq[j,n+j]=1
    eq=eq.tocsr(); ub=ub.tocsr()
    beq=np.array([demand[c] for c in cats],dtype=float)
    kwargs=dict(A_ub=ub if people else None,b_ub=[capacity[i] for i in people] if people else None,
                A_eq=eq,b_eq=beq,bounds=(0,None),method='highs')
    cvec=np.array([cost[p] for p in pairs]+[penalty]*len(cats),dtype=float)
    if backlog_first:
        first=linprog(np.r_[np.zeros(n),np.ones(len(cats))],**kwargs)
        if not first.success: raise RuntimeError(first.message)
        minimum=round(first.x[n:].sum())
        kwargs['A_eq']=vstack([eq,csr_matrix(np.r_[np.zeros(n),np.ones(len(cats))][None,:])])
        kwargs['b_eq']=np.r_[beq,minimum]
    result=linprog(cvec,**kwargs)
    if not result.success: raise RuntimeError(result.message)
    frac=float(np.max(np.abs(result.x-np.rint(result.x)),initial=0))
    if frac>1e-6: raise RuntimeError(f'Unexpected noninteger routing: {frac}')
    vals=np.rint(result.x).astype(int)
    x={p:int(vals[j]) for j,p in enumerate(pairs) if vals[j]>0}
    backlog={c:int(vals[n+j]) for j,c in enumerate(cats)}
    duals={i:float(result.ineqlin.marginals[j]) for j,i in enumerate(people)}
    return Routing(x,backlog,float(result.fun),duals,perf_counter()-start,frac)


def assign_tickets(routing,tickets):
    """Deterministic executable ticket mapping; None denotes weekly backlog."""
    remaining=dict(routing.x); assignments=[]
    for c,group in tickets.sort_values(['created','id']).groupby('category',sort=True):
        people=sorted(i for i,cat in remaining if cat==c)
        pointer=0
        for row in group.itertuples():
            while pointer<len(people) and remaining.get((people[pointer],c),0)==0: pointer+=1
            i=people[pointer] if pointer<len(people) else None
            if i is not None: remaining[i,c]-=1
            assignments.append((int(row.id),i,c))
    if sum(remaining.values()): raise ValueError('Routing counts exceed tickets')
    routing.assignments=assignments
    return routing


@dataclass
class CrossTraining:
    chosen: list
    objective: float
    status: int
    gap: float
    seconds: float
    message: str


def cross_train_mip(costs,demands,capacities,candidates,budget,penalty=100.,time_limit=30.):
    """Fixed-M M4; costs include existing pairs and explicitly assumed new pairs."""
    start=perf_counter(); candidates=sorted(candidates)
    _validate_counts([budget])
    if len(demands)!=len(capacities) or not demands: raise ValueError('Require matching nonempty weeks')
    if set(candidates)-set(costs): raise ValueError('Missing candidate costs')
    tindex={p:j for j,p in enumerate(candidates)}
    idx=len(candidates); xindex={}; uindex={}
    for w,(d,k) in enumerate(zip(demands,capacities)):
        _validate_counts(list(d.values())+list(k.values()))
        for p in sorted(costs):
            if p[0] in k and p[1] in d: xindex[w,p]=idx; idx+=1
        for c in d: uindex[w,c]=idx; idx+=1
    obj=np.zeros(idx); upper=np.full(idx,np.inf); integ=np.zeros(idx)
    upper[:len(candidates)]=1; integ[:len(candidates)]=1
    rows=[]; lower=[]; bound=[]
    def add(row,lo,hi): rows.append(row); lower.append(lo); bound.append(hi)
    add({j:1 for j in tindex.values()},0,budget)
    for w,(d,k) in enumerate(zip(demands,capacities)):
        for c,value in d.items():
            row={j:1 for (ww,p),j in xindex.items() if ww==w and p[1]==c}
            row[uindex[w,c]]=1; add(row,value,value); obj[uindex[w,c]]=penalty
        for i,value in k.items():
            add({j:1 for (ww,p),j in xindex.items() if ww==w and p[0]==i},0,value)
        for (ww,p),j in xindex.items():
            if ww!=w: continue
            obj[j]=costs[p]
            if p in tindex: add({j:1,tindex[p]:-k[p[0]]},-np.inf,0)
    mat=lil_matrix((len(rows),idx))
    for r,row in enumerate(rows):
        for j,v in row.items(): mat[r,j]=v
    res=milp(obj,integrality=integ,bounds=Bounds(np.zeros(idx),upper),
        constraints=LinearConstraint(mat.tocsr(),lower,bound),options={'time_limit':time_limit,'mip_rel_gap':1e-6})
    if res.x is None: raise RuntimeError(f'Cross-training has no incumbent: {res.message}')
    chosen=[p for p,j in tindex.items() if res.x[j]>.5]
    return CrossTraining(chosen,float(res.fun),int(res.status),float(getattr(res,'mip_gap',np.nan)),perf_counter()-start,str(res.message))
