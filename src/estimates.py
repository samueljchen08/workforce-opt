"""Training-only shrinkage, eligibility and integer weekly count capacity."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .data import CATEGORIES, stamp


@dataclass
class Estimates:
    cost: pd.DataFrame
    se: pd.DataFrame
    count: pd.DataFrame
    eligible: pd.DataFrame
    capacity: pd.Series
    means: pd.Series
    sigma: float

    def robust(self,z=0):
        if z<0: raise ValueError('z must be nonnegative')
        return self.cost+z*self.se


def fit(outcomes,arrivals,start,end,k=5,n_min=5,strict=False,capacity_mode='nonzero'):
    if k<=0 or n_min<1: raise ValueError('Require k > 0 and n_min >= 1')
    start,end=stamp(start),stamp(end)
    d=outcomes[(outcomes.created>=start)&(outcomes.created<end)].copy()
    if strict: d=d[d.resolved<end]
    if d.empty: raise ValueError('No observed outcomes before fitting cutoff')
    means=d.groupby('category').y.mean().reindex(CATEGORIES).fillna(d.y.mean())
    sigma=float(d.y.std(ddof=1)) if len(d)>1 else 0.
    ids=sorted(d.assignee.dropna().unique())
    count=d.groupby(['assignee','category']).size().unstack(fill_value=0).reindex(index=ids,columns=CATEGORIES,fill_value=0)
    sums=d.groupby(['assignee','category']).y.sum().unstack(fill_value=0).reindex(index=ids,columns=CATEGORIES,fill_value=0)
    cost=(sums+k*means)/(count+k)
    se=sigma/np.sqrt(count+k)
    arrivals=arrivals[(arrivals.created>=start)&(arrivals.created<end) & arrivals.assignee.notna()]
    if strict and 'assigned_at' in arrivals:
        arrivals=arrivals[arrivals.assigned_at<end]
    loads=arrivals.groupby(['assignee','week']).size().unstack(fill_value=0).reindex(index=ids,fill_value=0)
    if capacity_mode=='calendar':
        first=start.normalize()-pd.Timedelta(days=start.dayofweek)
        last=(end-pd.Timedelta(nanoseconds=1)).normalize()
        last-=pd.Timedelta(days=last.dayofweek)
        loads=loads.reindex(columns=pd.date_range(first,last,freq='7D'),fill_value=0)
        quantiles=loads.quantile(.9,axis=1)
    elif capacity_mode=='nonzero':
        quantiles=loads.where(loads>0).quantile(.9,axis=1)
    else: raise ValueError('capacity_mode must be nonzero or calendar')
    capacity=np.floor(quantiles.fillna(0)).clip(lower=2).astype(int)
    return Estimates(cost,se,count,count>=n_min,capacity,means,sigma)


def oracle(outcomes,k=5):
    """Retrospective held-out scoring proxy, not counterfactual ground truth."""
    if outcomes.empty: raise ValueError('No evaluation outcomes')
    means=outcomes.groupby('category').y.mean().reindex(CATEGORIES).fillna(outcomes.y.mean())
    g=outcomes.groupby(['assignee','category']).y.agg(['sum','count'])
    values={(i,c):(r['sum']+k*means[c])/(r['count']+k) for (i,c),r in g.iterrows()}
    return values,means
