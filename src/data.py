"""Audited ingestion; calendar duration is deliberately not called labor time."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

CATEGORIES = ['T-normal', 'T-high', 'T-urgent', 'Service']


def stamp(value):
    return pd.Timestamp(value, tz='UTC') if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert('UTC')


def prepare(issues, snapshots, start='2021-01-01', end='2023-03-01'):
    required = {'id','issue_type','issue_priority','issue_created','issue_resolution_date','issue_resolution'}
    if required - set(issues):
        raise ValueError(f'Missing issue columns: {required-set(issues)}')
    if {'id','turn','issue_assignee'} - set(snapshots):
        raise ValueError('Snapshots require id, turn and issue_assignee')
    a=issues.copy(); s=snapshots.copy(); audit={'raw_issues':len(a)}
    for frame in (a,s):
        ids=pd.to_numeric(frame.id,errors='coerce')
        if ((ids.dropna() % 1) != 0).any(): raise ValueError('Nonintegral issue ID')
        frame['id']=ids.astype('Int64')
    a=a.dropna(subset=['id'])
    if a.id.duplicated().any(): raise ValueError('Duplicate issue IDs')
    a['created']=pd.to_datetime(a.issue_created,utc=True,format='mixed',errors='coerce')
    a['resolved']=pd.to_datetime(a.issue_resolution_date,utc=True,format='mixed',errors='coerce')
    a=a[(a.created>=stamp(start)) & (a.created<stamp(end))].copy()
    audit['D1_period']=len(a); audit['D2_done']=int(a.issue_resolution.eq('Done').sum())
    a['category']=pd.Series(index=a.index,dtype='object')
    ticket=a.issue_type.eq('Ticket')
    a.loc[ticket & a.issue_priority.isin(['Lowest','Low','Medium']),'category']='T-normal'
    a.loc[ticket & a.issue_priority.eq('High'),'category']='T-high'
    a.loc[ticket & a.issue_priority.isin(['Highest','Blocker']),'category']='T-urgent'
    a.loc[a.issue_type.isin(['Service','HD Service']),'category']='Service'
    a=a.dropna(subset=['category']).copy()
    audit['D3_done_categories']=int(a.issue_resolution.eq('Done').sum())
    first=s.loc[pd.to_numeric(s.turn,errors='coerce').eq(1),['id','issue_assignee']].dropna(subset=['id'])
    if first.id.duplicated().any(): raise ValueError('Duplicate turn-1 snapshots')
    first=first.rename(columns={'issue_assignee':'assignee'})
    first['assignee']=first.assignee.astype('string').str.strip().replace('',pd.NA)
    a=a.merge(first,on='id',how='left',validate='one_to_one',indicator=True)
    audit['unmatched_snapshot_ids']=int(a['_merge'].eq('left_only').sum())
    audit['missing_first_assignee']=int(a.assignee.isna().sum())
    audit['D4_done_with_assignee']=int((a.issue_resolution.eq('Done') & a.assignee.notna()).sum())
    a['hours']=(a.resolved-a.created).dt.total_seconds()/3600
    a['week']=a.created.dt.normalize()-pd.to_timedelta(a.created.dt.dayofweek,unit='D')
    a['y']=np.log1p(a.hours.where(a.hours>=0))
    outcomes=a[a.issue_resolution.eq('Done') & a.y.notna() & a.assignee.notna()].copy()
    audit['D5_usable']=len(outcomes)
    audit['negative_duration']=int(a.hours.lt(0).sum())
    audit['all_category_arrivals']=len(a)
    audit['category_counts']=outcomes.category.value_counts().to_dict()
    audit['train_outcomes']=int(outcomes.created.lt(stamp('2022-07-01')).sum())
    audit['test_outcomes']=int(outcomes.created.ge(stamp('2022-07-01')).sum())
    return a.drop(columns='_merge'),outcomes.drop(columns='_merge'),audit


def load_data(raw):
    raw=Path(raw)
    snapshot=next((raw/n for n in ['issues_snapshot.csv','issues_snapshots.csv'] if (raw/n).exists()),None)
    if snapshot is None: raise FileNotFoundError('Place issues_snapshot.csv (or issues_snapshots.csv) in data/raw')
    paths=[raw/'issues.csv',snapshot]
    a,o,audit=prepare(*(pd.read_csv(p,low_memory=False) for p in paths))
    history=raw/'issues_change_history.csv'
    if history.exists():
        h=pd.read_csv(history,low_memory=False,usecols=['id','issueid','field','value','created'])
        h=h[h.field.eq('assignee') & h.value.notna()].copy()
        h['assigned_at']=pd.to_datetime(h.created,utc=True,format='mixed',errors='coerce')
        first=h.sort_values(['assigned_at','id']).drop_duplicates('issueid').set_index('issueid').assigned_at
        for frame in (a,o): frame['assigned_at']=frame.id.map(first)
        audit['arrival_assignment_time_missing']=int(a.assigned_at.isna().sum())
        paths.append(history)
    audit['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return a,o,audit


def split_weeks(frame,start,end,min_tickets=50,complete=True):
    """Half-open date interval; optionally exclude calendar weeks crossing it."""
    start,end=stamp(start),stamp(end)
    subset=frame[(frame.created>=start)&(frame.created<end)]
    for week, group in subset.groupby('week',sort=True):
        if complete and (week<start or week+pd.Timedelta(days=7)>end): continue
        if len(group)>=min_tickets:
            yield week,group.sort_values(['created','id']).copy()


def workflow(raw,issue_ids):
    h=pd.read_csv(Path(raw)/'issues_change_history.csv',low_memory=False)
    h=h[h.field.eq('status') & h.issueid.isin(issue_ids)].copy()
    h['created']=pd.to_datetime(h.created,utc=True,format='mixed',errors='coerce')
    h=h.sort_values(['issueid','created','id'])
    h['previous']=h.groupby('issueid')['value'].shift()
    h=h[h.previous.notna() & h.previous.ne(h.value)]
    return h.groupby(['previous','value']).size().rename('count').reset_index().sort_values('count',ascending=False)
