import unittest
import pandas as pd
from src.estimates import fit
from src.baselines import baseline


class BaselineTests(unittest.TestCase):
    def test_methods_respect_eligibility_capacity_and_conserve_tickets(self):
        dates=pd.to_datetime(['2022-01-03']*4,utc=True)
        d=pd.DataFrame({'id':range(4),'assignee':['a','a','b','b'],'category':['T-normal']*4,
            'created':dates,'resolved':dates,'week':dates,'y':[1.,1.,3.,3.]})
        e=fit(d,d,'2022-01-01','2022-02-01',n_min=1)
        tickets=pd.concat([d,d.assign(id=range(4,8))]).sort_values('id')
        for method in ['round_robin','greedy','proportions']:
            r=baseline(tickets,e,['a','b'],method,571)
            self.assertEqual(sum(r.x.values()),4)
            self.assertEqual(sum(r.backlog.values()),4)
            self.assertEqual(len(r.assignments),8)
            if method=='proportions': self.assertEqual(r.x,baseline(tickets,e,['a','b'],method,571).x)
