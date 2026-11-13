import unittest
import math
import pandas as pd
from src.models import Routing
from src.evaluate import score, summarize, route_week
from src.estimates import fit


class EvaluationTests(unittest.TestCase):
    def test_backlog_penalized_separately_from_assigned_hours(self):
        r=Routing({('a','x'):2},{'x':2},0.)
        m=score(r,({('a','x'):math.log(11)},pd.Series({'x':math.log(11)})),{'a':4},4)
        self.assertAlmostEqual(m['hours'],10)
        self.assertAlmostEqual(m['penalized_log_score'],(2*math.log(11)+200)/4)
        self.assertEqual(m['coverage'],.5)
        self.assertEqual(m['backlog'],2)

    def test_all_backlog_hours_are_missing_not_zero(self):
        m=score(Routing({}, {'x':3},0.),({},pd.Series({'x':2})),{},3)
        self.assertTrue(math.isnan(m['hours']))
        self.assertEqual(m['penalized_log_score'],100)

    def test_zero_load_staff_included_in_gini(self):
        m=score(Routing({('a','x'):4},{},0.),({('a','x'):2},pd.Series({'x':2})),{'a':4,'b':4},4)
        self.assertAlmostEqual(m['gini'],.5)

    def test_week_lp_produces_ticket_mapping_and_summary(self):
        dates=pd.to_datetime(['2022-01-03']*4,utc=True)
        d=pd.DataFrame({'id':range(4),'assignee':['a','a','b','b'],'category':['T-normal']*4,
            'created':dates,'resolved':dates,'week':dates,'y':[1.,1.,3.,3.]})
        e=fit(d,d,'2022-01-01','2022-02-01',n_min=1)
        r,cap=route_week(d,e,['a','b'],z=1)
        self.assertEqual(len(r.assignments),4)
        self.assertEqual(sum(r.x.values()),4)
        row=score(r,({('a','T-normal'):1,('b','T-normal'):3},e.means),cap,4)
        row.update(method='M2 robust',week='2022-01-03',seed=571)
        summary=summarize(pd.DataFrame([row]))
        self.assertEqual(summary.iloc[0]['weeks'],1)
