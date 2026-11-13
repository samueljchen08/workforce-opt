import unittest
import pandas as pd
from src.data import prepare, split_weeks


class DataTests(unittest.TestCase):
    def tables(self):
        a = pd.DataFrame(dict(id=[1.,2.,3.,4.], issue_type=['Ticket','HD Service','Ticket','Project'],
            issue_priority=['Medium','High','Blocker','Low'], issue_created=['2022-01-03']*4,
            issue_resolution_date=['2022-01-04','2022-01-05',None,'2022-01-06'],
            issue_resolution=['Done','Done',None,'Done']))
        s = pd.DataFrame(dict(id=[1.,1.,2.,3.], turn=[1,2,1,1], issue_assignee=['a','b','b','a']))
        return a,s

    def test_arrivals_include_unresolved_and_outcomes_use_first_assignee(self):
        a,s=self.tables(); arrivals,outcomes,audit=prepare(a,s)
        self.assertEqual(len(arrivals),3); self.assertEqual(len(outcomes),2)
        self.assertEqual(outcomes.loc[outcomes.id==1,'assignee'].iloc[0],'a')
        self.assertEqual(set(arrivals.category),{'T-normal','T-urgent','Service'})
        self.assertAlmostEqual(outcomes.loc[outcomes.id==1,'hours'].iloc[0],24)

    def test_duplicate_first_turn_rejected(self):
        a,s=self.tables()
        with self.assertRaises(ValueError): prepare(a,pd.concat([s,s.iloc[:1]]))

    def test_complete_weeks_exclude_boundary_week(self):
        a,s=self.tables(); arrivals,_,_=prepare(a,s)
        self.assertEqual(len(list(split_weeks(arrivals,'2022-01-04','2022-01-11',1,True))),0)

