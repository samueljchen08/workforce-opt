import unittest
import pandas as pd
from src.experiments import select_validation, roster_for, evaluation_cohort


class ExperimentTests(unittest.TestCase):
    def test_strict_validation_oracle_excludes_outcomes_unobserved_at_test_start(self):
        d=pd.DataFrame({'created':pd.to_datetime(['2022-02-01','2022-03-01'],utc=True),
            'resolved':pd.to_datetime(['2022-03-01','2022-08-01'],utc=True)})
        self.assertEqual(len(evaluation_cohort(d,'2022-01-01','2022-07-01',True)),1)
        self.assertEqual(len(evaluation_cohort(d,'2022-01-01','2022-07-01',False)),2)

    def test_validation_selects_coverage_before_conditional_service(self):
        d=pd.DataFrame({'k':[5,20],'z':[0,2],'backlog':[1.,0.],
            'penalized_log_score':[1.,5.]})
        self.assertEqual(select_validation(d),(20,2.))

    def test_prior_roster_does_not_use_current_or_future_assignments(self):
        a=pd.DataFrame({'assignee':['a','b','c'],
            'created':pd.to_datetime(['2022-06-20','2022-07-04','2022-06-20'],utc=True),
            'assigned_at':pd.to_datetime(['2022-06-21','2022-07-04','2022-07-06'],utc=True)})
        class E:
            capacity=pd.Series({'a':2,'b':2,'c':2})
            eligible=pd.DataFrame({'T-normal':[True,True,True]},index=['a','b','c'])
        got=roster_for(a,a.iloc[:0],E(),'2022-07-04',False)
        self.assertEqual(got,['a'])
