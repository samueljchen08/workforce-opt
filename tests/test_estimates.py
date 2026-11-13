import unittest
import numpy as np
import pandas as pd
from src.estimates import fit


class EstimateTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'id':[1,2], 'assignee':['a','a'],'category':['T-normal']*2,
            'created':pd.to_datetime(['2022-01-03','2022-01-04'],utc=True),
            'resolved':pd.to_datetime(['2022-01-05','2022-04-01'],utc=True),
            'week':pd.to_datetime(['2022-01-03']*2,utc=True),'y':[2.,6.]})

    def test_shrinkage_and_strict_resolution_cutoff(self):
        d=self.frame(); e=fit(d,d,'2022-01-01','2022-02-01',k=5,n_min=1,strict=True)
        self.assertEqual(e.count.loc['a','T-normal'],1)
        self.assertAlmostEqual(e.cost.loc['a','T-normal'],2)

    def test_zero_week_capacity_floor_and_eligibility(self):
        d=self.frame(); e=fit(d,d,'2022-01-01','2022-03-01',n_min=3)
        self.assertEqual(e.capacity['a'],2)
        self.assertFalse(e.eligible.loc['a','T-normal'])
        self.assertTrue(np.isfinite(e.cost.values).all())
