import unittest
import itertools
import numpy as np
from src.models import route_lp, cross_train_mip, fairness_capacity


class ModelTests(unittest.TestCase):
    def test_transport_optimum_and_integer_counts(self):
        r=route_lp({('a','x'):1,('a','y'):4,('b','x'):3,('b','y'):1},{'x':2,'y':2},{'a':2,'b':2})
        self.assertAlmostEqual(r.objective,4)
        self.assertEqual(r.x.get(('a','x')),2)
        self.assertEqual(sum(r.backlog.values()),0)
        self.assertLess(r.fractionality,1e-7)

    def test_no_roster_or_eligibility_still_conserves_demand(self):
        r=route_lp({}, {'x':3},{})
        self.assertEqual(r.backlog['x'],3)

    def test_backlog_first_even_when_service_cost_exceeds_penalty(self):
        r=route_lp({('a','x'):1000},{'x':1},{'a':1},backlog_first=True)
        self.assertEqual(r.backlog['x'],0)

    def test_dual_matches_finite_difference_away_from_breakpoint(self):
        costs={('a','x'):1,('b','x'):4}
        r=route_lp(costs,{'x':4},{'a':2,'b':4})
        s=route_lp(costs,{'x':4},{'a':3,'b':4})
        self.assertAlmostEqual(r.objective-s.objective,-r.duals['a'])
        self.assertAlmostEqual(r.objective-s.objective,3)

    def test_noninteger_capacity_rejected(self):
        with self.assertRaises(ValueError): route_lp({('a','x'):1},{'x':2},{'a':1.5})

    def test_fairness_capacity_uses_ceiling_and_existing_capacity(self):
        self.assertEqual(fairness_capacity({'a':10,'b':2},5,1.25),{'a':4,'b':2})

    def test_cross_training_optimum_equals_exhaustive_pair_choices(self):
        costs={('a','x'):1,('a','y'):2,('b','x'):5,('b','y'):1}
        capacities=[{'a':2,'b':2}]; demands=[{'x':2,'y':2}]
        candidates={('a','y'),('b','y')}
        m=cross_train_mip(costs,demands,capacities,candidates,1)
        alternatives=[]
        for pair in candidates:
            available={p:v for p,v in costs.items() if p not in candidates or p==pair}
            alternatives.append(route_lp(available,demands[0],capacities[0],backlog_first=False).objective)
        self.assertAlmostEqual(m.objective,min(alternatives))
        self.assertIn(('b','y'),m.chosen)
        self.assertEqual(m.status,0)

