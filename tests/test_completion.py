import unittest
from random import Random
from dataclasses import replace
from examples.demo import demo_inputs, DemoPerception
from majority.models import Coefficients, RunConfig
from majority.simulation import Simulation
from majority.randomness import RandomStreams
from majority.preparation import SearchConfig, search_seeds, seed_diagnostics
from majority.analysis import RunSummary, paired_inference

class AddedTests(unittest.TestCase):
    def test_concurrent_decisions_equal_serial(self):
        network, profiles, bank = demo_inputs()
        def run(workers):
            return tuple(Simulation(network, profiles, frozenset({0}), bank,
                Coefficients((0,0,1,0),1,1), RunConfig(rounds=3), DemoPerception(),
                RandomStreams(42), 'U0', workers=workers).run())
        self.assertEqual(run(1), run(4))

    def test_seed_constraints(self):
        network, profiles, _ = demo_inputs()
        result = search_seeds(network, profiles, 2, SearchConfig(starts=4, swaps=20,retry_starts=8), Random(1))
        for seeds in (result.high,result.low,result.random):
            if seeds is not None:
                self.assertEqual(len(seeds),2)
                self.assertFalse(seeds & network.observers)
        if result.high is not None:
            a,b=seed_diagnostics(network,profiles,result.high),seed_diagnostics(network,profiles,result.low)
            self.assertGreater(a.illusion,b.illusion)
            self.assertLessEqual(2*abs(a.fan_total-b.fan_total)/(a.fan_total+b.fan_total),.05)

    def test_paired_bootstrap(self):
        runs=[RunSummary('g',str(i),i,branch,8,3,value)
              for i in range(4) for branch,value in [('TH',.7+i*.01),('TL',.2+i*.01)]]
        result=paired_inference(runs,[('TH','TL')],repeats=100)[0]
        self.assertAlmostEqual(result.difference,.5)
        for bound in result.confidence_interval:
            self.assertAlmostEqual(bound,.5)
        self.assertGreater(result.adjusted_pvalue,0)
