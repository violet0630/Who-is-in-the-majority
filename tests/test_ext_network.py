import unittest

from majority.network import _connected, network_statistics
from scripts.ext_network import (NETWORK_CONDITIONS, group_seed, network_seed,
                                 profile_seed, search_seed)
from scripts.majority_study import make_network


class NetworkConditionTests(unittest.TestCase):
    def test_condition_overrides(self):
        self.assertEqual(set(NETWORK_CONDITIONS), {"NW1", "NW2", "NW3", "NW4"})
        self.assertEqual(NETWORK_CONDITIONS["NW1"], {"s_in": 0.5})
        self.assertEqual(NETWORK_CONDITIONS["NW2"], {"s_in": 1.5})
        self.assertEqual(NETWORK_CONDITIONS["NW3"], {"w": 0.05})
        self.assertEqual(NETWORK_CONDITIONS["NW4"], {"w": 0.5})

    def test_generated_networks_meet_frozen_constraints(self):
        base = {"mean_degree": 12, "s_out": 0.35, "s_in": 1.0, "w": 0.2, "fraction": 0.2,
                "rho": 6, "kappa": 0.5, "gamma": 1, "theta": [1, 0, -1, 0], "rounds": 10,
                "community_fractions": [0.15, 0.2, 0.25, 0.4]}
        for name, override in NETWORK_CONDITIONS.items():
            network = make_network(1000, {**base, **override}, network_seed(name, 0))
            edges = sum(map(len, network.following))
            self.assertTrue(11400 <= edges <= 12600, (name, edges))
            self.assertTrue(all(network.following), name)      # 每节点至少关注一人
            self.assertTrue(_connected(network.following), name)
            self.assertEqual(len(network.observers), 50, name)
            stats = network_statistics(network)
            self.assertTrue(0 <= stats.cross_community_fraction <= 1)

    def test_seed_formulas_are_deterministic(self):
        self.assertEqual(network_seed("NW1", 0), network_seed("NW1", 0))
        self.assertEqual(network_seed("NW1", 1), network_seed("NW1", 0) + 1)
        self.assertNotEqual(search_seed("NW1", 0, 0), group_seed("NW1", 0, 0))
        self.assertNotEqual(profile_seed("NW1", 0, 0), search_seed("NW1", 0, 0))


if __name__ == "__main__":
    unittest.main()
