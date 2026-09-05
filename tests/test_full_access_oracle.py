import contextlib
import io
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from models.Fed import get_model_list, select_clients


def _args(oracle=False):
    return SimpleNamespace(
        num_users=100,
        client_hetero_ration='4:3:3',
        client_chosen_mode='available',
        full_access_oracle=oracle,
    )


def _original_available_selection(args, ration_users):
    """Literal pre-oracle 7-profile available-selection logic for equivalence."""
    proportions = [round(value / 10.0, 2) for value in (4, 3, 3)]
    selected = []
    for model_type in ration_users:
        if int(model_type / 3) == 0:
            selected.append(random.randint(int(args.num_users * sum(proportions[:0])),
                                          args.num_users - 1))
        elif int(model_type / 3) == 1:
            selected.append(random.randint(int(args.num_users * sum(proportions[:1])),
                                          args.num_users - 1))
        elif int(model_type / 3) == 2:
            selected.append(random.randint(int(args.num_users * sum(proportions[:2])),
                                          args.num_users - 1))
    return selected


class FullAccessOracleTests(unittest.TestCase):
    def test_1_flag_false_is_exact_original_available_dispatch(self):
        sequences = [
            [0] * 20,
            [3, 4, 5] * 8,
            [6] * 20,
            [0, 6, 3, 1, 5, 2, 4, 6, 0, 6],
        ]
        for sequence in sequences:
            random.seed(12345)
            expected = _original_available_selection(_args(False), sequence)
            expected_state = random.getstate()
            random.seed(12345)
            actual = select_clients(_args(False), sequence, 7)
            self.assertEqual(actual, expected)
            self.assertEqual(random.getstate(), expected_state)

    def test_2_baseline_full_eligibility_is_70_to_99(self):
        random.seed(17)
        selected = select_clients(_args(False), [6] * 1000, 7)
        self.assertTrue(all(70 <= client_id <= 99 for client_id in selected))
        self.assertEqual(min(selected), 70)
        self.assertEqual(max(selected), 99)

    def test_3_oracle_full_has_uniform_access_to_all_clients(self):
        random.seed(17)
        selected = select_clients(_args(True), [6] * 10000, 7)
        self.assertTrue(all(0 <= client_id <= 99 for client_id in selected))
        self.assertEqual(min(selected), 0)
        self.assertEqual(max(selected), 99)
        self.assertEqual(len(set(selected)), 100)
        self.assertTrue(any(client_id < 70 for client_id in selected))
        blocks = [sum(lower <= client_id <= upper for client_id in selected)
                  for lower, upper in ((0, 39), (40, 69), (70, 99))]
        self.assertTrue(all(count > 0 for count in blocks))

    def test_4_profiles_zero_to_five_are_exactly_unchanged_in_oracle_mode(self):
        sequence = [0, 1, 2, 3, 4, 5] * 100
        random.seed(31415)
        baseline = select_clients(_args(False), sequence, 7)
        random.seed(31415)
        oracle = select_clients(_args(True), sequence, 7)
        self.assertEqual(oracle, baseline)
        for profile, client_id in zip(sequence, oracle):
            if profile <= 2:
                self.assertTrue(0 <= client_id <= 99)
            else:
                self.assertTrue(40 <= client_id <= 99)

    def test_5_duplicate_client_occurrences_remain_allowed(self):
        random.seed(2026)
        selected = select_clients(_args(True), [6] * 100, 7)
        self.assertLess(len(set(selected)), len(selected))

    def test_6_profile_dispatch_sequence_is_not_changed(self):
        profiles = np.random.RandomState(1).choice(range(7), 1000).tolist()
        before = list(profiles)
        random.seed(9)
        baseline_ids = select_clients(_args(False), profiles, 7)
        random.seed(9)
        oracle_ids = select_clients(_args(True), profiles, 7)
        self.assertEqual(profiles, before)
        self.assertEqual(len(baseline_ids), len(profiles))
        self.assertEqual(len(oracle_ids), len(profiles))
        self.assertEqual(
            [profile for profile in profiles],
            [profile for profile in before])

    def test_7_model_integrity_is_the_original_seven_profiles(self):
        args = SimpleNamespace(
            width_ration=[0.4, 0.66, 1.0], depth_saved=[2, 3, 4],
            model='resnet', dataset='cifar100', num_classes=100,
            num_channels=3, device=torch.device('cpu'))
        with contextlib.redirect_stdout(io.StringIO()):
            models, info = get_model_list(args)
        self.assertEqual([(float(width), int(depth)) for width, depth, _ in info], [
            (0.4, 2), (0.4, 3), (0.4, 4),
            (0.66, 2), (0.66, 3), (0.66, 4), (1.0, 2),
        ])
        self.assertAlmostEqual(info[-1][2], 11.424356, places=5)
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    self.assertFalse(module.track_running_stats)

    def test_8_cached_split_integrity(self):
        path = Path('data/cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json')
        self.assertTrue(path.is_file())
        with path.open(encoding='utf-8') as handle:
            split = json.load(handle)['train_data']
        sizes = [len(split[str(index)]) for index in range(100)]
        self.assertEqual((min(sizes), float(np.median(sizes)), max(sizes)), (262, 491.5, 757))
        self.assertEqual(sum(sizes), 50000)

    def test_9_training_path_uses_clean_adaptivefl_aggregation_only(self):
        source = Path('Algorithm/Training_AdaptiveFL.py').read_text(encoding='utf-8')
        self.assertIn('Aggregation_AdaptiveFL', source)
        self.assertNotIn('Aggregation_Support', source)
        self.assertNotIn('Aggregation_TemporalSupport', source)
        self.assertNotIn('full_access_oracle', source)
        self.assertIn('LocalUpdate_AdaptiveFL', source)

    def test_10_dispatch_smoke_has_no_exception_or_non_integer_ids(self):
        profiles = [0, 3, 6, 4, 1, 6, 2, 5, 6, 0]
        for oracle in (False, True):
            random.seed(77)
            selected = select_clients(_args(oracle), profiles, 7)
            self.assertEqual(len(selected), len(profiles))
            self.assertTrue(all(isinstance(client_id, int) for client_id in selected))
            self.assertTrue(all(0 <= client_id < 100 for client_id in selected))

    def test_11_mixed_profile_causal_isolation_preserves_non_full_ids_and_rng(self):
        sequence = [
            6, 0, 3, 1, 5, 2, 4, 6, 0, 6,
            3, 0, 6, 5, 1, 4, 6, 2, 3, 0,
        ]
        oracle_full_ids = []
        for seed in (0, 1, 2, 6, 9, 17, 12345, 2026):
            random.seed(seed)
            baseline = select_clients(_args(False), sequence, 7)
            baseline_state = random.getstate()
            random.seed(seed)
            oracle = select_clients(_args(True), sequence, 7)
            oracle_state = random.getstate()
            self.assertEqual(baseline_state, oracle_state)
            for profile, baseline_id, oracle_id in zip(sequence, baseline, oracle):
                if profile == 6:
                    self.assertTrue(70 <= baseline_id <= 99)
                    self.assertTrue(0 <= oracle_id <= 99)
                    oracle_full_ids.append(oracle_id)
                else:
                    self.assertEqual(baseline_id, oracle_id)
        self.assertTrue(any(client_id < 70 for client_id in oracle_full_ids))


if __name__ == '__main__':
    unittest.main(verbosity=2)
