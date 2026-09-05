import contextlib
import io
import inspect
import json
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from models.Fed import (Aggregation_AdaptiveFL, get_model_list, sample_profile_slots,
                        select_clients)
from utils.options import args_parser


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / 'data' / 'cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json'


def args(quota=False):
    return SimpleNamespace(
        fixed_profile_quota_oracle=quota,
        client_chosen_mode='available', client_hetero_ration='4:3:3', num_users=100,
        width_ration=[0.4, 0.66, 1.0], depth_saved=[2, 3, 4], model='resnet',
        dataset='cifar100', num_classes=100, num_channels=3, device=torch.device('cpu'))


class FixedProfileQuotaOracleTests(unittest.TestCase):
    def test_flag_defaults_to_false(self):
        with patch.object(sys, 'argv', ['test']):
            self.assertFalse(args_parser().fixed_profile_quota_oracle)

    def test_default_sampling_is_the_original_numpy_path(self):
        np.random.seed(17)
        expected = np.random.choice(range(7), 10)
        np.random.seed(17)
        actual = sample_profile_slots(args(False), 7, 10)
        self.assertTrue(np.array_equal(actual, expected))

    def test_quota_has_exact_group_counts_every_round(self):
        np.random.seed(1)
        for _ in range(100):
            profiles = sample_profile_slots(args(True), 7, 10)
            self.assertEqual(len(profiles), 10)
            self.assertEqual(sum(p in (0, 1, 2) for p in profiles), 4)
            self.assertEqual(sum(p in (3, 4, 5) for p in profiles), 3)
            self.assertEqual(sum(p == 6 for p in profiles), 3)

    def test_quota_profile_membership_and_shuffle(self):
        np.random.seed(9)
        profiles = sample_profile_slots(args(True), 7, 10).tolist()
        self.assertTrue(all(p in (0, 1, 2) for p in [p for p in profiles if p < 3]))
        self.assertTrue(all(p in (3, 4, 5) for p in [p for p in profiles if 3 <= p < 6]))
        self.assertEqual(profiles.count(6), 3)
        self.assertNotEqual(profiles, sorted(profiles))

    def test_quota_rejects_noncanonical_shape(self):
        with self.assertRaises(ValueError):
            sample_profile_slots(args(True), 6, 10)
        with self.assertRaises(ValueError):
            sample_profile_slots(args(True), 7, 9)

    def test_client_eligibility_is_unchanged(self):
        profiles = [0, 1, 2, 3, 4, 5, 6]
        random.seed(4)
        clients = select_clients(args(True), profiles, 7)
        self.assertTrue(all(0 <= clients[i] <= 99 for i in range(3)))
        self.assertTrue(all(40 <= clients[i] <= 99 for i in range(3, 6)))
        self.assertTrue(70 <= clients[6] <= 99)

    def test_duplicate_clients_remain_allowed(self):
        random.seed(0)
        clients = select_clients(args(True), [6] * 1000, 7)
        self.assertLess(len(set(clients)), len(clients))

    def test_training_remains_clean_adaptivefl(self):
        source = (ROOT / 'Algorithm' / 'Training_AdaptiveFL.py').read_text(encoding='utf-8')
        self.assertIn('Aggregation_AdaptiveFL', source)
        self.assertIn('sample_profile_slots(args, len(net_glob_list), m)', source)
        self.assertIn('ration_users = np.random.choice(range(len(net_glob_list)), m)', source)
        self.assertNotIn('full_access_oracle', source)
        self.assertNotIn('TemporalSupport', source)
        self.assertNotIn('SupportAggregation', source)
        self.assertNotIn('fixed_profile_quota_oracle', inspect.getsource(Aggregation_AdaptiveFL))
        update_source = (ROOT / 'models' / 'Update.py').read_text(encoding='utf-8')
        self.assertNotIn('fixed_profile_quota_oracle', update_source)

    def test_seven_profile_model_and_classifier_bias_integrity(self):
        with contextlib.redirect_stdout(io.StringIO()):
            models, _ = get_model_list(args(True))
        self.assertEqual(len(models), 7)
        for model in models:
            self.assertEqual(tuple(model.fc.bias.shape), (100,))
            batch_norms = [module for module in model.modules()
                           if isinstance(module, torch.nn.BatchNorm2d)]
            self.assertTrue(batch_norms)
            self.assertTrue(all(not module.track_running_stats for module in batch_norms))

    def test_cached_split_integrity(self):
        payload = json.loads(SPLIT.read_text(encoding='utf-8'))
        sizes = [len(payload['train_data'][str(i)]) for i in range(100)]
        self.assertEqual(min(sizes), 262)
        self.assertEqual(np.median(sizes), 491.5)
        self.assertEqual(max(sizes), 757)
        self.assertEqual(sum(sizes), 50000)


if __name__ == '__main__':
    unittest.main()
