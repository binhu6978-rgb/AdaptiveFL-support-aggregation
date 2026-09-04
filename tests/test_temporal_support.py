import contextlib
import io
import random
import unittest
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

from models.Fed import (
    Aggregation_AdaptiveFL,
    Aggregation_TemporalSupport,
    Initialize_TemporalSupportState,
    get_model_list,
    split_model,
)


def _clone_state(state):
    return OrderedDict((key, value.detach().clone()) for key, value in state.items())


def _clone_states(states):
    return [_clone_state(state) for state in states]


def _clone_temporal_state(state):
    return {name: _clone_state(values) for name, values in state.items()}


def _max_abs_diff(left, right):
    maximum = 0.0
    for key in left:
        if left[key].is_floating_point():
            maximum = max(maximum, float((left[key] - right[key]).abs().max().item()))
        elif not torch.equal(left[key], right[key]):
            return float('inf')
    return maximum


def _assert_states_equal(test_case, left, right):
    test_case.assertEqual(list(left.keys()), list(right.keys()))
    for key in left:
        test_case.assertTrue(torch.equal(left[key], right[key]), key)


def _assert_temporal_states_equal(test_case, left, right):
    test_case.assertEqual(set(left), set(right))
    for state_name in left:
        _assert_states_equal(test_case, left[state_name], right[state_name])


def _simple_full_inputs():
    global_state = OrderedDict([
        ('linear.weight', torch.tensor([[0.5, -1.0], [1.5, 2.0]])),
        ('linear.bias', torch.tensor([0.25, -0.5])),
    ])
    local_states = [
        OrderedDict([
            ('linear.weight', torch.tensor([[1.0, -0.5], [2.0, 1.0]])),
            ('linear.bias', torch.tensor([1.0, 0.0])),
        ]),
        OrderedDict([
            ('linear.weight', torch.tensor([[0.0, 0.5], [1.0, 3.0]])),
            ('linear.bias', torch.tensor([0.0, -1.0])),
        ]),
    ]
    return global_state, local_states, [3, 7]


def _partial_inputs(support_mass):
    return (
        OrderedDict([('linear.bias', torch.zeros(1))]),
        [OrderedDict([('linear.bias', torch.ones(1))]),
         OrderedDict([('linear.bias', torch.empty(0))])],
        [support_mass, 100 - support_mass],
    )


def _state_with_scalar(numerator, mass):
    return {
        'numerator': OrderedDict([('linear.bias', torch.tensor([numerator]))]),
        'mass': OrderedDict([('linear.bias', torch.tensor([mass]))]),
    }


class TemporalSupportAggregationTests(unittest.TestCase):
    def test_1_full_coverage_strict_equivalence(self):
        global_state, local_states, lens = _simple_full_inputs()
        baseline = Aggregation_AdaptiveFL(local_states, lens, global_state)
        for beta in (0.0, 0.5, 0.85, 0.99):
            output, _, _ = Aggregation_TemporalSupport(
                local_states, lens, global_state,
                Initialize_TemporalSupportState(global_state), beta=beta)
            self.assertLessEqual(_max_abs_diff(output, baseline), 1e-7)

    def test_2_zero_support_keeps_global_despite_history(self):
        global_state = OrderedDict([('linear.bias', torch.zeros(4))])
        local_states = [OrderedDict([('linear.bias', torch.ones(2))]),
                        OrderedDict([('linear.bias', torch.empty(0))])]
        state = {
            'numerator': OrderedDict([('linear.bias', torch.full((4,), 18.0))]),
            'mass': OrderedDict([('linear.bias', torch.full((4,), 2.0))]),
        }
        output, new_state, _ = Aggregation_TemporalSupport(
            local_states, [2, 8], global_state, state, beta=0.85)
        self.assertTrue(torch.equal(output['linear.bias'][2:], global_state['linear.bias'][2:]))
        self.assertTrue(torch.equal(new_state['numerator']['linear.bias'][2:],
                                    torch.full((2,), 15.3)))

    def test_3_beta_zero_is_adaptivefl(self):
        global_state = OrderedDict([('linear.weight', torch.zeros(3, 3)),
                                    ('linear.bias', torch.zeros(3))])
        local_states = [
            OrderedDict([('linear.weight', torch.ones(3, 3)), ('linear.bias', torch.ones(3))]),
            OrderedDict([('linear.weight', torch.full((2, 2), 2.0)),
                         ('linear.bias', torch.full((2,), 2.0))]),
        ]
        baseline = Aggregation_AdaptiveFL(local_states, [2, 8], global_state)
        state = {
            'numerator': OrderedDict((key, torch.full_like(value, 17.0))
                                     for key, value in global_state.items()),
            'mass': OrderedDict((key, torch.full_like(value, 3.0))
                                for key, value in global_state.items()),
        }
        output, _, _ = Aggregation_TemporalSupport(
            local_states, [2, 8], global_state, state, beta=0.0)
        self.assertLessEqual(_max_abs_diff(output, baseline), 1e-7)

    def test_4_fully_shared_coordinate_ignores_temporal_state(self):
        global_state, local_states, lens = _simple_full_inputs()
        baseline = Aggregation_AdaptiveFL(local_states, lens, global_state)
        state = {
            'numerator': OrderedDict((key, torch.full_like(value, 1000.0))
                                     for key, value in global_state.items()),
            'mass': OrderedDict((key, torch.full_like(value, 0.01))
                                for key, value in global_state.items()),
        }
        output, _, _ = Aggregation_TemporalSupport(
            local_states, lens, global_state, state, beta=0.85)
        self.assertLessEqual(_max_abs_diff(output, baseline), 1e-7)

    def test_5_first_observation_has_no_shrink(self):
        for support_mass in (10, 20, 50):
            global_state, local_states, lens = _partial_inputs(support_mass)
            output, new_state, _ = Aggregation_TemporalSupport(
                local_states, lens, global_state,
                Initialize_TemporalSupportState(global_state), beta=0.85)
            self.assertAlmostEqual(float(output['linear.bias'][0]), 1.0, places=7)
            self.assertAlmostEqual(float(new_state['numerator']['linear.bias'][0]),
                                   support_mass / 100.0, places=7)
            self.assertAlmostEqual(float(new_state['mass']['linear.bias'][0]),
                                   support_mass / 100.0, places=7)

    def test_6_support_weighted_evidence_gives_larger_support_more_influence(self):
        means = []
        for support_mass in (10, 50):
            global_state, local_states, lens = _partial_inputs(support_mass)
            _, new_state, _ = Aggregation_TemporalSupport(
                local_states, lens, global_state, _state_with_scalar(2.0, 4.0), beta=0.85)
            means.append(float(new_state['numerator']['linear.bias'][0]
                               / new_state['mass']['linear.bias'][0]))
        self.assertGreater(means[1] - 0.5, means[0] - 0.5)
        self.assertAlmostEqual(means[0], 1.8 / 3.5, places=7)
        self.assertAlmostEqual(means[1], 2.2 / 3.9, places=7)

    def test_7_zero_support_evidence_ages_without_updating_global(self):
        global_state = OrderedDict([('linear.bias', torch.zeros(1))])
        local_states = [OrderedDict([('linear.bias', torch.empty(0))])]
        output, new_state, _ = Aggregation_TemporalSupport(
            local_states, [100], global_state, _state_with_scalar(2.0, 4.0), beta=0.85)
        self.assertAlmostEqual(float(new_state['numerator']['linear.bias'][0]), 1.7, places=7)
        self.assertAlmostEqual(float(new_state['mass']['linear.bias'][0]), 3.4, places=6)
        self.assertAlmostEqual(float(new_state['numerator']['linear.bias'][0]
                                     / new_state['mass']['linear.bias'][0]), 0.5, places=7)
        self.assertTrue(torch.equal(output['linear.bias'], global_state['linear.bias']))

    def test_8_positive_support_has_no_threshold_jump(self):
        values = []
        for support_mass in (1, 10, 20, 50, 80, 95, 99, 100):
            global_state, local_states, lens = _partial_inputs(support_mass)
            output, _, _ = Aggregation_TemporalSupport(
                local_states, lens, global_state, _state_with_scalar(2.0, 4.0), beta=0.85)
            support = support_mass / 100.0
            temporal_mean = (1.7 + support) / (3.4 + support)
            expected = 1.0 if support == 1.0 else support + (1.0 - support) * temporal_mean
            self.assertAlmostEqual(float(output['linear.bias'][0]), expected, places=6)
            values.append(float(output['linear.bias'][0]))
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[-1], 1.0, places=7)

    def test_9_real_seven_profile_resnet_beta_zero_equivalence(self):
        args = SimpleNamespace(width_ration=[0.4, 0.66, 1.0], depth_saved=[2, 3, 4],
                               model='resnet', dataset='cifar100', num_classes=100,
                               num_channels=3, device=torch.device('cpu'))
        with contextlib.redirect_stdout(io.StringIO()):
            net_glob_list, net_slim_info = get_model_list(args)
        expected_profiles = [(0.4, 2), (0.4, 3), (0.4, 4), (0.66, 2),
                             (0.66, 3), (0.66, 4), (1.0, 2)]
        self.assertEqual([(float(info[0]), int(info[1])) for info in net_slim_info],
                         expected_profiles)
        global_state = _clone_state(net_glob_list[-1].state_dict())
        local_states = []
        for index, net in enumerate(net_glob_list):
            local_state = split_model(global_state, net.state_dict())
            for key, value in local_state.items():
                if value.is_floating_point():
                    local_state[key] = value + (index + 1) * 1e-4
            local_states.append(local_state)
        lens = [11, 12, 13, 14, 15, 16, 17]
        baseline = Aggregation_AdaptiveFL(local_states, lens, global_state)
        output, state, _ = Aggregation_TemporalSupport(
            local_states, lens, global_state,
            Initialize_TemporalSupportState(global_state), beta=0.0)
        maximum = _max_abs_diff(output, baseline)
        self.assertLessEqual(maximum, 1e-6)
        self.assertTrue(all(torch.isfinite(value).all() for group in state.values()
                            for value in group.values()))
        print('real_7_profile_beta0_max_abs_diff={:.9f}'.format(maximum))

    def test_10_inputs_are_not_modified(self):
        global_state, local_states, lens = _simple_full_inputs()
        state = Initialize_TemporalSupportState(global_state)
        for values in state.values():
            for value in values.values():
                value.fill_(0.75)
        global_before = _clone_state(global_state)
        locals_before = _clone_states(local_states)
        state_before = _clone_temporal_state(state)
        Aggregation_TemporalSupport(local_states, lens, global_state, state, beta=0.85)
        _assert_states_equal(self, global_state, global_before)
        _assert_temporal_states_equal(self, state, state_before)
        for local_state, local_before in zip(local_states, locals_before):
            _assert_states_equal(self, local_state, local_before)

    def test_11_rng_state_is_unchanged(self):
        global_state, local_states, lens = _simple_full_inputs()
        state = Initialize_TemporalSupportState(global_state)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        Aggregation_TemporalSupport(local_states, lens, global_state, state, beta=0.85)
        numpy_after = np.random.get_state()
        self.assertEqual(python_before, random.getstate())
        self.assertEqual(numpy_before[0], numpy_after[0])
        self.assertTrue(np.array_equal(numpy_before[1], numpy_after[1]))
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))

    def test_12_finite_checks_reject_nonfinite_state(self):
        global_state, local_states, lens = _simple_full_inputs()
        state = Initialize_TemporalSupportState(global_state)
        state['numerator']['linear.bias'][0] = float('nan')
        with self.assertRaises(FloatingPointError):
            Aggregation_TemporalSupport(local_states, lens, global_state, state, beta=0.85)


if __name__ == '__main__':
    unittest.main(verbosity=2)
