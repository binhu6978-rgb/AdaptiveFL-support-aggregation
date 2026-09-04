import contextlib
import copy
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
    Initialize_TemporalSupportMemory,
    get_model_list,
    split_model,
)


def _clone_state(state):
    return OrderedDict((key, value.detach().clone()) for key, value in state.items())


def _clone_states(states):
    return [_clone_state(state) for state in states]


def _max_abs_diff(left, right):
    maximum = 0.0
    for key in left:
        if left[key].is_floating_point():
            maximum = max(
                maximum,
                float((left[key] - right[key]).abs().max().item()))
        else:
            if not torch.equal(left[key], right[key]):
                return float('inf')
    return maximum


def _assert_states_equal(test_case, left, right):
    test_case.assertEqual(list(left.keys()), list(right.keys()))
    for key in left:
        test_case.assertTrue(torch.equal(left[key], right[key]), key)


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


class TemporalSupportAggregationTests(unittest.TestCase):
    def test_1_full_coverage_strict_equivalence(self):
        global_state, local_states, lens = _simple_full_inputs()
        baseline = Aggregation_AdaptiveFL(local_states, lens, global_state)
        for beta in (0.0, 0.5, 0.85, 0.99):
            memory = Initialize_TemporalSupportMemory(global_state)
            output, _, _ = Aggregation_TemporalSupport(
                local_states, lens, global_state, memory, beta=beta)
            maximum = _max_abs_diff(output, baseline)
            self.assertLessEqual(maximum, 1e-7)

    def test_2_zero_support_keeps_global(self):
        global_state = OrderedDict([
            ('linear.bias', torch.zeros(4)),
        ])
        local_states = [
            OrderedDict([('linear.bias', torch.ones(2))]),
            OrderedDict([('linear.bias', torch.empty(0))]),
        ]
        memory = OrderedDict([('linear.bias', torch.full((4,), 9.0))])
        output, new_memory, _ = Aggregation_TemporalSupport(
            local_states, [2, 8], global_state, memory, beta=0.85)
        self.assertTrue(torch.equal(output['linear.bias'][2:], global_state['linear.bias'][2:]))
        self.assertTrue(torch.equal(new_memory['linear.bias'][2:], memory['linear.bias'][2:]))

    def test_3_beta_zero_is_adaptivefl(self):
        global_state = OrderedDict([
            ('linear.weight', torch.zeros(3, 3)),
            ('linear.bias', torch.zeros(3)),
        ])
        local_states = [
            OrderedDict([
                ('linear.weight', torch.ones(3, 3)),
                ('linear.bias', torch.ones(3)),
            ]),
            OrderedDict([
                ('linear.weight', torch.full((2, 2), 2.0)),
                ('linear.bias', torch.full((2,), 2.0)),
            ]),
        ]
        baseline = Aggregation_AdaptiveFL(local_states, [2, 8], global_state)
        memory = OrderedDict([
            ('linear.weight', torch.full((3, 3), 17.0)),
            ('linear.bias', torch.full((3,), 17.0)),
        ])
        output, _, _ = Aggregation_TemporalSupport(
            local_states, [2, 8], global_state, memory, beta=0.0)
        self.assertLessEqual(_max_abs_diff(output, baseline), 1e-7)

    def test_4_fully_shared_region_ignores_memory(self):
        global_state, local_states, lens = _simple_full_inputs()
        baseline = Aggregation_AdaptiveFL(local_states, lens, global_state)
        memory = OrderedDict((key, torch.full_like(value, 1000.0))
                             for key, value in global_state.items())
        output, _, _ = Aggregation_TemporalSupport(
            local_states, lens, global_state, memory, beta=0.85)
        self.assertLessEqual(_max_abs_diff(output, baseline), 1e-7)

    def test_5_partial_support_matches_hand_calculation(self):
        global_state = OrderedDict([('linear.bias', torch.zeros(1))])
        local_states = [
            OrderedDict([('linear.bias', torch.ones(1))]),
            OrderedDict([('linear.bias', torch.empty(0))]),
        ]
        memory = OrderedDict([('linear.bias', torch.tensor([0.5]))])
        output, new_memory, _ = Aggregation_TemporalSupport(
            local_states, [2, 8], global_state, memory, beta=0.85)
        self.assertAlmostEqual(float(new_memory['linear.bias'][0]), 0.575, places=6)
        self.assertAlmostEqual(float(output['linear.bias'][0]), 0.66, places=6)

    def test_6_continuous_support_weighting(self):
        values = []
        for support_mass in (20, 50, 80, 95, 99, 100):
            global_state = OrderedDict([('linear.bias', torch.zeros(1))])
            if support_mass == 100:
                local_states = [OrderedDict([('linear.bias', torch.ones(1))])]
                lens = [100]
            else:
                local_states = [
                    OrderedDict([('linear.bias', torch.ones(1))]),
                    OrderedDict([('linear.bias', torch.empty(0))]),
                ]
                lens = [support_mass, 100 - support_mass]
            memory = OrderedDict([('linear.bias', torch.tensor([0.5]))])
            output, _, _ = Aggregation_TemporalSupport(
                local_states, lens, global_state, memory, beta=0.85)
            expected = (support_mass / 100.0) + (1.0 - support_mass / 100.0) * 0.575
            self.assertAlmostEqual(float(output['linear.bias'][0]), expected, places=6)
            values.append(float(output['linear.bias'][0]))
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[-1], 1.0, places=7)

    def test_7_real_seven_profile_resnet_beta_zero_equivalence(self):
        args = SimpleNamespace(
            width_ration=[0.4, 0.66, 1.0], depth_saved=[2, 3, 4],
            model='resnet', dataset='cifar100', num_classes=100,
            num_channels=3, device=torch.device('cpu'))
        with contextlib.redirect_stdout(io.StringIO()):
            net_glob_list, net_slim_info = get_model_list(args)
        expected_profiles = [
            (0.4, 2), (0.4, 3), (0.4, 4),
            (0.66, 2), (0.66, 3), (0.66, 4), (1.0, 2),
        ]
        self.assertEqual(len(net_glob_list), 7)
        self.assertEqual(
            [(float(info[0]), int(info[1])) for info in net_slim_info],
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
        memory = Initialize_TemporalSupportMemory(global_state)
        output, new_memory, _ = Aggregation_TemporalSupport(
            local_states, lens, global_state, memory, beta=0.0)
        maximum = _max_abs_diff(output, baseline)
        self.assertLessEqual(maximum, 1e-6)
        self.assertTrue(all(torch.isfinite(value).all()
                            for value in output.values()
                            if value.is_floating_point()))
        self.assertTrue(all(torch.isfinite(value).all()
                            for value in new_memory.values()))
        print('real_7_profile_beta0_max_abs_diff={:.9f}'.format(maximum))

    def test_8_inputs_are_not_modified(self):
        global_state, local_states, lens = _simple_full_inputs()
        memory = OrderedDict((key, torch.full_like(value, 0.75))
                             for key, value in global_state.items())
        global_before = _clone_state(global_state)
        locals_before = _clone_states(local_states)
        memory_before = _clone_state(memory)
        Aggregation_TemporalSupport(local_states, lens, global_state, memory, beta=0.85)
        _assert_states_equal(self, global_state, global_before)
        _assert_states_equal(self, memory, memory_before)
        for local_state, local_before in zip(local_states, locals_before):
            _assert_states_equal(self, local_state, local_before)

    def test_9_rng_state_is_unchanged(self):
        global_state, local_states, lens = _simple_full_inputs()
        memory = Initialize_TemporalSupportMemory(global_state)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.get_rng_state().clone()
        Aggregation_TemporalSupport(local_states, lens, global_state, memory, beta=0.85)
        numpy_after = np.random.get_state()
        self.assertEqual(python_before, random.getstate())
        self.assertEqual(numpy_before[0], numpy_after[0])
        self.assertTrue(np.array_equal(numpy_before[1], numpy_after[1]))
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
