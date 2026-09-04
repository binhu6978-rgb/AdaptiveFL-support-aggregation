import json
import os

import numpy as np
import torch


COSINE_DENOMINATOR_EPS = 1e-12


def _distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    return {
        'count': int(array.size),
        'mean': float(array.mean()),
        'median': float(np.median(array)),
        'min': float(array.min()),
        'p10': float(np.percentile(array, 10)),
        'p25': float(np.percentile(array, 25)),
        'negative_ratio': float((array < 0.0).mean()),
        'ratio_below_0_3': float((array < 0.3).mean()),
    }


def _format_round_distribution(values):
    stats = _distribution(values)
    if stats is None:
        return 'N/A'
    return ('mean={mean:.6f}, median={median:.6f}, min={min:.6f}, '
            'P10={p10:.6f}, P25={p25:.6f}, '
            'negative_ratio={negative_ratio:.6f}, '
            'ratio(cos < 0.3)={ratio_below_0_3:.6f}').format(**stats)


def _format_group(values):
    stats = _distribution(values)
    if stats is None:
        return 'N/A'
    return 'mean={mean:.6f}, median={median:.6f}'.format(**stats)


class CrossScaleConsistencyDiagnostics:
    """Read-only shared-core update consistency measurements."""

    def __init__(self, core_template, net_slim_info, result_dir):
        self.net_slim_info = list(net_slim_info)
        self.result_dir = result_dir
        self.core_shapes = {
            key: tuple(value.shape)
            for key, value in core_template.items()
            if value.is_floating_point()
        }
        self.core_keys = list(self.core_shapes.keys())
        self.shared_core_coordinates = int(sum(
            np.prod(shape, dtype=np.int64)
            for shape in self.core_shapes.values()))
        if not self.core_keys or self.shared_core_coordinates <= 0:
            raise AssertionError('Shared core contains no floating-point coordinates')

        self.client_records = []
        self.pair_records = []
        self.integrity_records = []
        print('Shared core total coordinates: {}'.format(
            self.shared_core_coordinates))

    @staticmethod
    def _snapshot(states):
        return [
            {key: value.detach().clone() for key, value in state.items()}
            for state in states
        ]

    @staticmethod
    def _checksum(states):
        return float(sum(
            value.detach().to(dtype=torch.float64).sum().item()
            for state in states
            for value in state.values()
            if value.is_floating_point()
        ))

    @staticmethod
    def _assert_unchanged(states, snapshots):
        max_abs_diff = 0.0
        for state, snapshot in zip(states, snapshots):
            if state.keys() != snapshot.keys():
                raise AssertionError('Diagnostic changed state_dict keys')
            for key, value in state.items():
                before = snapshot[key]
                if not torch.equal(value.detach(), before):
                    if value.numel() > 0:
                        difference = (value.detach().to(dtype=torch.float32)
                                      - before.to(dtype=torch.float32)).abs().max().item()
                        max_abs_diff = max(max_abs_diff, float(difference))
                    raise AssertionError(
                        'Diagnostic modified input tensor {} (max_abs_diff={})'.format(
                            key, max_abs_diff))
        return max_abs_diff

    def _flatten_core(self, state):
        pieces = []
        for key in self.core_keys:
            value = state[key].detach()
            target_shape = self.core_shapes[key]
            if value.dim() > 1:
                slices = [slice(None)] * value.dim()
                slices[0] = slice(0, target_shape[0])
                slices[1] = slice(0, target_shape[1])
                core_value = value[tuple(slices)]
            elif value.dim() == 1:
                core_value = value[:target_shape[0]]
            else:
                core_value = value
            if tuple(core_value.shape) != target_shape:
                raise AssertionError(
                    'Shared-core shape mismatch for {}: expected {}, got {}'.format(
                        key, target_shape, tuple(core_value.shape)))
            pieces.append(core_value.to(dtype=torch.float32).reshape(-1))
        vector = torch.cat(pieces)
        if vector.numel() != self.shared_core_coordinates:
            raise AssertionError(
                'Shared-core coordinate mismatch: expected {}, got {}'.format(
                    self.shared_core_coordinates, vector.numel()))
        return vector

    @staticmethod
    def _safe_cosine(left, right):
        denominator = torch.clamp(
            torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right),
            min=COSINE_DENOMINATOR_EPS)
        cosine = torch.dot(left, right) / denominator
        if not torch.isfinite(cosine):
            raise AssertionError('Non-finite cosine encountered')
        return float(cosine.item())

    def measure(self, round_idx, w_locals, lens, client_ids,
                model_indices, global_model_param):
        if round_idx < 1:
            raise AssertionError('Round 0 diagnostics must be skipped')
        if not (len(w_locals) == len(lens) == len(client_ids)
                == len(model_indices)):
            raise AssertionError('Diagnostic input lengths do not match')
        if len(w_locals) < 2:
            raise AssertionError('LOO consensus requires at least two clients')

        all_states = list(w_locals) + [global_model_param]
        snapshots = self._snapshot(all_states)
        checksum_before = self._checksum(all_states)

        with torch.no_grad():
            global_core = self._flatten_core(global_model_param)
            update_vectors = []
            for local_state in w_locals:
                update = self._flatten_core(local_state) - global_core
                if update.shape != global_core.shape:
                    raise AssertionError('Client shared-core vector shape mismatch')
                if not torch.isfinite(update).all():
                    raise AssertionError('Non-finite shared-core update encountered')
                update_vectors.append(update)

            updates = torch.stack(update_vectors)
            norms = torch.linalg.vector_norm(updates, dim=1)
            if not torch.isfinite(norms).all():
                raise AssertionError('Non-finite update norm encountered')

            pair_round_records = []
            for left_idx in range(len(w_locals)):
                left_model_idx = int(model_indices[left_idx])
                left_profile = self.net_slim_info[left_model_idx]
                for right_idx in range(left_idx + 1, len(w_locals)):
                    right_model_idx = int(model_indices[right_idx])
                    right_profile = self.net_slim_info[right_model_idx]
                    cosine = self._safe_cosine(
                        updates[left_idx], updates[right_idx])
                    record = {
                        'round': int(round_idx),
                        'client_i': int(client_ids[left_idx]),
                        'client_j': int(client_ids[right_idx]),
                        'model_idx_i': left_model_idx,
                        'model_idx_j': right_model_idx,
                        'width_i': float(left_profile[0]),
                        'width_j': float(right_profile[0]),
                        'depth_i': int(left_profile[1]),
                        'depth_j': int(right_profile[1]),
                        'cosine': cosine,
                        'same_profile': left_model_idx == right_model_idx,
                        'same_width': float(left_profile[0]) == float(right_profile[0]),
                        'cross_width': float(left_profile[0]) != float(right_profile[0]),
                    }
                    pair_round_records.append(record)
                    self.pair_records.append(record)

            weights = updates.new_tensor(lens, dtype=torch.float32)
            total_weight = weights.sum()
            weighted_update_sum = torch.sum(
                updates * weights.reshape(-1, 1), dim=0)
            loo_round_records = []
            for client_idx in range(len(w_locals)):
                denominator = total_weight - weights[client_idx]
                if denominator.item() <= 0:
                    raise AssertionError('LOO consensus weight must be positive')
                consensus = (
                    weighted_update_sum
                    - weights[client_idx] * updates[client_idx]
                ) / denominator
                cosine = self._safe_cosine(updates[client_idx], consensus)
                model_idx = int(model_indices[client_idx])
                profile = self.net_slim_info[model_idx]
                record = {
                    'round': int(round_idx),
                    'client_id': int(client_ids[client_idx]),
                    'model_idx': model_idx,
                    'width': float(profile[0]),
                    'depth': int(profile[1]),
                    'lens': int(lens[client_idx]),
                    'update_norm': float(norms[client_idx].item()),
                    'loo_cosine': cosine,
                }
                loo_round_records.append(record)
                self.client_records.append(record)

        max_abs_diff = self._assert_unchanged(all_states, snapshots)
        checksum_after = self._checksum(all_states)
        if checksum_before != checksum_after or max_abs_diff != 0.0:
            raise AssertionError(
                'Diagnostic input integrity failed: checksum {} -> {}, max_abs_diff={}'.format(
                    checksum_before, checksum_after, max_abs_diff))
        integrity = {
            'round': int(round_idx),
            'checksum_before': checksum_before,
            'checksum_after': checksum_after,
            'max_abs_diff': max_abs_diff,
        }
        self.integrity_records.append(integrity)

        pair_values = [record['cosine'] for record in pair_round_records]
        loo_values = [record['loo_cosine'] for record in loo_round_records]
        same_width_values = [
            record['cosine'] for record in pair_round_records
            if record['same_width']]
        cross_width_values = [
            record['cosine'] for record in pair_round_records
            if record['cross_width']]
        print('Round {} consistency:'.format(round_idx))
        print('Pairwise cosine: {}'.format(
            _format_round_distribution(pair_values)))
        print('LOO cosine: {}'.format(
            _format_round_distribution(loo_values)))
        print('Same-width pairs: {}'.format(
            _format_group(same_width_values)))
        print('Cross-width pairs: {}'.format(
            _format_group(cross_width_values)))
        print('Diagnostic input integrity: checksum_before={:.9f}, '
              'checksum_after={:.9f}, max_abs_diff={:.1f}'.format(
                  checksum_before, checksum_after, max_abs_diff))

    def save(self):
        os.makedirs(self.result_dir, exist_ok=True)
        common_metadata = {
            'experiment': 'Cross-Scale Update Consistency Diagnostic',
            'diagnostic_rounds': list(range(1, 50)),
            'shared_core_coordinates': self.shared_core_coordinates,
            'profiles': [
                {
                    'model_idx': index,
                    'width': float(profile[0]),
                    'depth': int(profile[1]),
                    'parameters_million': float(profile[2]),
                }
                for index, profile in enumerate(self.net_slim_info)
            ],
        }
        client_payload = dict(common_metadata)
        client_payload['records'] = self.client_records
        client_payload['input_integrity'] = self.integrity_records
        pair_payload = dict(common_metadata)
        pair_payload['records'] = self.pair_records

        client_path = os.path.join(
            self.result_dir, 'client_loo_consistency.json')
        pair_path = os.path.join(
            self.result_dir, 'pairwise_consistency.json')
        with open(client_path, 'w', encoding='utf-8') as output:
            json.dump(client_payload, output, indent=2)
        with open(pair_path, 'w', encoding='utf-8') as output:
            json.dump(pair_payload, output, indent=2)
        print('Saved client LOO consistency: {}'.format(client_path))
        print('Saved pairwise consistency: {}'.format(pair_path))
