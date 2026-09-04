"""Exact offline width-group geometry from saved client norms and cosines only."""

import json
import math
import os
from collections import Counter, defaultdict


WIDTHS = (0.4, 0.66, 1.0)
PAIR_NAMES = ((0.4, 0.66), (0.4, 1.0), (0.66, 1.0))
PHASES = {'early': range(1, 16), 'middle': range(16, 33), 'late': range(33, 50)}


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def describe(values):
    values = list(values)
    if not values:
        return {'valid_rounds': 0}
    mean_value = sum(values) / len(values)
    return {
        'valid_rounds': len(values), 'mean': mean_value,
        'std': math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values)),
        'median': percentile(values, 0.5), 'p10': percentile(values, 0.1),
        'p25': percentile(values, 0.25), 'p75': percentile(values, 0.75),
        'p90': percentile(values, 0.9), 'min': min(values), 'max': max(values),
        'negative_ratio': sum(value < 0 for value in values) / len(values),
    }


def rank_average(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2.0
        for original_index, _ in ordered[index:end]:
            ranks[original_index] = average
        index = end
    return ranks


def pearson(left, right):
    if len(left) < 2:
        return None
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left)
                            * sum((y - right_mean) ** 2 for y in right))
    return numerator / denominator if denominator else None


def trend(round_values):
    if len(round_values) < 2:
        return {'ols_slope_per_round': None, 'spearman_round_vs_cosine': None}
    rounds, values = zip(*round_values)
    round_mean = sum(rounds) / len(rounds)
    value_mean = sum(values) / len(values)
    denominator = sum((round_value - round_mean) ** 2 for round_value in rounds)
    slope = (sum((round_value - round_mean) * (value - value_mean)
                 for round_value, value in round_values) / denominator)
    return {
        'ols_slope_per_round': slope,
        'spearman_round_vs_cosine': pearson(rank_average(list(rounds)), rank_average(list(values))),
    }


def nonnegative(value, scale, label):
    tolerance = 1e-8 * max(1.0, scale)
    if value < -tolerance:
        raise ValueError('{} is materially negative: {}'.format(label, value))
    return max(0.0, value)


def vector_geometry(gram, weights_left, weights_right=None):
    if weights_right is None:
        weights_right = weights_left
    dot = sum(weights_left[i] * gram[i][j] * weights_right[j]
              for i in range(3) for j in range(3))
    return dot


def counterfactual(gram, natural, alternate):
    natural_sq = nonnegative(vector_geometry(gram, natural), 1.0, 'natural norm squared')
    alternate_sq = nonnegative(vector_geometry(gram, alternate), 1.0, 'counterfactual norm squared')
    cross = vector_geometry(gram, natural, alternate)
    if not natural_sq or not alternate_sq:
        return None
    cosine = max(-1.0, min(1.0, cross / math.sqrt(natural_sq * alternate_sq)))
    return {
        'cosine_to_natural': cosine,
        'angle_degrees': math.degrees(math.acos(cosine)),
        'norm_ratio_to_natural': math.sqrt(alternate_sq / natural_sq),
    }


def phase_counterfactual(values):
    if not values:
        return {'valid_rounds': 0}
    angles = [value['angle_degrees'] for value in values]
    ratios = [value['norm_ratio_to_natural'] for value in values]
    cosines = [value['cosine_to_natural'] for value in values]
    return {
        'valid_rounds': len(values), 'median_cosine_to_natural': percentile(cosines, 0.5),
        'median_angle_degrees': percentile(angles, 0.5), 'mean_angle_degrees': sum(angles) / len(angles),
        'angle_p25_degrees': percentile(angles, 0.25), 'angle_p75_degrees': percentile(angles, 0.75),
        'median_norm_ratio_to_natural': percentile(ratios, 0.5),
    }


def phase_cosines(values):
    if not values:
        return {'valid_rounds': 0}
    cosines = [value for _, value in values]
    return {
        'valid_rounds': len(cosines), 'mean_cosine': sum(cosines) / len(cosines),
        'median_cosine': percentile(cosines, 0.5), 'p25': percentile(cosines, 0.25),
        'p75': percentile(cosines, 0.75),
        'negative_ratio': sum(value < 0 for value in cosines) / len(cosines),
    }


def main():
    root = os.path.join('results', 'cross_scale_consistency_50')
    pair_path = os.path.join(root, 'pairwise_consistency.json')
    client_path = os.path.join(root, 'client_loo_consistency.json')
    with open(pair_path, encoding='utf-8') as handle:
        pair_data = json.load(handle)
    with open(client_path, encoding='utf-8') as handle:
        client_data = json.load(handle)

    clients_by_round = defaultdict(list)
    pairs_by_round = defaultdict(list)
    for record in client_data['records']:
        clients_by_round[record['round']].append(record)
    for record in pair_data['records']:
        pairs_by_round[record['round']].append(record)

    rounds = client_data['diagnostic_rounds']
    if rounds != pair_data['diagnostic_rounds']:
        raise ValueError('Diagnostic round lists do not match')

    alignment = {'rounds_checked': 0, 'all_passed': True, 'duplicate_client_occurrences': 0,
                 'rounds_with_duplicate_client_ids': 0}
    per_round = []
    pair_cosines = {name: [] for name in PAIR_NAMES}
    concentrations = {width: [] for width in WIDTHS}
    shares = {width: [] for width in WIDTHS}
    client_counts = {width: [] for width in WIDTHS}
    counterfactuals = {'equal_width': [], 'full_20_percent': []}

    for round_value in rounds:
        clients = clients_by_round[round_value]
        pairs = pairs_by_round[round_value]
        if len(clients) != 10 or len(pairs) != 45:
            raise ValueError('Round {} has {} clients and {} pairs; expected 10 and 45'.format(
                round_value, len(clients), len(pairs)))
        alignment['rounds_checked'] += 1
        duplicate_count = len(clients) - len(set(record['client_id'] for record in clients))
        alignment['duplicate_client_occurrences'] += duplicate_count
        if duplicate_count:
            alignment['rounds_with_duplicate_client_ids'] += 1

        gram = [[0.0 for _ in clients] for _ in clients]
        for index, record in enumerate(clients):
            gram[index][index] = record['update_norm'] ** 2
        expected_index = 0
        for left in range(10):
            for right in range(left + 1, 10):
                record = pairs[expected_index]
                expected_index += 1
                left_client, right_client = clients[left], clients[right]
                for pair_key, client_key in (('client_i', 'client_id'), ('model_idx_i', 'model_idx'),
                                             ('width_i', 'width'), ('client_j', 'client_id'),
                                             ('model_idx_j', 'model_idx'), ('width_j', 'width')):
                    expected = left_client[client_key] if pair_key.endswith('_i') else right_client[client_key]
                    if record[pair_key] != expected:
                        raise ValueError('Occurrence-order alignment failure at round {}, pair {}'.format(
                            round_value, expected_index - 1))
                dot = left_client['update_norm'] * right_client['update_norm'] * record['cosine']
                gram[left][right] = dot
                gram[right][left] = dot

        group_indices = {width: [index for index, record in enumerate(clients)
                                 if record['width'] == width] for width in WIDTHS}
        masses = {width: sum(clients[index]['lens'] for index in indices)
                  for width, indices in group_indices.items()}
        total_mass = sum(masses.values())
        round_result = {'round': round_value, 'groups': {}, 'pair_cosines': {}, 'counterfactuals': {}}

        group_dot = [[0.0] * 3 for _ in range(3)]
        for row, width_left in enumerate(WIDTHS):
            for col, width_right in enumerate(WIDTHS):
                group_dot[row][col] = sum(
                    clients[i]['lens'] * clients[j]['lens'] * gram[i][j]
                    for i in group_indices[width_left] for j in group_indices[width_right])

        for index, width in enumerate(WIDTHS):
            norm_sq = nonnegative(group_dot[index][index], abs(group_dot[index][index]),
                                  'group norm squared')
            norm = math.sqrt(norm_sq)
            denominator = sum(clients[item]['lens'] * clients[item]['update_norm']
                              for item in group_indices[width])
            concentration = norm / denominator if denominator else None
            share = masses[width] / total_mass
            round_result['groups'][str(width)] = {
                'client_count': len(group_indices[width]), 'sample_mass': masses[width],
                'sample_mass_share': share, 'concentration': concentration,
            }
            shares[width].append(share)
            client_counts[width].append(len(group_indices[width]))
            if concentration is not None:
                concentrations[width].append(concentration)

        for width_left, width_right in PAIR_NAMES:
            left_index, right_index = WIDTHS.index(width_left), WIDTHS.index(width_right)
            norm_left = math.sqrt(nonnegative(group_dot[left_index][left_index], 1.0, 'group norm'))
            norm_right = math.sqrt(nonnegative(group_dot[right_index][right_index], 1.0, 'group norm'))
            cosine = None if not norm_left or not norm_right else group_dot[left_index][right_index] / (norm_left * norm_right)
            name = '{}_vs_{}'.format(width_left, width_right)
            round_result['pair_cosines'][name] = cosine
            if cosine is not None:
                pair_cosines[(width_left, width_right)].append((round_value, cosine))

        if all(masses[width] > 0 for width in WIDTHS):
            gram_means = [[group_dot[i][j] / (masses[WIDTHS[i]] * masses[WIDTHS[j]])
                           for j in range(3)] for i in range(3)]
            natural = [masses[width] / total_mass for width in WIDTHS]
            alternatives = {
                'equal_width': [1.0 / 3.0] * 3,
                'full_20_percent': [0.8 * natural[0] / (natural[0] + natural[1]),
                                    0.8 * natural[1] / (natural[0] + natural[1]), 0.2],
            }
            for name, alternate in alternatives.items():
                result = counterfactual(gram_means, natural, alternate)
                if result is not None:
                    result['round'] = round_value
                    counterfactuals[name].append(result)
                    round_result['counterfactuals'][name] = result
        per_round.append(round_result)

    result = {
        'analysis': 'Width-Group Consensus Geometry Analysis',
        'input_files': [pair_path.replace('\\', '/'), client_path.replace('\\', '/')],
        'method': 'Exact Gram reconstruction from saved norms and pairwise cosines; no vector reconstruction.',
        'alignment': alignment,
        'group_sample_shares': {str(width): describe(shares[width]) for width in WIDTHS},
        'group_client_counts': {str(width): describe(client_counts[width]) for width in WIDTHS},
        'internal_concentration': {str(width): describe(concentrations[width]) for width in WIDTHS},
        'group_cosines': {}, 'counterfactuals': {}, 'per_round': per_round,
    }
    result['group_sample_shares_by_phase'] = {
        phase: {str(width): describe([round_result['groups'][str(width)]['sample_mass_share']
                                      for round_result in per_round if round_result['round'] in phase_rounds])
                for width in WIDTHS}
        for phase, phase_rounds in PHASES.items()
    }
    result['group_client_counts_by_phase'] = {
        phase: {str(width): describe([round_result['groups'][str(width)]['client_count']
                                      for round_result in per_round if round_result['round'] in phase_rounds])
                for width in WIDTHS}
        for phase, phase_rounds in PHASES.items()
    }
    result['internal_concentration_by_phase'] = {
        phase: {str(width): describe([round_result['groups'][str(width)]['concentration']
                                      for round_result in per_round
                                      if round_result['round'] in phase_rounds
                                      and round_result['groups'][str(width)]['concentration'] is not None])
                for width in WIDTHS}
        for phase, phase_rounds in PHASES.items()
    }
    result['full_group_single_client_rounds'] = sum(
        round_result['groups']['1.0']['client_count'] == 1 for round_result in per_round)
    result['full_group_absent_rounds'] = sum(
        round_result['groups']['1.0']['client_count'] == 0 for round_result in per_round)
    for pair, values in pair_cosines.items():
        name = '{}_vs_{}'.format(*pair)
        result['group_cosines'][name] = {
            'overall': describe(value for _, value in values), 'trend': trend(values),
            'phases': {phase: phase_cosines([(round_value, value) for round_value, value in values
                                              if round_value in phase_rounds])
                       for phase, phase_rounds in PHASES.items()},
        }
    for name, values in counterfactuals.items():
        result['counterfactuals'][name] = {
            'overall': phase_counterfactual(values),
            'phases': {phase: phase_counterfactual([value for value in values
                                                     if value['round'] in phase_rounds])
                       for phase, phase_rounds in PHASES.items()},
        }

    output_path = os.path.join(root, 'width_group_consensus_geometry.json')
    with open(output_path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(output_path)


if __name__ == '__main__':
    main()
