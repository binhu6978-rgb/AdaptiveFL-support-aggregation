"""Dispatch-only validation for the Full-access oracle; no models or training."""

import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.Fed import select_clients


def make_args(full_access_oracle):
    return SimpleNamespace(
        num_users=100,
        client_hetero_ration='4:3:3',
        client_chosen_mode='available',
        full_access_oracle=full_access_oracle,
    )


def summarize(profile_sequence, selected_ids, oracle):
    full_ids = [client_id for profile, client_id in zip(profile_sequence, selected_ids)
                if profile == 6]
    violations = {'small_0_to_2': 0, 'medium_3_to_5': 0, 'full': 0}
    for profile, client_id in zip(profile_sequence, selected_ids):
        if profile <= 2 and not 0 <= client_id <= 99:
            violations['small_0_to_2'] += 1
        elif 3 <= profile <= 5 and not 40 <= client_id <= 99:
            violations['medium_3_to_5'] += 1
        elif profile == 6 and not (0 <= client_id <= 99 if oracle else 70 <= client_id <= 99):
            violations['full'] += 1
    blocks = [sum(low <= client_id <= high for client_id in full_ids)
              for low, high in ((0, 39), (40, 69), (70, 99))]
    return {
        'full_occurrence_count': len(full_ids), 'full_client_min': min(full_ids),
        'full_client_max': max(full_ids), 'full_unique_client_count': len(set(full_ids)),
        'full_block_counts_0_39_40_69_70_99': blocks,
        'full_block_ratios_0_39_40_69_70_99': [count / len(full_ids) for count in blocks],
        'eligibility_violations': violations,
        'duplicate_full_occurrences_allowed': len(set(full_ids)) < len(full_ids),
    }


def main():
    profile_sequence = np.random.RandomState(1).choice(range(7), 10000).tolist()
    outputs = {}
    for name, oracle in (('baseline', False), ('full_access_oracle', True)):
        random.seed(123456)
        selected_ids = select_clients(make_args(oracle), profile_sequence, 7)
        outputs[name] = summarize(profile_sequence, selected_ids, oracle)
    result = {
        'simulation': 'dispatch-only; no model construction, local training, or aggregation',
        'profile_assignment_occurrences': len(profile_sequence),
        'profile_counts': [profile_sequence.count(profile) for profile in range(7)],
        'profile_sequence_identical_between_modes': True,
        'modes': outputs,
    }
    path = os.path.join('results', 'full_access_oracle_dispatch_simulation.json')
    os.makedirs('results', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(path)


if __name__ == '__main__':
    main()
