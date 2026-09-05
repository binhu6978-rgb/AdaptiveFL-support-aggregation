"""Offline simulation of the diagnostic fixed 4-small/3-medium/3-full quota."""
import argparse
import json
import random
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / 'data' / 'cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json'
DEFAULT_BASELINE = ROOT / 'results' / 'capacity_exposure_audit' / 'baseline_exposure.json'
DEFAULT_OUTPUT = ROOT / 'results' / 'capacity_exposure_audit'
REGIONS = {'shared_small': lambda p: p <= 6,
           'medium_capable': lambda p: 3 <= p <= 6,
           'full_exclusive': lambda p: p == 6}


def load_sizes(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    users = {int(k): v for k, v in payload['train_data'].items()}
    sizes = np.asarray([len(users[i]) for i in range(100)], dtype=int)
    if sizes.sum() != 50000:
        raise ValueError('cached split must total 50,000 examples')
    return sizes


def summary(values):
    values = np.asarray(values, dtype=float)
    return {'total': float(values.sum()), 'mean_per_round': float(values.mean()),
            'median_per_round': float(np.median(values)), 'p10_per_round': float(np.percentile(values, 10)),
            'p90_per_round': float(np.percentile(values, 90)), 'zero_update_round_ratio': float(np.mean(values == 0))}


def audit(records, sizes, lr=0.01, decay=0.998):
    profiles = np.asarray([p for row in records for p in row['ration_users']], dtype=int)
    profile_counts = [int(np.sum(profiles == p)) for p in range(7)]
    regions = {}
    for name, contains in REGIONS.items():
        occurrences, sample_counts, weighted_counts = [], [], []
        for row in records:
            selected = [(p, client) for p, client in zip(row['ration_users'], row['idx_users']) if contains(p)]
            samples = sum(int(sizes[client]) for _, client in selected)
            occurrences.append(len(selected)); sample_counts.append(samples)
            weighted_counts.append((lr * decay ** row['round']) * samples)
        regions[name] = {'occurrence_exposure': summary(occurrences),
                         'sample_weighted_exposure': summary(sample_counts),
                         'lr_weighted_sample_exposure_proxy': summary(weighted_counts)}
    shared = regions['shared_small']
    ratios = {name + '_over_shared': {
        'occurrence_ratio': regions[name]['occurrence_exposure']['total'] / shared['occurrence_exposure']['total'],
        'sample_weighted_ratio': regions[name]['sample_weighted_exposure']['total'] / shared['sample_weighted_exposure']['total'],
        'lr_weighted_sample_ratio': regions[name]['lr_weighted_sample_exposure_proxy']['total'] / shared['lr_weighted_sample_exposure_proxy']['total'],
    } for name in ('medium_capable', 'full_exclusive')}
    return {'rounds': len(records), 'slots_per_round': 10, 'total_slots': int(profiles.size),
            'profile_counts': profile_counts,
            'group_counts': {'small_profiles_0_2': int(sum(profile_counts[:3])),
                             'medium_profiles_3_5': int(sum(profile_counts[3:6])), 'full_profile_6': int(profile_counts[6])},
            'regions': regions, 'exposure_ratios': ratios}


def simulate(rounds, seed):
    profile_rng = random.Random(seed)
    client_rng = random.Random(seed)
    records = []
    for round_id in range(rounds):
        profiles = ([profile_rng.choice([0, 1, 2]) for _ in range(4)] +
                    [profile_rng.choice([3, 4, 5]) for _ in range(3)] + [6, 6, 6])
        profile_rng.shuffle(profiles)
        clients = [client_rng.randint(0, 99) if p <= 2 else
                   client_rng.randint(40, 99) if p <= 5 else
                   client_rng.randint(70, 99) for p in profiles]
        records.append({'round': round_id, 'ration_users': profiles, 'idx_users': clients})
    return records


def compare(simulated, baseline):
    result = {}
    for region in ('shared_small', 'medium_capable', 'full_exclusive'):
        result[region + '_over_baseline'] = {}
        for metric in ('occurrence_exposure', 'sample_weighted_exposure', 'lr_weighted_sample_exposure_proxy'):
            numerator = simulated['regions'][region][metric]['total']
            denominator = baseline['regions'][region][metric]['total']
            result[region + '_over_baseline'][metric + '_ratio'] = numerator / denominator
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--baseline', type=Path, default=DEFAULT_BASELINE)
    parser.add_argument('--rounds', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.rounds != 500:
        raise ValueError('this audit is defined for 500 rounds')
    sizes = load_sizes(args.split)
    records = simulate(args.rounds, args.seed)
    simulated = audit(records, sizes)
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))['baseline_exposure']
    result = {'input': {'split_json': str(args.split), 'baseline_exposure_json': str(args.baseline),
                        'rounds': args.rounds, 'seed': args.seed},
              'quota_definition': {'small_slots': 4, 'medium_slots': 3, 'full_slots': 3,
                                   'small_profiles': [0, 1, 2], 'medium_profiles': [3, 4, 5], 'full_profile': 6,
                                   'eligibility': {'small': '0-99', 'medium': '40-99', 'full': '70-99'},
                                   'duplicates_allowed': True},
              'simulation': simulated, 'relative_to_baseline': compare(simulated, baseline),
              'round_records': records}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'fixed_quota_simulation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    full = simulated['regions']['full_exclusive']; medium = simulated['regions']['medium_capable']; shared = simulated['regions']['shared_small']
    relative = result['relative_to_baseline']
    md = f'''# Fixed 4:3:3 quota offline dispatch simulation\n\nThis is a deterministic 500-round dispatch simulation (`seed=1`), not FL training. Each round has exactly four Small slots (profiles 0–2 chosen uniformly), three Medium slots (profiles 3–5 chosen uniformly), and three Full slots (profile 6); the 10 slots are shuffled. Client draws use the original feasible pools and continue to allow duplicate clients.\n\n## Simulated profile exposure\n\n- Profile 0–6 counts: {simulated['profile_counts']}\n- Small / Medium / Full counts: {simulated['group_counts']['small_profiles_0_2']} / {simulated['group_counts']['medium_profiles_3_5']} / {simulated['group_counts']['full_profile_6']}\n- Full occurrences per round: {full['occurrence_exposure']['mean_per_round']:.2f}\n- Full zero-occurrence-round ratio: {full['occurrence_exposure']['zero_update_round_ratio']:.2%}\n\n| Region | Slot total | Sample-weighted total | LR-weighted sample proxy | Ratio to baseline occurrence | Ratio to baseline samples | Ratio to baseline LR proxy |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n'''
    for name, region in [('shared_small', shared), ('medium_capable', medium), ('full_exclusive', full)]:
        change = relative[name + '_over_baseline']
        md += f"| {name} | {region['occurrence_exposure']['total']:.0f} | {region['sample_weighted_exposure']['total']:.0f} | {region['lr_weighted_sample_exposure_proxy']['total']:.2f} | {change['occurrence_exposure_ratio']:.6f} | {change['sample_weighted_exposure_ratio']:.6f} | {change['lr_weighted_sample_exposure_proxy_ratio']:.6f} |\\n"
    (args.output_dir / 'fixed_quota_simulation.md').write_text(md, encoding='utf-8')
    print(json.dumps({'output': str(args.output_dir), 'profile_counts': simulated['profile_counts']}, indent=2))


if __name__ == '__main__':
    main()
