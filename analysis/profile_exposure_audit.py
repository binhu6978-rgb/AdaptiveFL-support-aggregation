"""Offline exposure audit from the original AdaptiveFL dispatch log."""
import argparse
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = ROOT / 'resut_ori' / 'AdaptiveFL.log'
DEFAULT_SPLIT = ROOT / 'data' / 'cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json'
DEFAULT_OUTPUT = ROOT / 'results' / 'capacity_exposure_audit'
REGIONS = {'shared_small': lambda p: p <= 6,
           'medium_capable': lambda p: 3 <= p <= 6,
           'full_exclusive': lambda p: p == 6}


def load_sizes(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    users = {int(k): v for k, v in payload['train_data'].items()}
    if sorted(users) != list(range(100)):
        raise ValueError('cached split must contain client IDs 0..99')
    sizes = np.asarray([len(users[i]) for i in range(100)], dtype=int)
    if sizes.sum() != 50000:
        raise ValueError('cached split must total 50,000 examples')
    return sizes


def parse_array(text, expected, field):
    values = [int(value) for value in re.findall(r'-?\d+', text)]
    if len(values) != expected:
        raise ValueError('{} has {} values, expected {}'.format(field, len(values), expected))
    return values


def parse_dispatches(path, rounds=500, slots=10):
    text = Path(path).read_text(encoding='utf-8', errors='ignore')
    pattern = re.compile(
        r'Round\s+(\d+).*?this epoch choose:\s*(\[[^\n]*\]).*?'
        r'this epoch dispatch models:\s*(\[[^\n]*\])', re.S)
    records = []
    for round_text, chosen, profiles in pattern.findall(text):
        round_id = int(round_text)
        if round_id >= rounds:
            continue
        records.append({
            'round': round_id,
            'idx_users': parse_array(chosen, slots, 'chosen clients'),
            'ration_users': parse_array(profiles, slots, 'profile dispatch'),
        })
    records.sort(key=lambda row: row['round'])
    if [row['round'] for row in records] != list(range(rounds)):
        raise ValueError('did not parse exactly contiguous rounds 0..{}'.format(rounds - 1))
    return records


def summary(values):
    values = np.asarray(values, dtype=float)
    return {'total': float(values.sum()), 'mean_per_round': float(values.mean()),
            'median_per_round': float(np.median(values)),
            'p10_per_round': float(np.percentile(values, 10)),
            'p90_per_round': float(np.percentile(values, 90)),
            'zero_update_round_ratio': float(np.mean(values == 0))}


def audit(records, sizes, lr=0.01, decay=0.998):
    profiles = np.asarray([p for row in records for p in row['ration_users']], dtype=int)
    if profiles.size != 5000 or np.any((profiles < 0) | (profiles > 6)):
        raise ValueError('expected 5,000 profile slots in [0, 6]')
    profile_counts = [int(np.sum(profiles == p)) for p in range(7)]
    region_data = {}
    for name, contains in REGIONS.items():
        occurrence_per_round, samples_per_round, lr_samples_per_round = [], [], []
        for row in records:
            selected = [(p, client) for p, client in zip(row['ration_users'], row['idx_users']) if contains(p)]
            sample_count = sum(int(sizes[client]) for _, client in selected)
            occurrence_per_round.append(len(selected))
            samples_per_round.append(sample_count)
            lr_samples_per_round.append((lr * (decay ** row['round'])) * sample_count)
        region_data[name] = {
            'occurrence_exposure': summary(occurrence_per_round),
            'sample_weighted_exposure': summary(samples_per_round),
            'lr_weighted_sample_exposure_proxy': summary(lr_samples_per_round),
        }
    shared = region_data['shared_small']
    ratios = {}
    for name in ('medium_capable', 'full_exclusive'):
        ratios[name + '_over_shared'] = {
            'occurrence_ratio': region_data[name]['occurrence_exposure']['total'] / shared['occurrence_exposure']['total'],
            'sample_weighted_ratio': region_data[name]['sample_weighted_exposure']['total'] / shared['sample_weighted_exposure']['total'],
            'lr_weighted_sample_ratio': region_data[name]['lr_weighted_sample_exposure_proxy']['total'] / shared['lr_weighted_sample_exposure_proxy']['total'],
        }
    return {'rounds': len(records), 'slots_per_round': 10, 'total_slots': int(profiles.size),
            'profile_counts': profile_counts,
            'group_counts': {'small_profiles_0_2': int(sum(profile_counts[:3])),
                             'medium_profiles_3_5': int(sum(profile_counts[3:6])),
                             'full_profile_6': int(profile_counts[6])},
            'group_ratios': {'small_profiles_0_2': float(sum(profile_counts[:3]) / profiles.size),
                             'medium_profiles_3_5': float(sum(profile_counts[3:6]) / profiles.size),
                             'full_profile_6': float(profile_counts[6] / profiles.size)},
            'regions': region_data, 'exposure_ratios': ratios}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, default=DEFAULT_LOG)
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sizes = load_sizes(args.split)
    records = parse_dispatches(args.log)
    result = {'input': {'baseline_log': str(args.log), 'split_json': str(args.split),
                        'log_parse_source': 'per-round ration_users and idx_users'},
              'client_sample_sizes': {'mean': float(sizes.mean()), 'median': float(np.median(sizes)),
                                      'min': int(sizes.min()), 'max': int(sizes.max()), 'total': int(sizes.sum())},
              'baseline_exposure': audit(records, sizes)}
    exposure = result['baseline_exposure']
    full = exposure['regions']['full_exclusive']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'baseline_exposure.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    md = f'''# AdaptiveFL baseline profile-exposure audit\n\nSource: real `ration_users` and `idx_users` records parsed from the first 500 rounds of `resut_ori/AdaptiveFL.log`. Sample-weighted exposure sums the selected client dataset sizes; LR-weighted exposure is only the proxy `sum(lr_t * samples)` with `lr_t = 0.01 * 0.998^t`, not a gradient-contribution estimate.\n\n## Profile occurrence\n\n- Total slots: {exposure['total_slots']} (500 rounds × 10)\n- Profile 0–6 counts: {exposure['profile_counts']}\n- Small / Medium / Full counts: {exposure['group_counts']['small_profiles_0_2']} / {exposure['group_counts']['medium_profiles_3_5']} / {exposure['group_counts']['full_profile_6']}\n- Full mean occurrences per round: {full['occurrence_exposure']['mean_per_round']:.4f}\n- Full zero-update-round ratio: {full['occurrence_exposure']['zero_update_round_ratio']:.2%}\n\n## Parameter-region exposure\n\n| Region | Slot total | Sample-weighted total | LR-weighted sample proxy | Mean slots/round | Median | P10 | P90 | Zero-round ratio |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n'''
    for name in ('shared_small', 'medium_capable', 'full_exclusive'):
        region = exposure['regions'][name]
        occ = region['occurrence_exposure']; sample = region['sample_weighted_exposure']; weighted = region['lr_weighted_sample_exposure_proxy']
        md += f"| {name} | {occ['total']:.0f} | {sample['total']:.0f} | {weighted['total']:.2f} | {occ['mean_per_round']:.3f} | {occ['median_per_round']:.1f} | {occ['p10_per_round']:.1f} | {occ['p90_per_round']:.1f} | {occ['zero_update_round_ratio']:.2%} |\\n"
    md += '\n## Exposure ratios to shared\n\n| Region / shared | Occurrence | Sample-weighted | LR-weighted sample proxy |\n| --- | ---: | ---: | ---: |\n'
    for name, ratio in exposure['exposure_ratios'].items():
        md += f"| {name} | {ratio['occurrence_ratio']:.6f} | {ratio['sample_weighted_ratio']:.6f} | {ratio['lr_weighted_sample_ratio']:.6f} |\\n"
    (args.output_dir / 'baseline_exposure.md').write_text(md, encoding='utf-8')
    print(json.dumps({'output': str(args.output_dir), 'profile_counts': exposure['profile_counts']}, indent=2))


if __name__ == '__main__':
    main()
