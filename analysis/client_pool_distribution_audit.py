"""Offline label-distribution audit for the cached CIFAR-100 client split.

This script reads the cached JSON exactly as stored and the existing CIFAR-100
pickle labels directly.  It neither downloads data nor regenerates a split.
"""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / 'data' / 'cifar100_U100_seed1_dirichlet_a0.3_bal_minsz10_try500.json'
DEFAULT_OUTPUT = ROOT / 'results' / 'capacity_exposure_audit'
DEFAULT_CIFAR_ROOT = Path(os.environ.get(
    'SEMCONSFL_CIFAR_ROOT', r'C:\Users\admin\Desktop\paper_ori\all_data\data'))


def load_split(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or 'train_data' not in payload:
        raise ValueError('expected a JSON object with a train_data mapping')
    users = {int(k): [int(i) for i in v]
             for k, v in payload['train_data'].items()}
    if sorted(users) != list(range(100)):
        raise ValueError('expected exactly client IDs 0..99')
    return payload, users


def load_cifar100_labels(cifar_root):
    train_file = Path(cifar_root) / 'cifar-100-python' / 'train'
    with train_file.open('rb') as handle:
        record = pickle.load(handle, encoding='latin1')
    labels = np.asarray(record['fine_labels'], dtype=int)
    if labels.shape != (50000,) or labels.min() < 0 or labels.max() >= 100:
        raise ValueError('unexpected CIFAR-100 train label payload')
    return labels, train_file


def jsd(p, q):
    midpoint = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def distribution_stats(counts, reference):
    total = int(counts.sum())
    probs = counts / total
    entropy = float(-np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
    return {
        'raw_class_counts': counts.astype(int).tolist(),
        'normalized_class_probabilities': probs.tolist(),
        'total_samples': total,
        'l1_to_global': float(np.abs(probs - reference).sum()),
        'tvd_to_global': float(0.5 * np.abs(probs - reference).sum()),
        'jsd_to_global': jsd(probs, reference),
        'cosine_similarity_to_global': float(np.dot(probs, reference) /
                                             (np.linalg.norm(probs) * np.linalg.norm(reference))),
        'entropy_bits': entropy,
        'entropy_difference_to_global_bits': None,
        'class_coverage': int(np.count_nonzero(counts)),
    }


def quantile_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        'mean': float(values.mean()), 'median': float(np.median(values)),
        'p5': float(np.percentile(values, 5)), 'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)), 'p95': float(np.percentile(values, 95)),
    }


def null_test(histograms, subset_size, simulations, rng, reference):
    tvd_values, jsd_values = [], []
    for _ in range(simulations):
        client_ids = rng.choice(histograms.shape[0], subset_size, replace=False)
        probs = histograms[client_ids].sum(axis=0)
        probs = probs / probs.sum()
        tvd_values.append(0.5 * np.abs(probs - reference).sum())
        jsd_values.append(jsd(probs, reference))
    return np.asarray(tvd_values), np.asarray(jsd_values)


def pool_record(name, client_ids, histograms, sizes, global_probs, global_entropy):
    counts = histograms[client_ids].sum(axis=0)
    record = distribution_stats(counts, global_probs)
    selected_sizes = sizes[client_ids]
    record.update({
        'name': name, 'client_ids': [int(i) for i in client_ids],
        'number_of_clients': len(client_ids),
        'client_sample_size': {
            'mean': float(selected_sizes.mean()), 'median': float(np.median(selected_sizes)),
            'min': int(selected_sizes.min()), 'max': int(selected_sizes.max()),
        },
    })
    record['entropy_difference_to_global_bits'] = record['entropy_bits'] - global_entropy
    return record


def classify(strong_percentile):
    if strong_percentile >= 95.0:
        return 'STRONGLY_BIASED'
    if strong_percentile >= 75.0:
        return 'MILDLY_BIASED'
    return 'TYPICAL_FOR_30_CLIENTS'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, default=DEFAULT_SPLIT)
    parser.add_argument('--cifar-root', type=Path, default=DEFAULT_CIFAR_ROOT)
    parser.add_argument('--simulations', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload, users = load_split(args.split)
    labels, label_file = load_cifar100_labels(args.cifar_root)
    all_indices = [i for client in users.values() for i in client]
    if len(all_indices) != 50000 or len(set(all_indices)) != 50000 or set(all_indices) != set(range(50000)):
        raise ValueError('cached split does not partition the 50,000 CIFAR-100 training examples')

    histograms = np.zeros((100, 100), dtype=int)
    for client_id, indices in users.items():
        histograms[client_id] = np.bincount(labels[np.asarray(indices)], minlength=100)
    sizes = histograms.sum(axis=1)
    global_counts = histograms.sum(axis=0)
    global_probs = global_counts / global_counts.sum()
    global_entropy = float(-np.sum(global_probs * np.log2(global_probs)))

    pools = {
        'all_0_99': list(range(100)), 'non_strong_0_69': list(range(70)),
        'strong_70_99': list(range(70, 100)), 'weak_0_39': list(range(40)),
        'medium_40_69': list(range(40, 70)), 'strong_tier_70_99': list(range(70, 100)),
        'medium_capable_40_99': list(range(40, 100)),
    }
    pool_results = {name: pool_record(name, ids, histograms, sizes, global_probs, global_entropy)
                    for name, ids in pools.items()}

    rng = np.random.RandomState(args.seed)
    random30_tvd, random30_jsd = null_test(histograms, 30, args.simulations, rng, global_probs)
    random60_tvd, random60_jsd = null_test(histograms, 60, args.simulations, rng, global_probs)
    strong = pool_results['strong_70_99']
    medium_capable = pool_results['medium_capable_40_99']
    strong_tvd_pct = float(100.0 * np.mean(random30_tvd <= strong['tvd_to_global']))
    strong_jsd_pct = float(100.0 * np.mean(random30_jsd <= strong['jsd_to_global']))
    result = {
        'input': {'split_json': str(args.split), 'split_metadata': payload.get('extra_meta', {}),
                  'cifar100_train_label_file': str(label_file), 'simulations': args.simulations,
                  'null_seed': args.seed},
        'global_reference': pool_results['all_0_99'],
        'pools': pool_results,
        'theoretical_uniform_reference': {'class_probability': 0.01, 'entropy_bits': float(np.log2(100))},
        'random30_null': {
            'subset_size': 30, 'tvd': quantile_summary(random30_tvd), 'jsd': quantile_summary(random30_jsd),
            'strong_actual_tvd': strong['tvd_to_global'], 'strong_actual_jsd': strong['jsd_to_global'],
            'strong_tvd_percentile': strong_tvd_pct, 'strong_jsd_percentile': strong_jsd_pct,
        },
        'random60_null': {
            'subset_size': 60, 'tvd': quantile_summary(random60_tvd), 'jsd': quantile_summary(random60_jsd),
            'medium_capable_actual_tvd': medium_capable['tvd_to_global'],
            'medium_capable_actual_jsd': medium_capable['jsd_to_global'],
            'medium_capable_tvd_percentile': float(100.0 * np.mean(random60_tvd <= medium_capable['tvd_to_global'])),
            'medium_capable_jsd_percentile': float(100.0 * np.mean(random60_jsd <= medium_capable['jsd_to_global'])),
        },
        'classification': classify(max(strong_tvd_pct, strong_jsd_pct)),
        'classification_rule': 'STRONGLY_BIASED at >=P95, MILDLY_BIASED at P75-P95, otherwise TYPICAL_FOR_30_CLIENTS; applies only to pooled label histograms.',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'client_pool_distribution.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    md = f'''# Client-pool label-distribution audit\n\nThis is an offline pooled-label analysis of the existing cached split; it does not assess feature difficulty, optimization, model-conditioned utility, or finite-round client-selection variance.\n\n## Strong pool (clients 70–99)\n\n- Clients / samples: 30 / {strong['total_samples']}\n- TVD to actual all-client distribution: {strong['tvd_to_global']:.6f}\n- JSD to actual all-client distribution: {strong['jsd_to_global']:.6f} bits\n- L1 / cosine / entropy difference: {strong['l1_to_global']:.6f} / {strong['cosine_similarity_to_global']:.6f} / {strong['entropy_difference_to_global_bits']:.6f} bits\n- Coverage: {strong['class_coverage']}/100 classes\n\n## Random-subset null tests ({args.simulations:,} simulations)\n\n| Null | Metric | Mean | Median | P5 | P25 | P75 | P95 | Actual percentile |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n| Random 30 | TVD | {result['random30_null']['tvd']['mean']:.6f} | {result['random30_null']['tvd']['median']:.6f} | {result['random30_null']['tvd']['p5']:.6f} | {result['random30_null']['tvd']['p25']:.6f} | {result['random30_null']['tvd']['p75']:.6f} | {result['random30_null']['tvd']['p95']:.6f} | {strong_tvd_pct:.2f}% |\n| Random 30 | JSD | {result['random30_null']['jsd']['mean']:.6f} | {result['random30_null']['jsd']['median']:.6f} | {result['random30_null']['jsd']['p5']:.6f} | {result['random30_null']['jsd']['p25']:.6f} | {result['random30_null']['jsd']['p75']:.6f} | {result['random30_null']['jsd']['p95']:.6f} | {strong_jsd_pct:.2f}% |\n| Random 60 | TVD | {result['random60_null']['tvd']['mean']:.6f} | {result['random60_null']['tvd']['median']:.6f} | {result['random60_null']['tvd']['p5']:.6f} | {result['random60_null']['tvd']['p25']:.6f} | {result['random60_null']['tvd']['p75']:.6f} | {result['random60_null']['tvd']['p95']:.6f} | {result['random60_null']['medium_capable_tvd_percentile']:.2f}% |\n| Random 60 | JSD | {result['random60_null']['jsd']['mean']:.6f} | {result['random60_null']['jsd']['median']:.6f} | {result['random60_null']['jsd']['p5']:.6f} | {result['random60_null']['jsd']['p25']:.6f} | {result['random60_null']['jsd']['p75']:.6f} | {result['random60_null']['jsd']['p95']:.6f} | {result['random60_null']['medium_capable_jsd_percentile']:.2f}% |\n\n## Classification\n\n**{result['classification']}** for pooled label-distribution evidence only.\n'''
    (args.output_dir / 'client_pool_distribution.md').write_text(md, encoding='utf-8')
    print(json.dumps({'output': str(args.output_dir), 'classification': result['classification']}, indent=2))


if __name__ == '__main__':
    main()
