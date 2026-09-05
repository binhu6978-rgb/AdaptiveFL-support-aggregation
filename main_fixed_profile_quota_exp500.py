"""Fixed-configuration 500-round Fixed Profile-Quota Oracle entry point."""

import os
import sys
import time

import numpy as np
import torch

from Algorithm.Training_AdaptiveFL import AdaptiveFL
from utils.get_datatset_new import get_dataset
from utils.options import args_parser
from utils.set_seed import set_random_seed


class Tee:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def main():
    args = args_parser()
    args.algorithm = 'AdaptiveFL'
    args.dataset = 'cifar100'
    args.model = 'resnet'
    args.num_classes = 100
    args.iid = 0
    args.data_beta = 0.3
    args.generate_data = 0
    args.epochs = 500
    args.num_users = 100
    args.frac = 0.1
    args.local_ep = 5
    args.local_bs = 50
    args.optimizer = 'sgd'
    args.lr = 0.01
    args.lr_decay = 0.998
    args.seed = 1
    args.client_chosen_mode = 'available'
    args.depth_saved = [2, 3, 4]
    args.width_ration = [0.4, 0.66, 1.0]
    args.fixed_profile_quota_oracle = True
    args.experiment_result_dir = os.path.join('results', 'fixed_profile_quota_full500')
    args.device = torch.device(
        'cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    os.makedirs(args.experiment_result_dir, exist_ok=True)
    original_stdout = sys.stdout
    tee = Tee(os.path.join(args.experiment_result_dir, 'fixed_profile_quota_500.log'))
    sys.stdout = tee
    started = time.time()
    try:
        print('Fixed Profile-Quota Oracle: 500 rounds, all-profile evaluation')
        print('aggregation=Aggregation_AdaptiveFL; fixed_profile_quota_oracle=True; device={}'.format(args.device))
        set_random_seed(args.seed)
        dataset_train, dataset_test, dict_users = get_dataset(args)
        client_sizes = [len(dict_users[index]) for index in range(args.num_users)]
        if (min(client_sizes), float(np.median(client_sizes)), max(client_sizes), sum(client_sizes)) != (
                262, 491.5, 757, 50000):
            raise RuntimeError('Unexpected cached split integrity: {}'.format(client_sizes))
        print('cached split min/median/max={}/{}/{}'.format(
            min(client_sizes), float(np.median(client_sizes)), max(client_sizes)))
        result = AdaptiveFL(args, dataset_train, dataset_test, dict_users)
        elapsed = time.time() - started
        print('completed 500/500 rounds; elapsed_seconds={:.3f}; average_round_seconds={:.3f}'.format(
            elapsed, elapsed / args.epochs))
        print('recorded_profiles={}; dispatch_rounds={}'.format(
            len(result['accuracy']), len(result['dispatch_records'])))
    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == '__main__':
    main()
