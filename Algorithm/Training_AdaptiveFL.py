#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time
import json
import os
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch import nn
import copy
import numpy as np
from tqdm import tqdm

from models.Fed import (get_model_list, Aggregation_AdaptiveFL, split_model,
                        select_clients)
from models.vgg import vgg_16_bn
from models.resnet import ResNet18_cifar
from models.resnet import ResNet18_widar
from utils.HeteroClients import HeteroClients
from models.test import test_img, test
from models.Update import DatasetSplit, LocalUpdate_AdaptiveFL
from optimizer.Adabelief import AdaBelief
from utils.cross_scale_consistency import CrossScaleConsistencyDiagnostics


def _assert_prefix_consistency(net_glob_list):
    largest_state = net_glob_list[-1].state_dict()
    for model_idx, net in enumerate(net_glob_list):
        expected_state = split_model(largest_state, net.state_dict())
        actual_state = net.state_dict()
        for key in actual_state:
            if not torch.equal(actual_state[key], expected_state[key]):
                max_abs_diff = (actual_state[key] - expected_state[key]).abs().max().item()
                raise AssertionError(
                    "Round 1 prefix inconsistency: model={}, parameter={}, max_abs_diff={}".format(
                        model_idx, key, max_abs_diff))
    print("Round 1 prefix consistency assertion: PASSED")


def _capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        'python': random.getstate(),
        'numpy': (
            numpy_state[0], numpy_state[1].copy(), numpy_state[2],
            numpy_state[3], numpy_state[4]),
        'torch_cpu': torch.get_rng_state().clone(),
        'torch_cuda': [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _assert_rng_state_unchanged(before):
    after = _capture_rng_state()
    numpy_equal = (
        before['numpy'][0] == after['numpy'][0]
        and np.array_equal(before['numpy'][1], after['numpy'][1])
        and before['numpy'][2:] == after['numpy'][2:])
    cuda_equal = (
        len(before['torch_cuda']) == len(after['torch_cuda'])
        and all(torch.equal(left, right) for left, right in zip(
            before['torch_cuda'], after['torch_cuda'])))
    if not (before['python'] == after['python'] and numpy_equal
            and torch.equal(before['torch_cpu'], after['torch_cpu'])
            and cuda_equal):
        raise AssertionError('Diagnostics changed an RNG state')


def AdaptiveFL(args, dataset_train, dataset_test, dict_users):
    net_glob_list, net_slim_info = get_model_list(args)

    expected_profiles = [
        (0.4, 2), (0.4, 3), (0.4, 4),
        (0.66, 2), (0.66, 3), (0.66, 4),
        (1.0, 2),
    ]
    actual_profiles = [
        (float(profile[0]), int(profile[1]))
        for profile in net_slim_info
    ]
    if len(net_glob_list) != 7 or actual_profiles != expected_profiles:
        raise AssertionError(
            "Expected seven heterogeneous profiles {}, got {}".format(
                expected_profiles, actual_profiles))
    full_idx = actual_profiles.index((1.0, 2))
    full_profile = net_slim_info[full_idx]

    batch_norm_modules = [
        module for net in net_glob_list for module in net.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    bn_track_running_stats = [
        module.track_running_stats for module in batch_norm_modules
    ]
    if not batch_norm_modules or any(bn_track_running_stats):
        raise AssertionError(
            "Expected all BatchNorm modules to use track_running_stats=False, got {}".format(
                bn_track_running_stats))

    print("Experiment: Cross-Scale Update Consistency Diagnostic")
    experiment_config = [
        ("algorithm", args.algorithm),
        ("dataset", args.dataset),
        ("iid", args.iid),
        ("data_beta", args.data_beta),
        ("seed", args.seed),
        ("epochs", args.epochs),
        ("num_users", args.num_users),
        ("frac", args.frac),
        ("local_ep", args.local_ep),
        ("local_bs", args.local_bs),
        ("optimizer", args.optimizer),
        ("lr", args.lr),
        ("lr_decay", args.lr_decay),
        ("client_chosen_mode", args.client_chosen_mode),
        ("depth_saved", args.depth_saved),
        ("width_ration", args.width_ration),
        ("len(net_glob_list)", len(net_glob_list)),
        ("profiles", net_slim_info),
        ("Full evaluation profile", full_profile),
        ("BN module count", len(batch_norm_modules)),
        ("BN track_running_stats", sorted(set(bn_track_running_stats))),
    ]
    for key, value in experiment_config:
        print("{}: {}".format(key, value))

    result_dir = getattr(
        args, 'result_dir', 'results/cross_scale_consistency_50')
    diagnostics = CrossScaleConsistencyDiagnostics(
        net_glob_list[0].state_dict(), net_slim_info, result_dir)

    # training
    total_time = 0
    time_list = []
    full_acc = []
    clients = HeteroClients(args, net_slim_info)

    # 开始训练
    for iter in tqdm(range(args.epochs)):  # tqdm 进度条库

        print('*' * 80)
        print('Round {:3d}'.format(iter))

        if iter == 1:
            _assert_prefix_consistency(net_glob_list)

        w_locals = []
        lens = []

        m = max(int(args.frac * args.num_users), 1)

        # 用户选择
        if args.client_chosen_mode == 'RL':  # RL-Selection
            # ration_users = np.random.choice([2, 5, 6], m)  # 模型选择
            ration_users = np.random.choice(range(len(net_glob_list)), m)  # 模型选择
            idx_users = clients.select_clients(ration_users)  # 基于强化学习的客户端选择
        elif args.client_chosen_mode == 'greedy':  # greedy 选择
            ration_users = np.random.choice([len(net_glob_list)-1], m)  # 模型选择
            idx_users = random.sample(range(args.num_users), len(ration_users))  # 基于强化学习的客户端选择
        else:
            ration_users = np.random.choice(range(len(net_glob_list)), m)  # 模型选择
            idx_users = select_clients(args, ration_users, len(net_glob_list))  # 基于规则的客户端选择

        round_global_state = None
        if iter >= 1:
            round_global_state = OrderedDict(
                (key, value.detach().clone())
                for key, value in net_glob_list[-1].state_dict().items())

        feedback_list = []
        max_time = 0
        for id, idx in enumerate(idx_users):
            begin = time.time()
            local = LocalUpdate_AdaptiveFL(args=args, dataset=dataset_train, idxs=dict_users[idx])
            if args.client_chosen_mode == 'RL' or args.client_chosen_mode == 'random' or args.client_chosen_mode == 'greedy':  # 在客户端本地做自适应裁剪
                feedback_model_idx = clients.train(idx, ration_users[id])
            elif args.client_chosen_mode == 'available' or args.client_chosen_mode == 'fit':
                feedback_model_idx = ration_users[id]  # 这里是基于规则的，不会出错，分发的是什么模型，返回的就是什么模型
            feedback_list.append(feedback_model_idx)

            w = local.train(round=iter, net=copy.deepcopy(net_glob_list[feedback_model_idx]).to(args.device))  # 这里开始正式训练

            w_locals.append(copy.deepcopy(w))
            lens.append(len(dict_users[idx]))
            time_epoch = time.time() - begin
            max_time = max(max_time, time_epoch)

        total_time += max_time

        time_list.append(total_time)
        print(f"this epoch cost time:{max_time}")

        print(f"this epoch choose: {idx_users}")  # 这一轮选择的用户下标
        print(f"this epoch dispatch models: {ration_users}")  # 每个用户对应的模型的比例, 初始分发
        print(f"this epoch received models: {feedback_list}")  # 每个用户对应的模型的比例, 初始分发
        print(f"hetero_proportion: \t{args.client_hetero_ration}")
        # 需要print 每个客户端的计算资源

        if iter == 0:
            print("Round 0 consistency diagnostic: SKIPPED (independent profile initialization)")
        else:
            rng_state_before = _capture_rng_state()
            diagnostics.measure(
                round_idx=iter,
                w_locals=w_locals,
                lens=lens,
                client_ids=idx_users,
                model_indices=ration_users,
                global_model_param=round_global_state)
            _assert_rng_state_unchanged(rng_state_before)
            if iter == 1:
                print("Diagnostic RNG state assertion: PASSED")

            current_global_state = net_glob_list[-1].state_dict()
            for key, value in current_global_state.items():
                if not torch.equal(value.detach(), round_global_state[key]):
                    difference = (value.detach().to(dtype=torch.float32)
                                  - round_global_state[key].to(
                                      dtype=torch.float32)).abs().max().item()
                    raise AssertionError(
                        "Pre-aggregation global changed during local training/diagnostics: "
                        "key={}, max_abs_diff={}".format(key, difference))

        w_glob_param = Aggregation_AdaptiveFL(
            w_locals, lens, net_glob_list[-1].state_dict())

        for net in net_glob_list:
            net.load_state_dict(split_model(w_glob_param, net.state_dict()))

        print("Full evaluation profile: {}".format(full_profile))
        full_acc.append(test(net_glob_list[full_idx], dataset_test, args))

    os.makedirs(result_dir, exist_ok=True)
    accuracy_path = os.path.join(result_dir, 'full_accuracy.json')
    result = {
        'full_profile': {
            'width': float(full_profile[0]),
            'depth': int(full_profile[1]),
            'parameters_million': float(full_profile[2]),
        },
        'experiment': 'Cross-Scale Update Consistency Diagnostic',
        'aggregation': 'Aggregation_AdaptiveFL',
        'bn_track_running_stats': False,
        'rounds': len(full_acc),
        'accuracy': full_acc,
        'cumulative_max_client_train_time_seconds': time_list,
    }
    with open(accuracy_path, 'w', encoding='utf-8') as result_file:
        json.dump(result, result_file, indent=2)
    diagnostics.save()
    print("Saved Full integrity accuracy: {}".format(accuracy_path))
    return full_acc


'''
                    
            for id, p in enumerate(model_proportion):
                if id == 0:  # small model
                    idx_users = np.random.choice(range(args.num_users), int(m * p), replace=False)
                    ration_users = [random.randint(0, 2) for _ in range(int(m * p))]
                elif id == 1:  # medium model
                    idx_users = np.hstack((idx_users,
                                           np.random.choice(
                                               [i for i in
                                                range(int(args.num_users * sum(hetero_proportion[:id])), args.num_users)  # 做个sum
                                                if i not in idx_users],
                                               int(m * p), replace=False)))
                    ration_users.extend([random.randint(3, 5) for _ in range(int(m * p))])
                elif id == 2:  # large model
                    idx_users = np.hstack((idx_users,
                                           np.random.choice(
                                               [i for i in
                                                range(int(args.num_users * sum(hetero_proportion[:id])), args.num_users)  # 做个sum
                                                if i not in idx_users],
                                               int(m * p), replace=False)))
                    ration_users.extend([6 for _ in range(int(m * p))])
'''
