#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import math
import random
from collections import OrderedDict

import numpy as np
import torch

from models import vgg_16_bn, ResNet18_widar, ResNet18_cifar, MobileNetV2


def Aggregation(w, lens):
    w_avg = None
    total_count = sum(lens)

    for i in range(0, len(w)):
        if i == 0:
            w_avg = copy.deepcopy(w[0])
            for k in w_avg.keys():
                w_avg[k] = w[i][k] * lens[i]
        else:
            for k in w_avg.keys():
                w_avg[k] += w[i][k] * lens[i]

    for k in w_avg.keys():
        w_avg[k] = torch.div(w_avg[k], total_count)

    return w_avg


def split_model(global_param, slim_param):
    param = copy.deepcopy(slim_param)
    for k, v in param.items():  # 遍历所有层，每一层遍历
        if v.dim() > 1:
            d1 = v.shape[0]
            d2 = v.shape[1]
            param[k] = global_param[k][:d1, :d2]
        elif v.dim() == 1:
            d1 = v.shape[0]
            param[k] = global_param[k][:d1]
        else:
            param[k] = global_param[k]
    return param


def Aggregation_AdaptiveFL(w, lens, global_model_param):
    w_avg = copy.deepcopy(global_model_param)  # largest model
    count = OrderedDict()
    for k, v in w_avg.items():  # 遍历所有层，每一层遍历
        parameter_type = k.split('.')[-1]

        count[k] = v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
        for m in range(len(w)):  # 遍历所有用户
            if parameter_type == 'weight':
                if v.dim() > 1:  # 卷积  和 线性层
                    d1 = w[m][k].shape[0]
                    d2 = w[m][k].shape[1]
                    tmp_v[:d1, :d2] += w[m][k] * lens[m]  # 第m个客户端的 k 层参数
                    count[k][:d1, :d2] += lens[m]
                else:  # BN层
                    d1 = w[m][k].shape[0]
                    tmp_v[:d1] += w[m][k] * lens[m]
                    count[k][:d1] += lens[m]
            else:
                d1 = w[m][k].shape[0]
                tmp_v[:d1] += w[m][k] * lens[m]
                count[k][:d1] += lens[m]

        tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
        tmp_v[count[k] == 0] = global_model_param[k][count[k] == 0]
        w_avg[k] = tmp_v

    return w_avg


def Initialize_TemporalSupportMemory(global_model_param):
    """Create zero temporal memory for floating largest-model tensors."""
    return OrderedDict(
        (key, torch.zeros_like(value, dtype=torch.float32))
        for key, value in global_model_param.items()
        if value.is_floating_point())


def Aggregation_TemporalSupport(w, lens, global_model_param, memory, beta=0.85):
    """Aggregate AdaptiveFL updates with continuous temporal support smoothing.

    For every floating coordinate with sample-weighted support ``s > 0``, this
    returns ``theta_t + s*u + (1-s)*m_new`` where ``u`` is the original
    AdaptiveFL update and ``m_new = beta*m_prev + (1-beta)*u``. Coordinates
    with zero support remain at ``theta_t`` and retain their previous memory.
    Fully supported coordinates are explicitly copied from AdaptiveFL so they
    are strictly independent of temporal memory.
    """
    if len(w) != len(lens) or not w:
        raise ValueError("w and lens must be non-empty and have the same length")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    total_count = sum(lens)
    if total_count <= 0:
        raise ValueError("sum(lens) must be positive")

    theta_ada = Aggregation_AdaptiveFL(w, lens, global_model_param)
    new_global_param = copy.deepcopy(global_model_param)
    new_memory = OrderedDict()

    total_coordinates = 0
    support_sum = 0.0
    nonzero_coordinates = 0
    memory_contribution_sq = 0.0
    current_update_sq = 0.0
    final_update_sq = 0.0

    for key, global_tensor in global_model_param.items():
        if not global_tensor.is_floating_point():
            new_global_param[key] = copy.deepcopy(theta_ada[key])
            continue

        if key not in memory:
            raise KeyError("Missing temporal memory for floating tensor: {}".format(key))
        previous_memory = memory[key]
        if (tuple(previous_memory.shape) != tuple(global_tensor.shape)
                or not previous_memory.is_floating_point()):
            raise ValueError(
                "Invalid temporal memory for {}: expected floating shape {}, got {}".format(
                    key, tuple(global_tensor.shape), tuple(previous_memory.shape)))

        base = global_tensor.detach().to(dtype=torch.float32)
        adaptive_tensor = theta_ada[key].detach().to(
            device=global_tensor.device, dtype=torch.float32)
        current_update = adaptive_tensor - base

        # Accumulate integer sample mass first so fully covered coordinates
        # obtain support exactly equal to one after division.
        support_mass = global_tensor.new_zeros(
            global_tensor.size(), dtype=torch.float32)
        for local_state, client_len in zip(w, lens):
            local_tensor = local_state[key]
            if global_tensor.dim() > 1:
                d1 = local_tensor.shape[0]
                d2 = local_tensor.shape[1]
                support_mass[:d1, :d2] += float(client_len)
            elif global_tensor.dim() == 1:
                d1 = local_tensor.shape[0]
                support_mass[:d1] += float(client_len)
            else:
                support_mass += float(client_len)

        support = support_mass / float(total_count)
        observed = support > 0
        fully_supported = support == 1

        previous_memory_float = previous_memory.detach().to(
            device=global_tensor.device, dtype=torch.float32)
        updated_memory = previous_memory_float.clone()
        updated_memory[observed] = (
            beta * previous_memory_float[observed]
            + (1.0 - beta) * current_update[observed])

        final_update = torch.zeros_like(current_update)
        memory_component = (1.0 - support) * updated_memory
        final_update[observed] = (
            support[observed] * current_update[observed]
            + memory_component[observed])

        result_tensor = base + final_update
        # This explicit assignment protects the mathematical s=1 identity from
        # any floating-point cancellation in base + (theta_ada - base).
        result_tensor[fully_supported] = adaptive_tensor[fully_supported]
        if beta == 0.0:
            # With zero beta, all covered coordinates algebraically reduce to
            # AdaptiveFL; assigning it directly gives exact baseline equality.
            result_tensor = adaptive_tensor

        if not torch.isfinite(result_tensor).all():
            raise FloatingPointError("Non-finite TemporalSupport output: {}".format(key))
        if not torch.isfinite(updated_memory).all():
            raise FloatingPointError("Non-finite TemporalSupport memory: {}".format(key))

        new_global_param[key] = result_tensor.to(dtype=global_tensor.dtype)
        new_memory[key] = updated_memory

        total_coordinates += support.numel()
        support_sum += support.sum().item()
        nonzero_coordinates += observed.sum().item()
        memory_contribution_sq += memory_component[observed].pow(2).sum().item()
        current_update_sq += current_update.pow(2).sum().item()
        final_update_sq += final_update.pow(2).sum().item()

    diagnostics = {
        'support_mean': support_sum / total_coordinates if total_coordinates else 0.0,
        'support_nonzero_ratio': (
            nonzero_coordinates / total_coordinates if total_coordinates else 0.0),
        'memory_contribution_l2': math.sqrt(memory_contribution_sq),
        'current_update_l2': math.sqrt(current_update_sq),
        'final_update_l2': math.sqrt(final_update_sq),
    }
    return new_global_param, new_memory, diagnostics


def _support_quantile(histogram, total_count, quantile):
    """Return a nearest-rank quantile from a compact support histogram."""
    target = quantile * total_count
    cumulative = 0
    for value, count in sorted(histogram.items()):
        cumulative += count
        if cumulative >= target:
            return value
    return max(histogram)


def Aggregation_Support(w, lens, global_model_param, epsilon=0.2):
    """Aggregate prefix-shaped client updates with coordinate-wise support shrinkage."""
    if len(w) != len(lens) or not w:
        raise ValueError("w and lens must be non-empty and have the same length")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")

    total_count = sum(lens)
    if total_count <= 0:
        raise ValueError("sum(lens) must be positive")

    w_avg = copy.deepcopy(global_model_param)
    support_histogram = {}
    total_coordinates = 0
    support_sum = 0.0
    min_nonzero = None
    region_counts = [0, 0, 0]
    region_update_sq = [0.0, 0.0, 0.0]
    global_update_sq = 0.0

    for k, v in global_model_param.items():
        base = v.detach().to(dtype=torch.float32)
        num = v.new_zeros(v.size(), dtype=torch.float32)
        support = v.new_zeros(v.size(), dtype=torch.float32)

        for m in range(len(w)):
            local_tensor = w[m][k].detach().to(device=v.device, dtype=torch.float32)
            client_weight = float(lens[m]) / float(total_count)

            if v.dim() > 1:
                d1 = local_tensor.shape[0]
                d2 = local_tensor.shape[1]
                theta_start = base[:d1, :d2]
                num[:d1, :d2] += client_weight * (local_tensor - theta_start)
                support[:d1, :d2] += client_weight
            elif v.dim() == 1:
                d1 = local_tensor.shape[0]
                theta_start = base[:d1]
                num[:d1] += client_weight * (local_tensor - theta_start)
                support[:d1] += client_weight
            else:
                raise ValueError("Aggregation_Support only supports 1-D or higher tensors: {}".format(k))

        covered = support > 0
        aggregate_update = torch.zeros_like(num)
        aggregate_update[covered] = ((1.0 + epsilon) * num[covered]
                                     / (support[covered] + epsilon))
        w_avg[k] = (base + aggregate_update).to(dtype=v.dtype)

        total_coordinates += support.numel()
        support_sum += support.sum().item()
        nonzero_support = support[covered]
        if nonzero_support.numel() > 0:
            parameter_min = nonzero_support.min().item()
            min_nonzero = parameter_min if min_nonzero is None else min(min_nonzero, parameter_min)

        unique_support, unique_counts = torch.unique(support, return_counts=True)
        for value, count in zip(unique_support.tolist(), unique_counts.tolist()):
            support_histogram[value] = support_histogram.get(value, 0) + count

        region_masks = [support < 0.3,
                        (support >= 0.3) & (support < 0.8),
                        support >= 0.8]
        for idx, mask in enumerate(region_masks):
            region_counts[idx] += mask.sum().item()
            region_update_sq[idx] += aggregate_update[mask].pow(2).sum().item()
        global_update_sq += aggregate_update.pow(2).sum().item()

    support_mean = support_sum / total_coordinates
    support_p25 = _support_quantile(support_histogram, total_coordinates, 0.25)
    support_p50 = _support_quantile(support_histogram, total_coordinates, 0.50)
    support_p75 = _support_quantile(support_histogram, total_coordinates, 0.75)
    region_ratios = [count / total_coordinates for count in region_counts]
    region_update_l2 = [math.sqrt(value) for value in region_update_sq]

    print(
        "Support aggregation diagnostics: "
        "min_nonzero={:.6f}, mean={:.6f}, P25={:.6f}, P50={:.6f}, P75={:.6f}; "
        "coordinate_ratio(s<0.3/0.3<=s<0.8/s>=0.8)={:.6f}/{:.6f}/{:.6f}; "
        "update_l2(s<0.3/0.3<=s<0.8/s>=0.8)={:.6f}/{:.6f}/{:.6f}; "
        "global_update_l2={:.6f}".format(
            min_nonzero if min_nonzero is not None else 0.0,
            support_mean, support_p25, support_p50, support_p75,
            region_ratios[0], region_ratios[1], region_ratios[2],
            region_update_l2[0], region_update_l2[1], region_update_l2[2],
            math.sqrt(global_update_sq)))

    return w_avg


def Aggregation_ScaleFL(w, lens, grad_info, global_model_param):
    w_avg = copy.deepcopy(global_model_param)  # largest model
    count = OrderedDict()
    for idx, (k, v) in enumerate(w_avg.items()):  # 遍历所有层，每一层遍历
        parameter_type = k.split('.')[-1]

        count[k] = v.new_zeros(v.size(), dtype=torch.float32)
        tmp_v = v.new_zeros(v.size(), dtype=torch.float32)
        for m in range(len(w)):  # 遍历所有用户
            if grad_info[m][idx]:
                if parameter_type == 'weight':
                    if v.dim() > 1:  # 卷积  和 线性层
                        d1 = w[m][k].shape[0]
                        d2 = w[m][k].shape[1]
                        tmp_v[:d1, :d2] += w[m][k] * lens[m]  # 第m个客户端的 k 层参数
                        count[k][:d1, :d2] += lens[m]
                    else:  # BN层
                        d1 = w[m][k].shape[0]
                        tmp_v[:d1] += w[m][k] * lens[m]
                        count[k][:d1] += lens[m]
                else:
                    d1 = w[m][k].shape[0]
                    tmp_v[:d1] += w[m][k] * lens[m]
                    count[k][:d1] += lens[m]


        tmp_v[count[k] > 0] = tmp_v[count[k] > 0].div_(count[k][count[k] > 0])
        tmp_v[count[k] == 0] = global_model_param[k][count[k] == 0]
        w_avg[k] = tmp_v

    return w_avg


def get_model_list(args):
    model_rate = args.width_ration
    depth_list = args.depth_saved

    net_glob_list = []
    net_slim_info = []
    for i in model_rate:
        for depth in depth_list:
            if args.model == 'vgg':
                net = vgg_16_bn(num_classes=args.num_classes, track_running_stats=False, num_channels=args.num_channels, slim_idx=depth, scale=i)
            elif args.model == 'resnet':
                if args.dataset == 'widar':
                    net = ResNet18_widar(num_classes=args.num_classes, track_running_stats=False, slim_idx=depth, scale=i)
                else:
                    net = ResNet18_cifar(num_channels=args.num_channels, num_classes=args.num_classes, track_running_stats=False, slim_idx=depth, scale=i)

            elif args.model == 'mobilenet':
                net = MobileNetV2(channels= args.num_channels, num_classes=args.num_classes, trs=False, slim_idx=depth, scale=i)

            total = sum([param.nelement() for param in net.parameters()])
            net.to(args.device)
            net.train()
            print("==" * 50)
            print('【model config】  model_name:{}, width:{} , depth:{}, param:{}MB'.format(args.model, i, depth, total * 4 / 1e6))
            print(net)
            net_glob_list.append(net)
            net_slim_info.append((i, depth, total / 1e6))  # 宽度 深度 参数量

            if i == 1.0:
                break
    return net_glob_list, net_slim_info


def select_clients(args, ration_users, net_glob_list_len):
    my_list = list(map(float, args.client_hetero_ration.split(':')))
    hetero_proportion = [round(x / sum(my_list), 2) for x in my_list]

    idx_users = []
    if net_glob_list_len == 7:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if int(model_type / 3) == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
                elif int(model_type / 3) == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), args.num_users - 1))
                elif int(model_type / 3) == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if int(model_type / 3) == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
                elif int(model_type / 3) == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), int(args.num_users * sum(hetero_proportion[:2])) - 1))
                elif int(model_type / 3) == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), int(args.num_users * sum(hetero_proportion[:3])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))
    else:
        if args.client_chosen_mode == 'available':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), args.num_users - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), args.num_users - 1))
                elif model_type == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), args.num_users - 1))
        elif args.client_chosen_mode == 'fit':
            for model_type in ration_users:
                if model_type == 0:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:0])), int(args.num_users * sum(hetero_proportion[:1])) - 1))
                elif model_type == 1:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:1])), int(args.num_users * sum(hetero_proportion[:2])) - 1))
                elif model_type == 2:
                    idx_users.append(random.randint(int(args.num_users * sum(hetero_proportion[:2])), int(args.num_users * sum(hetero_proportion[:3])) - 1))
        elif args.client_chosen_mode == 'random':
            idx_users = random.sample(range(args.num_users), len(ration_users))

    return idx_users
