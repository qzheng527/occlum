#!/usr/bin/env python
# coding=utf-8
# @Author: xinyu.kxy
# @Date: Fri 17 July 2023 09:22:56 PM CST


def pairwise_to_winrate(data_annotated: list,
                        annotation_key: str,
                        golden_label_key: str) -> dict:
    n_wins_a = 0.
    n_wins_b = 0.
    n_equal = 0.
    n_failed = 0.
    n_consistency = 0.
    n_total = len(data_annotated)

    for item in data_annotated:
        if item[annotation_key] == 0:
            n_wins_a += 1
        elif item[annotation_key] == 1:
            n_wins_b += 1
        elif item[annotation_key] == 2:
            n_equal += 1

        else:
            n_failed += 1
        # 一致性统计
        if item.get(golden_label_key, None) == item[annotation_key]:
            n_consistency += 1
    win_rate_a = round(n_wins_a / n_total, 4)
    win_rate_b = round(n_wins_b / n_total, 4)
    equal_rate = round(n_equal / n_total, 4)
    failed_rate = round(n_failed / n_total, 4)
    consistency_rate = round(n_consistency / n_total, 4)
    return dict(
        win_rate_a=win_rate_a * 100,
        win_rate_b=win_rate_b * 100,
        equal_rate=equal_rate * 100,
        failed_rate=failed_rate * 100,
        consistency_rate=consistency_rate * 100,
        n_total=n_total
    )


def single_to_acc(data_annotated: list,
                  annotation_key: str,
                  golden_label_key: str) -> dict:
    n_acc = 0.
    n_failed = 0.
    n_consistency = 0.
    n_total = len(data_annotated)

    for item in data_annotated:
        if item[annotation_key] == 1:
            n_acc += 1
        elif item[annotation_key] == -1:
            n_failed += 1
        # 一致性统计
        if item.get(golden_label_key, None) == item[annotation_key] or \
                item.get(golden_label_key, None) == str(item[annotation_key]):
            n_consistency += 1
    acc = round(n_acc / n_total, 4)
    consistency_rate = round(n_consistency / n_total, 4)
    failed_rate = round(n_failed / n_total, 4)
    return dict(
        acc=acc * 100,
        failed_rate=failed_rate * 100,
        consistency_rate=consistency_rate,
        n_total=n_total
    )
