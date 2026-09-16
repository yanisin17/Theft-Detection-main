# -*- coding: utf-8 -*-
"""报警策略:统一判定"什么条件下触发偷盗报警"。"""

from detection_engine import detector_config

# 向后兼容:旧代码/测试直接引用 alert_policy.THEFT_ALERT_THRESHOLD
THEFT_ALERT_THRESHOLD = detector_config.THEFT_ALERT_THRESHOLD

_ALERTABLE_BEHAVIORS = {'rapid_item_concealment'}


def normalize_behavior_name(behavior_type):
    return str(behavior_type or '').strip().lower().replace(' ', '_')


def is_alertable_behavior(behavior_type):
    """该行为是否属于可报警的偷盗行为(不含置信度/序列判断)。"""
    return normalize_behavior_name(behavior_type) in _ALERTABLE_BEHAVIORS


def should_trigger_theft_alert(behavior_type, confidence, sequence_confirmed=False,
                               threshold=None):
    """判定是否触发偷盗报警。

    要求:可报警行为 + 序列已确认 + 置信度达标。
    threshold 参数供网格搜索/评估时覆盖,不传则用 detector_config 的当前值。
    """
    if not is_alertable_behavior(behavior_type):
        return False
    if not sequence_confirmed:
        return False
    if threshold is None:
        threshold = detector_config.THEFT_ALERT_THRESHOLD
    return float(confidence) >= threshold


def should_run_aggregate_detector(realtime_confirmed):
    return not realtime_confirmed
