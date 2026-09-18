# -*- coding: utf-8 -*-
"""
检测阈值集中配置
================
此前阈值散落在 alert_policy.py / realtime_behavior.py / backend.py 三处,
无法系统地调优和对比。本模块统一管理所有影响识别精度的阈值;

覆盖方式(优先级从高到低):
    1. 显式传参(网格搜索 / 单元测试用构造函数参数覆盖)
    2. 项目根目录 detector_tuning.json(运维不改代码调参用)
    3. 本文件中的默认值

detector_tuning.json 示例:
    {
        "THEFT_ALERT_THRESHOLD": 0.65,
        "RETREAT_MIN": 0.12,
        "MEDIAPIPE_MODEL_COMPLEXITY": 1
    }
"""

import json
import os

# === 报警策略 ===
# P1 校准：0.7 与藏匿事件 alert_score 分布错配（旧公式下典型真实事件只有
# 0.55~0.65，全部漏报）。降至 0.6 配合 realtime_behavior 的新分数公式，
# 典型事件(回缩距离 1.2~1.5×阈值)得分 0.65~0.75 可稳定触发报警
THEFT_ALERT_THRESHOLD = 0.6     # 偷盗行为报警的最低置信度

# === 实时行为状态机(RealtimeBehaviorDetector) ===
VIS_MIN = 0.35                  # MediaPipe 关键点可见度下限
OUT_HITS_TO_REACH = 2           # 连续多少次"探出"判定才进入 reaching 阶段
REACH_DOWN_BELOW_HIP = 0.03     # 手腕低于髋部多少算向下探(归一化)
REACH_OUT_SIDE_RATIO = 1.05     # 手腕水平距身体中轴 > 肩宽*该系数算向外探
REACH_ARM_ANGLE_MIN = 120       # 向外探判定所需的最小手臂伸直角度(度)
RETREAT_MIN = 0.14              # 从最远点回缩到躯干的最小归一化距离
REACH_TIMEOUT = 3.0             # 探出后超过该秒数未回缩则放弃该序列
# P1：0 是最粗糙的姿态模型，子码流下远处的人裁剪图仅 100~200px 高，
# 手腕/手肘关键点抖动大、可见度低。complexity=1 显著更稳，速度可接受
# （逐人裁剪图 + static_image_mode，单人 ~50-100ms/帧 CPU）
MEDIAPIPE_MODEL_COMPLEXITY = 1  # MediaPipe Pose 复杂度 0/1/2,越大越准越慢
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.4

# === 佐证确认机制(降低单次事件误报) ===
# 单次"伸手→回缩藏匿"事件先挂起,以下任一条件满足才真正报警:
#   a) 同一 track 在 CORROBORATION_WINDOW 秒内出现第二次藏匿事件
#   b) 藏匿后确认手中持有可偷物品(check_object_in_hand)
# P1: 暂时关闭。fallback 物体模型是通用 COCO，TARGET_CLASSES 里没有货架
# 商品(包装食品/小商品等)，"手中持物"佐证路径基本永远不成立；真实小偷
# 通常只做一次 → 大量真实偷窃停留在 PENDING CONFIRM 永远不报警。
# 先关闭跑通召回，待商品检测模型就绪、误报需要压制时再开启
REQUIRE_CORROBORATION = False
CORROBORATION_WINDOW = 10.0     # 二次藏匿事件佐证的时间窗(秒)

_TUNING_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "detector_tuning.json",
)


def apply_overrides(path=None):
    """用 JSON 文件覆盖默认值;返回未能识别的键列表。"""
    path = path or _TUNING_FILE
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            overrides = json.load(f)
    except (OSError, ValueError) as e:
        print(f"detector_config: 无法读取 {path}: {e}")
        return []
    unknown = []
    for key, value in overrides.items():
        if key.startswith("_") or key in ("apply_overrides",):
            unknown.append(key)
            continue
        if key in globals():
            globals()[key] = value
        else:
            unknown.append(key)
    if unknown:
        print(f"detector_config: 忽略未知的配置键: {unknown}")
    return unknown


apply_overrides()
