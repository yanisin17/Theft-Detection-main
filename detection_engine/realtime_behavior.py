# -*- coding: utf-8 -*-
"""
实时行为检测器（项目A 21种行为引擎的实时可运行版本）
=====================================================
背景：
    原 video_behavior.py 的实时入口 detect_behaviors_in_image 存在内部数据结构不一致
    （姿态提取返回 33 点 list，特征提取却要求整数 key 的 dict；规则子检测器又按 17 点
    COCO 索引访问），且随包 XGBoost 在实时流上输出退化（二分类恒 0 / 多分类恒定值）。

本模块改用 MediaPipe Pose 的 33 个关键点（项目A的姿态源），以正确的 33 点索引实现
几何 + 时序规则，聚焦零售盗窃最确定的动作链：

        伸手(Taking Object) → 手回缩到腰腹/胯部(Rapid Item Concealment)

与初版 Theft-Detection“手腕一靠近髋部就报警”相比，本实现要求：
    1) 手腕先明确探出（低于髋部或水平探出身体且手臂伸直）；
    2) 随后回缩到躯干/腰腹/胯部区域；
    3) 回缩位移达到阈值。
正常站立时手即便在胯侧，因没有“探出→回缩”位移，不会触发，显著降低误报。

每个 track_id 维护独立状态机；一个“伸手→藏匿”周期产出一次藏匿事件。
是否对同一个人只报警一次 / 峰值置信度回写，由 backend 层负责。
"""

import os
import time
import math
import cv2
import numpy as np

from detection_engine import detector_config

# MediaPipe Pose 33 关键点索引
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16
_L_HIP, _R_HIP = 23, 24

# 判定阈值的默认值统一收在 detector_config.py(可被 detector_tuning.json
# 或构造函数参数覆盖),此处仅保留索引常量。


class _TrackState:
    __slots__ = ("phase", "out_hits", "far", "reach_t0", "last_seen",
                 "taking_conf", "taking_hits")

    def __init__(self):
        self.phase = "idle"        # idle -> reaching -> (fire) -> idle
        self.out_hits = 0
        self.far = None            # 探出阶段手腕到达的最远点 (x, y)
        self.reach_t0 = 0.0
        self.last_seen = time.time()
        self.taking_conf = 0.0
        self.taking_hits = 0

    def reset(self):
        self.phase = "idle"
        self.out_hits = 0
        self.far = None
        self.reach_t0 = 0.0
        self.taking_conf = 0.0
        self.taking_hits = 0


class RealtimeBehaviorDetector:
    """对单个人物裁剪图做 MediaPipe 33 点分析，返回当前周期最强行为事件。"""

    def __init__(self, min_detection_confidence=None, model_complexity=None,
                 vis_min=None, out_hits_to_reach=None, retreat_min=None,
                 reach_timeout=None, reach_down_below_hip=None,
                 reach_out_side_ratio=None, reach_arm_angle_min=None,
                 clock=None):
        """
        所有参数不传时取 detector_config 的当前值;显式传参用于网格搜索/评估。
        clock: 返回秒数的可调用对象,默认 time.time;离线回放评估时传视频时间,
               否则以快于实时的速度回放会让 _REACH_TIMEOUT 等超时逻辑失真。
        """
        cfg = detector_config
        self.vis_min = cfg.VIS_MIN if vis_min is None else vis_min
        self.out_hits_to_reach = (cfg.OUT_HITS_TO_REACH if out_hits_to_reach is None
                                  else out_hits_to_reach)
        self.retreat_min = cfg.RETREAT_MIN if retreat_min is None else retreat_min
        self.reach_timeout = cfg.REACH_TIMEOUT if reach_timeout is None else reach_timeout
        self.reach_down_below_hip = (cfg.REACH_DOWN_BELOW_HIP if reach_down_below_hip is None
                                     else reach_down_below_hip)
        self.reach_out_side_ratio = (cfg.REACH_OUT_SIDE_RATIO if reach_out_side_ratio is None
                                     else reach_out_side_ratio)
        self.reach_arm_angle_min = (cfg.REACH_ARM_ANGLE_MIN if reach_arm_angle_min is None
                                    else reach_arm_angle_min)
        self._clock = clock or time.time

        self._mp_pose = None
        self.pose = None
        self._states = {}          # track_key -> _TrackState
        if min_detection_confidence is None:
            min_detection_confidence = cfg.MEDIAPIPE_MIN_DETECTION_CONFIDENCE
        if model_complexity is None:
            model_complexity = cfg.MEDIAPIPE_MODEL_COMPLEXITY
        self._init_pose(min_detection_confidence, model_complexity)

    def _init_pose(self, min_det, model_complexity):
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            # 逐人裁剪图 + 自带状态机，使用静态图模式；complexity 可调(0快/1准)
            self.pose = self._mp_pose.Pose(
                static_image_mode=True,
                model_complexity=model_complexity,
                enable_segmentation=False,
                min_detection_confidence=min_det,
                min_tracking_confidence=0.4,
            )
            print(f"RealtimeBehaviorDetector: MediaPipe Pose (33pt, complexity={model_complexity}) ready.")
        except Exception as e:
            self.pose = None
            print(f"RealtimeBehaviorDetector init failed: {e}")

    @staticmethod
    def _angle(a, b, c):
        """以 b 为顶点的夹角（度），输入为 (x, y)"""
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        na = math.hypot(*ba)
        nc = math.hypot(*bc)
        if na < 1e-6 or nc < 1e-6:
            return 180.0
        cosv = (ba[0] * bc[0] + ba[1] * bc[1]) / (na * nc)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosv))))

    def _hand_signals(self, lm, side):
        """计算单手的几何信号。返回 dict 或 None（关键点不可用时）。"""
        ish, ishr = (_L_SHOULDER, _R_SHOULDER) if side == "L" else (_R_SHOULDER, _L_SHOULDER)
        iel, ier = (_L_ELBOW, _R_ELBOW) if side == "L" else (_R_ELBOW, _L_ELBOW)
        iwr, iwl = (_L_WRIST, _R_WRIST) if side == "L" else (_R_WRIST, _L_WRIST)
        ihip, ihipr = (_L_HIP, _R_HIP) if side == "L" else (_R_HIP, _L_HIP)

        sh, shr = lm[ish], lm[ishr]
        el = lm[iel]
        wr, wrl = lm[iwr], lm[iwl]
        hip, hipr = lm[ihip], lm[ihipr]

        # 可见度过滤
        if sh[2] < self.vis_min or el[2] < self.vis_min or wr[2] < self.vis_min or hip[2] < self.vis_min:
            return None

        smx, smy = (sh[0] + shr[0]) / 2.0, (sh[1] + shr[1]) / 2.0
        hmx, hmy = (hip[0] + hipr[0]) / 2.0, (hip[1] + hipr[1]) / 2.0
        tw = max(abs(shr[0] - sh[0]), 1e-3)
        th = max(abs(hip[1] - sh[1]), abs(hipr[1] - shr[1]), 1e-3)

        wx, wy = wr[0], wr[1]
        arm_ang = self._angle((sh[0], sh[1]), (el[0], el[1]), (wx, wy))

        # 1) 手探出：向下（低于髋）或向外（水平离开身体且手臂较直）
        reach_down = wy > hmy + self.reach_down_below_hip
        reach_out = abs(wx - smx) > self.reach_out_side_ratio * tw and arm_ang > self.reach_arm_angle_min
        reaching = reach_down or reach_out

        # 2) 手回缩到躯干 / 腰腹 / 胯部区域（藏匿位置）
        x_inside = (smx - 1.15 * tw) <= wx <= (smx + 1.15 * tw)
        on_torso = x_inside and (smy - 0.05) <= wy <= (hmy + 0.18)
        on_hip_bag = x_inside and (hmy - 0.25 * th) <= wy <= (hmy + 0.22)
        conceal_zone = on_torso or on_hip_bag

        return {
            "w": (wx, wy),
            "reaching": reaching,
            "reach_down": reach_down,
            "conceal_zone": conceal_zone,
            "on_hip_bag": on_hip_bag,
            "arm_ang": arm_ang,
        }

    def extract_landmarks(self, person_bgr):
        """
        跑 MediaPipe Pose,返回 {索引: (x, y, visibility)} 或 None。
        拆出这一步是为了离线评估/网格搜索:关键点推理与阈值无关,可缓存后
        用不同参数反复重放状态机(step),避免重复做昂贵的推理。
        """
        if self.pose is None or person_bgr is None or person_bgr.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)
        except Exception:
            return None
        if not res.pose_landmarks:
            return None
        pts = res.pose_landmarks.landmark
        return {i: (p.x, p.y, float(p.visibility)) for i, p in enumerate(pts)}

    def analyze(self, person_bgr, track_key):
        """提取关键点 + 推进状态机(在线实时路径的便捷入口)。"""
        lm = self.extract_landmarks(person_bgr)
        return self.step(lm, track_key)

    def step(self, lm, track_key, now=None):
        """
        纯状态机:给定关键点 lm(extract_landmarks 的输出,可为 None)推进一个
        周期,返回事件或 None。now 用于注入时间(离线回放传视频时间)。

        Returns:
            None
            或 {"type": str, "confidence": float, "phase": str}
            type ∈ {"Taking Object"(提示,低置信), "Rapid Item Concealment"(报警级)}
        """
        if now is None:
            now = self._clock()

        st = self._states.get(track_key)
        if st is None:
            st = _TrackState()
            self._states[track_key] = st
        st.last_seen = now

        if lm is None:
            return None

        sig_l = self._hand_signals(lm, "L")
        sig_r = self._hand_signals(lm, "R")
        sigs = [s for s in (sig_l, sig_r) if s is not None]
        if not sigs:
            return None

        # 主导手：探出阶段取“更靠外/靠下”的手，藏匿阶段取“在藏匿区”的手
        reaching_sig = max(sigs, key=lambda s: (s["reaching"], s["w"][1], abs(s["w"][0] - 0.5)))
        conceal_sig = next((s for s in sigs if s["conceal_zone"]), None)

        # ---------- idle：确认手确实探出 ----------
        if st.phase == "idle":
            if reaching_sig["reaching"]:
                st.out_hits += 1
                st.far = reaching_sig["w"]
                st.reach_t0 = now
                # 伸手拿取的中间提示（低于报警阈值，仅画面标注）
                st.taking_hits += 1
                st.taking_conf = min(0.45 + 0.05 * st.taking_hits, 0.65)
                if st.out_hits >= self.out_hits_to_reach:
                    st.phase = "reaching"
                return {"type": "Taking Object", "confidence": st.taking_conf, "phase": "reach"}
            else:
                st.out_hits = max(0, st.out_hits - 1)
                st.taking_hits = max(0, st.taking_hits - 1)
                return None

        # ---------- reaching：继续追踪最远点，或判定回缩藏匿 ----------
        if st.phase == "reaching":
            if reaching_sig["reaching"] and conceal_sig is None:
                # 手仍在探出，持续刷新最远点与时间
                if st.far is None:
                    st.far = reaching_sig["w"]
                else:
                    # 取离身体中轴更远 / 更低的点作为“最远点”
                    cand = reaching_sig["w"]
                    if cand[1] > st.far[1] or abs(cand[0] - 0.5) > abs(st.far[0] - 0.5):
                        st.far = cand
                st.reach_t0 = now
                return {"type": "Taking Object", "confidence": 0.65, "phase": "reach"}

            if conceal_sig is not None and st.far is not None:
                retreat = math.hypot(conceal_sig["w"][0] - st.far[0],
                                     conceal_sig["w"][1] - st.far[1])
                if retreat >= self.retreat_min:
                    # 完成 “探出拿取 → 回缩藏匿” 序列
                    # 报警分：基于回缩距离相对于阈值的超额倍数 + 藏匿位置 + 探出猛烈度
                    # 校准(P1)：旧公式 base=0.45+0.25×ratio 需要回缩距离达到 2×阈值
                    # 才能稳过 0.7 报警线，典型真实事件(1.2~1.5×阈值)得分只有
                    # 0.55~0.65，全部漏报。现改为 base=0.50+0.30×ratio 并配合
                    # 阈值降至 0.6：刚达标≈0.50~0.59(临界观察)，典型 0.65~0.75
                    # (触发报警)，极端 0.85(封顶)
                    excess = max(0.0, retreat - self.retreat_min)  # 超出阈值的部分
                    excess_ratio = min(1.0, excess / self.retreat_min)  # 0~1
                    base = 0.50 + excess_ratio * 0.30  # 0.50~0.80

                    # 藏匿位置加成：胯部/裤袋(+0.05) > 躯干藏匿(+0.02)
                    pos_bonus = 0.05 if conceal_sig["on_hip_bag"] else (0.02 if conceal_sig["conceal_zone"] else 0.0)

                    # 探出猛烈度加成：手臂伸直角度越大 = 探出越果决
                    reach_fierce = max(0.0, (reaching_sig.get("arm_ang", 180) - self.reach_arm_angle_min) / 60)  # 0~1
                    fierce_bonus = reach_fierce * 0.04

                    conf = float(max(0.45, min(0.85, base + pos_bonus + fierce_bonus)))
                    st.reset()
                    return {"type": "Rapid Item Concealment", "alert_score": conf,
                            "phase": "conceal"}

            if now - st.reach_t0 > self.reach_timeout:
                st.reset()
            return None

        return None

    def prune(self, max_age_seconds=20.0):
        """清理长期消失的 track 状态"""
        now = self._clock()
        dead = [k for k, s in self._states.items() if now - s.last_seen > max_age_seconds]
        for k in dead:
            self._states.pop(k, None)

    def close(self):
        if self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass
