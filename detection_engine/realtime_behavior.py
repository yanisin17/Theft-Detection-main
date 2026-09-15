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

# MediaPipe Pose 33 关键点索引
_L_SHOULDER, _R_SHOULDER = 11, 12
_L_ELBOW, _R_ELBOW = 13, 14
_L_WRIST, _R_WRIST = 15, 16
_L_HIP, _R_HIP = 23, 24

# 判定阈值（均基于归一化坐标 / 身体比例，分辨率无关）
_VIS_MIN = 0.35          # 关键点可见度下限
_OUT_HITS_TO_REACH = 2   # 连续多少次“探出”判定才进入 reaching 阶段
_REACH_DOWN_BELOW_HIP = 0.03    # 手腕低于髋部多少算向下探
_REACH_OUT_SIDE_RATIO = 1.05    # 手腕水平距身体中轴 > 肩宽*该系数 算向外探
_ARM_EXTENDED_ANGLE = 150       # 手臂伸直角度
_RETREAT_MIN = 0.14      # 从最远点回缩到躯干的最小归一化距离
_REACH_TIMEOUT = 3.0     # 探出后超过该秒数未回缩则放弃该序列


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

    def __init__(self, min_detection_confidence=0.4):
        self._mp_pose = None
        self.pose = None
        self._states = {}          # track_key -> _TrackState
        self._init_pose(min_detection_confidence)

    def _init_pose(self, min_det):
        try:
            import mediapipe as mp
            self._mp_pose = mp.solutions.pose
            # 逐人裁剪图 + 自带状态机，使用静态图模式；complexity=0 保证实时性能
            self.pose = self._mp_pose.Pose(
                static_image_mode=True,
                model_complexity=0,
                enable_segmentation=False,
                min_detection_confidence=min_det,
                min_tracking_confidence=0.4,
            )
            print("RealtimeBehaviorDetector: MediaPipe Pose (33pt) ready.")
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
        if sh[2] < _VIS_MIN or el[2] < _VIS_MIN or wr[2] < _VIS_MIN or hip[2] < _VIS_MIN:
            return None

        smx, smy = (sh[0] + shr[0]) / 2.0, (sh[1] + shr[1]) / 2.0
        hmx, hmy = (hip[0] + hipr[0]) / 2.0, (hip[1] + hipr[1]) / 2.0
        tw = max(abs(shr[0] - sh[0]), 1e-3)
        th = max(abs(hip[1] - sh[1]), abs(hipr[1] - shr[1]), 1e-3)

        wx, wy = wr[0], wr[1]
        arm_ang = self._angle((sh[0], sh[1]), (el[0], el[1]), (wx, wy))

        # 1) 手探出：向下（低于髋）或向外（水平离开身体且手臂较直）
        reach_down = wy > hmy + _REACH_DOWN_BELOW_HIP
        reach_out = abs(wx - smx) > _REACH_OUT_SIDE_RATIO * tw and arm_ang > 120
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

    def analyze(self, person_bgr, track_key):
        """
        分析单个人物裁剪图。

        Returns:
            None
            或 {"type": str, "confidence": float, "phase": str}
            type ∈ {"Taking Object"(提示,低置信), "Rapid Item Concealment"(报警级)}
        """
        if self.pose is None or person_bgr is None or person_bgr.size == 0:
            return None

        st = self._states.get(track_key)
        if st is None:
            st = _TrackState()
            self._states[track_key] = st
        st.last_seen = time.time()

        try:
            rgb = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)
        except Exception:
            return None

        if not res.pose_landmarks:
            return None

        pts = res.pose_landmarks.landmark
        lm = {i: (p.x, p.y, float(p.visibility)) for i, p in enumerate(pts)}

        sig_l = self._hand_signals(lm, "L")
        sig_r = self._hand_signals(lm, "R")
        sigs = [s for s in (sig_l, sig_r) if s is not None]
        if not sigs:
            return None

        now = time.time()
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
                if st.out_hits >= _OUT_HITS_TO_REACH:
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
                if retreat >= _RETREAT_MIN:
                    # 完成 “探出拿取 → 回缩藏匿” 序列
                    # 置信度：基于回缩距离相对于阈值的超额倍数 + 藏匿位置 + 探出猛烈度
                    # 预期分布：正常 0.72~0.85，典型 0.80~0.92，极端 0.93~0.96
                    excess = max(0.0, retreat - _RETREAT_MIN)  # 超出阈值的部分
                    excess_ratio = min(1.0, excess / _RETREAT_MIN)  # 0~1
                    base = 0.72 + excess_ratio * 0.18  # 0.72~0.90

                    # 藏匿位置加成：胯部/裤袋(+0.05) > 躯干藏匿(+0.02)
                    pos_bonus = 0.05 if conceal_sig["on_hip_bag"] else (0.02 if conceal_sig["conceal_zone"] else 0.0)

                    # 探出猛烈度加成：手臂伸直角度越大 = 探出越果决
                    reach_fierce = max(0.0, (reaching_sig.get("arm_ang", 180) - 120) / 60)  # 0~1
                    fierce_bonus = reach_fierce * 0.04

                    conf = float(max(0.72, min(0.96, base + pos_bonus + fierce_bonus)))
                    st.reset()
                    return {"type": "Rapid Item Concealment", "confidence": conf,
                            "phase": "conceal"}

            if now - st.reach_t0 > _REACH_TIMEOUT:
                st.reset()
            return None

        return None

    def prune(self, max_age_seconds=20.0):
        """清理长期消失的 track 状态"""
        now = time.time()
        dead = [k for k, s in self._states.items() if now - s.last_seen > max_age_seconds]
        for k in dead:
            self._states.pop(k, None)

    def close(self):
        if self.pose is not None:
            try:
                self.pose.close()
            except Exception:
                pass
