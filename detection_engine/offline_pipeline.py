# -*- coding: utf-8 -*-
"""
离线回放检测引擎
================
复刻 backend.video_loop 的核心检测判定链路(YOLO pose 追踪 → 人物裁剪 →
MediaPipe 关键点 → 实时状态机 → 佐证确认 → 报警策略),但不依赖
FastAPI/SQLite/WebSocket,用于:

    1. 对历史视频/证据帧目录回放,评估查全率(漏报)
    2. 网格搜索调参(关键点推理与阈值无关,可缓存后重放状态机)

与 backend.py 的对应关系:
    - YOLO pose track + classes=[0]                ← backend.py video_loop 第 969 行
    - 物品检测(回退模型 conf=0.3, 每 5 帧)      ← backend.py 第 996-1010 行
    - check_object_in_hand(120px 阈值)             ← backend.py 第 834 行
    - RealtimeBehaviorDetector(每 6 帧, ±60px 裁剪)← backend.py 第 1100-1151 行
    - 佐证确认(二次事件/手中持物)                ← backend.py realtime 分支
    - should_trigger_theft_alert                   ← alert_policy.py
"""

import os
import glob
import cv2
import numpy as np

from detection_engine.alert_policy import should_trigger_theft_alert
from detection_engine import detector_config
from detection_engine.realtime_behavior import RealtimeBehaviorDetector

# COCO 可偷物品类别(与 backend.py 的 TARGET_CLASSES 一致)
TARGET_CLASSES = [24, 25, 26, 28, 39, 40, 41, 42, 43, 67, 73, 74, 75, 76, 77, 78, 79]
OBJ_CONF = 0.3
OBJ_CHECK_INTERVAL = 5     # 物品检测每 N 帧一次
BEHAVIOR_CHECK_INTERVAL = 6  # 行为分析每 N 帧一次(与 backend 一致)
CROP_PAD = 60
DEFAULT_FPS = 30.0         # 帧目录(无 fps 元数据)按此值换算视频时间


def _check_object_in_hand(kpts, object_boxes, hand="LEFT"):
    """backend.check_object_in_hand 的离线副本(避免 import backend 拉起 FastAPI)。"""
    if len(kpts) < 11:
        return False
    wrist = kpts[9] if hand == "LEFT" else kpts[10]
    if wrist[0] == 0:
        return False
    for box in object_boxes:
        box_cx = (box[0] + box[2]) / 2
        box_cy = (box[1] + box[3]) / 2
        dist = np.sqrt((wrist[0] - box_cx) ** 2 + (wrist[1] - box_cy) ** 2)
        if dist < 120:
            return True
        if box[0] < wrist[0] < box[2] and box[1] < wrist[1] < box[3]:
            return True
    return False


class _ReplayTrackState:
    """对应 backend.PersonState 中与偷盗报警相关的字段。"""

    def __init__(self):
        self.holding_object = False
        self.last_holding_time = 0.0
        self.conceal_event_times = []
        self.alert_fired = False
        self.max_confidence = 0.0


class OfflinePipeline:
    """
    两阶段回放:
        build_timeline(source)  -> 逐帧跑 YOLO + MediaPipe,产出与阈值无关的
                                    时间线(关键点已缓存,耗时大头,只跑一次)
        replay(timeline, ...)   -> 用给定参数重放状态机 + 佐证 + 报警策略,
                                   秒级完成,网格搜索每个参数组合调一次
    """

    def __init__(self, pose_model='yolov8n-pose.pt', obj_model='models/yolov11n.pt',
                 realtime_kwargs=None, verbose=False):
        from ultralytics import YOLO
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pose_path = pose_model if os.path.isabs(pose_model) else os.path.join(root, pose_model)
        obj_path = obj_model if os.path.isabs(obj_model) else os.path.join(root, obj_model)
        self.model_pose = YOLO(pose_path)
        self.model_obj = YOLO(obj_path) if os.path.exists(obj_path) else None
        self.realtime_kwargs = realtime_kwargs or {}
        self.verbose = verbose

    # ------------------------------------------------------------------
    # 阶段 1:构建时间线(参数无关,可缓存)
    # ------------------------------------------------------------------
    def _iter_frames(self, source):
        """source 为视频文件路径或帧图片目录。产出 (frame_idx, frame_bgr)。"""
        if os.path.isdir(source):
            files = sorted(glob.glob(os.path.join(source, '*.jpg'))
                           + glob.glob(os.path.join(source, '*.png')))
            if not files:
                raise ValueError(f"目录中没有帧图片: {source}")
            for i, fp in enumerate(files):
                yield i, cv2.imread(fp)
        else:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {source}")
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield i, frame
                i += 1
            cap.release()

    def _fps_of(self, source):
        if os.path.isdir(source):
            return DEFAULT_FPS
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps and fps > 0 else DEFAULT_FPS

    def build_timeline(self, source):
        """
        返回 dict:
            name, fps, frames: 总帧数,
            samples: [{frame, t, track_id, kpts, lm, holding_hint}]
                每隔 BEHAVIOR_CHECK_INTERVAL 帧为每个人物采一条;
                lm 为 MediaPipe 33 点(裁剪自 ±60px 人物框),holding_hint 为
                该时刻手腕附近是否检测到可偷物品(佐证用)。
        """
        fps = self._fps_of(source)
        name = os.path.basename(os.path.normpath(source))
        detector = RealtimeBehaviorDetector(**self.realtime_kwargs)
        samples, total_frames = [], 0
        last_objects = []
        pending_holdings = {}   # track_id -> (t, holding)

        for frame_idx, frame in self._iter_frames(source):
            total_frames = frame_idx + 1
            if frame is None:
                continue
            t = frame_idx / fps

            results_pose = self.model_pose.track(frame, persist=True, verbose=False, classes=[0])
            if not results_pose or results_pose[0].boxes is None:
                continue

            # 物品检测(每 N 帧,与 backend 一致;中间帧沿用上次结果)
            if frame_idx % OBJ_CHECK_INTERVAL == 0 and self.model_obj is not None:
                detected_objects = []
                results_obj = self.model_obj(frame, verbose=False, conf=OBJ_CONF)
                if results_obj and results_obj[0].boxes is not None:
                    boxes_obj = results_obj[0].boxes.xyxy.cpu().numpy().astype(int)
                    cls_obj = results_obj[0].boxes.cls.cpu().numpy().astype(int)
                    for b, c in zip(boxes_obj, cls_obj):
                        if c in TARGET_CLASSES:
                            detected_objects.append(b)
                last_objects = detected_objects

            if results_pose[0].boxes.id is None:
                continue
            boxes = results_pose[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results_pose[0].boxes.id.cpu().numpy().astype(int)
            try:
                keypoints_all = results_pose[0].keypoints.xy.cpu().numpy()
            except Exception:
                keypoints_all = []

            for i, track_id in enumerate(track_ids):
                box = boxes[i]
                kpts = keypoints_all[i] if len(keypoints_all) > i else []
                if len(kpts) < 11:
                    continue

                # 手中持物(佐证信号,backend 每 5s 持久 3s,这里直接算当前帧)
                left_has = _check_object_in_hand(kpts, last_objects, "LEFT")
                right_has = _check_object_in_hand(kpts, last_objects, "RIGHT")
                holding = left_has or right_has
                if holding:
                    pending_holdings[track_id] = (t, True)

                # 每 N 帧采一条行为样本(与 backend 的检查频率一致)
                if frame_idx % BEHAVIOR_CHECK_INTERVAL == 0:
                    sx1 = max(0, box[0] - CROP_PAD)
                    sy1 = max(0, box[1] - CROP_PAD)
                    sx2 = min(frame.shape[1], box[2] + CROP_PAD)
                    sy2 = min(frame.shape[0], box[3] + CROP_PAD)
                    person_frame = frame[sy1:sy2, sx1:sx2].copy()
                    lm = detector.extract_landmarks(person_frame)
                    # 持物状态:当前帧检测到,或 3 秒内检测到过(与 backend 持久逻辑一致)
                    ht, hv = pending_holdings.get(track_id, (None, None))
                    holding_now = holding or (hv is True and t - ht <= 3.0)
                    samples.append({
                        "frame": frame_idx, "t": t,
                        "track_id": int(track_id), "kpts": kpts,
                        "lm": lm, "holding": holding_now,
                    })

        if self.verbose:
            print(f"[timeline] {name}: {total_frames} 帧, {len(samples)} 条人物样本")
        detector.close()
        return {"name": name, "source": source, "fps": fps,
                "frames": total_frames, "samples": samples}

    # ------------------------------------------------------------------
    # 阶段 2:重放状态机 + 佐证 + 报警策略(参数相关,秒级)
    # ------------------------------------------------------------------
    def replay(self, timeline, alert_threshold=None, require_corroboration=None,
               corroboration_window=None, realtime_kwargs=None, with_pending=False):
        """
        返回 dict: alerts(报警事件列表)、pendings(报警级但缺佐证的事件)、
        taking_events(中间提示事件数)。

        alerts 元素: {t, track_id, behavior, confidence, corroborated_by}
        """
        if alert_threshold is None:
            alert_threshold = detector_config.THEFT_ALERT_THRESHOLD
        if require_corroboration is None:
            require_corroboration = detector_config.REQUIRE_CORROBORATION
        if corroboration_window is None:
            corroboration_window = detector_config.CORROBORATION_WINDOW
        kwargs = dict(self.realtime_kwargs)
        if realtime_kwargs:
            kwargs.update(realtime_kwargs)

        detector = RealtimeBehaviorDetector(**kwargs)
        states = {}
        alerts, pendings, taking_events = [], [], 0

        for s in timeline["samples"]:
            key = f"{timeline['name']}:{s['track_id']}"
            event = detector.step(s["lm"], key, now=s["t"])
            if event is None:
                continue
            etype = event.get("type", "")
            econf = float(event.get("confidence", 0.0))
            if etype == "Taking Object":
                taking_events += 1
                continue

            if not should_trigger_theft_alert(etype, econf, sequence_confirmed=True,
                                              threshold=alert_threshold):
                continue

            st = states.setdefault(key, _ReplayTrackState())
            if econf > st.max_confidence:
                st.max_confidence = econf

            st.conceal_event_times.append(s["t"])
            st.conceal_event_times = [
                ts for ts in st.conceal_event_times
                if s["t"] - ts <= corroboration_window
            ]
            corroborated_by = None
            if not require_corroboration:
                corroborated_by = "disabled"
            elif len(st.conceal_event_times) >= 2:
                corroborated_by = "repeat"
            elif s["holding"]:
                corroborated_by = "holding"

            record = {"t": round(s["t"], 2), "track_id": s["track_id"],
                      "behavior": "rapid_item_concealment", "confidence": round(econf, 3)}
            if corroborated_by:
                record["corroborated_by"] = corroborated_by
                if not st.alert_fired:
                    alerts.append(record)
                    st.alert_fired = True
            else:
                pendings.append(record)  # 报警级事件但缺佐证

        detector.close()
        return {"alerts": alerts, "pendings": pendings,
                "taking_events": taking_events}
