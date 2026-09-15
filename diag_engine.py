"""
诊断脚本 v2：测试 VideoBehaviorDetector 引擎是否正常工作
对着摄像头做偷东西动作（伸手拿→缩回口袋），看输出结果
"""
import cv2
import sys
import os
import json
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("行为引擎诊断脚本 v2")
print("=" * 60)

# 1. 加载引擎
try:
    from detection_engine.models.behavior.video_behavior import VideoBehaviorDetector
    print("[OK] VideoBehaviorDetector 导入成功")
except Exception as e:
    print(f"[FAIL] 导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    detector = VideoBehaviorDetector()
    print("[OK] VideoBehaviorDetector 实例化成功")
    print(f"  - 标准模型: {hasattr(detector, 'standard_model') and detector.standard_model is not None}")
    print(f"  - 姿态检测器: {hasattr(detector, 'pose_detector') and detector.pose_detector is not None}")
    print(f"  - model_type: {getattr(detector, 'model_type', '未设置')}")
except Exception as e:
    print(f"[FAIL] 实例化失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 2. 加载 YOLO
print("\n--- 加载 YOLO ---")
try:
    from ultralytics import YOLO
    yolo = YOLO('models/yolov11n.pt')
    print("[OK] YOLO 加载成功")
except Exception as e:
    print(f"[FAIL] YOLO 加载失败: {e}")
    sys.exit(1)

# 3. 打开 RTSP 流
print("\n--- 打开摄像头 ---")
RTSP_URL = "rtsp://admin:a12345678@172.30.20.63:554/Streaming/Channels/102"
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print("[FAIL] 无法打开 RTSP 流")
    sys.exit(1)
print("[OK] 摄像头已打开")
print("现在对着摄像头做偷东西动作（伸手拿→缩回口袋）")
print("跑 60 帧后自动结束\n")

# 4. 跑 60 帧
frame_count = 0
behavior_counts = {}
max_confidences = {}
errors = []
total_behaviors = 0
raw_returns = []  # 记录前几帧的原始返回值

for i in range(60):
    ret, frame = cap.read()
    if not ret:
        print(f"[WARN] 第 {i} 帧读取失败")
        continue

    frame_count += 1

    # YOLO 检测 - 降低置信度阈值到 0.1 看看能不能检测到
    try:
        yolo_results = yolo(frame, verbose=False, conf=0.1)

        # 统计检测到的人 + 打印所有检测
        person_count = 0
        all_detections = []
        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                cls_name = yolo.names.get(cls_id, str(cls_id))
                if cls_id == 0:
                    person_count += 1
                all_detections.append(f"{cls_name}({conf_val:.2f})")

        if i % 10 == 0:
            print(f"帧 {i}: 检测到 {person_count} 人, 所有检测: {all_detections[:10]}")

        # 第 0 帧保存到磁盘
        if i == 0:
            cv2.imwrite("diag_frame_0.jpg", frame)
            print(f"  [DEBUG] 第 0 帧已保存到 diag_frame_0.jpg ({frame.shape[1]}x{frame.shape[0]})")

        # 调用引擎 - detections 传 ultralytics Results 对象
        behaviors = detector.detect_behaviors_in_image(frame, yolo_results[0] if yolo_results else None)

        if i < 5:
            raw_returns.append(f"帧{i}: type={type(behaviors).__name__}, len={len(behaviors) if behaviors else 0}")
            if behaviors:
                raw_returns.append(f"  内容: {behaviors[:2]}")

        if behaviors:
            total_behaviors += len(behaviors)
            for b in behaviors:
                if isinstance(b, dict):
                    btype = b.get('type', b.get('behavior_type', 'unknown'))
                    bconf = b.get('confidence', b.get('score', 0))
                else:
                    btype = str(b)
                    bconf = 0

                behavior_counts[btype] = behavior_counts.get(btype, 0) + 1
                if btype not in max_confidences or bconf > max_confidences[btype]:
                    max_confidences[btype] = bconf

                if bconf > 0.1:
                    print(f"  帧 {i}: {btype} = {bconf:.3f}")

    except Exception as e:
        err_msg = str(e)
        if len(errors) < 10:
            errors.append(f"帧 {i}: {err_msg}")
            if isinstance(e, KeyError):
                print(f"  [ERROR] KeyError 帧 {i}: {err_msg}")
            elif isinstance(e, TypeError):
                print(f"  [ERROR] TypeError 帧 {i}: {err_msg}")

cap.release()
cv2.destroyAllWindows()

# 5. 汇总
print("\n" + "=" * 60)
print("诊断结果汇总")
print("=" * 60)
print(f"总帧数: {frame_count}")
print(f"检测到的行为总数: {total_behaviors}")
print(f"行为类型数: {len(behavior_counts)}")

if raw_returns:
    print("\n--- 前 5 帧原始返回 ---")
    for r in raw_returns:
        print(f"  {r}")

if behavior_counts:
    print("\n--- 行为类型统计 (按出现次数排序) ---")
    for btype, count in sorted(behavior_counts.items(), key=lambda x: -x[1]):
        max_conf = max_confidences.get(btype, 0)
        print(f"  {btype}: 出现 {count} 次, 最高置信度 {max_conf:.3f}")
else:
    print("\n[WARNING] 60 帧内没有检测到任何行为！")

if errors:
    print(f"\n--- 错误 ({len(errors)} 条, 显示前 10) ---")
    for e in errors:
        print(f"  {e}")
else:
    print("\n--- 无错误 ---")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
