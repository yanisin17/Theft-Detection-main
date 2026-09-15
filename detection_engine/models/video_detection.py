import os
import cv2
import time
import torch
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from .common_detector import CommonDetector
from .behavior.video_behavior import VideoBehaviorDetector

# 配置日志
logger = logging.getLogger('video_detection')

class VideoDetector(CommonDetector):
    """专门处理视频的检测器"""
    
    def __init__(self, model_path="models/yolov11n.pt"):
        """初始化视频检测器
        
        Args:
            model_path: YOLO模型路径
        """
        super().__init__(model_path)
        
        # 创建视频行为检测器
        self.behavior_detector = VideoBehaviorDetector()
        
        # 用于跟踪的数据
        self.tracked_objects = defaultdict(list)
        
        logger.info("视频检测器初始化成功")
    
    def analyze_video(self, video_path, output_path=None, max_frames=None, frame_interval=1, save_frames=False, 
                      draw_boxes=True, conf_threshold=0.25, progress_callback=None, callback=None, 
                      callback_data=None):
        """
        分析视频文件中的盗窃行为
        
        Args:
            video_path: 视频文件路径
            output_path: 输出视频路径
            max_frames: 最大处理帧数
            frame_interval: 帧间隔
            save_frames: 是否保存每一帧
            draw_boxes: 是否绘制检测框
            conf_threshold: 置信度阈值
            progress_callback: 进度回调函数
            callback: 每帧处理后的回调函数
            callback_data: 回调函数的附加数据
            
        Returns:
            dict: 视频分析结果
        """
        if not os.path.exists(video_path):
            logging.error(f"视频文件不存在: {video_path}")
            return {"error": f"视频文件不存在: {video_path}"}
            
        try:
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
                
            # 获取视频信息
            fps = video.get(cv2.CAP_PROP_FPS)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 处理max_frames参数
            if max_frames is None or max_frames <= 0 or max_frames > frame_count:
                max_frames = frame_count
                
            # 设置输出视频
            video_writer = None
            if output_path:
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
            # 跟踪状态
            frames_processed = 0
            suspicious_behavior_frames = []
            total_suspicious_behaviors = 0
            start_time = time.time()
            skip_count = 0
            
            # 用于记录零售环境相关帧数
            retail_frames = 0
            non_retail_frames = 0
            
            # 开始逐帧处理
            while frames_processed < max_frames:
                # 读取帧
                ret, frame = video.read()
                if not ret:
                    break  # 视频结束
                    
                # 实现帧间隔
                if skip_count < frame_interval - 1:
                    skip_count += 1
                    continue
                skip_count = 0
                
                # 转换为RGB (YOLO模型需要RGB格式)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 使用模型进行预测
                results = self.model.predict(frame_rgb, conf=conf_threshold)[0]
                
                # 标记为视频帧处理
                results.video_frame = True
                
                # 检查是否为零售环境
                is_retail = self._is_retail_environment(results)
                if is_retail:
                    retail_frames += 1
                else:
                    non_retail_frames += 1
                    
                # 分析行为
                behaviors = self.analyze_behavior(results)
                
                # 如果检测到可疑行为，保存帧索引和行为
                if behaviors:
                    current_frame_index = frames_processed * frame_interval
                    suspicious_behavior_frames.append((current_frame_index, behaviors))
                    total_suspicious_behaviors += len(behaviors)
                    
                # 绘制检测框
                if draw_boxes:
                    frame = results.plot()
                    
                # 保存处理后的帧
                if video_writer:
                    video_writer.write(frame)
                    
                # 保存单独的帧
                if save_frames:
                    frames_dir = os.path.join(os.path.dirname(output_path), "frames")
                    os.makedirs(frames_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(frames_dir, f"frame_{frames_processed:04d}.jpg"), frame)
                    
                # 更新进度
                if progress_callback:
                    progress = (frames_processed + 1) / max_frames * 100
                    progress_callback(progress)
                    
                # 调用回调函数
                if callback:
                    callback(frame, results, behaviors, callback_data)
                    
                frames_processed += 1
                
            # 计算处理时间和速度
            end_time = time.time()
            processing_time = end_time - start_time
            processing_speed = frames_processed / processing_time if processing_time > 0 else 0
            
            # 计算零售环境判断结果
            is_retail_environment = False
            retail_confidence = 0.0
            
            # 至少处理了10帧才做判断
            if (retail_frames + non_retail_frames) >= 10:
                retail_confidence = retail_frames / (retail_frames + non_retail_frames)
                is_retail_environment = retail_confidence >= 0.6  # 如果超过60%的帧被判断为零售环境
            
            # 释放资源
            video.release()
            if video_writer:
                video_writer.release()
                
            logging.info(f"视频分析完成。处理了 {frames_processed} 帧，检测到 {total_suspicious_behaviors} 个可疑行为。")
            logging.info(f"零售环境判断: {'是' if is_retail_environment else '否'} (置信度: {retail_confidence:.2f})")
            
            # 返回分析结果
            return {
                "frames_processed": frames_processed,
                "processing_time": processing_time,
                "processing_speed": processing_speed,
                "suspicious_behavior_frames": suspicious_behavior_frames,
                "total_suspicious_behaviors": total_suspicious_behaviors,
                "is_retail_environment": is_retail_environment,  # 添加零售环境判断结果
                "retail_confidence": retail_confidence  # 添加零售环境置信度
            }
            
        except Exception as e:
            logging.error(f"视频分析失败: {str(e)}")
            logging.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _update_tracking(self, frame_idx, detections):
        """更新物体跟踪数据
        
        Args:
            frame_idx: 当前帧索引
            detections: 检测结果
        """
        if detections is None:
            return
            
        # 提取当前帧中检测到的物体
        current_objects = []
        
        if hasattr(detections, 'pandas') and callable(getattr(detections, 'pandas')):
            df = detections.pandas().xyxy[0]
            
            for _, row in df.iterrows():
                obj = {
                    'class': row['name'],
                    'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                    'confidence': row['confidence']
                }
                current_objects.append(obj)
        
        # 根据类别跟踪物体
        for obj in current_objects:
            class_name = obj['class']
            self.tracked_objects[class_name].append((frame_idx, obj))
    
    def draw_detection(self, frame, detections, frame_idx=0):
        """在视频帧上绘制检测结果
        
        Args:
            frame: 视频帧
            detections: 检测结果
            frame_idx: 当前帧索引
            
        Returns:
            drawn_frame: 标记后的帧
        """
        # 首先绘制基本的检测框
        drawn_frame = self.draw_basic_detection(frame, detections)
        
        # 添加帧计数
        height, width = drawn_frame.shape[:2]
        cv2.putText(
            drawn_frame,
            f"Frame: {frame_idx}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        
        # 添加跟踪信息（如果有）
        if frame_idx > 0:
            # 计算每个类别的物体计数
            class_counts = {}
            for class_name, objects in self.tracked_objects.items():
                # 只统计最近50帧内的物体
                recent_objects = [obj for idx, obj in objects if frame_idx - idx <= 50]
                if recent_objects:
                    class_counts[class_name] = len(recent_objects)
            
            # 显示物体计数
            y_pos = 50
            for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                cv2.putText(
                    drawn_frame,
                    f"{class_name}: {count}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
                y_pos += 20
        
        return drawn_frame 