import os
import sys
import cv2
import torch
import logging
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('common_detector')

class CommonDetector:
    """基础检测器类，提供共享的模型加载和基础检测功能"""
    
    def __init__(self, model_path="models/yolov11n.pt"):
        """初始化检测器
        
        Args:
            model_path: YOLO模型路径
        """
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # 加载模型
        self.load_yolo_model()
    
    def load_yolo_model(self):
        """加载YOLOv11n模型"""
        try:
            # 首先尝试使用ultralytics YOLO加载模型
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                logger.info(f"Model loaded from {self.model_path} using ultralytics YOLO")
                return
            except ImportError:
                logger.warning("ultralytics package not found, trying alternative loading methods...")
            except Exception as e:
                logger.warning(f"Failed to load model with ultralytics YOLO: {e}, trying alternative loading methods...")
            
            # 尝试直接使用PyTorch加载模型
            if os.path.exists(self.model_path):
                self.model = torch.load(self.model_path, map_location=self.device)
                if hasattr(self.model, 'model'):
                    self.model = self.model.model  # 获取实际模型
                self.model.eval()
                logger.info(f"Model loaded from {self.model_path} using torch.load")
                return
            else:
                logger.warning(f"Model file {self.model_path} not found, trying to download...")
            
            # 最后，尝试从torch hub加载模型
            self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', self.model_path)
            self.model.eval()
            logger.info(f"Model loaded from torch hub")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv11n model: {e}")
            raise ValueError(f"Failed to load YOLOv11n model: {e}")
    
    def detect_objects(self, img):
        """检测图像中的对象
        
        Args:
            img: 输入图像
            
        Returns:
            detections: YOLO检测结果
        """
        if self.model is None:
            logger.error("Model not loaded. Please load the model first.")
            return None
        
        # 转换图像为适当格式（如果必要）
        if isinstance(img, str):
            img = cv2.imread(img)
        
        try:
            # 对于ultralytics YOLO模型
            if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
                results = self.model.predict(source=img, conf=0.25, verbose=False)[0]
                return results
            
            # 对于torch hub模型
            elif hasattr(self.model, '__call__'):
                results = self.model(img)
                if results.pandas().xyxy[0].empty:
                    logger.info("No objects detected in the image.")
                return results
            
            else:
                logger.error("Model doesn't have the expected interface for detection.")
                return None
                
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return None
    
    def draw_basic_detection(self, img, detections):
        """在图像上绘制基本检测框
        
        Args:
            img: 原始图像
            detections: 检测结果
            
        Returns:
            marked_img: 标记后的图像
        """
        if detections is None:
            return img.copy()
            
        marked_img = img.copy()
        
        # 处理ultralytics YOLO结果
        if hasattr(detections, 'boxes') and hasattr(detections.boxes, 'data'):
            boxes = detections.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 获取类名
                if hasattr(detections, 'names'):
                    cls_name = detections.names[int(cls)]
                else:
                    cls_name = f"class {int(cls)}"
                
                # 绘制矩形
                cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签 - 修改到边界框内部而非上方
                label = f"{cls_name}: {conf:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # 绘制标签背景 - 在边界框内部顶部
                cv2.rectangle(marked_img, (x1, y1), (x1 + label_width, y1 + label_height + 5), (0, 255, 0), -1)
                # 绘制标签文本 - 在边界框内部顶部
                cv2.putText(marked_img, label, (x1, y1 + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # 处理torch hub模型结果
        elif hasattr(detections, 'pandas') and callable(getattr(detections, 'pandas')):
            df = detections.pandas().xyxy[0]
            for _, det in df.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                conf = det['confidence']
                cls_name = det['name']
                
                # 绘制矩形
                cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 绘制标签 - 修改到边界框内部而非上方
                label = f"{cls_name}: {conf:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                # 绘制标签背景 - 在边界框内部顶部
                cv2.rectangle(marked_img, (x1, y1), (x1 + label_width, y1 + label_height + 5), (0, 255, 0), -1)
                # 绘制标签文本 - 在边界框内部顶部
                cv2.putText(marked_img, label, (x1, y1 + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
        return marked_img 

    def _is_retail_environment(self, detection_result):
        """判断图像是否为店铺或超市环境
        
        Args:
            detection_result: 检测结果
            
        Returns:
            bool: 是否为店铺环境
        """
        # 如果检测结果带有video_frame标记，则调用视频帧专用的零售环境识别
        if hasattr(detection_result, 'video_frame') and detection_result.video_frame:
            # 尝试导入并调用TheftDetector中的视频环境判断方法
            try:
                from .detection import TheftDetector
                return TheftDetector._is_retail_environment_video(TheftDetector(), detection_result)
            except (ImportError, AttributeError):
                # 如果不能导入或方法不存在，则使用简化的判断逻辑
                return self._simple_retail_environment_check(detection_result)
        else:
            # 对于非视频帧，使用简化的环境检查
            return self._simple_retail_environment_check(detection_result)
    
    def _simple_retail_environment_check(self, detection_result):
        """简化版的零售环境检查，适用于CommonDetector"""
        if detection_result is None:
            return False
            
        # 零售环境指标物体
        retail_indicators = [
            "shelf", "shopping cart", "cashier", "cash register", "checkout",
            "bottle", "refrigerator", "food", "fruit", "vegetable",
            "counter", "display", "store", "supermarket", "retail"
        ]
        
        # 检查是否存在零售指标物体
        if hasattr(detection_result, 'boxes'):
            for box in detection_result.boxes:
                cls_id = int(box.cls[0])
                class_name = detection_result.names.get(cls_id, "").lower()
                
                if any(indicator in class_name for indicator in retail_indicators):
                    return True
                
        return False 