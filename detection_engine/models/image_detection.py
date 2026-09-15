import os
import cv2
import torch
import logging
import numpy as np
import joblib
from pathlib import Path
from .common_detector import CommonDetector
from .behavior.image_behavior import ImageBehaviorDetector
import pickle  # 添加模块导入

# 配置日志
logger = logging.getLogger('image_detection')

class ImageDetector(CommonDetector):
    """专门处理图像的检测器"""
    
    def __init__(self, model_path="models/yolov11n.pt"):
        """初始化图像检测器
        
        Args:
            model_path: YOLO模型路径
        """
        super().__init__(model_path)
        
        # 添加XGBoost模型加载
        self.standard_model = None
        self.enhanced_model = None
        self.model_type = "enhanced"  # 默认使用增强模型
        self.load_ml_models()
        
        # 创建行为检测器
        self.behavior_detector = ImageBehaviorDetector()
        
        # 常见的盗窃相关物品类别
        self.theft_related_classes = [
            'person', 'backpack', 'handbag', 'suitcase', 'bottle', 
            'wine glass', 'cup', 'knife', 'spoon', 'bowl', 'banana', 
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'laptop', 
            'remote', 'keyboard', 'cell phone', 'book', 'clock'
        ]
        
        logger.info("图像检测器初始化成功")
    
    def load_ml_models(self):
        """加载机器学习模型，如XGBoost用于盗窃行为分类"""
        try:
            # 尝试加载标准模型
            standard_model_path = Path("models/standard/theft_xgb_model.pkl")
            if standard_model_path.exists():
                with open(standard_model_path, 'rb') as f:
                    self.standard_model = pickle.load(f)
                logger.info("成功加载标准XGBoost模型")
            else:
                logger.warning(f"标准XGBoost模型文件不存在: {standard_model_path}")
            
            # 尝试加载增强模型
            enhanced_model_path = Path("models/enhanced/enhanced_theft_xgb_model.pkl")
            if enhanced_model_path.exists():
                with open(enhanced_model_path, 'rb') as f:
                    self.enhanced_model = pickle.load(f)
                logger.info("成功加载增强XGBoost模型")
            else:
                logger.warning(f"增强XGBoost模型文件不存在: {enhanced_model_path}")
            
            # 确定使用哪个模型
            if self.enhanced_model:
                self.model_type = "enhanced"
            elif self.standard_model:
                self.model_type = "standard"
            else:
                self.model_type = None
                logger.warning("未能加载任何XGBoost模型，将使用规则引擎进行行为检测")
        
        except Exception as e:
            logger.error(f"加载XGBoost模型出错: {e}")
            logger.exception("模型加载异常详情")
            self.model_type = None
    
    def detect_theft(self, image):
        """在图像中检测盗窃行为
        
        Args:
            image: 输入图像
            
        Returns:
            detections: 物体检测结果
            theft_probability: 盗窃行为的概率
        """
        try:
            # 检测物体
            detections = self.detect_objects(image)
            
            # 提取特征
            features = self.extract_features(detections)
            
            # 分类盗窃行为
            theft_prob = self.classify_theft(features)
            
            return detections, theft_prob
            
        except Exception as e:
            logger.error(f"盗窃行为检测错误: {e}")
            return None, 0.0
    
    def extract_features(self, detections):
        """从检测结果中提取特征
        
        Args:
            detections: 物体检测结果
            
        Returns:
            features: 特征向量
        """
        if detections is None:
            return np.zeros(len(self.theft_related_classes) + 3)
        
        # 初始化特征
        person_count = 0
        object_count = 0
        person_object_interaction = 0
        class_counts = {cls: 0 for cls in self.theft_related_classes}
        
        try:
            # 从检测结果中提取数据
            if hasattr(detections, 'pandas') and callable(getattr(detections, 'pandas')):
                df = detections.pandas().xyxy[0]
                
                for _, row in df.iterrows():
                    class_name = row['name']
                    
                    # 计数检测到的相关物体
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    
                    # 计算人数
                    if class_name == 'person':
                        person_count += 1
                    
                    # 计算物体数（不包括人）
                    if class_name != 'person' and class_name in self.theft_related_classes:
                        object_count += 1
                
                # 检查人与物体的交互
                if person_count > 0 and object_count > 0:
                    person_boxes = df[df['name'] == 'person'][['xmin', 'ymin', 'xmax', 'ymax']].values
                    
                    for _, row in df.iterrows():
                        if row['name'] != 'person' and row['name'] in self.theft_related_classes:
                            obj_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                            
                            # 检查人与物体是否有交互
                            for person_box in person_boxes:
                                if self._check_interaction(person_box, obj_box):
                                    person_object_interaction += 1
                                    break
            
            # 将类别计数转换为特征向量
            class_features = [class_counts[cls] for cls in self.theft_related_classes]
            
            # 构建最终特征向量
            features = np.array(class_features + [person_count, object_count, person_object_interaction])
            
            return features
            
        except Exception as e:
            logger.error(f"特征提取错误: {e}")
            return np.zeros(len(self.theft_related_classes) + 3)
    
    def _check_interaction(self, person_box, obj_box, threshold=0.2):
        """检查人与物体是否有交互
        
        Args:
            person_box: 人物边界框 [xmin, ymin, xmax, ymax]
            obj_box: 物体边界框
            threshold: 交互阈值
            
        Returns:
            is_interacting: 是否有交互
        """
        # 计算IoU或距离
        px1, py1, px2, py2 = person_box
        ox1, oy1, ox2, oy2 = obj_box
        
        # 检查是否有重叠
        if ox2 < px1 or px2 < ox1 or oy2 < py1 or py2 < oy1:
            # 计算中心点
            person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            obj_center = ((ox1 + ox2) / 2, (oy1 + oy2) / 2)
            
            # 计算距离
            distance = np.sqrt((person_center[0] - obj_center[0])**2 + (person_center[1] - obj_center[1])**2)
            
            # 计算人物对角线长度
            person_diag = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
            
            # 如果距离小于人物对角线长度的阈值，认为有交互
            return distance < person_diag * threshold
        else:
            # 有重叠，肯定有交互
            return True
    
    def classify_theft(self, features):
        """分类盗窃行为
        
        Args:
            features: 特征向量
            
        Returns:
            theft_probability: 盗窃行为的概率
        """
        if self.model_type is None:
            return 0.5  # 默认值
        
        try:
            # 预测盗窃概率
            if hasattr(self.standard_model, 'predict_proba'):
                probs = self.standard_model.predict_proba([features])
                theft_prob = probs[0][1]  # 假设第二个类别是盗窃
            else:
                theft_pred = self.standard_model.predict([features])
                theft_prob = float(theft_pred[0])
            
            return theft_prob
            
        except Exception as e:
            logger.error(f"盗窃行为分类错误: {e}")
            return 0.5  # 默认值
    
    def detect_behaviors(self, image, detections=None):
        """检测图像中的可疑行为
        
        Args:
            image: 输入图像
            detections: 物体检测结果（如果已经检测过）
            
        Returns:
            behaviors: 检测到的行为列表
            marked_image: 标记了行为的图像
        """
        # 如果没有提供检测结果，先进行检测
        if detections is None:
            detections = self.detect_objects(image)
            
        # 使用行为检测器检测可疑行为
        return self.behavior_detector.detect_behaviors_in_image(image, detections)
    
    def draw_detection(self, image, detections, theft_prob=0.0):
        """在图像上绘制检测结果和盗窃概率
        
        Args:
            image: 原始图像
            detections: 检测结果
            theft_prob: 盗窃行为概率
            
        Returns:
            marked_image: 标记后的图像
        """
        # 首先绘制基本检测框
        marked_image = self.draw_basic_detection(image, detections)
        
        # 添加盗窃概率信息
        height, width = marked_image.shape[:2]
        
        # 绘制盗窃概率条
        bar_height = 30
        bar_width = 150
        margin = 20
        
        # 创建背景
        cv2.rectangle(
            marked_image, 
            (margin, margin), 
            (margin + bar_width + 10, margin + bar_height + 10), 
            (255, 255, 255), 
            -1
        )
        
        # 绘制概率条
        filled_width = int(bar_width * theft_prob)
        
        # 设置颜色（从绿色到红色）
        if theft_prob < 0.3:
            color = (0, 255, 0)  # 绿色
        elif theft_prob < 0.7:
            color = (0, 165, 255)  # 橙色
        else:
            color = (0, 0, 255)  # 红色
        
        cv2.rectangle(
            marked_image, 
            (margin + 5, margin + 5), 
            (margin + 5 + filled_width, margin + 5 + bar_height), 
            color, 
            -1
        )
        
        cv2.rectangle(
            marked_image, 
            (margin + 5, margin + 5), 
            (margin + 5 + bar_width, margin + 5 + bar_height), 
            (0, 0, 0), 
            1
        )
        
        # 添加文本
        cv2.putText(
            marked_image,
            f"Theft Probability: {theft_prob:.2f}",
            (margin + bar_width + 15, margin + bar_height // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

        
        # 如果检测结果包含可疑行为，也绘制出来
        if hasattr(detections, 'suspicious_behaviors') and detections.suspicious_behaviors:
            behaviors = detections.suspicious_behaviors
            
            # 在图像上标记可疑行为
            y_pos = margin + bar_height + 20
            
            for i, behavior in enumerate(behaviors):
                if i > 5:  # 最多显示6个行为
                    cv2.putText(
                        marked_image,
                        f"... Additionally {len(behaviors) - 6} behaviors",
                        (margin, y_pos + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )
                    break
                    
                desc = behavior.get('description', 'Unknown')
                conf = behavior.get('confidence', 0.0)
                
                cv2.putText(
                    marked_image,
                    f"{desc}: {conf:.2f}",
                    (margin, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1
                )
                
                y_pos += 20
                
            # 如果有边界框信息，也绘制出来
            for behavior in behaviors:
                if 'bbox' in behavior:
                    x1, y1, x2, y2 = behavior['bbox']
                    cv2.rectangle(marked_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    desc = behavior.get('description', 'Unknown')
                    if len(desc) > 15:
                        desc = desc[:12] + "..."
                        
                    cv2.putText(
                        marked_image,
                        desc,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1
                    )
        
        return marked_image 