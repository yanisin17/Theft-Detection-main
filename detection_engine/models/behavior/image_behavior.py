import cv2
import numpy as np
import logging
import os
from pathlib import Path
import time
import pickle
import json
import math

# 添加OpenPose相关导入
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe未安装，将使用替代方法进行姿态估计")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('image_behavior_detector')

class ImageBehaviorDetector:
    """检测图像中的可疑盗窃行为特征"""
    
    def __init__(self, config_path=None, confidence_threshold=0.5):
        """初始化行为检测器
        
        Args:
            config_path: 配置文件路径
            confidence_threshold: 行为检测阈值
        """
        # 加载配置
        self.config = self._load_config(config_path)
        self.confidence_threshold = self.config.get('confidence_threshold', confidence_threshold)
        self.detected_behaviors = []
        
        # 定义可检测的行为类型及描述
        self.behavior_types = {
            'elbow_inward': '手肘内收角度＞60°',
            'hand_pressing_pocket': '单手持续按压口袋/裤腰',
            'umbrella_covering': '遮挡商品区域',
            'baby_adjustment': '怀抱婴儿/手提袋反复调整',
            'back_to_camera': '背对摄像头整理包内物品',
            'repeated_path': '同一区域反复折返',
            'sudden_squat': '突然蹲下并伴随手臂后缩',
            'avoiding_camera': '刻意绕行监控死角',
            'tearing_tag': '撕标签动作',
            'package_shaking': '拆包装抖动',
            'empty_box_restore': '空盒复原',
            'abnormal_arm_position': '手臂姿势异常',
            'suspicious_crouching': '可疑蹲姿',
            'unusual_reaching': '不自然的伸手姿势',
            'body_shielding': '身体屏蔽姿势',
            'abnormal_head_movement': '头部异常转动',
            'single_arm_hiding': '单手/手臂遮挡或隐藏', 
            'suspicious_arm_posture': '可疑的手臂姿势/手臂靠近腰部',
            'Item Concealed in Pocket': '将物品放入口袋'
        }
        
        # 初始化姿态估计模型
        self.pose_detector = None
        self.initialize_pose_detector()
        
        # 初始化机器学习模型
        self.standard_model = None
        self.enhanced_model = None
        self.model_type = "enhanced"  # 默认使用增强模型
        self.load_xgboost_models()
        
        # 加载行为分类器
        self.behavior_classifier = None
        self.load_behavior_classifier()
        
        # 加载特征归一化参数
        self.feature_normalization = self.config.get('feature_normalization', {'enabled': False})
        
        # 零售环境检测
        self.retail_environment_detected = False
        
        logger.info("图像行为检测器初始化完成")
    
    def _load_config(self, config_path=None):
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            config: 配置字典
        """
        if config_path is None:
            # 尝试在默认位置加载配置
            default_paths = [
                'src/config/detector_config.json',
                'config/detector_config.json',
                '../config/detector_config.json',
                'detector_config.json'
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"已加载配置文件: {config_path}")
                return config
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
        
        logger.warning("未找到有效配置文件，使用默认配置")
        return {
            "confidence_threshold": 0.5,
            "language": "zh",
            "use_adaptive_weights": True
        }
    
    def load_xgboost_models(self):
        """加载XGBoost模型"""
        # 标准模型
        standard_model_path = "models/standard/theft_xgb_model.pkl"
        try:
            if os.path.exists(standard_model_path):
                self.standard_model = pickle.load(open(standard_model_path, 'rb'))
                logger.info(f"成功加载标准XGBoost模型: {standard_model_path}")
            else:
                logger.warning(f"标准XGBoost模型文件不存在: {standard_model_path}")
        except Exception as e:
            logger.error(f"加载标准XGBoost模型出错: {e}")
        
        # 增强模型
        enhanced_model_path = "models/enhanced/enhanced_theft_xgb_model.pkl"
        try:
            if os.path.exists(enhanced_model_path):
                self.enhanced_model = pickle.load(open(enhanced_model_path, 'rb'))
                logger.info(f"成功加载增强XGBoost模型: {enhanced_model_path}")
            else:
                logger.warning(f"增强XGBoost模型文件不存在: {enhanced_model_path}")
        except Exception as e:
            logger.error(f"加载增强XGBoost模型出错: {e}")
        
        # 如果增强模型不可用，但标准模型可用，则使用标准模型
        if self.enhanced_model is None and self.standard_model is not None:
            self.model_type = "standard"
            logger.info("增强模型不可用，将使用标准模型")
        
        # 如果两种模型都不可用，发出警告
        if self.enhanced_model is None and self.standard_model is None:
            logger.warning("未能加载任何XGBoost模型，将使用规则引擎进行行为检测")
    
    def load_behavior_classifier(self):
        """加载行为分类器模型"""
        # 从配置中获取路径，或使用默认路径
        model_path = self.config.get('behavior_classifier_path', 'models/behavior_classifier.pkl')
        
        # 尝试不同的可能路径
        possible_paths = [
            model_path,
            os.path.join(os.getcwd(), model_path),
            os.path.join(os.getcwd(), "models", "behavior_classifier.pkl"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), model_path)
        ]
        
        # 尝试加载模型
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    loaded_data = pickle.load(open(path, 'rb'))
                    
                    # 检查是否是新格式的模型包（字典，包含模型和标签编码器）
                    if isinstance(loaded_data, dict) and 'model' in loaded_data and 'label_encoder' in loaded_data:
                        self.behavior_classifier = loaded_data['model']
                        self.label_encoder = loaded_data['label_encoder']
                        logger.info(f"成功加载新格式行为分类器模型: {path}")
                    else:
                        # 旧格式模型，直接使用
                        self.behavior_classifier = loaded_data
                        logger.info(f"成功加载旧格式行为分类器模型: {path}")
                    return
            except Exception as e:
                logger.error(f"尝试加载行为分类器模型失败 ({path}): {e}")
        
        # 如果没有找到现有模型，尝试使用XGBoost模型作为替代
        if self.enhanced_model is not None or self.standard_model is not None:
            # 使用已有的XGBoost模型创建简单的行为分类器
            try:
                from sklearn.ensemble import RandomForestClassifier
                # 创建一个简单的分类器，使用基本行为类型
                basic_types = ['abnormal_arm_position', 'suspicious_arm_posture', 'hands_behind_back']
                
                # 创建一个简单的分类器作为后备
                dummy_classifier = RandomForestClassifier(n_estimators=10)
                # 简单初始化它，这样它至少有classes_属性
                X_dummy = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                y_dummy = np.array([basic_types[0]])  # 使用第一个行为类型
                dummy_classifier.fit(X_dummy, y_dummy)
                dummy_classifier.classes_ = np.array(basic_types)
                
                self.behavior_classifier = dummy_classifier
                logger.info("使用简单分类器作为行为分类器的后备")
                return
            except Exception as e:
                logger.error(f"创建后备分类器失败: {e}")
        
        # 如果所有尝试都失败
        logger.warning(f"行为分类器模型不存在，部分高级行为分析功能将不可用")
    
    def predict_with_xgboost(self, features):
        """使用XGBoost模型预测
        
        Args:
            features: 特征向量
            
        Returns:
            prediction: 预测结果 (0 或 1)
            probability: 概率
        """
        if features is None:
            return 0, 0.0
        
        # 应用特征归一化（如果启用）
        if self.feature_normalization.get('enabled', False):
            features = self._normalize_features(features)
        
        # 根据模型类型选择模型
        model = self.enhanced_model if self.model_type == "enhanced" else self.standard_model
        
        # 如果没有可用模型，返回默认值
        if model is None:
            return 0, 0.0
        
        # 将特征转换为numpy数组并重塑为一个样本
        features_np = np.array(features).reshape(1, -1)
        
        try:
            # 预测概率
            probabilities = model.predict_proba(features_np)[0]
            # 获取正类概率
            positive_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            # 预测类别
            prediction = 1 if positive_prob >= 0.5 else 0
            
            return prediction, float(positive_prob)
        except Exception as e:
            logger.error(f"XGBoost预测出错: {e}")
            return 0, 0.0
    
    def _normalize_features(self, features):
        """归一化特征
        
        Args:
            features: 原始特征向量
            
        Returns:
            normalized_features: 归一化后的特征
        """
        if not self.feature_normalization.get('enabled', False):
            return features
        
        method = self.feature_normalization.get('method', 'minmax')
        params = self.feature_normalization.get('params', {})
        
        if method == 'minmax':
            normalized_features = []
            feature_names = [
                'left_wrist_dist', 'right_wrist_dist', 'torso_height', 
                'left_arm_angle', 'right_arm_angle', 'wrist_hip_ratio_left', 
                'wrist_hip_ratio_right', 'arms_crossed', 'torso_rotation',
                'shoulder_width', 'head_position', 'knee_hip_ratio'
            ]
            
            # 确保特征数量一致，不足的补0
            features_dict = {name: (features[i] if i < len(features) else 0) 
                           for i, name in enumerate(feature_names[:len(features)])}
            
            # 应用归一化
            for i, name in enumerate(feature_names):
                if i < len(features):
                    param = params.get(name, {'min': 0, 'max': 1})
                    min_val, max_val = param.get('min', 0), param.get('max', 1)
                    range_val = max_val - min_val
                    
                    if range_val == 0:
                        normalized_features.append(0)
                    else:
                        # 将值限制在[min_val, max_val]范围内
                        clamped_val = max(min_val, min(max_val, features[i]))
                        # 归一化到[0, 1]
                        normalized_val = (clamped_val - min_val) / range_val
                        normalized_features.append(normalized_val)
                else:
                    normalized_features.append(0)
            
            return normalized_features
        
        # 目前只支持minmax方法
        return features
    
    def predict_behaviors(self, features):
        """使用行为分类器预测具体行为类型
        
        Args:
            features: 特征向量
            
        Returns:
            behaviors: 预测的行为列表，包含类型和置信度
        """
        if self.behavior_classifier is None or features is None:
            return []
            
        try:
            # 应用特征归一化（如果启用）
            if self.feature_normalization.get('enabled', False):
                features = self._normalize_features(features)
            
            # 确保特征格式正确
            X = np.array([features])
            
            # 预测行为
            probabilities = self.behavior_classifier.predict_proba(X)[0]
            
            # 获取类别名称
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                # 使用新格式模型中的标签编码器
                class_names = self.label_encoder.classes_
            else:
                # 使用旧格式模型中的类别名称
                class_names = self.behavior_classifier.classes_
            
            # 获取配置中的行为阈值
            behavior_thresholds = self.config.get('behavior_thresholds', {})
            default_threshold = self.config.get('default_behavior_threshold', 0.5)
            
            # 提取有意义的行为预测（概率超过阈值）
            behaviors = []
            for class_idx, class_name in enumerate(class_names):
                prob = probabilities[class_idx]
                # 获取该行为的特定阈值，如果没有则使用默认值
                threshold = behavior_thresholds.get(class_name, default_threshold)
                
                if prob >= threshold:
                    behaviors.append({
                        'type': class_name,
                        'confidence': float(prob),
                        'model_based': True  # 标记为模型预测的行为
                    })
            
            # 按置信度排序
            behaviors.sort(key=lambda x: x['confidence'], reverse=True)
            return behaviors
        except Exception as e:
            logger.error(f"行为预测失败: {e}")
            return []
    
    def _extract_standard_features(self, landmarks):
        """提取标准特征集
        
        Args:
            landmarks: 姿态关键点字典
            
        Returns:
            features: 标准特征向量
        """
        if not landmarks or len(landmarks) < 12:
            return None
            
        try:
            features = []
            
            # 1. 计算左右手腕距离
            left_wrist_dist = 0
            right_wrist_dist = 0
            
            # 左手腕(15)和髋部中点的距离
            if 15 in landmarks and 23 in landmarks and 24 in landmarks:
                left_wrist = (landmarks[15]['x'], landmarks[15]['y'])
                left_hip = (landmarks[23]['x'], landmarks[23]['y'])
                right_hip = (landmarks[24]['x'], landmarks[24]['y'])
                hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                
                left_wrist_dist = math.sqrt((left_wrist[0] - hip_center[0]) ** 2 + 
                                         (left_wrist[1] - hip_center[1]) ** 2)
            
            # 右手腕(16)和髋部中点的距离
            if 16 in landmarks and 23 in landmarks and 24 in landmarks:
                right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
                left_hip = (landmarks[23]['x'], landmarks[23]['y'])
                right_hip = (landmarks[24]['x'], landmarks[24]['y'])
                hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                
                right_wrist_dist = math.sqrt((right_wrist[0] - hip_center[0]) ** 2 + 
                                          (right_wrist[1] - hip_center[1]) ** 2)
            
            features.append(left_wrist_dist)
            features.append(right_wrist_dist)
            
            # 2. 计算躯干高度
            torso_height = 0
            if 11 in landmarks and 24 in landmarks:
                # 使用右肩(11)和右髋(24)计算躯干高度
                torso_height = abs(landmarks[11]['y'] - landmarks[24]['y'])
            
            features.append(torso_height)
            
            # 3. 计算左右手臂角度
            left_arm_angle = 0
            right_arm_angle = 0
            
            # 左手臂角度：使用左肩(11)、左肘(13)和左手腕(15)
            if 11 in landmarks and 13 in landmarks and 15 in landmarks:
                shoulder = (landmarks[11]['x'], landmarks[11]['y'])
                elbow = (landmarks[13]['x'], landmarks[13]['y'])
                wrist = (landmarks[15]['x'], landmarks[15]['y'])
                
                # 计算肩到肘的向量
                se_vector = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
                # 计算肘到手腕的向量
                ew_vector = (wrist[0] - elbow[0], wrist[1] - elbow[1])
                
                # 计算两个向量之间的角度
                dot_product = se_vector[0] * ew_vector[0] + se_vector[1] * ew_vector[1]
                se_magnitude = math.sqrt(se_vector[0] ** 2 + se_vector[1] ** 2)
                ew_magnitude = math.sqrt(ew_vector[0] ** 2 + ew_vector[1] ** 2)
                
                if se_magnitude * ew_magnitude != 0:
                    cos_angle = dot_product / (se_magnitude * ew_magnitude)
                    # 防止因浮点数精度问题导致的溢出
                    cos_angle = max(-1, min(1, cos_angle))
                    # 弧度转角度
                    left_arm_angle = math.degrees(math.acos(cos_angle))
            
            # 右手臂角度：使用右肩(12)、右肘(14)和右手腕(16)
            if 12 in landmarks and 14 in landmarks and 16 in landmarks:
                shoulder = (landmarks[12]['x'], landmarks[12]['y'])
                elbow = (landmarks[14]['x'], landmarks[14]['y'])
                wrist = (landmarks[16]['x'], landmarks[16]['y'])
                
                # 计算肩到肘的向量
                se_vector = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
                # 计算肘到手腕的向量
                ew_vector = (wrist[0] - elbow[0], wrist[1] - elbow[1])
                
                # 计算两个向量之间的角度
                dot_product = se_vector[0] * ew_vector[0] + se_vector[1] * ew_vector[1]
                se_magnitude = math.sqrt(se_vector[0] ** 2 + se_vector[1] ** 2)
                ew_magnitude = math.sqrt(ew_vector[0] ** 2 + ew_vector[1] ** 2)
                
                if se_magnitude * ew_magnitude != 0:
                    cos_angle = dot_product / (se_magnitude * ew_magnitude)
                    # 防止因浮点数精度问题导致的溢出
                    cos_angle = max(-1, min(1, cos_angle))
                    # 弧度转角度
                    right_arm_angle = math.degrees(math.acos(cos_angle))
            
            features.append(left_arm_angle)
            features.append(right_arm_angle)
            
            # 4. 计算手腕与髋部比例
            wrist_hip_ratio_left = 0
            wrist_hip_ratio_right = 0
            
            # 计算左手腕(15)和左髋(23)的比例
            if 15 in landmarks and 23 in landmarks and 24 in landmarks:
                wrist_y = landmarks[15]['y']
                left_hip_y = landmarks[23]['y']
                right_hip_y = landmarks[24]['y']
                hip_center_y = (left_hip_y + right_hip_y) / 2
                
                # 比例：手腕Y坐标与髋部中点的差值，除以躯干高度（如果有）
                if torso_height > 0:
                    wrist_hip_ratio_left = (wrist_y - hip_center_y) / torso_height
            
            # 计算右手腕(16)和右髋(24)的比例
            if 16 in landmarks and 23 in landmarks and 24 in landmarks:
                wrist_y = landmarks[16]['y']
                left_hip_y = landmarks[23]['y']
                right_hip_y = landmarks[24]['y']
                hip_center_y = (left_hip_y + right_hip_y) / 2
                
                # 比例：手腕Y坐标与髋部中点的差值，除以躯干高度（如果有）
                if torso_height > 0:
                    wrist_hip_ratio_right = (wrist_y - hip_center_y) / torso_height
            
            features.append(wrist_hip_ratio_left)
            features.append(wrist_hip_ratio_right)
            
            # 5. 计算手臂交叉状态
            arms_crossed = 0
            if 13 in landmarks and 14 in landmarks and 15 in landmarks and 16 in landmarks:
                left_elbow_x = landmarks[13]['x']
                right_elbow_x = landmarks[14]['x']
                left_wrist_x = landmarks[15]['x']
                right_wrist_x = landmarks[16]['x']
                
                # 判断手臂是否交叉
                # 交叉条件：左手腕在右手肘的右侧，或右手腕在左手肘的左侧
                if (left_wrist_x > right_elbow_x) or (right_wrist_x < left_elbow_x):
                    arms_crossed = 1
            
            features.append(arms_crossed)
            
            # 6. 躯干旋转（肩膀和髋部的水平对齐程度）
            torso_rotation = 0
            if 11 in landmarks and 12 in landmarks and 23 in landmarks and 24 in landmarks:
                # 计算肩膀线的角度
                shoulder_angle = math.atan2(landmarks[12]['y'] - landmarks[11]['y'], 
                                           landmarks[12]['x'] - landmarks[11]['x'])
                # 计算髋部线的角度
                hip_angle = math.atan2(landmarks[24]['y'] - landmarks[23]['y'], 
                                      landmarks[24]['x'] - landmarks[23]['x'])
                # 两条线之间的角度差异（弧度）
                angle_diff = abs(shoulder_angle - hip_angle)
                # 转换为角度
                torso_rotation = math.degrees(angle_diff)
            
            features.append(torso_rotation)
            
            # 7. 肩膀宽度
            shoulder_width = 0
            if 11 in landmarks and 12 in landmarks:
                # 计算肩膀宽度
                shoulder_width = math.sqrt((landmarks[12]['x'] - landmarks[11]['x'])**2 + 
                                          (landmarks[12]['y'] - landmarks[11]['y'])**2)
            
            features.append(shoulder_width)
            
            return features
            
        except Exception as e:
            logger.error(f"提取标准特征出错: {e}")
            return None
    
    def initialize_pose_detector(self):
        """初始化姿态估计模型"""
        if MEDIAPIPE_AVAILABLE:
            try:
                # 检查项目内的MediaPipe资源目录
                possible_mediapipe_dirs = [
                    os.path.join(os.getcwd(), "models", "mediapipe"),
                    os.path.join(os.getcwd(), "mediapipe_resources"),
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "mediapipe")
                ]
                
                # 首先检查环境变量
                mediapipe_resource_dir = os.environ.get('MEDIAPIPE_RESOURCE_DIR')
                if mediapipe_resource_dir and os.path.exists(mediapipe_resource_dir):
                    logger.info(f"使用环境变量中的MediaPipe资源目录: {mediapipe_resource_dir}")
                else:
                    # 查找项目内可能的目录
                    for dir_path in possible_mediapipe_dirs:
                        if os.path.exists(dir_path):
                            os.environ['MEDIAPIPE_RESOURCE_DIR'] = dir_path
                            mediapipe_resource_dir = dir_path
                            logger.info(f"使用项目内的MediaPipe资源目录: {mediapipe_resource_dir}")
                            break
                    
                    # 如果仍未找到且将要显示警告，首先检查MediaPipe是否能正常初始化
                    if not mediapipe_resource_dir:
                        # 尝试不设置资源目录初始化，如果成功就不显示警告
                        test_detector = mp.solutions.pose.Pose(
                            static_image_mode=True,
                            model_complexity=1,
                            min_detection_confidence=0.5
                        )
                        # 如果能成功初始化，就不显示警告了
                        logger.info("MediaPipe无需额外资源目录，可以正常初始化")
                    else:
                        logger.warning("环境变量MEDIAPIPE_RESOURCE_DIR未设置或目录不存在")
                
                # 初始化姿态估计模块
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                
                # 初始化姿态估计器
                self.pose_detector = self.mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=2,        # 提高模型复杂度以提高背向姿势检测能力
                    enable_segmentation=True,
                    min_detection_confidence=0.4  # 降低检测置信度以更好地捕捉背向姿势
                )
                logger.info("MediaPipe姿态估计模型初始化成功")
            except Exception as e:
                logger.error(f"MediaPipe姿态估计模型初始化失败: {e}")
                self.pose_detector = None
                self.mp_pose = None
                self.mp_drawing = None
                self.mp_drawing_styles = None
        else:
            logger.warning("MediaPipe不可用，姿态估计功能将降级")
            self.pose_detector = None
            self.mp_pose = None
            self.mp_drawing = None
            self.mp_drawing_styles = None
    
    def detect_behaviors_in_image(self, image, debug=False):
        """检测图像中的可疑行为
        
        Args:
            image: 输入图像
            debug: 是否开启调试模式
            
        Returns:
            result_image: 标记了可疑行为的图像
            behaviors: 检测到的可疑行为列表
        """
        # 清空上一次的记录
        self.behaviors = []
        self.object_count = 0
        self.person_count = 0
        
        # 获取图像尺寸
        h, w = image.shape[:2]
        
        # 复制图像用于标记
        marked_image = image.copy()
        
        # 提取人物和物体的检测结果
        persons = self._extract_person_detections(image)
        self.person_count = len(persons)
        
        objects = self._extract_object_detections(image)
        self.object_count = len(objects)
        
        # 使用姿态估计检测行为 (如果可用)
        pose_results = None
        if self.pose_detector is not None:
            pose_results = self._extract_pose_landmarks(image)
            
            # 如果有姿态估计结果，绘制关键点
            if pose_results:
                marked_image = self._draw_pose_landmarks(marked_image, pose_results)
                pose_results = [pose_results['landmarks']]
        
        # 检测各种可疑行为
        self._detect_occlusion_behaviors(image, persons, objects)
        self._detect_abnormal_posture(image, persons, objects)
        self._detect_abnormal_item_handling(image, persons, objects)
        
        # 直接调用袖口藏物检测
        self._detect_sleeve_concealment(image, persons, objects)
        
        # 调用口袋藏物检测，即使没有姿态估计结果也尝试检测
        self._detect_pocket_concealment(image, pose_results, persons, objects)
        
        # 基于姿态估计的行为检测(如果有姿态检测结果)
        if pose_results and len(pose_results) > 0:
            self._detect_pose_based_behaviors(image, pose_results, persons, objects)
        
        # 绘制检测结果
        if self.behaviors:
            marked_image = self._draw_detected_behaviors(marked_image)
        
        # 在调试模式下，绘制所有检测到的边界框和姿态关键点
        if debug:
            marked_image = self._draw_debug_info(marked_image, persons, objects, pose_results)
        
        return marked_image, self.behaviors
    
    def _detect_occlusion_behaviors(self, image, persons, objects):
        """检测遮挡类可疑行为"""
        height, width = image.shape[:2]
        
        for person_bbox, person_conf in persons:
            x1, y1, x2, y2 = person_bbox
            person_width = x2 - x1
            person_height = y2 - y1
            
            # 检测背对摄像头
            # 简单模拟：如果宽高比大于一定值且处于画面上方，可能是背对摄像头
            aspect_ratio = person_width / person_height if person_height > 0 else 0
            if aspect_ratio > 0.55 and y1 < height * 0.4:
                confidence = 0.7
                behavior = 'back_to_camera'
                if confidence > self.confidence_threshold:
                    self._add_behavior(behavior, person_bbox, confidence, image)
            
            # 检测用外套/雨伞遮挡
            # 简单模拟：检查人物区域的颜色分布是否不均匀
            if person_height > 50:  # 确保人物足够大才检测
                person_roi = image[y1:y2, x1:x2]
                
                if person_roi.size > 0:  # 确保ROI有效
                    # 计算颜色标准差
                    color_std = np.std(person_roi.reshape(-1, 3), axis=0).mean()
                    
                    # 颜色变化大可能有外套遮挡
                    if color_std > 50:
                        confidence = 0.6 + min(0.3, color_std / 300)
                        behavior = 'umbrella_covering'
                        if confidence > self.confidence_threshold:
                            self._add_behavior(behavior, person_bbox, confidence, image)
        
        # 检测手提袋交互
        for person_bbox, person_conf in persons:
            for obj_bbox, obj_conf, obj_class in objects:
                if obj_class in ['backpack', 'handbag', 'suitcase']:
                    if self._is_interacting(person_bbox, obj_bbox):
                        confidence = 0.65
                        behavior = 'baby_adjustment'
                        if confidence > self.confidence_threshold:
                            self._add_behavior(behavior, obj_bbox, confidence, image)
    
    def _detect_posture_anomalies(self, image, persons):
        """检测姿态异常的可疑行为"""
        for person_bbox, person_conf in persons:
            x1, y1, x2, y2 = person_bbox
            person_width = x2 - x1
            person_height = y2 - y1
            
            # 检测手肘内收
            # 简单模拟：根据人物轮廓的形状分析
            if person_height > 150:  # 确保人物足够大才检测
                # 提取人物区域
                person_roi = image[y1:y2, x1:x2]
                
                if person_roi.size > 0:
                    # 转为灰度图
                    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    
                    # 边缘检测
                    edges = cv2.Canny(blur, 35, 125)
                    
                    # 寻找轮廓
                    try:
                        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                        
                        # 过滤小轮廓
                        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
                        
                        # 分析轮廓形状
                        for cnt in large_contours:
                            # 计算轮廓的中心位置和方向
                            if len(cnt) > 5:  # 确保有足够的点拟合椭圆
                                try:
                                    (cx, cy), (ma, mi), angle = cv2.fitEllipse(cnt)
                                    
                                    # 根据椭圆的扁平率和角度判断是否是手肘内收
                                    if ma > 0 and mi / ma < 0.3 and (angle < 45 or angle > 135):
                                        confidence = 0.65
                                        behavior = 'elbow_inward'
                                        if confidence > self.confidence_threshold:
                                            self._add_behavior(behavior, person_bbox, confidence, image)
                                        break  # 找到一个符合条件的就跳出
                                except:
                                    pass  # 拟合椭圆可能失败
                    except:
                        pass  # 轮廓查找可能失败
            
            # 检测手按口袋
            # 简单模拟：根据手位置与身体下部的相对位置
            if person_height > 200:  # 确保人物足够大才检测
                lower_body_start = y1 + person_height * 0.5
                
                # 提取下半身区域
                if lower_body_start < y2:
                    lower_body_roi = image[int(lower_body_start):y2, x1:x2]
                    
                    if lower_body_roi.size > 0:
                        # 转为HSV空间以更好地检测肤色
                        hsv = cv2.cvtColor(lower_body_roi, cv2.COLOR_BGR2HSV)
                        
                        # 肤色范围
                        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                        
                        # 肤色掩码
                        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                        
                        # 计算下半身区域的肤色像素比例
                        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size if skin_mask.size > 0 else 0
                        
                        # 如果肤色比例适中(不太高也不太低)，可能是手在口袋上
                        if 0.05 < skin_ratio < 0.2:
                            confidence = 0.6 + skin_ratio
                            behavior = 'hand_pressing_pocket'
                            if confidence > self.confidence_threshold:
                                self._add_behavior(behavior, person_bbox, confidence, image)
    
    def _detect_product_handling_anomalies(self, image, persons, objects):
        """检测商品处理异常的可疑行为"""
        for person_bbox, person_conf in persons:
            for obj_bbox, obj_conf, obj_class in objects:
                # 检查人与物体是否有交互
                if self._is_interacting(person_bbox, obj_bbox):
                    ox1, oy1, ox2, oy2 = obj_bbox
                    
                    # 提取物体区域
                    obj_roi = image[oy1:oy2, ox1:ox2]
                    
                    if obj_roi.size > 0:
                        # 转为灰度图
                        gray = cv2.cvtColor(obj_roi, cv2.COLOR_BGR2GRAY)
                        blur = cv2.GaussianBlur(gray, (5, 5), 0)
                        
                        # 边缘检测
                        edges = cv2.Canny(blur, 50, 150)
                        
                        # 计算边缘密度
                        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
                        
                        # 边缘密度高可能是在撕标签或拆包装
                        if edge_density > 0.1:
                            if obj_class in ['bottle', 'cup', 'cell phone']:
                                confidence = 0.6 + edge_density
                                behavior = 'tearing_tag'
                                if confidence > self.confidence_threshold:
                                    self._add_behavior(behavior, obj_bbox, confidence, image)
                            else:
                                confidence = 0.6 + edge_density
                                behavior = 'package_shaking'
                                if confidence > self.confidence_threshold:
                                    self._add_behavior(behavior, obj_bbox, confidence, image)
    
    def _is_interacting(self, person_bbox, obj_bbox):
        """检查人与物体是否有交互"""
        x1_1, y1_1, x2_1, y2_1 = person_bbox
        x1_2, y1_2, x2_2, y2_2 = obj_bbox
        
        # 计算IoU
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            # 没有重叠区域，检查距离
            person_center = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
            obj_center = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
            
            # 计算距离
            distance = np.sqrt((person_center[0] - obj_center[0])**2 + (person_center[1] - obj_center[1])**2)
            
            # 人物尺寸
            person_size = np.sqrt((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)
            
            # 如果距离小于人物尺寸的一半，认为有交互
            return distance < person_size * 0.5
        else:
            # 有重叠区域，肯定有交互
            return True
    
    def _add_behavior(self, behavior_key, bbox, confidence, image):
        """添加检测到的行为"""
        if behavior_key in self.behavior_types:
            behavior_desc = self.behavior_types[behavior_key]
            self.detected_behaviors.append({
                'type': behavior_key,
                'description': behavior_desc,
                'confidence': confidence,
                'bbox': bbox
            })
            
            # 在图像上标记检测结果
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    def _draw_behavior_detections(self, image):
        """在图像上绘制检测结果"""
        for behavior in self.detected_behaviors:
            bbox = behavior['bbox']
            x1, y1, x2, y2 = bbox
            
            # 绘制矩形框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # 绘制行为类型
            behavior_text = behavior['type']
            text_size = cv2.getTextSize(behavior_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # 背景框
            cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), (0, 0, 255), -1)
            
            # 文字
            cv2.putText(image, behavior_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 置信度
            conf_text = f"{behavior['confidence']:.2f}"
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            
            cv2.rectangle(image, (x2 - conf_size[0], y1 - text_size[1] - 5), (x2, y1), (0, 0, 255), -1)
            cv2.putText(image, conf_text, (x2 - conf_size[0], y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return image
    
    def _create_behavior_panel(self, height):
        """创建显示可疑行为的信息面板"""
        # 面板宽度
        panel_width = 250
        
        # 创建白色背景
        panel = np.ones((height, panel_width, 3), dtype=np.uint8) * 255
        
        # 添加标题
        cv2.putText(panel, "Suspicious Behavior Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.line(panel, (10, 40), (panel_width-10, 40), (0, 0, 0), 1)
        
        if not self.detected_behaviors:
            cv2.putText(panel, "No suspicious behaviors detected", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 0), 1)
            return panel
        
        # 显示检测到的行为
        cv2.putText(panel, f"Detected {len(self.detected_behaviors)} suspicious behaviors", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # 列出每个行为
        y_pos = 100
        for i, behavior in enumerate(self.detected_behaviors):
            if y_pos > height - 50:
                cv2.putText(panel, "...(More)", (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                break
                
            behavior_type = behavior['description']
            confidence = behavior['confidence']
            
            # 截断过长的行为描述
            if len(behavior_type) > 20:
                behavior_type = behavior_type[:17] + "..."
                
            # 添加行为类型
            cv2.putText(panel, f"{i+1}. {behavior_type}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # 添加置信度
            conf_text = f"{confidence:.2f}"
            cv2.putText(panel, conf_text, (panel_width-60, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            y_pos += 30
        
        return panel
    
    def analyze_image(self, image_path, detector):
        """分析图像中的可疑行为
        
        Args:
            image_path: 图像文件路径
            detector: 物体检测器
            
        Returns:
            output_path: 处理后的图像路径
            behaviors: 检测到的可疑行为列表
        """
        # 创建输出目录
        output_dir = os.path.join("static", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Cannot read image: {image_path}")
            return None, []
        
        # 执行物体检测
        detections = detector.detect_objects(image)
        
        # 检测可疑行为
        behaviors, marked_image = self.detect_behaviors_in_image(image, detections)
        
        # 保存结果
        output_path = os.path.join(output_dir, f"behavior_analysis_{Path(image_path).stem}_{int(time.time())}.jpg")
        cv2.imwrite(output_path, marked_image)
        
        return output_path, behaviors
    
    def _detect_group_behaviors(self, image, persons):
        """检测团伙协作类盗窃行为"""
        if len(persons) < 2:
            return  # 至少需要两个人才能检测团伙行为
            
        height, width = image.shape[:2]
        
        # 计算人员分布特征
        person_positions = []
        for person_bbox, _ in persons:
            x1, y1, x2, y2 = person_bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            person_positions.append((center_x, center_y))
            
        # 计算人员之间的距离矩阵
        distances = []
        for i in range(len(person_positions)):
            for j in range(i+1, len(person_positions)):
                p1 = person_positions[i]
                p2 = person_positions[j]
                dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                distances.append((i, j, dist))
                
        # 根据距离排序
        distances.sort(key=lambda x: x[2])
        
        # 检测协同掩护行为 - 检查是否有人员形成特定的空间分布
        if len(distances) > 0:
            # 如果最近的两人间距离适中(不是太近也不太远)且位置接近某些区域
            closest_pair = distances[0]
            i, j, dist = closest_pair
            
            # 计算图像对角线长度作为参考
            diag_length = np.sqrt(width**2 + height**2)
            
            # 如果两人距离在合理范围内(比如对角线的10%-25%)
            if 0.1 * diag_length < dist < 0.25 * diag_length:
                # 检查他们的位置分布是否符合掩护模式
                p1 = person_positions[i]
                p2 = person_positions[j]
                
                # 计算两人的中点位置
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                
                # 如果中点位置在商品架附近(假设商品架位于图像中部)
                image_center_x = width / 2
                image_center_y = height / 2
                
                center_dist = np.sqrt((mid_x - image_center_x)**2 + (mid_y - image_center_y)**2)
                
                if center_dist < 0.3 * diag_length:
                    confidence = 0.7
                    behavior = 'coordinated_movement'
                    common_bbox = [
                        int(min(persons[i][0][0], persons[j][0][0])),
                        int(min(persons[i][0][1], persons[j][0][1])),
                        int(max(persons[i][0][2], persons[j][0][2])),
                        int(max(persons[i][0][3], persons[j][0][3]))
                    ]
                    if confidence > self.confidence_threshold:
                        self._add_behavior(behavior, common_bbox, confidence, image)
        
        # 检测望风行为 - 检查是否有人位于出口附近且朝向外部
        for idx, (person_bbox, person_conf) in enumerate(persons):
            x1, y1, x2, y2 = person_bbox
            
            # 检查是否靠近图像边缘(可能是出入口)
            is_near_edge = (x1 < 0.1 * width or x2 > 0.9 * width or 
                           y1 < 0.1 * height or y2 > 0.9 * height)
            
            if is_near_edge:
                # 检查是否有人相对静止(作为望风者)
                is_stationary = True  # 在静态图像中无法准确判断，默认为True
                
                # 检查是否面向出口(简单模拟)
                facing_outward = True  # 在静态图像中难以判断朝向，默认为True
                
                if is_stationary and facing_outward:
                    confidence = 0.65
                    behavior = 'lookout_positioning'
                    if confidence > self.confidence_threshold:
                        self._add_behavior(behavior, person_bbox, confidence, image)
        
        # 检测分散注意力行为
        # 分析人员异常动作和位置
        for idx, (person_bbox, _) in enumerate(persons):
            x1, y1, x2, y2 = person_bbox
            person_roi = image[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
                
            # 分析姿势和动作的异常性(简化版)
            gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 计算边缘密度，作为动作幅度的粗略指标
            edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
            
            # 判断是否有大幅度动作(边缘密度高)
            if edge_density > 0.15:
                # 检查该人是否远离其他人(可能是试图吸引注意力)
                is_isolated = True
                for j, (other_bbox, _) in enumerate(persons):
                    if j == idx:
                        continue
                    
                    other_center = [(other_bbox[0] + other_bbox[2]) / 2, 
                                   (other_bbox[1] + other_bbox[3]) / 2]
                    person_center = [(x1 + x2) / 2, (y1 + y2) / 2]
                    
                    dist = np.sqrt((other_center[0] - person_center[0])**2 + 
                                   (other_center[1] - person_center[1])**2)
                    
                    if dist < 0.3 * diag_length:
                        is_isolated = False
                        break
                
                if is_isolated:
                    confidence = 0.6 + 0.2 * edge_density
                    behavior = 'distraction_behavior'
                    if confidence > self.confidence_threshold:
                        self._add_behavior(behavior, person_bbox, confidence, image)
    
    def _detect_environmental_behaviors(self, image, persons, objects):
        """检测与商店环境相关的可疑行为"""
        height, width = image.shape[:2]
        
        # 检测盲区逗留行为
        # 在静态图像中，我们假设图像角落可能是监控盲区
        corner_regions = [
            [0, 0, width * 0.2, height * 0.2],                    # 左上角
            [width * 0.8, 0, width, height * 0.2],                # 右上角
            [0, height * 0.8, width * 0.2, height],               # 左下角
            [width * 0.8, height * 0.8, width, height]            # 右下角
        ]
        
        for person_bbox, _ in persons:
            x1, y1, x2, y2 = person_bbox
            person_center = [(x1 + x2) / 2, (y1 + y2) / 2]
            
            # 检查人是否在角落区域
            for corner in corner_regions:
                cx1, cy1, cx2, cy2 = corner
                
                if (cx1 <= person_center[0] <= cx2 and cy1 <= person_center[1] <= cy2):
                    confidence = 0.65
                    behavior = 'blind_spot_lingering'
                    if confidence > self.confidence_threshold:
                        self._add_behavior(behavior, person_bbox, confidence, image)
                    break
        
        # 检测临近关店行为(在静态图像中使用暗光检测来模拟)
        # 分析图像整体亮度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # 如果亮度较低(可能是关店前)，检查人们的行动
        if avg_brightness < 100:  # 假设阈值为100
            for person_bbox, _ in persons:
                # 检查人是否靠近商品区域
                x1, y1, x2, y2 = person_bbox
                person_center = [(x1 + x2) / 2, (y1 + y2) / 2]
                
                # 假设图像中心区域是商品区
                is_near_products = (width * 0.3 < person_center[0] < width * 0.7 and 
                                   height * 0.3 < person_center[1] < height * 0.7)
                
                if is_near_products:
                    confidence = 0.6 + (100 - avg_brightness) / 100 * 0.3  # 越暗置信度越高
                    behavior = 'closing_time_activity'
                    if confidence > self.confidence_threshold:
                        self._add_behavior(behavior, person_bbox, confidence, image)
    
    def _detect_high_value_product_behaviors(self, image, persons, objects):
        """检测与高价值商品相关的盗窃行为"""
        # 检测多件相同商品行为
        # 统计每种物品类型的数量
        object_counts = {}
        for obj_bbox, _, obj_class in objects:
            if obj_class not in object_counts:
                object_counts[obj_class] = []
            object_counts[obj_class].append(obj_bbox)
        
        # 检查是否有多件相同物品
        for obj_class, bboxes in object_counts.items():
            if len(bboxes) >= 3:  # 如果同一类物品有3件或以上
                # 计算物品的平均区域大小
                areas = [(box[2]-box[0])*(box[3]-box[1]) for box in bboxes]
                avg_area = sum(areas) / len(areas)
                
                # 排除太小的物品(可能是误检)
                if avg_area > 1000:  # 假设面积阈值
                    # 计算物品的边界框
                    min_x = min([box[0] for box in bboxes])
                    min_y = min([box[1] for box in bboxes])
                    max_x = max([box[2] for box in bboxes])
                    max_y = max([box[3] for box in bboxes])
                    
                    group_bbox = [min_x, min_y, max_x, max_y]
                    confidence = 0.6 + 0.05 * len(bboxes)  # 物品数量越多置信度越高
                    behavior = 'multiple_identical_items'
                    if confidence > self.confidence_threshold:
                        self._add_behavior(behavior, group_bbox, confidence, image)
        
        # 检测价格标签互换行为
        for person_bbox, _ in persons:
            for obj_bbox, _, obj_class in objects:
                # 检查人与物体是否有交互
                if self._is_interacting(person_bbox, obj_bbox):
                    # 提取手部区域(简化版，假设手在人物框的下半部分)
                    x1, y1, x2, y2 = person_bbox
                    hand_region = [x1, y1 + (y2-y1)//2, x2, y2]
                    
                    # 检查手部区域是否与物体有重叠
                    ox1, oy1, ox2, oy2 = obj_bbox
                    
                    overlap_x1 = max(hand_region[0], ox1)
                    overlap_y1 = max(hand_region[1], oy1)
                    overlap_x2 = min(hand_region[2], ox2)
                    overlap_y2 = min(hand_region[3], oy2)
                    
                    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                        # 提取重叠区域
                        overlap_roi = image[overlap_y1:overlap_y2, overlap_x1:overlap_x2]
                        
                        if overlap_roi.size > 0:
                            # 转为灰度图
                            gray = cv2.cvtColor(overlap_roi, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 50, 150)
                            
                            # 计算边缘密度作为手部精细操作的指标
                            edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
                            
                            # 如果边缘密度适中(不太高也不太低)，可能是在调整标签
                            if 0.05 < edge_density < 0.15:
                                confidence = 0.6 + edge_density
                                behavior = 'price_tag_switching'
                                if confidence > self.confidence_threshold:
                                    self._add_behavior(behavior, obj_bbox, confidence, image)
    
    def _detect_digital_behaviors(self, image, persons, objects):
        """检测与数字化防盗设备相关的盗窃行为"""
        # 检测信号屏蔽行为
        for person_bbox, _ in persons:
            x1, y1, x2, y2 = person_bbox
            
            # 提取人物区域
            person_roi = image[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
                
            # 转换到HSV颜色空间来检测金属箔的特征(银色)
            hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
            
            # 定义金属箔的颜色范围(银色)
            lower_silver = np.array([0, 0, 150])
            upper_silver = np.array([180, 30, 255])
            
            # 创建掩码
            silver_mask = cv2.inRange(hsv, lower_silver, upper_silver)
            
            # 计算银色区域比例
            silver_ratio = np.sum(silver_mask > 0) / silver_mask.size if silver_mask.size > 0 else 0
            
            # 如果银色比例超过阈值，可能是使用了铝箔袋
            if silver_ratio > 0.1:
                confidence = 0.6 + silver_ratio
                behavior = 'signal_blocking_behavior'
                if confidence > self.confidence_threshold:
                    self._add_behavior(behavior, person_bbox, confidence, image)
        
        # 检测防盗标签篡改行为
        for person_bbox, _ in persons:
            for obj_bbox, _, obj_class in objects:
                if self._is_interacting(person_bbox, obj_bbox):
                    x1, y1, x2, y2 = person_bbox
                    ox1, oy1, ox2, oy2 = obj_bbox
                    
                    # 提取人与物体交互的区域
                    interaction_x1 = max(x1, ox1)
                    interaction_y1 = max(y1, oy1)
                    interaction_x2 = min(x2, ox2)
                    interaction_y2 = min(y2, oy2)
                    
                    if interaction_x2 > interaction_x1 and interaction_y2 > interaction_y1:
                        interaction_roi = image[interaction_y1:interaction_y2, interaction_x1:interaction_x2]
                        
                        if interaction_roi.size > 0:
                            # 分析纹理特征(使用边缘检测)
                            gray = cv2.cvtColor(interaction_roi, cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 100, 200)
                            
                            # 计算边缘密度
                            edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
                            
                            # 如果边缘密度高，可能正在篡改防盗标签
                            if edge_density > 0.2:
                                confidence = 0.65 + 0.25 * edge_density
                                behavior = 'security_tag_tampering'
                                if confidence > self.confidence_threshold:
                                    self._add_behavior(behavior, obj_bbox, confidence, image)
    
    def _extract_pose_landmarks(self, image):
        """从图像中提取姿态关键点
        
        Args:
            image: 输入图像
            
        Returns:
            landmarks: 姿态关键点坐标字典或None
        """
        if self.pose_detector is None:
            return None
            
        try:
            # 转换为RGB (MediaPipe需要RGB图像)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 执行姿态估计
            results = self.pose_detector.process(rgb_image)
            
            if not results.pose_landmarks:
                # 为了更好地处理背对姿势的情况，可以尝试调整图像后再次检测
                # 尝试水平翻转图像可能会提高检测率
                flipped_image = cv2.flip(rgb_image, 1)
                flipped_results = self.pose_detector.process(flipped_image)
                
                if flipped_results.pose_landmarks:
                    logger.info("在翻转图像中检测到姿态关键点")
                    # 需要调整翻转后的坐标
                    width = image.shape[1]
                    landmarks = {}
                    for idx, landmark in enumerate(flipped_results.pose_landmarks.landmark):
                        # 还原翻转后的x坐标 (1.0 - x)
                        landmarks[idx] = {
                            'x': int((1.0 - landmark.x) * width),
                            'y': int(landmark.y * image.shape[0]),
                            'visibility': landmark.visibility
                        }
                    
                    # 保存原始MediaPipe结果用于绘图
                    return {
                        'landmarks': landmarks,
                        'mp_results': flipped_results,
                        'is_flipped': True
                    }
                else:
                    logger.info("未检测到姿态关键点，尝试降低检测置信度")
                    # 如果仍然没有检测到，处理背对姿势的特殊情况
                    # 有时对于背向摄像头的姿势，可能需要特殊处理
                    return None
                
            # 提取关键点坐标并规范化为图像尺寸
            landmarks = {}
            height, width = image.shape[:2]
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[idx] = {
                    'x': int(landmark.x * width),
                    'y': int(landmark.y * height),
                    'visibility': landmark.visibility
                }
            
            # 返回整个结果，包括分割掩码和姿态关键点
            return {
                'landmarks': landmarks,
                'mp_results': results,
                'is_flipped': False
            }
            
        except Exception as e:
            logger.error(f"姿态关键点提取失败: {e}")
            return None
            
    def _draw_pose_landmarks(self, image, pose_results):
        """在图像上绘制姿态关键点
        
        Args:
            image: 原始图像
            pose_results: 姿态估计结果
            
        Returns:
            marked_image: 绘制了姿态关键点的图像
        """
        if pose_results is None or 'mp_results' not in pose_results:
            return image
            
        marked_image = image.copy()
        
        try:
            # 转换为RGB (MediaPipe绘图需要RGB图像)
            rgb_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)
            
            # 检查必要的绘图组件是否存在
            if hasattr(self, 'mp_drawing') and hasattr(self, 'mp_pose') and hasattr(self, 'mp_drawing_styles'):
                # 绘制姿态关键点
                self.mp_drawing.draw_landmarks(
                    rgb_image,
                    pose_results['mp_results'].pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
                
                # 转换回BGR
                marked_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            else:
                logger.warning("缺少MediaPipe绘图组件，无法绘制姿态关键点")
            
            return marked_image
            
        except Exception as e:
            logger.error(f"绘制姿态关键点失败: {e}")
            return image
            
    def _detect_pose_based_behaviors(self, image, pose_results, persons, objects):
        """基于姿态估计检测可疑行为
        
        Args:
            image: 输入图像
            pose_results: 姿态估计结果
            persons: 人物检测结果
            objects: 物体检测结果
        """
        if pose_results is None or 'landmarks' not in pose_results:
            return
            
        landmarks = pose_results['landmarks']
        
        # 异常手臂位置检测
        self._detect_abnormal_arm_positions(image, landmarks, persons)
        
        # 可疑蹲姿检测
        self._detect_suspicious_crouching(image, landmarks, persons)
        
        # 不自然伸手姿势检测
        self._detect_unusual_reaching(image, landmarks, persons, objects)
        
        # 身体屏蔽姿态检测
        self._detect_body_shielding(image, landmarks, persons, objects)
        
        # 头部异常转动检测
        self._detect_abnormal_head_movement(image, landmarks, persons)
        
        # 双手背后姿势检测
        self._detect_hands_behind_back(image, landmarks, persons)
        
        # 新增：可疑手臂姿势检测
        self._detect_suspicious_arm_posture(image, landmarks, persons)
    
    def _get_person_for_landmarks(self, landmarks, persons):
        """找到与姿态关键点匹配的人物边界框
        
        Args:
            landmarks: 姿态关键点
            persons: 人物检测结果
            
        Returns:
            person_bbox: 匹配的人物边界框或None
        """
        if not landmarks or not persons:
            return None
        
        try:
            # 计算姿态关键点的边界框
            valid_points = []
            for i in landmarks:
                if 'visibility' in landmarks[i] and 'x' in landmarks[i] and 'y' in landmarks[i] and landmarks[i]['visibility'] > 0.5:
                    valid_points.append((landmarks[i]['x'], landmarks[i]['y']))
            
            if not valid_points:
                # 如果没有可见点，尝试降低阈值
                for i in landmarks:
                    if 'x' in landmarks[i] and 'y' in landmarks[i]:
                        valid_points.append((landmarks[i]['x'], landmarks[i]['y']))
                
                if not valid_points:
                    return None
            
            # 根据有效点计算边界
            xs = [p[0] for p in valid_points]
            ys = [p[1] for p in valid_points]
            
            min_x = min(xs)
            min_y = min(ys)
            max_x = max(xs)
            max_y = max(ys)
            
            pose_center_x = (min_x + max_x) / 2
            pose_center_y = (min_y + max_y) / 2
            
            # 找到与姿态关键点中心最接近的人物边界框
            best_person = None
            min_distance = float('inf')
            
            # 增加对persons参数的类型检查和处理
            for person_data in persons:
                # 处理不同格式的person_data
                person_bbox = None
                
                if isinstance(person_data, list) or isinstance(person_data, tuple):
                    # 兼容[(bbox, conf), ...]格式
                    if len(person_data) >= 1:
                        person_bbox = person_data[0]
                elif isinstance(person_data, dict):
                    # 兼容[{'bbox': bbox, 'confidence': conf}, ...]格式
                    person_bbox = person_data.get('bbox')
                
                if person_bbox is None:
                    continue
                    
                # 确保bbox有4个元素
                if len(person_bbox) != 4:
                    continue
                
                px1, py1, px2, py2 = person_bbox
                person_center_x = (px1 + px2) / 2
                person_center_y = (py1 + py2) / 2
                
                distance = ((pose_center_x - person_center_x) ** 2 + 
                            (pose_center_y - person_center_y) ** 2) ** 0.5
                            
                if distance < min_distance:
                    min_distance = distance
                    best_person = person_bbox
            
            return best_person
        except Exception as e:
            import logging
            logger = logging.getLogger('image_behavior_detector')
            logger.error(f"匹配姿态关键点与人物边界框出错: {e}")
            return None
    
    def _detect_abnormal_arm_positions(self, image, landmarks, persons):
        """检测异常手臂位置
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            persons: 人物检测结果
        """
        # MediaPipe关键点索引
        # 左肩: 11, 左肘: 13, 左腕: 15
        # 右肩: 12, 右肘: 14, 右腕: 16
        
        required_points = [11, 13, 15, 12, 14, 16]
        
        # 确保所有需要的关键点可见
        if not all(i in landmarks and landmarks[i]['visibility'] > 0.3 for i in required_points):
            return
            
        # 检查左臂弯曲
        left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
        left_elbow = (landmarks[13]['x'], landmarks[13]['y'])
        left_wrist = (landmarks[15]['x'], landmarks[15]['y'])
        
        # 检查右臂弯曲
        right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
        right_elbow = (landmarks[14]['x'], landmarks[14]['y'])
        right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
        
        # 计算左臂弯曲角度
        left_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        # 计算右臂弯曲角度
        right_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        # 获取对应的人物边界框
        person_bbox = self._get_person_for_landmarks(landmarks, persons)
        if person_bbox is None:
            return
            
        # 检测异常手臂位置的条件
        if left_angle < 90 or right_angle < 90:
            # 检查是否有手腕靠近身体中心线
            mid_hip_x = (landmarks.get(23, {}).get('x', 0) + landmarks.get(24, {}).get('x', 0)) / 2
            
            left_wrist_to_mid = abs(left_wrist[0] - mid_hip_x)
            right_wrist_to_mid = abs(right_wrist[0] - mid_hip_x)
            
            # 如果手腕靠近中线且手臂弯曲
            if left_wrist_to_mid < 100 or right_wrist_to_mid < 100:
                confidence = 0.7
                behavior = 'abnormal_arm_position'
                if confidence > self.confidence_threshold:
                    self._add_behavior(behavior, person_bbox, confidence, image)
    
    def _detect_suspicious_crouching(self, image, landmarks, persons):
        """检测可疑蹲姿
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            persons: 人物检测结果
        """
        # 检查是否有髋部和膝盖关键点
        hip_points = [23, 24]  # 左右髋部
        knee_points = [25, 26]  # 左右膝盖
        
        if not all(i in landmarks and landmarks[i]['visibility'] > 0.3 for i in hip_points + knee_points):
            return
            
        # 获取髋部和膝盖的平均高度
        hip_y = (landmarks[23]['y'] + landmarks[24]['y']) / 2
        knee_y = (landmarks[25]['y'] + landmarks[26]['y']) / 2
        
        # 检查髋部和膝盖的高度差
        hip_knee_distance = knee_y - hip_y
        
        # 获取对应的人物边界框
        person_bbox = self._get_person_for_landmarks(landmarks, persons)
        if person_bbox is None:
            return
        
        # 人物边界框高度
        person_height = person_bbox[3] - person_bbox[1]
        
        # 判断是否是蹲姿 (膝盖位置相对髋部较低)
        if hip_knee_distance > 0.15 * person_height:
            # 额外检查手是否靠近地面(判断是否在底层货架区域操作)
            wrist_points = [15, 16]  # 左右手腕
            
            if any(i in landmarks and landmarks[i]['visibility'] > 0.3 and 
                   landmarks[i]['y'] > hip_y + 0.2 * person_height 
                   for i in wrist_points):
                confidence = 0.75
                behavior = 'suspicious_crouching'
                if confidence > self.confidence_threshold:
                    self._add_behavior(behavior, person_bbox, confidence, image)
    
    def _detect_unusual_reaching(self, image, landmarks, persons, objects):
        """检测不自然的伸手姿势
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            persons: 人物检测结果
            objects: 物体检测结果
        """
        # 检查是否有肩部和手腕关键点
        shoulder_points = [11, 12]  # 左右肩部
        wrist_points = [15, 16]  # 左右手腕
        
        if not all(i in landmarks and landmarks[i]['visibility'] > 0.3 for i in shoulder_points) or \
           not any(i in landmarks and landmarks[i]['visibility'] > 0.3 for i in wrist_points):
            return
            
        # 获取对应的人物边界框
        person_bbox = self._get_person_for_landmarks(landmarks, persons)
        if person_bbox is None:
            return
            
        # 检查手腕是否伸向高处
        left_shoulder_y = landmarks.get(11, {}).get('y', float('inf'))
        right_shoulder_y = landmarks.get(12, {}).get('y', float('inf'))
        left_wrist_y = landmarks.get(15, {}).get('y', float('inf'))
        right_wrist_y = landmarks.get(16, {}).get('y', float('inf'))
        
        # 计算手臂伸展程度 (手腕高于肩膀)
        unusual_left = 15 in landmarks and left_wrist_y < left_shoulder_y - 50
        unusual_right = 16 in landmarks and right_wrist_y < right_shoulder_y - 50
        
        if unusual_left or unusual_right:
            # 检查这个姿势是否对应商品区域
            for obj_bbox, _, obj_class in objects:
                ox1, oy1, ox2, oy2 = obj_bbox
                
                # 检查是否有手腕靠近物体
                for wrist_idx in wrist_points:
                    if wrist_idx in landmarks and landmarks[wrist_idx]['visibility'] > 0.3:
                        wrist_x, wrist_y = landmarks[wrist_idx]['x'], landmarks[wrist_idx]['y']
                        
                        # 判断手腕是否在物体附近
                        is_near_object = (ox1 - 50 <= wrist_x <= ox2 + 50) and (oy1 - 50 <= wrist_y <= oy2 + 50)
                        
                        if is_near_object:
                            confidence = 0.65
                            behavior = 'unusual_reaching'
                            if confidence > self.confidence_threshold:
                                self._add_behavior(behavior, person_bbox, confidence, image)
                                return
    
    def _detect_body_shielding(self, image, landmarks, persons, objects):
        """检测身体屏蔽姿态
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            persons: 人物检测结果
            objects: 物体检测结果
        """
        # 获取对应的人物边界框
        person_bbox = self._get_person_for_landmarks(landmarks, persons)
        if person_bbox is None:
            return
            
        # 检查是否有肩部关键点
        if not (11 in landmarks and 12 in landmarks and 
                landmarks[11]['visibility'] > 0.3 and landmarks[12]['visibility'] > 0.3):
            return
            
        # 计算肩部旋转角度
        left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
        right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
        
        # 计算肩部线与水平线的角度
        shoulder_angle = np.degrees(np.arctan2(right_shoulder[1] - left_shoulder[1], 
                                              right_shoulder[0] - left_shoulder[0]))
        
        # 检查手腕位置
        wrist_points = [15, 16]  # 左右手腕
        visible_wrists = [i for i in wrist_points if i in landmarks and landmarks[i]['visibility'] > 0.3]
        
        if not visible_wrists:
            return
            
        # 检查是否有物体在身体前方
        has_object_interaction = False
        for obj_bbox, _, _ in objects:
            if self._is_interacting(person_bbox, obj_bbox):
                has_object_interaction = True
                break
        
        # 如果肩部倾斜超过一定角度，且手在身体前方与物体交互
        if abs(shoulder_angle) > 20 and has_object_interaction:
            confidence = 0.7
            behavior = 'body_shielding'
            if confidence > self.confidence_threshold:
                self._add_behavior(behavior, person_bbox, confidence, image)
    
    def _detect_abnormal_head_movement(self, image, landmarks, persons):
        """检测头部异常转动
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            persons: 人物检测结果
        """
        # 检查是否有头部关键点
        head_points = list(range(0, 11))  # 头部关键点
        visible_head_points = [i for i in head_points if i in landmarks and landmarks[i]['visibility'] > 0.3]
        
        if len(visible_head_points) < 5:  # 至少需要5个可见的头部关键点
            return
            
        # 获取对应的人物边界框
        person_bbox = self._get_person_for_landmarks(landmarks, persons)
        if person_bbox is None:
            return
            
        # 鼻子和耳朵位置
        nose = landmarks.get(0, None)
        left_ear = landmarks.get(7, None)
        right_ear = landmarks.get(8, None)
        
        if not nose or (not left_ear and not right_ear):
            return
            
        # 检查头部偏转
        head_turn = False
        
        # 如果一只耳朵可见而另一只不可见（或可见度低），说明头部偏转
        if left_ear and right_ear:
            left_ear_vis = left_ear['visibility']
            right_ear_vis = right_ear['visibility']
            
            if (left_ear_vis > 0.7 and right_ear_vis < 0.3) or (right_ear_vis > 0.7 and left_ear_vis < 0.3):
                head_turn = True
        
        # 或者通过计算鼻子相对于脸部中心的位置来判断
        elif nose:
            # 计算脸部边界框
            face_points = [i for i in range(0, 11) if i in landmarks and landmarks[i]['visibility'] > 0.3]
            if face_points:
                face_x_coords = [landmarks[i]['x'] for i in face_points]
                face_center_x = sum(face_x_coords) / len(face_x_coords)
                
                # 如果鼻子位置偏离脸部中心，说明头部偏转
                if abs(nose['x'] - face_center_x) > 20:
                    head_turn = True
        
        if head_turn:
            confidence = 0.65
            behavior = 'abnormal_head_movement'
            if confidence > self.confidence_threshold:
                self._add_behavior(behavior, person_bbox, confidence, image)
    
    def _detect_hands_behind_back(self, image, landmarks, persons):
        """检测双手放在背后姿势
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            persons: 人物检测结果
        """
        # 检查是否有肩部、手肘和手腕关键点
        shoulder_points = [11, 12]  # 左右肩膀
        elbow_points = [13, 14]    # 左右手肘
        wrist_points = [15, 16]    # 左右手腕
        hip_points = [23, 24]      # 左右髋部
        
        # 确保至少一边肩膀是可见的
        if not any(i in landmarks and landmarks[i]['visibility'] > 0.05 for i in shoulder_points):
            return
            
        # 获取对应的人物边界框
        person_bbox = self._get_person_for_landmarks(landmarks, persons)
        if person_bbox is None:
            return
        
        # 如果双手腕关键点可见度都很低，可能是因为手在背后 - 进一步降低阈值
        wrists_visible = [i for i in wrist_points if i in landmarks and landmarks[i]['visibility'] > 0.12]
        elbows_visible = [i for i in elbow_points if i in landmarks and landmarks[i]['visibility'] > 0.15]
        
        # 条件1: 手腕关键点不可见或可见度很低
        low_wrist_visibility = (len(wrists_visible) == 0 or 
                               any(landmarks[i]['visibility'] < 0.12 for i in wrist_points if i in landmarks))
        
        # 条件2: 手肘关键点可见且角度显示手臂弯曲 - 更宽松的条件
        bent_arms = False
        bent_arm_angles = []
        for i, elbow in enumerate(elbow_points):
            if elbow in landmarks and landmarks[elbow]['visibility'] > 0.15:
                shoulder = shoulder_points[i]
                wrist = wrist_points[i]
                
                # 如果肩膀可见但手腕不可见或可见度低，视为弯曲手臂
                if shoulder in landmarks and landmarks[shoulder]['visibility'] > 0.15:
                    if wrist not in landmarks or landmarks[wrist]['visibility'] < 0.12:
                        bent_arms = True
                        bent_arm_angles.append(90)  # 假设角度为90度
                        break
                    
                    # 或者计算手臂角度
                    elif wrist in landmarks:
                        shoulder_pos = (landmarks[shoulder]['x'], landmarks[shoulder]['y'])
                        elbow_pos = (landmarks[elbow]['x'], landmarks[elbow]['y'])
                        wrist_pos = (landmarks[wrist]['x'], landmarks[wrist]['y'])
                        
                        angle = self._calculate_angle(shoulder_pos, elbow_pos, wrist_pos)
                        bent_arm_angles.append(angle)
                        if angle < 150:  # 更宽松的角度阈值
                            bent_arms = True
                            break
        
        # 条件3: 检测背对摄像头的姿势
        back_view = False
        
        # 简单条件：如果鼻子可见度低，可能是背对摄像头
        if 0 in landmarks and landmarks[0]['visibility'] < 0.6:  # 调整阈值
            back_view = True
            
        # 检查肩膀位置，判断是否是背面视角
        if all(i in landmarks for i in shoulder_points):
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            shoulder_width = abs(left_shoulder['x'] - right_shoulder['x'])
            
            # 如果肩膀宽度小于预期宽度，可能是侧面或背面视角
            person_width = person_bbox[2] - person_bbox[0]
            if shoulder_width < person_width * 0.65:  # 调整阈值为人物宽度的65%
                back_view = True
        
        # 条件4: 检查手腕和躯干的相对位置
        behind_torso = False
        if any(i in landmarks for i in wrist_points) and any(i in landmarks for i in hip_points):
            # 计算躯干中心线
            visible_hips = [i for i in hip_points if i in landmarks]
            if visible_hips:
                hips_center_x = sum(landmarks[i]['x'] for i in visible_hips) / len(visible_hips)
                
                for wrist in wrists_visible:
                    # 手腕水平位置接近躯干中心，但可见度低 - 可能在背后或侧面
                    wrist_dist = abs(landmarks[wrist]['x'] - hips_center_x)
                    if wrist_dist < 90 and landmarks[wrist]['visibility'] < 0.5:  # 增加检测距离，降低可见度要求
                        behind_torso = True
                        break
                    
                    # 新增：检测手腕位置是否处于人体边界之外但很近
                    person_right_edge = person_bbox[2]
                    person_left_edge = person_bbox[0]
                    wrist_x = landmarks[wrist]['x']
                    
                    # 手腕接近人体边缘（12%误差范围内）
                    edge_margin = (person_right_edge - person_left_edge) * 0.12  # 增加误差范围
                    near_right_edge = abs(wrist_x - person_right_edge) < edge_margin
                    near_left_edge = abs(wrist_x - person_left_edge) < edge_margin
                    
                    if near_right_edge or near_left_edge:
                        behind_torso = True
                        break
        
        # 条件5: 只有一只手可见且手臂弯曲
        single_arm_visible = len(wrists_visible) == 1 and bent_arms
        
        # 条件6: 检测手腕高度异常（手腕位置高于肩膀或过低）
        unusual_wrist_height = False
        for wrist in wrists_visible:
            if wrist in landmarks and any(shoulder in landmarks for shoulder in shoulder_points):
                # 获取可见肩膀的平均高度
                visible_shoulders = [s for s in shoulder_points if s in landmarks]
                avg_shoulder_y = sum(landmarks[s]['y'] for s in visible_shoulders) / len(visible_shoulders)
                
                wrist_y = landmarks[wrist]['y']
                
                # 如果手腕明显高于肩膀或明显低于髋部
                if wrist_y < avg_shoulder_y - 15 or (any(hip in landmarks for hip in hip_points) and 
                        wrist_y > sum(landmarks[h]['y'] for h in hip_points if h in landmarks) / 
                        len([h for h in hip_points if h in landmarks]) + 35):  # 调整阈值
                    unusual_wrist_height = True
                    break
        
        # 条件7: 检测手臂靠近腰部或口袋区域
        hands_near_waist = False
        if any(i in landmarks for i in wrist_points) and any(i in landmarks for i in hip_points):
            visible_hips = [i for i in hip_points if i in landmarks]
            if visible_hips:
                avg_hip_y = sum(landmarks[i]['y'] for i in visible_hips) / len(visible_hips)
                waist_y_range = (avg_hip_y - 65, avg_hip_y + 20)  # 扩大腰部区域范围
                
                for wrist in wrists_visible:
                    wrist_y = landmarks[wrist]['y']
                    # 手腕在腰部区域范围内
                    if waist_y_range[0] <= wrist_y <= waist_y_range[1]:
                        hands_near_waist = True
                        break
        
        # 条件8: 检测急速的肘部运动或不自然的弯曲角度
        unusual_elbow_angle = False
        if bent_arm_angles:
            # 检查是否有不自然的弯曲角度 (通常人体自然肘部弯曲在80-120度之间)
            unusual_angles = [angle for angle in bent_arm_angles if angle < 75 or angle > 165]  # 调整角度范围
            if unusual_angles:
                unusual_elbow_angle = True
                
        # 条件9: 身体呈现不自然姿势 (例如弯腰或侧倾)
        unusual_body_posture = False
        if all(i in landmarks for i in hip_points) and all(i in landmarks for i in shoulder_points):
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            
            # 计算躯干的倾斜度
            shoulders_mid_x = (left_shoulder['x'] + right_shoulder['x']) / 2
            hips_mid_x = (left_hip['x'] + right_hip['x']) / 2
            shoulders_mid_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            hips_mid_y = (left_hip['y'] + right_hip['y']) / 2
            
            # 计算躯干倾斜角度
            dx = shoulders_mid_x - hips_mid_x
            dy = shoulders_mid_y - hips_mid_y
            torso_angle = abs(math.degrees(math.atan2(dx, dy)))
            
            # 正常站姿躯干角度应接近180度，偏差大则异常
            if torso_angle > 12:  # 降低阈值使检测更敏感
                unusual_body_posture = True

        # 条件10: 检测单臂在旁边或单臂靠近腰部/口袋
        side_arm_position = False
        if len(elbows_visible) >= 1 and len(wrists_visible) >= 1:
            for wrist in wrists_visible:
                wrist_pos_x = landmarks[wrist]['x']
                wrist_pos_y = landmarks[wrist]['y']
                
                # 检查手腕是否在身体一侧
                if person_bbox is not None:
                    bbox_center_x = (person_bbox[0] + person_bbox[2]) / 2
                    bbox_width = person_bbox[2] - person_bbox[0]
                    
                    # 手腕位于身体侧面区域
                    side_margin = bbox_width * 0.3  # 调整为bbox宽度的30%
                    at_side = (abs(wrist_pos_x - bbox_center_x) > side_margin)
                    
                    # 检查手腕是否在腰部高度区域
                    in_waist_area = False
                    if any(i in landmarks for i in hip_points):
                        avg_hip_y = sum(landmarks[i]['y'] for i in hip_points if i in landmarks) / len([i for i in hip_points if i in landmarks])
                        waist_area_range = (avg_hip_y - 65, avg_hip_y + 30)  # 腰部区域范围
                        in_waist_area = waist_area_range[0] <= wrist_pos_y <= waist_area_range[1]
                    
                    if at_side and in_waist_area:
                        side_arm_position = True
                        break
        
        # 条件11: 检测手腕是否位于图像边缘（可能正在伸手拿取物品）
        wrist_at_edge = False
        image_height, image_width = image.shape[:2]
        edge_threshold = 0.1  # 图像10%的边缘区域
        
        for wrist in wrist_points:
            if wrist in landmarks and landmarks[wrist]['visibility'] > 0.15:
                wrist_x_norm = landmarks[wrist]['x'] / image_width
                wrist_y_norm = landmarks[wrist]['y'] / image_height
                
                if (wrist_x_norm < edge_threshold or wrist_x_norm > (1 - edge_threshold) or
                    wrist_y_norm < edge_threshold or wrist_y_norm > (1 - edge_threshold)):
                    wrist_at_edge = True
                    break
        
        # 条件12: 检测手是否在衣物下方（手腕可见度低但肘部可见）
        hands_under_clothing = False
        for i, wrist in enumerate(wrist_points):
            elbow = elbow_points[i]
            if (elbow in landmarks and landmarks[elbow]['visibility'] > 0.5 and
                (wrist not in landmarks or landmarks[wrist]['visibility'] < 0.2)):
                hands_under_clothing = True
                break
        
        # 条件13: 检测蹲姿伴随伸手动作
        crouching_with_reaching = False
        # 检查膝盖位置判断是否蹲姿
        knee_points = [25, 26]  # 左右膝盖
        visible_knees = [i for i in knee_points if i in landmarks and landmarks[i]['visibility'] > 0.3]
        
        if visible_knees and any(i in landmarks for i in shoulder_points):
            # 计算膝盖与肩膀的高度差
            avg_knee_y = sum(landmarks[i]['y'] for i in visible_knees) / len(visible_knees)
            avg_shoulder_y = sum(landmarks[i]['y'] for i in shoulder_points if i in landmarks) / len([i for i in shoulder_points if i in landmarks])
            
            # 膝盖位置显著高于正常站姿的位置（相对肩膀）表示可能是蹲姿
            knee_shoulder_ratio = (avg_knee_y - avg_shoulder_y) / (person_bbox[3] - person_bbox[1])
            if knee_shoulder_ratio < 0.6:  # 调整阈值，正常站姿约为0.7-0.8
                # 同时检查是否有手臂在伸展状态
                if any(angle > 140 for angle in bent_arm_angles):
                    crouching_with_reaching = True
        
        # 根据满足的条件数量计算可能性分数
        conditions_met = sum([
            low_wrist_visibility, 
            bent_arms,
            back_view,
            behind_torso,
            single_arm_visible,
            unusual_wrist_height,
            hands_near_waist,
            unusual_elbow_angle,
            unusual_body_posture,
            side_arm_position,
            wrist_at_edge,
            hands_under_clothing,
            crouching_with_reaching
        ])
        
        # 设置基础置信度，并根据满足的条件数增加置信度
        # 更多条件满足，置信度更高
        confidence_base = 0.57  # 基础置信度
        confidence_boost = min(0.38, conditions_met * 0.04)  # 每满足一个条件增加0.04，最多增加0.38
        confidence = confidence_base + confidence_boost
        
        # 确定行为类型
        behavior = None
        
        # 根据满足的条件选择最合适的行为类型
        if back_view and (behind_torso or low_wrist_visibility):
            behavior = 'hands_behind_back'
        elif hands_near_waist or side_arm_position:
            behavior = 'suspicious_arm_posture'
        elif unusual_elbow_angle or unusual_body_posture:
            behavior = 'abnormal_arm_position'
        elif single_arm_visible and (bent_arms or unusual_wrist_height):
            behavior = 'single_arm_hiding'
        elif crouching_with_reaching:
            behavior = 'unusual_reaching'
        else:
            behavior = 'suspicious_arm_posture'  # 默认行为类型
        
        # 如果置信度超过阈值，添加检测到的行为
        if confidence > self.confidence_threshold:
            self._add_behavior(behavior, person_bbox, confidence, image)
    
    def _calculate_angle(self, a, b, c):
        """计算三点形成的角度
        
        Args:
            a: 第一个点坐标 (x, y)
            b: 中间点坐标 (x, y)
            c: 第三个点坐标 (x, y)
            
        Returns:
            angle: 角度值（度）
        """
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        
        # 计算两个向量的点积
        dot_product = ba[0] * bc[0] + ba[1] * bc[1]
        
        # 计算两个向量的模
        magnitude_ba = np.sqrt(ba[0]**2 + ba[1]**2)
        magnitude_bc = np.sqrt(bc[0]**2 + bc[1]**2)
        
        # 计算夹角的余弦值
        cos_angle = dot_product / (magnitude_ba * magnitude_bc)
        
        # 防止数值误差导致cos_angle超出[-1, 1]范围
        cos_angle = max(-1, min(1, cos_angle))
        
        # 计算角度并转换为度
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle 

    def _extract_enhanced_features(self, landmarks, frame=None):
        """提取增强特征集
        
        Args:
            landmarks: 姿态关键点字典
            frame: 原始帧图像数据（可选）
            
        Returns:
            features: 增强特征向量
        """
        # 首先获取标准特征
        standard_features = self._extract_standard_features(landmarks)
        if standard_features is None:
            return None
            
        try:
            # 标准特征已经包含：
            # 0-1: 左右手腕距离
            # 2: 躯干高度
            # 3-4: 左右手臂角度
            # 5-6: 左右手腕与髋部比例
            # 7: 手臂交叉状态
            # 8: 躯干旋转
            # 9: 肩膀宽度
            
            features = list(standard_features)  # 复制标准特征
            
            # 10. 添加头部位置特征
            head_position = 0
            if 0 in landmarks and 23 in landmarks and 24 in landmarks:
                nose_y = landmarks[0]['y']
                left_hip_y = landmarks[23]['y']
                right_hip_y = landmarks[24]['y']
                hip_center_y = (left_hip_y + right_hip_y) / 2
                
                # 头部位置和髋部中点的相对位置
                head_position = nose_y - hip_center_y
                
                # 如果有躯干高度，则归一化
                if features[2] > 0:  # torso_height
                    head_position /= features[2]
            
            features.append(head_position)
            
            # 11. 计算膝盖-髋部比例（蹲下姿势检测）
            knee_hip_ratio = 0
            if 25 in landmarks and 26 in landmarks and 23 in landmarks and 24 in landmarks:
                left_knee_y = landmarks[25]['y']
                right_knee_y = landmarks[26]['y']
                left_hip_y = landmarks[23]['y']
                right_hip_y = landmarks[24]['y']
                
                knee_center_y = (left_knee_y + right_knee_y) / 2
                hip_center_y = (left_hip_y + right_hip_y) / 2
                
                # 膝盖中点到髋部中点的距离
                # 这个比例增加表示膝盖弯曲（蹲下）
                knee_hip_distance = knee_center_y - hip_center_y
                
                if features[2] > 0:  # torso_height
                    knee_hip_ratio = knee_hip_distance / features[2]
            
            features.append(knee_hip_ratio)
            
            # 12. 手腕速度（如果有帧图像和前一帧数据）
            wrist_velocity_left = 0
            wrist_velocity_right = 0
            
            # 需要在类实例中存储前一帧的数据才能计算速度
            if hasattr(self, 'prev_landmarks') and self.prev_landmarks is not None:
                prev_landmarks = self.prev_landmarks
                
                # 计算左手腕速度
                if 15 in landmarks and 15 in prev_landmarks:
                    curr_wrist = (landmarks[15]['x'], landmarks[15]['y'])
                    prev_wrist = (prev_landmarks[15]['x'], prev_landmarks[15]['y'])
                    
                    # 简单的欧几里得距离作为位移
                    displacement = math.sqrt((curr_wrist[0] - prev_wrist[0])**2 + 
                                           (curr_wrist[1] - prev_wrist[1])**2)
                    
                    # 由于是图像处理，没有实际时间，使用相对速度
                    wrist_velocity_left = displacement
                
                # 计算右手腕速度
                if 16 in landmarks and 16 in prev_landmarks:
                    curr_wrist = (landmarks[16]['x'], landmarks[16]['y'])
                    prev_wrist = (prev_landmarks[16]['x'], prev_landmarks[16]['y'])
                    
                    # 简单的欧几里得距离作为位移
                    displacement = math.sqrt((curr_wrist[0] - prev_wrist[0])**2 + 
                                           (curr_wrist[1] - prev_wrist[1])**2)
                    
                    # 由于是图像处理，没有实际时间，使用相对速度
                    wrist_velocity_right = displacement
            
            # 存储当前帧的关键点用于下一帧计算
            self.prev_landmarks = landmarks.copy()
            
            features.append(wrist_velocity_left)
            features.append(wrist_velocity_right)
            
            # 13. 姿势稳定性（计算关键点的抖动程度）
            posture_stability = 0
            if hasattr(self, 'prev_landmarks_list') and len(self.prev_landmarks_list) > 0:
                # 计算关键点的平均位移
                total_displacement = 0
                num_points = 0
                
                for point_id in range(33):  # MediaPipe姿势估计有33个关键点
                    if point_id in landmarks and point_id in self.prev_landmarks_list[-1]:
                        curr_point = (landmarks[point_id]['x'], landmarks[point_id]['y'])
                        prev_point = (self.prev_landmarks_list[-1][point_id]['x'], 
                                    self.prev_landmarks_list[-1][point_id]['y'])
                        
                        displacement = math.sqrt((curr_point[0] - prev_point[0])**2 + 
                                             (curr_point[1] - prev_point[1])**2)
                        
                        total_displacement += displacement
                        num_points += 1
                
                if num_points > 0:
                    # 平均位移作为稳定性的逆指标
                    avg_displacement = total_displacement / num_points
                    # 转换为稳定性指标：位移越大，稳定性越低
                    posture_stability = 1.0 / (1.0 + avg_displacement)
            
            # 保存关键点历史（最多5帧）
            if not hasattr(self, 'prev_landmarks_list'):
                self.prev_landmarks_list = []
            
            self.prev_landmarks_list.append(landmarks.copy())
            if len(self.prev_landmarks_list) > 5:
                self.prev_landmarks_list.pop(0)
            
            features.append(posture_stability)
            
            return features
            
        except Exception as e:
            logger.error(f"提取增强特征出错: {e}")
            return None
    
    def detect_behaviors(self, image, pose_data=None, objects=None, retail_info=None):
        """检测图像中的可疑行为
        
        Args:
            image: 输入图像(numpy数组)
            pose_data: 可选的预计算姿态数据
            objects: 检测到的物体列表
            retail_info: 零售环境信息
            
        Returns:
            detected_behaviors: 检测到的行为列表
        """
        if image is None:
            logger.error("输入图像为空")
            return []
        
        # 记录环境信息
        if retail_info:
            self.retail_environment_detected = True
        
        # 如果没有提供姿态数据，尝试检测
        if pose_data is None:
            pose_data = self.detect_pose(image)
        
        if pose_data is None or len(pose_data) == 0:
            logger.info("未检测到人体姿态")
            return []
        
        # 重置检测结果
        self.detected_behaviors = []
        
        # 提取姿态关键点
        landmarks = self.extract_pose_landmarks(pose_data)
        
        # 使用机器学习模型预测
        ml_predictions = self.predict_with_ml_models(landmarks, image)
        
        # 使用规则引擎预测
        rule_predictions = self.predict_with_rules(landmarks, objects, retail_info)
        
        # 合并预测结果
        self.detected_behaviors = self.combine_predictions(ml_predictions, rule_predictions)
        
        # 按照置信度排序
        self.detected_behaviors.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return self.detected_behaviors
    
    def predict_with_ml_models(self, landmarks, image=None):
        """使用机器学习模型预测行为
        
        Args:
            landmarks: 姿态关键点
            image: 原始图像（可选）
            
        Returns:
            predictions: 预测的行为
        """
        predictions = []
        
        # 检查是否有有效的特征
        if not landmarks or len(landmarks) < 5:
            return predictions
        
        # 提取特征
        if self.model_type == "enhanced" and self.enhanced_model is not None:
            features = self._extract_enhanced_features(landmarks, image)
            model = self.enhanced_model
        else:
            features = self._extract_standard_features(landmarks)
            model = self.standard_model
        
        if features is None or model is None:
            return predictions
        
        # 使用XGBoost模型进行预测
        prediction, confidence = self.predict_with_xgboost(features)
        
        # 如果预测为盗窃行为
        if prediction == 1 and confidence >= self.confidence_threshold:
            predictions.append({
                'type': 'theft_suspicious',
                'confidence': confidence,
                'model_based': True,
                'features': features
            })
        
        # 使用行为分类器预测具体行为类型
        specific_behaviors = self.predict_behaviors(features)
        predictions.extend(specific_behaviors)
        
        return predictions
    
    def predict_with_rules(self, landmarks, objects=None, retail_info=None):
        """使用规则引擎预测行为
        
        Args:
            landmarks: 姿态关键点
            objects: 检测到的物体列表（可选）
            retail_info: 零售环境信息（可选）
            
        Returns:
            predictions: 预测的行为
        """
        import logging
        logger = logging.getLogger('image_behavior_detector')
        logger.info("执行规则引擎行为预测...")
        
        predictions = []
        
        # 降低对关键点数量的要求
        if not landmarks:
            logger.info("没有可用的关键点数据")
            return predictions
        
        # 输出可见关键点信息
        visible_points = sum(1 for point_id in landmarks if point_id in landmarks and landmarks[point_id]['visibility'] > 0.2)
        logger.info(f"可见关键点数量: {visible_points}")
        
        # 即使关键点很少，也尝试进行检测
        
        # 检查手臂是否交叉
        arms_crossed = self._check_arms_crossed(landmarks)
        if arms_crossed:
            logger.info("检测到手臂交叉姿势")
            predictions.append({
                'type': 'arms_crossed',
                'confidence': 0.8,
                'model_based': False
            })
        
        # 检查是否手按口袋
        hand_pocket = self._check_hand_pocket(landmarks)
        if hand_pocket:
            logger.info("检测到手按口袋姿势")
            predictions.append({
                'type': 'hand_pressing_pocket',
                'confidence': 0.9,  # 提高置信度
                'model_based': False
            })
            
        # 检查是否有将物品放入口袋的行为
        item_to_pocket = self._check_item_to_pocket(landmarks, objects)
        if item_to_pocket:
            logger.info("检测到物品放入口袋行为")
            predictions.append({
                'type': 'Item Concealed in Pocket',  # 使用英文类型，与behavior_type_map中的键匹配
                'confidence': 0.95,  # 提高置信度
                'model_based': False
            })
            
            # 如果检测到物品放入口袋，也添加隐藏物品行为
            predictions.append({
                'type': 'Hiding Item',
                'confidence': 0.9,
                'model_based': False
            })
        
        # 检测可疑的手部动作 - 即使没有检测到口袋行为，如果有手腕接近髋部的情况也视为可疑
        if 15 in landmarks and 23 in landmarks:  # 左手腕和左髋
            left_wrist = (landmarks[15]['x'], landmarks[15]['y'])
            left_hip = (landmarks[23]['x'], landmarks[23]['y'])
            left_dist = math.sqrt((left_wrist[0] - left_hip[0])**2 + (left_wrist[1] - left_hip[1])**2)
            
            if left_dist < 0.3:  # 宽松的阈值
                logger.info(f"检测到左手接近髋部，距离: {left_dist}")
                predictions.append({
                    'type': 'Suspicious Hand Movement',
                    'confidence': 0.7,
                    'model_based': False
                })
        
        if 16 in landmarks and 24 in landmarks:  # 右手腕和右髋
            right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
            right_hip = (landmarks[24]['x'], landmarks[24]['y'])
            right_dist = math.sqrt((right_wrist[0] - right_hip[0])**2 + (right_wrist[1] - right_hip[1])**2)
            
            if right_dist < 0.3:  # 宽松的阈值
                logger.info(f"检测到右手接近髋部，距离: {right_dist}")
                predictions.append({
                    'type': 'Suspicious Hand Movement',
                    'confidence': 0.7,
                    'model_based': False
                })
        
        # 检测背对姿势 - 如果肩部宽度很小可能表示背对
        if 11 in landmarks and 12 in landmarks:
            left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
            right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
            shoulder_width = math.sqrt((left_shoulder[0] - right_shoulder[0])**2 + (left_shoulder[1] - right_shoulder[1])**2)
            
            if shoulder_width < 0.15:  # 很窄的肩宽表示可能背对
                logger.info(f"检测到可能的背对姿势，肩宽: {shoulder_width}")
                predictions.append({
                    'type': 'Back Facing Posture',
                    'confidence': 0.8,
                    'model_based': False
                })
                
                # 如果是背对姿势，增加可疑行为的置信度
                predictions.append({
                    'type': 'Suspicious Posture',
                    'confidence': 0.85,
                    'model_based': False
                })
        
        # 如果在零售环境中，检查特定的零售相关行为
        if self.retail_environment_detected or retail_info:
            logger.info("在零售环境中检查特定行为")
            retail_predictions = self._check_retail_specific_behaviors(landmarks, objects, retail_info)
            predictions.extend(retail_predictions)
        
        # 记录所有检测到的行为
        if predictions:
            logger.info(f"规则引擎检测到 {len(predictions)} 个可疑行为:")
            for pred in predictions:
                logger.info(f" - {pred['type']}: {pred['confidence']:.2f}")
        else:
            logger.info("规则引擎未检测到任何可疑行为")
        
        return predictions

    def _detect_pocket_concealment(self, image, pose_results, persons, objects):
        """专门检测将物品放入口袋的行为
        
        Args:
            image: 输入图像
            pose_results: 姿态估计结果
            persons: 检测到的人物列表
            objects: 检测到的物体列表
        """
        if not pose_results or not persons:
            # 如果没有姿态估计结果，但检测到了人物，尝试基于人物轮廓检测背向姿势
            if persons:
                for person_bbox, person_conf in persons:
                    # 检查是否为背向姿势（基于边界框特征）
                    x1, y1, x2, y2 = person_bbox
                    aspect_ratio = (x2-x1) / (y2-y1) if (y2-y1) > 0 else 0
                    
                    # 如果人物处于画面中央且宽高比适中，可能是背对摄像头
                    center_x = (x1 + x2) / 2
                    image_width = image.shape[1]
                    
                    is_center = (0.2 * image_width < center_x < 0.8 * image_width)  # 扩大检测范围
                    
                    # 放宽背向姿势检测的条件
                    if 0.25 < aspect_ratio < 0.85 and is_center:  # 扩大宽高比的范围
                        # 提取人物ROI并分析
                        person_roi = image[y1:y2, x1:x2]
                        if person_roi.size > 0:
                            # 尝试通过图像处理检测手臂位置和口袋区域交互
                            try:
                                # 检查是否有手臂交叉或手在腰部的可能
                                h, w = person_roi.shape[:2]
                                
                                # 获取下半部分图像(腰部以下)
                                lower_part = person_roi[h//2:, :]
                                # 获取中部区域(可能的手臂区域)
                                mid_part = person_roi[h//3:2*h//3, :]
                                
                                # 检查颜色和边缘分布
                                gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                                edges = cv2.Canny(blur, 30, 100)
                                
                                # 分析腰部区域的边缘密度
                                h_edges, w_edges = edges.shape[:2]
                                waist_area = edges[h_edges//3:2*h_edges//3, :]
                                waist_edge_density = np.sum(waist_area > 0) / waist_area.size if waist_area.size > 0 else 0
                                
                                # 分析手臂区域的边缘和颜色分布
                                side_area = edges[:, :w_edges//3]
                                side_edge_density = np.sum(side_area > 0) / side_area.size if side_area.size > 0 else 0
                                
                                # 【增强的检测】检查是否有手臂在背后/侧面区域
                                # 使用颜色分析来检测可能的手部区域
                                if waist_edge_density > 0.05 or side_edge_density > 0.08:
                                    # 对下半部分图像进行颜色分析
                                    lower_hsv = cv2.cvtColor(lower_part, cv2.COLOR_BGR2HSV)
                                    # 检查肤色区域 - 简化为检测不同于主要服装颜色的区域
                                    dominant_color = np.median(lower_part.reshape(-1, 3), axis=0)
                                    color_diff = np.sum(np.abs(lower_part - dominant_color), axis=2)
                                    color_variation = np.std(color_diff)
                                    
                                    # 判断条件：边缘密度高 或 颜色变化在腰部区域明显
                                    if (waist_edge_density > 0.05 and color_variation > 15) or side_edge_density > 0.1:
                                        # 人在画面中央、身体比例正常、腰部有明显边缘且颜色分布不均匀，高概率是在操作口袋
                                        confidence = 0.75 + min(0.15, waist_edge_density * 2)
                                        behavior = 'Item Concealed in Pocket'
                                        self._add_behavior(behavior, person_bbox, confidence, image)
                                        return
                            except Exception as e:
                                logger.warning(f"背向姿势分析错误: {e}")
                                # 兜底检测逻辑: 如果人物在合理位置且姿态比例正常，给予基本置信度
                                if 0.3 < aspect_ratio < 0.8 and is_center:
                                    confidence = 0.65  # 基本置信度
                                    behavior = 'Item Concealed in Pocket'
                                    self._add_behavior(behavior, person_bbox, confidence, image)
                                    return
            return
        
    def _check_item_to_pocket(self, landmarks, objects=None):
        """检查是否有将物品放入口袋的行为
        
        Args:
            landmarks: 姿态关键点
            objects: 检测到的物体列表
            
        Returns:
            item_to_pocket: 是否将物品放入口袋
        """
        import logging
        logger = logging.getLogger('image_behavior_detector')
        
        # 需要左右手腕点、左右肘部点和左右髋点（降低要求，只需部分关键点可见）
        min_required_points = [15, 16, 23, 24]  # 左右手腕和左右髋部最低要求
        visible_required = sum(1 for point in min_required_points if point in landmarks and landmarks[point]['visibility'] > 0.2)
        logger.info(f"口袋检测 - 可见关键点数量: {visible_required}/{len(min_required_points)}")
        
        # 降低阈值，只需要2个关键点可见
        if visible_required < 2:  # 允许至少2个关键点可见
            logger.info("口袋检测 - 可见关键点不足，返回False")
            return False
        
        # 获取关键点坐标（如果存在）
        left_elbow = (landmarks[13]['x'], landmarks[13]['y']) if 13 in landmarks else None
        right_elbow = (landmarks[14]['x'], landmarks[14]['y']) if 14 in landmarks else None
        left_wrist = (landmarks[15]['x'], landmarks[15]['y']) if 15 in landmarks else None
        right_wrist = (landmarks[16]['x'], landmarks[16]['y']) if 16 in landmarks else None
        left_hip = (landmarks[23]['x'], landmarks[23]['y']) if 23 in landmarks else None
        right_hip = (landmarks[24]['x'], landmarks[24]['y']) if 24 in landmarks else None
        
        logger.info(f"口袋检测 - 左手腕: {left_wrist}, 右手腕: {right_wrist}")
        logger.info(f"口袋检测 - 左髋部: {left_hip}, 右髋部: {right_hip}")
        
        # 计算手腕到髋部的距离（如果两个点都存在）
        left_wrist_to_hip = None
        right_wrist_to_hip = None
        
        if left_wrist and left_hip:
            left_wrist_to_hip = math.sqrt((left_wrist[0] - left_hip[0])**2 + (left_wrist[1] - left_hip[1])**2)
        
        if right_wrist and right_hip:
            right_wrist_to_hip = math.sqrt((right_wrist[0] - right_hip[0])**2 + (right_wrist[1] - right_hip[1])**2)
            
        logger.info(f"口袋检测 - 左手腕到髋部距离: {left_wrist_to_hip}, 右手腕到髋部距离: {right_wrist_to_hip}")
        
        # 计算肘部弯曲角度 - 需要肩部点
        left_elbow_angle = None
        right_elbow_angle = None
        
        if 11 in landmarks and 13 in landmarks and 15 in landmarks and landmarks[11]['visibility'] > 0.2:  # 左肩
            left_shoulder = (landmarks[11]['x'], landmarks[11]['y'])
            left_elbow_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
        
        if 12 in landmarks and 14 in landmarks and 16 in landmarks and landmarks[12]['visibility'] > 0.2:  # 右肩
            right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
            right_elbow_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            
        logger.info(f"口袋检测 - 左肘角度: {left_elbow_angle}, 右肘角度: {right_elbow_angle}")
        
        # 计算参考距离（肩宽）
        shoulder_width = 0
        if 11 in landmarks and 12 in landmarks:
            left_shoulder = (landmarks[11]['x'], landmarks[11]['y']) 
            right_shoulder = (landmarks[12]['x'], landmarks[12]['y'])
            shoulder_width = math.sqrt((left_shoulder[0] - right_shoulder[0])**2 + 
                                    (left_shoulder[1] - right_shoulder[1])**2)
            
        logger.info(f"口袋检测 - 肩宽: {shoulder_width}")
        
        # 增加检测逻辑：对于背对姿势的特殊处理
        # 检查两个手腕是否都在中心线附近（可能表示背对姿势下的手在腰部/口袋）
        is_back_posture = False
        if left_wrist and right_wrist:
            # 检查两个手腕是否在身体中轴附近
            center_x = 0.5  # 归一化坐标中的中心线
            left_to_center = abs(left_wrist[0] - center_x)
            right_to_center = abs(right_wrist[0] - center_x)
            
            logger.info(f"口袋检测 - 左手腕到中轴距离: {left_to_center}, 右手腕到中轴距离: {right_to_center}")
            
            # 放宽背对姿势的检测条件
            if left_to_center < 0.25 and right_to_center < 0.25:
                is_back_posture = True
                logger.info("口袋检测 - 检测到背对姿势特征")
        
        # 判断条件：
        # 1. 手腕接近髋部
        # 2. 肘部明显弯曲（角度小于140度，进一步放宽标准）
        # 3. 或者是背对姿势检测
        
        wrist_hip_threshold = shoulder_width * 0.5 if shoulder_width > 0 else 0.3  # 进一步放宽阈值
        logger.info(f"口袋检测 - 手腕到髋部距离阈值: {wrist_hip_threshold}")
        
        left_arm_to_pocket = (left_wrist_to_hip is not None and 
                             left_wrist_to_hip < wrist_hip_threshold and 
                             (left_elbow_angle is None or left_elbow_angle < 140))
                             
        right_arm_to_pocket = (right_wrist_to_hip is not None and 
                              right_wrist_to_hip < wrist_hip_threshold and 
                              (right_elbow_angle is None or right_elbow_angle < 140))
                              
        logger.info(f"口袋检测 - 左手臂放入口袋: {left_arm_to_pocket}, 右手臂放入口袋: {right_arm_to_pocket}")
        
        # 如果有物体检测结果，可以进一步判断手是否拿着物体
        has_object_in_hand = False
        if objects:
            logger.info(f"口袋检测 - 检测到 {len(objects)} 个物体")
            # 简单判断：如果有小物体和手腕在接近的位置
            for obj in objects:
                if obj.get('class') in ['bottle', 'cup', 'cell phone', 'book', 'remote', 'scissors', 'teddy bear', 'toothbrush']:
                    obj_bbox = obj.get('bbox', [0, 0, 0, 0])
                    obj_center_x = obj_bbox[0] + obj_bbox[2]/2
                    obj_center_y = obj_bbox[1] + obj_bbox[3]/2
                    
                    logger.info(f"口袋检测 - 检测到物体 {obj.get('class')} 在位置 ({obj_center_x}, {obj_center_y})")
                    
                    # 计算物体到手腕的距离
                    if left_wrist:
                        left_dist = math.sqrt((obj_center_x - left_wrist[0])**2 + (obj_center_y - left_wrist[1])**2)
                        logger.info(f"口袋检测 - 物体到左手腕距离: {left_dist}")
                        if left_dist < 0.25:  # 进一步放宽阈值
                            has_object_in_hand = True
                            logger.info("口袋检测 - 检测到物体在左手中")
                            break
                    
                    if right_wrist:
                        right_dist = math.sqrt((obj_center_x - right_wrist[0])**2 + (obj_center_y - right_wrist[1])**2)
                        logger.info(f"口袋检测 - 物体到右手腕距离: {right_dist}")
                        if right_dist < 0.25:  # 进一步放宽阈值
                            has_object_in_hand = True
                            logger.info("口袋检测 - 检测到物体在右手中")
                            break
        else:
            logger.info("口袋检测 - 没有物体检测数据")
            # 即使没有物体检测数据，也尝试判断姿势
            has_object_in_hand = True  # 假设有物体，放宽条件
        
        # 如果检测到物体在手中并且手臂姿势符合放入口袋的动作，返回True
        if has_object_in_hand and (left_arm_to_pocket or right_arm_to_pocket):
            logger.info("口袋检测 - 检测到物体在手中且手臂姿势符合放入口袋的动作，返回True")
            return True
        
        # 即使没有检测到物体，如果有明显的手部向口袋移动的姿势或背对姿势特征，也判断为可能将物品放入口袋
        if (left_arm_to_pocket or right_arm_to_pocket) or is_back_posture:
            logger.info("口袋检测 - 检测到手臂向口袋移动姿势或背对姿势特征，返回True")
            return True
            
        logger.info("口袋检测 - 没有检测到将物品放入口袋的行为，返回False")
        return False
        
    def _check_retail_specific_behaviors(self, landmarks, objects=None, retail_info=None):
        """检查零售环境特定的行为
        
        Args:
            landmarks: 姿态关键点
            objects: 检测到的物体列表
            retail_info: 零售环境信息
            
        Returns:
            predictions: 预测的行为
        """
        predictions = []
        
        # 如果没有零售环境信息，无法进行特定检测
        if not retail_info:
            return predictions
        
        # 示例：检查是否在商品区域有可疑动作
        if objects and 'product_shelf' in [obj.get('class') for obj in objects]:
            # 检查手臂是否伸向商品架
            if self._check_reaching_products(landmarks, objects):
                predictions.append({
                    'type': 'product_interaction',
                    'confidence': 0.7,
                    'model_based': False
                })
                
                # 如果同时低头，可能是在查看产品
                if self._check_looking_down(landmarks):
                    predictions.append({
                        'type': 'examining_product',
                        'confidence': 0.75,
                        'model_based': False
                    })
        
        # 示例：检查是否有遮挡行为
        if self._check_covering_action(landmarks):
            predictions.append({
                'type': 'umbrella_covering',
                'confidence': 0.8,
                'model_based': False
            })
        
        return predictions
    
    def _check_reaching_products(self, landmarks, objects):
        """检查是否伸手拿取商品
        
        Args:
            landmarks: 姿态关键点
            objects: 检测到的物体列表
            
        Returns:
            reaching: 是否伸手拿取
        """
        # 简单实现：检查手腕是否靠近商品架
        if not (15 in landmarks and 16 in landmarks) or not objects:
            return False
        
        left_wrist = (landmarks[15]['x'], landmarks[15]['y'])
        right_wrist = (landmarks[16]['x'], landmarks[16]['y'])
        
        for obj in objects:
            if obj.get('class') == 'product_shelf':
                bbox = obj.get('bbox', [0, 0, 0, 0])
                shelf_x, shelf_y, shelf_w, shelf_h = bbox
                
                # 检查手腕是否在货架附近
                shelf_center_x = shelf_x + shelf_w / 2
                shelf_center_y = shelf_y + shelf_h / 2
                
                left_dist = math.sqrt((left_wrist[0] - shelf_center_x)**2 + 
                                   (left_wrist[1] - shelf_center_y)**2)
                right_dist = math.sqrt((right_wrist[0] - shelf_center_x)**2 + 
                                    (right_wrist[1] - shelf_center_y)**2)
                
                # 阈值可以根据实际情况调整
                threshold = min(shelf_w, shelf_h) * 0.5
                
                if left_dist < threshold or right_dist < threshold:
                    return True
        
        return False
    
    def _check_looking_down(self, landmarks):
        """检查是否低头
        
        Args:
            landmarks: 姿态关键点
            
        Returns:
            looking_down: 是否低头
        """
        if not (0 in landmarks and 11 in landmarks and 12 in landmarks):
            return False
        
        nose = landmarks[0]['y']
        left_shoulder = landmarks[11]['y']
        right_shoulder = landmarks[12]['y']
        
        # 肩膀Y坐标的平均值
        shoulder_y = (left_shoulder + right_shoulder) / 2
        
        # 如果鼻子的Y坐标明显大于肩膀的Y坐标，则认为是低头
        # Y坐标在图像中通常是从上往下递增
        return nose > shoulder_y + 0.1  # 阈值可调整
    
    def _check_covering_action(self, landmarks):
        """检查是否有遮挡行为
        
        Args:
            landmarks: 姿态关键点
            
        Returns:
            covering: 是否有遮挡行为
        """
        # 简单实现：检查手臂是否抬高遮挡
        if not (13 in landmarks and 14 in landmarks and 11 in landmarks and 12 in landmarks):
            return False
        
        left_elbow_y = landmarks[13]['y']
        right_elbow_y = landmarks[14]['y']
        left_shoulder_y = landmarks[11]['y']
        right_shoulder_y = landmarks[12]['y']
        
        # 如果肘部高于肩部，可能是抬手遮挡
        left_raised = left_elbow_y < left_shoulder_y
        right_raised = right_elbow_y < right_shoulder_y
        
        return left_raised or right_raised
        
    def _detect_suspicious_arm_posture(self, image, landmarks, persons):
        """检测可疑的手臂姿势，特别是手部靠近腰部/后背的情况
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            persons: 人物检测结果
        """
        try:
            # 检查关键点
            shoulder_points = [11, 12]  # 左右肩膀
            elbow_points = [13, 14]    # 左右手肘
            wrist_points = [15, 16]    # 左右手腕
            hip_points = [23, 24]      # 左右髋部
            
            # 确保至少一边肩膀和肘部可见，并且landmarks数据结构完整
            has_valid_shoulder = False
            has_valid_elbow = False
            
            for i in shoulder_points:
                if i in landmarks and 'visibility' in landmarks[i] and landmarks[i]['visibility'] > 0.2:
                    has_valid_shoulder = True
                    break
                    
            for i in elbow_points:
                if i in landmarks and 'visibility' in landmarks[i] and landmarks[i]['visibility'] > 0.2:
                    has_valid_elbow = True
                    break
            
            if not (has_valid_shoulder and has_valid_elbow):
                return
            
            # 获取对应的人物边界框
            person_bbox = self._get_person_for_landmarks(landmarks, persons)
            if person_bbox is None:
                return
            
            # 检查手腕位置与髋部位置的关系
            suspicious_posture = False
            confidence = 0.0
            
            # 检查手腕是否在髋部附近
            valid_wrists = []
            valid_hips = []
            
            for i in wrist_points:
                if i in landmarks and 'visibility' in landmarks[i] and landmarks[i]['visibility'] > 0.2:
                    valid_wrists.append(i)
            
            for i in hip_points:
                if i in landmarks and 'x' in landmarks[i] and 'y' in landmarks[i]:
                    valid_hips.append(i)
            
            if valid_wrists and valid_hips:
                for wrist in valid_wrists:
                    wrist_x, wrist_y = landmarks[wrist]['x'], landmarks[wrist]['y']
                    
                    # 计算髋部区域
                    hip_y_coords = [landmarks[hip]['y'] for hip in valid_hips]
                    hip_x_coords = [landmarks[hip]['x'] for hip in valid_hips]
                    
                    if hip_y_coords and hip_x_coords:
                        avg_hip_y = sum(hip_y_coords) / len(hip_y_coords)
                        avg_hip_x = sum(hip_x_coords) / len(hip_x_coords)
                        
                        # 计算手腕与髋部的距离
                        dist_to_hip = np.sqrt((wrist_x - avg_hip_x)**2 + (wrist_y - avg_hip_y)**2)
                        
                        # 根据人物尺寸归一化距离
                        person_height = person_bbox[3] - person_bbox[1]
                        normalized_dist = dist_to_hip / person_height if person_height > 0 else float('inf')
                        
                        # 如果手腕接近髋部
                        if normalized_dist < 0.25:  # 手腕在髋部附近（人物高度的25%范围内）
                            suspicious_posture = True
                            confidence = 0.65 + (0.25 - normalized_dist)  # 距离越近，置信度越高
                            break
            
            # 检查手臂姿势 - 手肘弯曲且手腕不太可见
            if not suspicious_posture:
                for i, elbow in enumerate(elbow_points):
                    if elbow in landmarks and 'visibility' in landmarks[elbow] and landmarks[elbow]['visibility'] > 0.3:
                        shoulder = shoulder_points[i]
                        wrist = wrist_points[i]
                        
                        # 确保肩膀可见
                        if shoulder in landmarks and 'visibility' in landmarks[shoulder] and landmarks[shoulder]['visibility'] > 0.3:
                            # 情况1: 检查手腕可见度
                            wrist_visible = wrist in landmarks and 'visibility' in landmarks[wrist] and landmarks[wrist]['visibility'] > 0.2
                            
                            if wrist_visible:
                                # 计算手肘角度
                                shoulder_pos = (landmarks[shoulder]['x'], landmarks[shoulder]['y'])
                                elbow_pos = (landmarks[elbow]['x'], landmarks[elbow]['y'])
                                wrist_pos = (landmarks[wrist]['x'], landmarks[wrist]['y'])
                                
                                angle = self._calculate_angle(shoulder_pos, elbow_pos, wrist_pos)
                                
                                # 如果手臂弯曲且手腕在人体侧面/背面区域
                                if angle < 120:
                                    # 检查手腕是否位于躯干侧面
                                    person_center_x = (person_bbox[0] + person_bbox[2]) / 2
                                    side_margin = (person_bbox[2] - person_bbox[0]) * 0.2  # 人物宽度的20%
                                    
                                    is_at_side = (abs(wrist_pos[0] - person_center_x) > side_margin)
                                    
                                    if is_at_side:
                                        suspicious_posture = True
                                        confidence = 0.7 + (120 - angle) / 120  # 角度越小，置信度越高
                                        break
                            else:
                                # 情况2: 手腕不可见但手肘弯曲 - 可能表示手臂在背后
                                shoulder_pos = (landmarks[shoulder]['x'], landmarks[shoulder]['y'])
                                elbow_pos = (landmarks[elbow]['x'], landmarks[elbow]['y'])
                                
                                # 检查手肘相对于肩膀的位置
                                dx = elbow_pos[0] - shoulder_pos[0]
                                dy = elbow_pos[1] - shoulder_pos[1]
                                
                                # 如果手肘明显在身体后侧(x方向变化不大，但y方向明显下移)
                                if abs(dx) < 30 and dy > 20:
                                    suspicious_posture = True
                                    confidence = 0.75
                                    break
            
            if suspicious_posture and confidence > self.confidence_threshold:
                behavior = 'suspicious_arm_posture'
                self._add_behavior(behavior, person_bbox, confidence, image)
                
        except Exception as e:
            import logging
            logger = logging.getLogger('image_behavior_detector')
            logger.error(f"检测可疑手臂姿势出错: {e}")
            return