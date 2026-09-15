import os
import cv2
import time
import torch
import numpy as np
import logging
import traceback
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import random
import pickle  # 添加pickle模块导入用于加载模型
import json  # 用于加载配置
import math  # 添加math模块导入用于计算

# 添加MediaPipe导入
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not installed, will use alternative pose estimation method")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('video_behavior_detector')

# 简单的对象跟踪器实现
class Tracker:
    """简单的目标跟踪器，用于跟踪视频中检测到的对象"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = self.config.get('max_disappeared', 30)  # 对象可以消失的最大帧数
        self.distance_threshold = self.config.get('distance_threshold', 100)  # 距离阈值
    def update(self, boxes):
        """
        更新跟踪器状态
        
        Args:
            boxes: 边界框列表 [x1, y1, x2, y2]
        
        Returns:
            跟踪状态字典 {id: box}
        """
        # 如果没有跟踪对象，初始化所有框
        if len(self.tracks) == 0:
            for box in boxes:
                self.tracks[self.next_id] = {
                    'box': box,
                    'disappeared': 0
                }
                self.next_id += 1
            return self.tracks
            
        # 如果没有检测，更新所有跟踪对象的消失计数
        if len(boxes) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracks[track_id]
            return self.tracks
            
        # 计算现有跟踪对象和新检测框之间的距离
        matched_tracks = {}
        unmatched_tracks = list(self.tracks.keys())
        unmatched_boxes = list(range(len(boxes)))
        
        # 简单的贪心匹配算法
        for track_id in list(self.tracks.keys()):
            if track_id not in unmatched_tracks:
                continue
                
            track_box = self.tracks[track_id]['box']
            min_distance = float('inf')
            min_idx = None
            
            for i in unmatched_boxes:
                box = boxes[i]
                # 计算框中心点距离
                track_center = [(track_box[0] + track_box[2]) / 2, (track_box[1] + track_box[3]) / 2]
                box_center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                distance = np.sqrt((track_center[0] - box_center[0])**2 + (track_center[1] - box_center[1])**2)
                
                if distance < min_distance:
                    min_distance = distance
                    min_idx = i
            
            # 如果找到匹配，且距离在阈值内
            if min_idx is not None and min_distance < self.distance_threshold:
                matched_tracks[track_id] = min_idx
                unmatched_tracks.remove(track_id)
                unmatched_boxes.remove(min_idx)
        
        # 更新匹配的跟踪对象
        for track_id, box_idx in matched_tracks.items():
            self.tracks[track_id]['box'] = boxes[box_idx]
            self.tracks[track_id]['disappeared'] = 0
            
        # 更新未匹配的跟踪
        for track_id in unmatched_tracks:
            self.tracks[track_id]['disappeared'] += 1
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                del self.tracks[track_id]
                
        # 为未匹配的框创建新跟踪
        for box_idx in unmatched_boxes:
            self.tracks[self.next_id] = {
                'box': boxes[box_idx],
                'disappeared': 0
            }
            self.next_id += 1
            
        return self.tracks

class VideoBehaviorDetector:
    """检测视频中的可疑盗窃行为特征"""
    
    def __init__(self, config_path=None):
        """初始化视频行为检测器"""
        # 加载配置文件
        self.config = self._load_config(config_path)
        
        # 使用配置中的阈值，如果没有配置则使用默认值
        self.confidence_threshold = self.config.get('confidence_threshold', 0.3)
        self.prev_frame = None  # 添加前一帧属性
        self.prev_gray_frame = None  # 添加前一灰度帧属性，用于运动检测
        self.prev_landmarks = None  # 用于跟踪前一帧的姿态关键点
        self.history_buffer = []  # 历史行为缓冲区
        self.history_size = self.config.get('history_size', 15)  # 历史缓冲区大小
        
        # 加载本地化文本
        self.localization = self._load_localization()
        
        # 初始化行为类型和权重
        self._init_behavior_types()
        
        # 更新进度回调
        self.update_progress_callback = None
        self.frame_processed_callback = None
        
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("video_behavior_detector")
        
        # 检查日志处理器是否已存在
        if not logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # 添加控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # 添加文件处理器
            log_dir = os.path.join("logs")
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "video_behavior.log"), encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        self.logger = logger
        self.logger.info(self.localization.get("initialization_complete", "Video behavior detector initialized"))
        
        # 初始化姿态估计模型 - 移到logger初始化之后
        self.initialize_pose_detector()
        
        # 初始化行为检测模型
        self.standard_model = None
        self.enhanced_model = None
        self.model_type = "enhanced"  # 默认使用增强模型
        self.load_xgboost_models()
        
        # 可能的行为分类模型
        self.behavior_classifier = None
        self.load_behavior_classifier()
        
        # 初始化自适应权重系统
        self.adaptive_weights = {}
        self._initialize_adaptive_weights()
        
        # 用于存储检测到的零售环境信息
        self.retail_environment_detected = False
        self.retail_objects = {}
        
        # 用于跟踪行为连续性
        self.consecutive_behaviors = defaultdict(int)
        
        # 缓存ML模型预测结果
        self.prediction_cache = {}
        
        # 存储过去的帧，用于计算动态特征
        self.frame_history = []
        self.max_frame_history = 5
        
        # 统计模型性能
        self.model_stats = {
            'standard': {'true_positives': 0, 'false_positives': 0, 'predictions': 0},
            'enhanced': {'true_positives': 0, 'false_positives': 0, 'predictions': 0},
            'behavior_classifier': {'true_positives': 0, 'false_positives': 0, 'predictions': 0}
        }
        
        # 人物历史记录
        self.person_history = {}
        
        self.logger.info(f"自适应权重系统{'已启用' if self.config.get('use_adaptive_weights', True) else '已禁用'}")
        self.logger.info(f"零售环境检测{'已启用' if self.config.get('retail_environment', {}).get('enabled', True) else '已禁用'}")
    
    def _load_config(self, config_path=None):
        """加载配置文件"""
        default_config_path = Path("src/config/detector_config.json")
        
        # 尝试使用提供的路径或默认路径
        if config_path is None and default_config_path.exists():
            config_path = default_config_path
        
        # 默认配置
        default_config = {
            "confidence_threshold": 0.3,
            "history_size": 15,
            "use_adaptive_weights": True,
            "learning_rate": 0.1,
            "initial_behavior_weights": {
                "Covering Product Area": 0.5,
                "Unusual Elbow Position": 0.5,
                "Repetitive Position Adjustment": 0.5,
                "Suspected Tag Removal": 0.4,
                "Suspicious Item Handling": 0.7,
                "Rapid Item Concealment": 0.8,
                "Abnormal Arm Position": 0.6,
                "Suspicious Crouching": 0.6,
                "Unusual Reaching": 0.7,
                "Body Shielding": 0.7,
                "Abnormal Head Movement": 0.5,
                "Single Arm Hiding": 0.7,
                "Concealment Gesture": 0.7,
                "Distraction Behavior": 0.8,
                "Group Theft Coordination": 0.8
            },
            "model_weights": {
                "ml_model_weight": 0.6,
                "rule_engine_weight": 0.4
            },
            "detection_parameters": {
                "motion_threshold": 0.01,
                "rapid_motion_threshold": 0.03,
                "concentrated_motion_ratio": 0.2,
                "arm_angle_threshold": 90,
                "wrist_distance_threshold": 100,
                "hip_knee_ratio": 0.15,
                "proximity_threshold": 50,
                "head_angle_threshold": 45,
                "group_distance_threshold": 150,
                "interaction_time_threshold": 3,
                "distraction_movement_threshold": 0.05
            },
            "tracker": {
                "max_disappeared": 30,
                "distance_threshold": 100
            }
        }
        
        config = default_config
        
        # 如果提供了配置文件路径，则尝试加载
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                    # 深度更新配置
                    self._update_config(config, user_config)
                    print(f"已从 {config_path} 加载配置")
            except Exception as e:
                print(f"加载配置文件失败: {e}")
                print("使用默认配置")
        
        return config
    
    def _update_config(self, config, new_config):
        """递归更新配置字典"""
        for key, value in new_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                self._update_config(config[key], value)
            else:
                config[key] = value
    
    def _load_localization(self):
        """加载本地化文本"""
        # 默认英文
        en_localization = {
            "initialization_complete": "Video behavior detector initialized",
            "loading_model_success": "Successfully loaded {} XGBoost model",
            "loading_model_error": "Error loading XGBoost model: {}",
            "model_file_not_exist": "{} XGBoost model file does not exist: {}",
            "no_model_warning": "No XGBoost models loaded, will use rule engine for behavior detection",
            "pose_detector_initialized": "Pose detector initialized successfully",
            "pose_detection_error": "Error in pose detection: {}",
            "behavior_detected": "Detected {}",
            "theft_probability": "Theft probability: {:.2f}",
            "behaviors_title": "Behaviors:",
            "standard_model": "standard",
            "enhanced_model": "enhanced"
        }
        
        # 中文
        zh_localization = {
            "initialization_complete": "视频行为检测器初始化完成",
            "loading_model_success": "成功加载{}XGBoost模型",
            "loading_model_error": "加载XGBoost模型出错: {}",
            "model_file_not_exist": "{}XGBoost模型文件不存在: {}",
            "no_model_warning": "未能加载任何XGBoost模型，将使用规则引擎进行行为检测",
            "pose_detector_initialized": "姿态检测器初始化成功",
            "pose_detection_error": "姿态检测出错: {}",
            "behavior_detected": "检测到{}",
            "theft_probability": "盗窃概率: {:.2f}",
            "behaviors_title": "行为:",
            "standard_model": "标准",
            "enhanced_model": "增强"
        }
        
        # 可以在这里添加更多语言
        
        # 从环境变量或配置中获取语言设置
        lang = os.environ.get('APP_LANGUAGE', 'zh')
        
        if lang == 'zh':
            return zh_localization
        else:
            return en_localization
    
    def _init_behavior_types(self):
        """初始化行为类型及相关配置"""
        # 定义可疑行为类型（英文）
        self.behavior_types = [
            "Covering Product Area", 
            "Unusual Elbow Position",
            "Repetitive Position Adjustment",
            "Suspected Tag Removal",
            "Suspicious Item Handling",
            "Rapid Item Concealment",
            # 添加基于姿态的新行为类型            
            "Abnormal Arm Position",
            "Suspicious Crouching",
            "Unusual Reaching",
            "Body Shielding",
            "Abnormal Head Movement",
            "Single Arm Hiding",
            "Concealment Gesture",
            # 新增加的行为类型
            "Distraction Behavior",
            "Group Theft Coordination"
        ]
        
        # 添加中英文行为类型映射       
        self.behavior_type_map = {
            "Covering Product Area": "遮挡商品区域", 
            "Unusual Elbow Position": "手肘内收姿态异常",
            "Repetitive Position Adjustment": "反复调整位置",
            "Suspected Tag Removal": "疑似撕标签动作",
            "Suspicious Item Handling": "可疑商品处理",
            "Rapid Item Concealment": "快速藏匿物品",
            # 添加基于姿态的新行为类型映射         
            "Abnormal Arm Position": "手臂位置异常",
            "Suspicious Crouching": "可疑蹲姿",
            "Unusual Reaching": "不自然的伸手姿势",
            "Body Shielding": "身体屏蔽姿势",
            "Abnormal Head Movement": "头部异常转动",
            "Single Arm Hiding": "单臂遮挡",
            "Concealment Gesture": "遮掩隐藏手势",
            # 新增加的行为类型映射
            "Distraction Behavior": "分散注意力行为",
            "Group Theft Coordination": "团伙盗窃配合"
        }
        
        # 定义行为类型对应的颜色(BGR格式)
        self.behavior_color_map = {
           "Covering Product Area": (0, 0, 255),    # 红色
            "Unusual Elbow Position": (0, 165, 255),  # 橙色
            "Repetitive Position Adjustment": (0, 255, 0),    # 绿色
            "Suspected Tag Removal": (255, 0, 255),  # 品红
            "Suspicious Item Handling": (255, 255, 0),  # 青色
            "Rapid Item Concealment": (0, 0, 255),     # 蓝色
            # 添加基于姿态的新行为类型颜色            
            "Abnormal Arm Position": (0, 165, 255),    # 橙色
            "Suspicious Crouching": (0, 255, 255),     # 黄色
            "Unusual Reaching": (0, 255, 0),           # 绿色  
            "Body Shielding": (255, 0, 0),             # 蓝色
            "Abnormal Head Movement": (128, 0, 128),   # 紫色
            "Single Arm Hiding": (255, 0, 128),        # 品红            
            "Concealment Gesture": (128, 128, 0),      # 青绿          
            # 新增行为类型颜色
            "Distraction Behavior": (100, 100, 200),   # 淡蓝            
            "Group Theft Coordination": (200, 50, 50)  # 深红    
        }
        
        # 从配置中获取行为权重
        self.behavior_weights = self.config.get('behavior_weights', {})
        
        # 添加行为历史记录
        self.person_history = defaultdict(list)  # 用于跟踪每个人的历史行为
        self.group_interactions = {}  # 用于跟踪多人互动
    
    def _initialize_adaptive_weights(self):
        """初始化自适应权重系统"""
        # 为每种行为类型初始化自适应权重
        initial_weights = self.config.get('initial_behavior_weights', {})
        
        for behavior_type in self.behavior_types:
            # 初始权重来自配置或默认值           
            initial_weight = initial_weights.get(behavior_type, 0.5)
            
            self.adaptive_weights[behavior_type] = {
                'weight': initial_weight,
                'true_positives': 0,
                'false_positives': 0,
                'last_update': time.time(),
                'total_detections': 0,
                'recent_accuracy': 0.5  # 初始准确率假设为0.5
            }
    
    def update_adaptive_weight(self, behavior_type, is_correct, confidence_factor=1.0):
        """更新自适应权重
        
        Args:
            behavior_type: 行为类型
            is_correct: 检测是否正确
            confidence_factor: 置信度因子, 影响权重调整速度
        """
        if behavior_type not in self.adaptive_weights or not self.config.get('use_adaptive_weights', True):
            return
            
        # 更新计数
        if is_correct:
            self.adaptive_weights[behavior_type]['true_positives'] += 1
        else:
            self.adaptive_weights[behavior_type]['false_positives'] += 1
            
        self.adaptive_weights[behavior_type]['total_detections'] += 1
            
        # 计算准确率
        tp = self.adaptive_weights[behavior_type]['true_positives']
        fp = self.adaptive_weights[behavior_type]['false_positives']
        total = self.adaptive_weights[behavior_type]['total_detections']
        
        # 计算整体准确率和近期准确率
        overall_accuracy = tp / total if total > 0 else 0.5
        
        # 根据最近的表现计算加权准确率
        recent_weight = 0.7  # 最近表现的权重
        recent_accuracy = self.adaptive_weights[behavior_type]['recent_accuracy']
        new_recent_accuracy = recent_accuracy * (1 - recent_weight) + (1 if is_correct else 0) * recent_weight
        self.adaptive_weights[behavior_type]['recent_accuracy'] = new_recent_accuracy
        
        # 结合整体准确率和近期准确率
        combined_accuracy = overall_accuracy * 0.3 + new_recent_accuracy * 0.7
        
        # 获取学习率
        learning_rate = self.config.get('learning_rate', 0.1) * confidence_factor
        
        # 调整权重 (指数移动平均)
        current_weight = self.adaptive_weights[behavior_type]['weight']
        new_weight = current_weight * (1 - learning_rate) + combined_accuracy * learning_rate
        
        # 确保权重在合理范围内
        new_weight = max(0.1, min(0.95, new_weight))
        
        # 更新权重
        self.adaptive_weights[behavior_type]['weight'] = new_weight
        self.adaptive_weights[behavior_type]['last_update'] = time.time()
        
        # 如果是零售环境相关行为, 根据环境检测结果进行额外调整
        if self.retail_environment_detected and behavior_type in self.config.get('retail_environment', {}).get('boost_behaviors', []):
            boost = self.config.get('retail_environment', {}).get('confidence_boost', 0.2)
            new_weight = min(0.95, new_weight + boost * (1 - new_weight))
            self.adaptive_weights[behavior_type]['weight'] = new_weight
        
        self.logger.debug(f"更新权重 {behavior_type}: {new_weight:.3f} (TP:{tp}, FP:{fp}, 准确率:{combined_accuracy:.3f})")
    
    def get_behavior_weight(self, behavior_type):
        """获取行为类型的当前权重
        
        Args:
            behavior_type: 行为类型
            
        Returns:
            weight: 当前权重
        """
        # 如果有自适应权重，使用自适应权重
        if behavior_type in self.adaptive_weights and self.config.get('use_adaptive_weights', True):
            return self.adaptive_weights[behavior_type]['weight']
            
        # 否则使用配置中的权重或默认值
        initial_weights = self.config.get('initial_behavior_weights', {})
        return initial_weights.get(behavior_type, 0.5)
        
    def load_xgboost_models(self):
        """加载XGBoost模型用于盗窃行为分类"""
        try:
            # 尝试加载标准模型
            standard_model_path = Path("models/standard/theft_xgb_model.pkl")
            alt_standard_path = Path("models/theft_xgb_model.pkl")
            std_path = standard_model_path if standard_model_path.exists() else alt_standard_path
            if std_path.exists():
                try:
                    with open(std_path, 'rb') as f:
                        loaded = pickle.load(f)
                    # 处理 dict 格式 {'model': XGBClassifier, 'label_encoder': ...}
                    if isinstance(loaded, dict) and 'model' in loaded:
                        loaded = loaded['model']
                    # 检查是否有 predict 方法（训练过的模型才有）
                    if hasattr(loaded, 'predict'):
                        self.standard_model = loaded
                        self.logger.info(self.localization["loading_model_success"].format(
                            self.localization["standard_model"]))
                    else:
                        self.logger.debug(f"标准模型无predict方法(未训练), 跳过: {std_path}")
                        self.standard_model = None
                except Exception as e:
                    self.logger.debug(f"标准模型加载失败: {e}")
                    self.standard_model = None
            else:
                self.logger.debug(self.localization["model_file_not_exist"].format(
                    self.localization["standard_model"], std_path))
            
            # 尝试加载增强模型
            enhanced_model_path = Path("models/enhanced/enhanced_theft_xgb_model.pkl")
            if enhanced_model_path.exists():
                try:
                    with open(enhanced_model_path, 'rb') as f:
                        loaded = pickle.load(f)
                    if isinstance(loaded, dict) and 'model' in loaded:
                        loaded = loaded['model']
                    if hasattr(loaded, 'predict'):
                        self.enhanced_model = loaded
                        self.logger.info(self.localization["loading_model_success"].format(
                            self.localization["enhanced_model"]))
                except Exception as e:
                    self.logger.debug(f"增强模型加载失败: {e}")
                    self.enhanced_model = None
            else:
                self.logger.debug(self.localization["model_file_not_exist"].format(
                    self.localization["enhanced_model"], enhanced_model_path))
            
            # 确定使用哪个模型
            if self.enhanced_model:
                self.model_type = "enhanced"
            elif self.standard_model:
                self.model_type = "standard"
            else:
                self.model_type = None
                self.logger.warning(self.localization["no_model_warning"])
        
        except Exception as e:
            self.logger.error(self.localization["loading_model_error"].format(str(e)))
            self.logger.error(traceback.format_exc())
            self.model_type = None
    
    def predict_with_xgboost(self, features, behavior_context=None):
        """使用XGBoost模型预测行为
        
        Args:
            features: 特征向量
            behavior_context: 行为上下文信息
            
        Returns:
            prediction: 预测结果 (0: 正常, 1: 盗窃)
            probability: 盗窃行为的概率
        """
        if features is None:
            return 0, 0.0
        
        # 检查是否使用增强特征，如果是但模型不支持，则转换为标准特征
        is_enhanced = len(features) > 8
        
        # 计算特征的缓存键
        feature_key = hash(str(features))
        
        # 检查缓存中是否有结果
        if feature_key in self.prediction_cache:
            cached_result = self.prediction_cache[feature_key]
            # 如果缓存结果还新鲜（不超过1秒），直接返回
            if time.time() - cached_result['timestamp'] < 1.0:
                return cached_result['prediction'], cached_result['probability']
        
        # 进行预测
        if self.model_type == "enhanced" and self.enhanced_model and is_enhanced:
            # 使用增强模型
            try:
                prediction = self.enhanced_model.predict([features])[0]
                probability = self.enhanced_model.predict_proba([features])[0][1]
                
                # 更新模型统计
                self.model_stats['enhanced']['predictions'] += 1
                
                # 更新预测缓存
                self.prediction_cache[feature_key] = {
                    'prediction': prediction,
                    'probability': probability,
                    'timestamp': time.time()
                }
                
                return prediction, probability
            except Exception as e:
                self.logger.error(f"增强模型预测错误: {e}")
                # 回退到标准模型
                if self.standard_model:
                    self.model_type = "standard"
                    # 如果是增强特征，转换为标准特征
                    if is_enhanced:
                        features = self._convert_to_standard_features(features)
        
        # 使用标准模型
        if self.model_type == "standard" and self.standard_model:
            try:
                # 确保特征是标准格式
                if is_enhanced:
                    features = self._convert_to_standard_features(features)
                
                prediction = self.standard_model.predict([features])[0]
                probability = self.standard_model.predict_proba([features])[0][1]
                
                # 更新模型统计
                self.model_stats['standard']['predictions'] += 1
                
                # 缓存预测结果
                self.prediction_cache[feature_key] = {
                    'prediction': prediction,
                    'probability': probability,
                    'timestamp': time.time()
                }
                
                return prediction, probability
            except Exception as e:
                self.logger.error(f"标准模型预测错误: {e}")
        
        # 如果没有可用的模型或者预测失败
        return 0, 0.0
    
    def _convert_to_standard_features(self, enhanced_features):
        """将增强特征转换为标准特征
        
        Args:
            enhanced_features: 增强特征向量
            
        Returns:
            standard_features: 标准特征向量
        """
        # 标准特征是增强特征的子集
        if len(enhanced_features) >= 8:
            return enhanced_features[:8]
        return enhanced_features
    
    def combine_predictions(self, ml_prediction, rule_prediction, context=None):
        """智能组合ML模型和规则引擎的预测结果
        
        Args:
            ml_prediction: ML模型预测结果 (预测值, 概率)
            rule_prediction: 规则引擎预测结果 (预测值, 概率)
            context: 上下文信息, 例如检测到的行为、环境等
            
        Returns:
            final_prediction: 组合后的预测值(0: 正常, 1: 盗窃)
            final_probability: 组合后的概率
        """
        ml_pred, ml_prob = ml_prediction
        rule_pred, rule_prob = rule_prediction
        
        # 优化模型权重 - 增加ML权重，减少规则引擎权重
        ml_weight = self.config.get('model_weights', {}).get('ml_model_weight', 0.75)  # 提高ML权重
        rule_weight = self.config.get('model_weights', {}).get('rule_engine_weight', 0.25)  # 降低规则权重
        
        # 如果ML预测为盗窃且概率较高，进一步增强其权重
        if ml_pred == 1 and ml_prob > 0.65:
            ml_weight += 0.1
            rule_weight = max(0.1, rule_weight - 0.1)
            
            # 重新归一化权重
            total = ml_weight + rule_weight
            ml_weight /= total
            rule_weight /= total
        
        # 计算加权概率
        weighted_prob = ml_prob * ml_weight + rule_prob * rule_weight
        
        # 如果有连续性行为，增加概率
        if context and 'consecutive_behaviors' in context:
            for behavior, count in context['consecutive_behaviors'].items():
                if count >= 3:  # 如果某个行为连续出现3次以上                 
                    # 增加概率，但不超0.95
                    weighted_prob = min(0.95, weighted_prob + 0.05 * (count - 2))
        
        # 根据加权概率确定最终预判
        final_pred = 1 if weighted_prob >= 0.5 else 0
        
        return final_pred, weighted_prob
    
    def detect_retail_environment(self, detections):
        """检测图像是否为零售/超市环境
        
        Args:
            detections: 检测结果            
        Returns:
            is_retail: 是否检测到零售环境
        """
        if not self.config.get('retail_environment', {}).get('enabled', True):
            return False
            
        try:
            # 获取零售环境所需物体
            required_objects = self.config.get('retail_environment', {}).get('required_objects', 
                                                                          ["shelf", "product", "display"])
            
            # 从检测结果中提取类别
            detected_classes = []
            if hasattr(detections, 'pandas') and callable(getattr(detections, 'pandas')):
                df = detections.pandas().xyxy[0]
                detected_classes = df['name'].tolist()
            
            # 检查是否检测到零售环境所需物体
            detected_retail_objects = {}
            for obj_class in detected_classes:
                lower_class = obj_class.lower()
                # 检查是否包含零售相关词
                for retail_obj in required_objects:
                    if retail_obj in lower_class:
                        if retail_obj not in detected_retail_objects:
                            detected_retail_objects[retail_obj] = 0
                        detected_retail_objects[retail_obj] += 1
            
            # 如果检测到至少一个所需物体，认为是零售环境
            is_retail = len(detected_retail_objects) > 0
            
            # 更新零售环境检测结果
            if is_retail:
                self.retail_environment_detected = True
                self.retail_objects.update(detected_retail_objects)
                self.logger.info(self.localization.get("retail_environment_detected", "零售环境已检测到"))
            
            return is_retail
            
        except Exception as e:
            self.logger.error(f"零售环境检测错误: {e}")
            return False
    
    def _extract_standard_features(self, landmarks):
        """提取标准特征
        
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
                left_wrist = (landmarks[15][0], landmarks[15][1])
                left_hip = (landmarks[23][0], landmarks[23][1])
                right_hip = (landmarks[24][0], landmarks[24][1])
                hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                
                left_wrist_dist = math.sqrt((left_wrist[0] - hip_center[0]) ** 2 + 
                                         (left_wrist[1] - hip_center[1]) ** 2)
            
            # 右手腕(16)和髋部中点的距离
            if 16 in landmarks and 23 in landmarks and 24 in landmarks:
                right_wrist = (landmarks[16][0], landmarks[16][1])
                left_hip = (landmarks[23][0], landmarks[23][1])
                right_hip = (landmarks[24][0], landmarks[24][1])
                hip_center = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
                
                right_wrist_dist = math.sqrt((right_wrist[0] - hip_center[0]) ** 2 + 
                                          (right_wrist[1] - hip_center[1]) ** 2)
            
            features.append(left_wrist_dist)
            features.append(right_wrist_dist)
            
            # 2. 计算躯干高度
            torso_height = 0
            if 11 in landmarks and 24 in landmarks:
                # 使用右肩(11)和右髋(24)计算躯干高度
                torso_height = abs(landmarks[11][1] - landmarks[24][1])
            
            features.append(torso_height)
            
            # 3. 计算左右手臂角度
            left_arm_angle = 0
            right_arm_angle = 0
            
            # 左手臂角度：使用左肩(11)、左肘(13)和左手腕(15)
            if 11 in landmarks and 13 in landmarks and 15 in landmarks:
                shoulder = (landmarks[11][0], landmarks[11][1])
                elbow = (landmarks[13][0], landmarks[13][1])
                wrist = (landmarks[15][0], landmarks[15][1])
                
                # 计算肩到肘的向量
                se_vector = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
                # 计算肘到手腕的向量
                ew_vector = (wrist[0] - elbow[0], wrist[1] - elbow[1])
                
                # 计算两个向量之间的角
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
                shoulder = (landmarks[12][0], landmarks[12][1])
                elbow = (landmarks[14][0], landmarks[14][1])
                wrist = (landmarks[16][0], landmarks[16][1])
                
                # 计算肩到肘的向量
                se_vector = (elbow[0] - shoulder[0], elbow[1] - shoulder[1])
                # 计算肘到手腕的向量
                ew_vector = (wrist[0] - elbow[0], wrist[1] - elbow[1])
                
                # 计算两个向量之间的角
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
            
            # 4. 计算手腕与髋部比值
            wrist_hip_ratio_left = 0
            wrist_hip_ratio_right = 0
            
            # 计算左手腕(15)和左髋(23)的比值
            if 15 in landmarks and 23 in landmarks and 24 in landmarks:
                wrist_y = landmarks[15][1]
                left_hip_y = landmarks[23][1]
                right_hip_y = landmarks[24][1]
                hip_center_y = (left_hip_y + right_hip_y) / 2
                
                # 比例：手腕Y坐标与髋部中点的差值，除以躯干高度（如果有）
                if torso_height > 0:
                    wrist_hip_ratio_left = (wrist_y - hip_center_y) / torso_height
            
            # 计算右手腕(16)和右髋(24)的比值
            if 16 in landmarks and 23 in landmarks and 24 in landmarks:
                wrist_y = landmarks[16][1]
                left_hip_y = landmarks[23][1]
                right_hip_y = landmarks[24][1]
                hip_center_y = (left_hip_y + right_hip_y) / 2
                
                # 比例：手腕Y坐标与髋部中点的差值，除以躯干高度（如果有）
                if torso_height > 0:
                    wrist_hip_ratio_right = (wrist_y - hip_center_y) / torso_height
            
            features.append(wrist_hip_ratio_left)
            features.append(wrist_hip_ratio_right)
            
            # 5. 计算手臂交叉状态
            arms_crossed = 0
            if 13 in landmarks and 14 in landmarks and 15 in landmarks and 16 in landmarks:
                left_elbow_x = landmarks[13][0]
                right_elbow_x = landmarks[14][0]
                left_wrist_x = landmarks[15][0]
                right_wrist_x = landmarks[16][0]
                
                # 判断手臂是否交叉
                # 交叉条件：左手腕在右手肘的右侧，或右手腕在左手肘的左侧
                if (left_wrist_x > right_elbow_x) or (right_wrist_x < left_elbow_x):
                    arms_crossed = 1
            
            features.append(arms_crossed)
            
            # 6. 躯干旋转（肩膀和髋部的水平对齐程度）
            torso_rotation = 0
            if 11 in landmarks and 12 in landmarks and 23 in landmarks and 24 in landmarks:
                # 计算肩膀线的角度
                shoulder_angle = math.atan2(landmarks[12][1] - landmarks[11][1], 
                                           landmarks[12][0] - landmarks[11][0])
                # 计算髋部线的角度
                hip_angle = math.atan2(landmarks[24][1] - landmarks[23][1], 
                                      landmarks[24][0] - landmarks[23][0])
                # 两条线之间的角度差异（弧度）
                angle_diff = abs(shoulder_angle - hip_angle)
                # 转换为角度
                torso_rotation = math.degrees(angle_diff)
            
            features.append(torso_rotation)
            
            # 7. 肩膀宽度
            shoulder_width = 0
            if 11 in landmarks and 12 in landmarks:
                # 计算肩膀宽度
                shoulder_width = math.sqrt((landmarks[12][0] - landmarks[11][0])**2 + 
                                          (landmarks[12][1] - landmarks[11][1])**2)
            
            features.append(shoulder_width)
            
            # 8. 头部位置
            head_position = 0
            if 0 in landmarks and 11 in landmarks and 12 in landmarks:
                # 头部(0)相对于肩膀中点的位置
                shoulder_center_x = (landmarks[11][0] + landmarks[12][0]) / 2
                head_x = landmarks[0][0]
                # 头部水平偏移
                head_position = head_x - shoulder_center_x
            
            features.append(head_position)
            
            # 9. 膝盖与胯部的关系
            knee_hip_ratio = 0
            if 25 in landmarks and 26 in landmarks and 23 in landmarks and 24 in landmarks:
                # 计算膝盖中点和胯部中点的位置
                knee_center_y = (landmarks[25][1] + landmarks[26][1]) / 2
                hip_center_y = (landmarks[23][1] + landmarks[24][1]) / 2
                # 膝盖与胯部的距离
                knee_hip_distance = abs(knee_center_y - hip_center_y)
                # 与躯干高度的比例
                if torso_height > 0:
                    knee_hip_ratio = knee_hip_distance / torso_height
            
            features.append(knee_hip_ratio)
            
            # 标准化特征（可选）
            # 这一步可以提高模型的泛化能力
            # 但需要确保训练和推理时使用相同的标准化方法
            return features
            
        except Exception as e:
            self.logger.error(f"提取标准特征出错: {e}")
            return None
    
    def _extract_enhanced_features(self, landmarks, prev_landmarks=None):
        """提取增强特征（动态特征）
        
        Args:
            landmarks: 当前帧的姿态关键点
            prev_landmarks: 前一帧的姿态关键点（可选）
            
        Returns:
            features: 增强特征向量
        """
        # 首先提取标准特征（静态特征）
        std_features = self._extract_standard_features(landmarks)
        
        if std_features is None:
            return None
            
        # 如果没有前一帧关键点，只返回标准特征
        if prev_landmarks is None:
            # 添加动态特征的默认值（6个动态特征）
            dynamic_features = [0.0] * 6  # 6个动态特征
            return std_features + dynamic_features
        
        try:
            # 提取动态特征
            dynamic_features = []
            
            # 1. 左右手腕移动程度
            left_wrist_movement = 0.0
            right_wrist_movement = 0.0
            
            if 15 in landmarks and 15 in prev_landmarks:
                left_wrist_curr = (landmarks[15][0], landmarks[15][1])
                left_wrist_prev = (prev_landmarks[15][0], prev_landmarks[15][1])
                left_wrist_movement = math.sqrt(
                    (left_wrist_curr[0] - left_wrist_prev[0])**2 + 
                    (left_wrist_curr[1] - left_wrist_prev[1])**2
                )
            
            if 16 in landmarks and 16 in prev_landmarks:
                right_wrist_curr = (landmarks[16][0], landmarks[16][1])
                right_wrist_prev = (prev_landmarks[16][0], prev_landmarks[16][1])
                right_wrist_movement = math.sqrt(
                    (right_wrist_curr[0] - right_wrist_prev[0])**2 + 
                    (right_wrist_curr[1] - right_wrist_prev[1])**2
                )
            
            dynamic_features.append(left_wrist_movement)
            dynamic_features.append(right_wrist_movement)
            
            # 2. 躯干旋转角度变化
            torso_rotation_change = 0.0
            # 计算躯干旋转 - 使用肩膀和髋部的相对位置
            if all(i in landmarks and i in prev_landmarks for i in [11, 12, 23, 24]):
                # 当前帧肩膀中心位置
                curr_shoulder_center_x = (landmarks[11][0] + landmarks[12][0]) / 2
                # 前一帧肩膀中心位置
                prev_shoulder_center_x = (prev_landmarks[11][0] + prev_landmarks[12][0]) / 2
                
                # 当前帧髋部中心点位置
                curr_hip_center_x = (landmarks[23][0] + landmarks[24][0]) / 2
                # 前一帧髋部中心点位置
                prev_hip_center_x = (prev_landmarks[23][0] + prev_landmarks[24][0]) / 2
                
                # 计算旋转变化
                curr_rotation = curr_shoulder_center_x - curr_hip_center_x
                prev_rotation = prev_shoulder_center_x - prev_hip_center_x
                torso_rotation_change = abs(curr_rotation - prev_rotation)
            
            dynamic_features.append(torso_rotation_change)
            
            # 3. 头部移动
            head_movement = 0.0
            if 0 in landmarks and 0 in prev_landmarks:
                head_curr = (landmarks[0][0], landmarks[0][1])
                head_prev = (prev_landmarks[0][0], prev_landmarks[0][1])
                head_movement = math.sqrt(
                    (head_curr[0] - head_prev[0])**2 + 
                    (head_curr[1] - head_prev[1])**2
                )
            
            dynamic_features.append(head_movement)
            
            # 4. 肩膀宽度变化
            shoulder_width_change = 0.0
            if 11 in landmarks and 12 in landmarks and 11 in prev_landmarks and 12 in prev_landmarks:
                curr_shoulder_width = abs(landmarks[11][0] - landmarks[12][0])
                prev_shoulder_width = abs(prev_landmarks[11][0] - prev_landmarks[12][0])
                shoulder_width_change = abs(curr_shoulder_width - prev_shoulder_width)
            
            dynamic_features.append(shoulder_width_change)
            
            # 5. 躯干高度变化
            torso_height_change = 0.0
            if 11 in landmarks and 24 in landmarks and 11 in prev_landmarks and 24 in prev_landmarks:
                # 使用右肩(11)和右髋(24)计算躯干高度
                curr_torso_height = abs(landmarks[11][1] - landmarks[24][1])
                prev_torso_height = abs(prev_landmarks[11][1] - prev_landmarks[24][1])
                torso_height_change = abs(curr_torso_height - prev_torso_height)
            
            dynamic_features.append(torso_height_change)
            
            # 合并标准特征和动态特征
            return std_features + dynamic_features
            
        except Exception as e:
            self.logger.error(f"提取增强特征出错: {e}")
            return std_features + [0.0] * 6  # 出错时返回标准特征，默认动态特征
    
    def initialize_pose_detector(self):
        """初始化姿态检测器"""
        try:
            import mediapipe as mp
            
            # 初始化绘图组件
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mp_pose = mp.solutions.pose
            
            # 创建姿态检测器实例
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,  # 视频模式
                model_complexity=1,      # 中等复杂度
                smooth_landmarks=True,   # 平滑关键点
                enable_segmentation=False, # 关闭分割以提高性能
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.logger.info("姿态检测器初始化成功")
            
        except ImportError:
            self.logger.error("无法导入MediaPipe，请确保已安装")
            self.pose_detector = None
            
        except Exception as e:
            self.logger.error(f"初始化姿态检测器时出错: {str(e)}")
            traceback.print_exc()
            self.pose_detector = None
    
    def detect_behaviors_in_image(self, image, detections):
        """
        检测单帧图像中的可疑行为
        
        Args:
            image: 输入图像
            detections: 检测到的人员和物品
            
        Returns:
            behaviors: 检测到的可疑行为列表
        """
        behaviors = []
        
        # 使用规则引擎检测行为
        rule_behaviors = self._detect_behaviors_with_rules(image, detections)
        if rule_behaviors:
            behaviors.extend(rule_behaviors)
        
        # 使用姿态估计和ML模型检测行为
        if self.pose_detector:
            pose_results = self._extract_pose_landmarks(image)
            if pose_results and 'landmarks' in pose_results:
                # 关键修复：把 list 格式转换成 dict 格式（下游方法期望 dict with integer keys）
                landmarks_dict = self._convert_to_landmarks_dict(pose_results['landmarks'])
                
                if landmarks_dict and len(landmarks_dict) >= 12:
                    # 如果有pose数据，使用XGBoost模型进行预测
                    if self.model_type:
                        if self.model_type == "enhanced":
                            prev_dict = self._convert_to_landmarks_dict(
                                self.prev_landmarks['landmarks'] if self.prev_landmarks and isinstance(self.prev_landmarks, dict) and 'landmarks' in self.prev_landmarks else None
                            ) if self.prev_landmarks else None
                            features = self._extract_enhanced_features(landmarks_dict, prev_dict)
                        else:
                            features = self._extract_standard_features(landmarks_dict)
                        
                        if features:
                            prediction, probability = self.predict_with_xgboost(features)
                            if prediction == 1 and probability > 0.6:
                                behaviors.append({
                                    "type": "ML Detected Theft",
                                    "confidence": float(probability),
                                    "position": "Full Frame",
                                    "bbox": [0, 0, image.shape[1], image.shape[0]],
                                    "label": f"盗窃行为(置信度{probability:.2f})"
                                })
                    
                    # 保存当前帧的pose_results用于下一帧（保留原始格式）
                    self.prev_landmarks = pose_results
                    
                    # 使用基于规则的姿态行为检测（传 dict 格式）
                    ml_behaviors = self._detect_pose_based_behaviors(image, landmarks_dict, None, detections)
                    if ml_behaviors:
                        behaviors.extend(ml_behaviors)
        
        return behaviors
    
    def _detect_behaviors_with_rules(self, image, detections):
        """
        使用基本规则检测行为（不依赖于姿态关键点）
        
        Args:
            image: 输入图像
            detections: 检测结果
            
        Returns:
            behaviors: 检测到的行为列表
        """
        behaviors = []
        
        try:
            # 检测零售环境
            is_retail = self.detect_retail_environment(detections)
            
            # 分离人物和物体检测
            person_detections = []
            object_detections = []
            
            if hasattr(detections, 'xyxy'):
                # 适配ultralytics格式
                for i in range(len(detections.xyxy)):
                    class_id = int(detections.cls[i])
                    class_name = detections.names.get(class_id, "unknown")
                    
                    if class_name.lower() == 'person':
                        person_detections.append({
                            'class': 'person',
                            'confidence': float(detections.conf[i]),
                            'bbox': detections.xyxy[i].tolist()
                        })
                    else:
                        object_detections.append({
                            'class': class_name,
                            'confidence': float(detections.conf[i]),
                            'bbox': detections.xyxy[i].tolist()
                        })
                        
            elif isinstance(detections, list):
                # 适配列表格式
                for d in detections:
                    if isinstance(d, dict) and 'class' in d:
                        if d['class'].lower() == 'person':
                            person_detections.append(d)
                        else:
                            object_detections.append(d)
            
            # 分析人物位置和行为
            person_count = len(person_detections)
            object_count = len(object_detections)
            
            # 检测人物是否存在
            if person_count > 0:
                # 重要修改：移除商品检测依赖，即使没有检测到商品也继续分析人物行为
                # 之前代码在这里判断 object_count > 0，如果没有物品就不进行分析
                
                # 记录基本信息
                self.logger.info(f"人物检测: {person_count > 0}, 商品检测: {object_count > 0}")
                
                # 即使没有检测到商品，也继续分析人物行为
                if True:  # 原条件为 object_count > 0
                    # 检测基本行为
                    # TODO: 添加更多规则检测
                    
                    # 简单的逗留行为检测
                    if hasattr(self, '_last_person_positions'):
                        for person in person_detections:
                            if 'bbox' in person:
                                bbox = person['bbox']
                                person_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                                
                                # 检查是否与之前位置接近（表示逗留）
                                for last_pos in self._last_person_positions:
                                    dist = math.sqrt((person_center[0] - last_pos[0])**2 + (person_center[1] - last_pos[1])**2)
                                    if dist < 50:  # 小距离移动
                                        self._loitering_count += 1
                                        if self._loitering_count > 10:  # 持续逗留
                                            behavior = {
                                                'type': 'loitering',
                                                'confidence': min(0.5 + 0.02 * self._loitering_count, 0.9),
                                                'bbox': bbox,
                                                'color': (0, 0, 255)  # 红色
                                            }
                                            behaviors.append(behavior)
                                        break
                                else:
                                    # 如果没有与任何上一个位置接近，重置计数
                                    self._loitering_count = max(0, self._loitering_count - 1)
                                
                                # 更新位置历史
                                self._last_person_positions.append(person_center)
                                if len(self._last_person_positions) > 30:  # 保留最近30帧
                                    self._last_person_positions.pop(0)
                    else:
                        # 初始化位置历史
                        self._last_person_positions = []
                        self._loitering_count = 0
                    
                    # 分析人物与商品的互动
                    if object_count > 0:
                        for person in person_detections:
                            if 'bbox' not in person:
                                continue
                                
                            p_bbox = person['bbox']
                            person_center = ((p_bbox[0] + p_bbox[2]) / 2, (p_bbox[1] + p_bbox[3]) / 2)
                            
                            for obj in object_detections:
                                if 'bbox' not in obj:
                                    continue
                                    
                                o_bbox = obj['bbox']
                                obj_center = ((o_bbox[0] + o_bbox[2]) / 2, (o_bbox[1] + o_bbox[3]) / 2)
                                
                                # 计算人与物品的距离
                                distance = math.sqrt(
                                    (person_center[0] - obj_center[0])**2 + 
                                    (person_center[1] - obj_center[1])**2
                                )
                                
                                # 如果人与物品距离很近，可能正在互动
                                if distance < 100:
                                    behavior = {
                                        'type': 'product_interaction',
                                        'confidence': 0.6,
                                        'details': f"与{obj.get('class', 'object')}互动",
                                        'bbox': p_bbox,
                                        'color': (0, 255, 0)  # 绿色
                                    }
                                    behaviors.append(behavior)
                else:
                    self.logger.info(f"缺少盗窃关键元素: 人物存在={person_count > 0}, 商品存在={object_count > 0}")
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"基于规则的行为检测出错: {str(e)}")
            traceback.print_exc()
            return behaviors
    
    def analyze_video_behavior(self, video_path, detector, callback=None, frame_callback=None):
        """
        分析视频中的可疑行为
        
        Args:
            video_path: 视频路径
            detector: 检测器对象
            callback: 进度回调函数
            frame_callback: 帧处理回调函数
            
        Returns:
            output_path: 处理后的视频路径
            suspicious_frames: 可疑帧列表
            behaviors: 检测到的可疑行为列表[(帧索引, 行为), ...]
        """
        # 设置回调函数
        if callback:
            self.update_progress_callback = callback
        if frame_callback:
            self.frame_processed_callback = frame_callback
        
        # 创建输出目录
        output_dir = os.path.join("static", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建输出文件名- 使用正斜杠拼接路径，避免生成反斜杠路径
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 使用正斜杠拼接路径，避免生成反斜杠路径
        output_base = "static/output/result_" + current_time
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            error_msg = f"无法打开视频文件: {video_path}"
            self.logger.error(error_msg)
            return None, [], []
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 总帧数")
        
        # 创建视频输出
        try:
            # 优先使用MP4格式编解码器，避免后续转换步骤          
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = f"{output_base}.mp4"
            
            # 初始化VideoWriter
            out = cv2.VideoWriter(
                output_path, 
                fourcc, 
                fps, 
                (width, height), 
                isColor=True
            )
            
            if not out.isOpened():
                # 如果MP4V失败，尝试MJPG (AVI格式)
                self.logger.warning("MP4V编解码器初始化失败，尝试MJPG")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out.release()  # 释放之前的资源
                output_temp = f"{output_base}.avi"
                out = cv2.VideoWriter(output_temp, fourcc, fps, (width, height))
                output_path = output_temp
        except Exception as e:
            self.logger.error(f"创建视频编写器时出错: {str(e)}")
            cap.release()
            return None, [], []
        
        if not out.isOpened():
            self.logger.error("无法创建输出视频文件")
            cap.release()
            return None, [], []
        
        # 跟踪相关变量
        frame_index = 0
        suspicious_frames = []
        all_behaviors = []
        
        # 历史帧和光流计算
        prev_gray = None
        flow_history = []
        
        # 可疑度历史记录
        theft_probability_history = []
        
        # 行为次数统计
        behavior_count = {behavior: 0 for behavior in self.behavior_types}
        
        # 上一帧的关键点数据
        prev_landmarks = None
        
        # 初始化手臂和头部历史位置追踪
        self._head_position_history = []
        self._left_wrist_history = []
        self._right_wrist_history = []
        
        # 初始化行为检测计数器
        no_detection_count = 0
        
        try:
            while True:
                # 读取帧
                ret, frame = cap.read()
                if not ret:
                    self.logger.info(f"视频处理完成，总共处理{frame_index}帧")
                    break
                
                # 计算总体进度和更新UI（确保进度条显示整体处理进度）
                progress = min(100, int(100 * frame_index / total_frames))
                if self.update_progress_callback and frame_index % 5 == 0:
                    # 使用回调函数更新整体进度
                    progress_text = f"正在处理视频 {progress}% - {frame_index}/{total_frames}"
                    self.update_progress_callback(progress, progress_text)
                    
                # 计算时间
                frame_time = frame_index / fps if fps > 0 else 0
                time_str = time.strftime('%H:%M:%S', time.gmtime(frame_time)) + f'.{int((frame_time % 1) * 100):02d}'
                
                # 执行检测 - 优化检测频率
                detections = None
                flow_data = None
                behaviors = []
                theft_probability = 0.0
                
                # 降低处理间隔，视频中每帧都处理，以确保不错过关键动作
                if frame_index % 1 == 0:  # 每帧检测
                    # 使用传入的检测器检测
                    try:
                        # 创建检测器适配函数
                        def get_detection_result(frame):
                            if hasattr(detector, 'detect_theft') and callable(detector.detect_theft):
                                return detector.detect_theft(frame)
                            elif hasattr(detector, 'detect') and callable(detector.detect):
                                detections = detector.detect(frame)
                                # 根据检测结果计算更合理的盗窃概率                          
                                if detections and len(detections) > 0:
                                    # 基于检测对象数量和置信度计算概率                                    
                                    person_count = sum(1 for d in detections 
                                        if hasattr(d, 'class_name') and d.class_name == "person")
                                    avg_confidence = sum(getattr(d, 'confidence', 0.5) for d in detections) / len(detections)
                                    # 人数和置信度共同决定盗窃可能
                                    theft_probability = min(0.4 + 0.1 * person_count + 0.2 * avg_confidence, 0.9)
                                else:
                                    theft_probability = 0.1  # 没有检测到物体时低概率
                                return detections, theft_probability
                            elif hasattr(detector, 'detect_objects') and callable(detector.detect_objects):
                                detections = detector.detect_objects(frame)
                                # 动态计算盗窃概率                                
                                if detections and len(detections) > 0:
                                    # 获取所有检测对象的类别
                                    object_classes = [getattr(d, 'class', getattr(d, 'class_name', '')) for d in detections]
                                    person_present = any('person' in str(cls).lower() for cls in object_classes)
                                    
                                    # 根据检测对象和置信度动态计算概率                                    
                                    if person_present:
                                        # 人在场景中时，基于检测数量和置信度计算
                                        avg_confidence = sum(getattr(d, 'confidence', 0.5) for d in detections) / len(detections)
                                        # 随机因素使不同帧的概率有所变化
                                        import random
                                        theft_probability = 0.4 + 0.3 * avg_confidence + random.uniform(-0.1, 0.1)
                                        theft_probability = max(0.1, min(0.9, theft_probability))  # 确保在合理范围内
                                    else:
                                        theft_probability = 0.2  # 没有人时低概率
                                else:
                                    theft_probability = 0.1  # 没有检测到物体时低概率
                                return detections, theft_probability
                            else:
                                self.logger.warning("检测器对象没有合适的检测方法")
                                return None, 0.0
                            
                        # 执行检测
                        detections, theft_probability = get_detection_result(frame)
                        
                        # 执行姿态估计 - 更频繁地执行姿态估计
                        pose_results = None
                        if self.pose_detector is not None and frame_index % 2 == 0:  # 每2帧进行一次姿态估计
                            pose_results = self._extract_pose_landmarks(frame)
                        
                    except Exception as e:
                        self.logger.error(f"检测过程中出错: {str(e)}")
                        traceback.print_exc()
                        
                    # 计算光流（增加频率）
                    if frame_index % 3 == 0:  # 每3帧计算一次光流
                        try:
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            if prev_gray is not None:
                                flow = cv2.calcOpticalFlowFarneback(
                                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                                
                                # 计算光流大小和方向
                                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                                
                                # 存储光流数据
                                flow_data = {
                                    'magnitude': magnitude,
                                    'angle': angle,
                                    'average_motion': np.mean(magnitude)
                                }
                            prev_gray = gray.copy()
                        except Exception as e:
                            self.logger.error(f"光流计算错误: {str(e)}")
                    
                    # 检测行为 - 视频专用行为检测逻辑 (类似图片处理但独立实现)
                    try:
                        if detections:
                            # 检测零售环境
                            is_retail = self.detect_retail_environment(detections)
                            if frame_index % 30 == 0:  # 每30帧记录一次环境信息
                                self.logger.info(f"环境检测: {'零售环境' if is_retail else '非零售环境'}")
                            
                            # 视频处理特有的人物检测逻辑
                            person_detections = []
                            object_detections = []
                            
                            # 处理不同格式的检测结果
                            if hasattr(detections, 'boxes'):
                                # 适配ultralytics格式
                                for i, box in enumerate(detections.boxes):
                                    if hasattr(box, 'cls') and detections.names.get(int(box.cls[0]), "") == "person":
                                        # 构建人物检测对象
                                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                        person_detections.append({
                                            'class': 'person',
                                            'confidence': float(box.conf[0]),
                                            'bbox': [x1, y1, x2, y2]
                                        })
                                    else:
                                        # 构建物体检测对象
                                        if hasattr(box, 'cls') and hasattr(box, 'xyxy'):
                                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                            class_id = int(box.cls[0])
                                            class_name = detections.names.get(class_id, "unknown")
                                            object_detections.append({
                                                'class': class_name,
                                                'confidence': float(box.conf[0]),
                                                'bbox': [x1, y1, x2, y2]
                                            })
                            elif isinstance(detections, list):
                                # 适配列表格式
                                for d in detections:
                                    if isinstance(d, dict):
                                        if d.get('class') == 'person':
                                            person_detections.append(d)
                                        else:
                                            object_detections.append(d)
                            
                            # 处理姿态检测结果
                            landmarks = None
                            if pose_results and 'landmarks' in pose_results:
                                landmarks = pose_results['landmarks']
                            
                            # 1. 基于规则的行为检测
                            rule_behaviors = self._detect_behaviors_with_rules(frame, detections)
                            behaviors.extend(rule_behaviors)
                            
                            # 2. 姿态和高级行为检测 - 降低检测阈值提高敏感度
                            if landmarks is not None and isinstance(landmarks, list) and len(landmarks) > 0:
                                no_detection_count = 0  # 重置计数器
                                
                                for person in person_detections:
                                    # 获取人物边界框
                                    if 'bbox' in person:
                                        person_bbox = person['bbox']
                                        person_id = person.get('id', None)
                                        
                                        # 执行姿态行为检测
                                        try:
                                            # 检测各种行为
                                            try:
                                                # 1. 检测异常手臂位置 (优先检测)
                                                arm_behaviors = self._detect_abnormal_arm_positions(
                                                    frame, landmarks, person_bbox, person_id
                                                )
                                                behaviors.extend(arm_behaviors)
                                                
                                                # 2. 检测可疑蹲姿
                                                crouch_behaviors = self._detect_suspicious_crouching(
                                                    frame, landmarks, person_bbox
                                                )
                                                behaviors.extend(crouch_behaviors)
                                                
                                                # 3. 检测异常伸手行为 (关键检测点)
                                                reach_behaviors = self._detect_unusual_reaching(
                                                    frame, landmarks, person_bbox, object_detections
                                                )
                                                behaviors.extend(reach_behaviors)
                                                
                                                # 4. 检测身体遮挡行为
                                                shield_behaviors = self._detect_body_shielding(
                                                    frame, landmarks, person_bbox, object_detections
                                                )
                                                behaviors.extend(shield_behaviors)
                                                
                                                # 5. 检测异常头部动作 (关键检测点)
                                                head_behaviors = self._detect_abnormal_head_movement(
                                                    frame, landmarks, person_bbox
                                                )
                                                behaviors.extend(head_behaviors)
                                                
                                                # 6. 检测双手放在背后
                                                hands_behind_behaviors = self._detect_hands_behind_back(
                                                    frame, landmarks, person_bbox
                                                )
                                                behaviors.extend(hands_behind_behaviors)
                                                
                                                # 7. 新增: 检测物品拿取和隐藏行为 (不依赖商品检测)
                                                grabbing_behaviors = self._detect_item_grabbing_and_concealment(
                                                    frame, landmarks, person_bbox
                                                )
                                                behaviors.extend(grabbing_behaviors)
                                                
                                            except Exception as e:
                                                self.logger.error(f"特定行为检测错误: {e}")
                                                traceback.print_exc()
                                            
                                            # 更新人物历史
                                            if person_id is not None:
                                                self._update_person_history(person_id, behaviors, landmarks)
                                            
                                        except Exception as e:
                                            self.logger.error(f"姿态行为检测错误: {e}")
                                            traceback.print_exc()
                            else:
                                no_detection_count += 1
                            
                            # 3. 执行机器学习模型预测
                            # 视频处理特有: 使用当前帧和前一帧的关键点进行预测
                            if landmarks is not None and isinstance(landmarks, list) and len(landmarks) > 0:
                                try:
                                    # 提取标准特征
                                    current_features = self._extract_standard_features(landmarks)
                                    
                                    # 如果有前一帧的特征，则使用增强特征
                                    if prev_landmarks is not None:
                                        # 提取增强特征
                                        enhanced_features = self._extract_enhanced_features(landmarks, prev_landmarks)
                                        # 预测行为
                                        ml_behaviors = self.predict_behaviors(current_features, self._extract_standard_features(prev_landmarks))
                                    else:
                                        # 仅使用当前帧特征
                                        ml_behaviors = self.predict_behaviors(current_features)
                                    
                                    # 添加机器学习预测的行为
                                    behaviors.extend(ml_behaviors)
                                    
                                except Exception as e:
                                    self.logger.error(f"机器学习预测错误: {e}")
                                    traceback.print_exc()
                            
                            # 4. 基于光流检测运动相关行为
                            if flow_data:
                                try:
                                    motion_behaviors = self._detect_motion_behaviors(frame, flow_data)
                                    behaviors.extend(motion_behaviors)
                                except Exception as e:
                                    self.logger.error(f"基于运动的行为检测错误: {e}")
                            
                            # 5. 调整零售环境中的盗窃概率
                            behaviors = self._adjust_theft_probability_in_retail(behaviors, is_retail)
                            
                            # 如果连续多帧没有检测到关键点，尝试重新检测
                            if no_detection_count > 10 and frame_index % 2 == 0:
                                self.logger.info("连续多帧未检测到关键点，尝试重新初始化姿态检测器")
                                try:
                                    self.initialize_pose_detector()
                                    no_detection_count = 0
                                except Exception as e:
                                    self.logger.error(f"重新初始化姿态检测器失败: {e}")
                            
                            # 更新前一帧的关键点
                            prev_landmarks = landmarks
                            
                    except Exception as e:
                        self.logger.error(f"行为检测错误: {str(e)}")
                        traceback.print_exc()
                
                # 添加时间戳和盗窃概率
                theft_probability_history.append(theft_probability)
                avg_theft_prob = np.mean(theft_probability_history[-10:]) if theft_probability_history else theft_probability
                
                # 创建输出帧
                output_frame = frame.copy()
                
                # 添加姿态关键点（如果有）
                if pose_results:
                    output_frame = self._draw_pose_landmarks(output_frame, pose_results)
                
                # 添加时间信息
                text_y = 30
                cv2.putText(output_frame, f"Time: {time_str}", (10, text_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                text_y += 30
                
                # 添加检测到的行为
                if behaviors:
                    # 更新行为计数
                    for behavior in behaviors:
                        behavior_type = behavior['type']
                        if behavior_type in behavior_count:
                            behavior_count[behavior_type] += 1
                    
                    # 显示行为信息 - 使用英文
                    cv2.putText(output_frame, "Behaviors:", (10, text_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    text_y += 25
                    
                    # 记录已使用的位置以避免标签重叠
                    used_label_positions = []
                    
                    for i, behavior in enumerate(behaviors):
                        if i >= 3:  # 最多显示3个行为                            
                            cv2.putText(output_frame, "...", (10, text_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            text_y += 20
                            break
                            
                        # 获取行为类型和置信度
                        behavior_type = behavior['type']
                        confidence = behavior['confidence']
                        
                        # 直接使用英文行为类型，不使用中文映射
                        # display_type = self.behavior_type_map.get(behavior_type, behavior_type)
                        display_type = behavior_type
                        
                        # 显示行为信息
                        cv2.putText(output_frame, f"{i+1}. {display_type} ({confidence:.2f})", 
                                  (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        text_y += 20
                        
                        # 在相应位置绘制边界框
                        if 'bbox' in behavior:
                            bbox = behavior['bbox']
                            color = behavior.get('color', (0, 0, 255))  # 默认红色
                            cv2.rectangle(output_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                            
                            # 添加标签 - 避免重叠
                            label_x, label_y = bbox[0], bbox[1] - 5
                            
                            # 检查是否与已有标签重叠
                            overlap = False
                            for pos in used_label_positions:
                                if abs(pos[0] - label_x) < 100 and abs(pos[1] - label_y) < 20:
                                    overlap = True
                                    break
                            
                            # 如果重叠，调整位置
                            if overlap:
                                label_y -= 20  # 向上移动标签
                            
                            # 记录使用的位置
                            used_label_positions.append((label_x, label_y))
                            
                            try:
                                cv2.putText(output_frame, f"{display_type}", (label_x, label_y), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            except Exception as e:
                                self.logger.error(f"添加标签错误: {e}")
                    
                    # 降低可疑行为阈值，使视频检测更敏感
                    significant_behaviors = [b for b in behaviors if b.get('confidence', 0) > 0.55]
                    if significant_behaviors:
                        suspicious_frames.append(frame_index)
                        all_behaviors.append((frame_index, behaviors))
                
                # 写入帧到输出视频
                out.write(output_frame)
                
                # 调用帧处理回调                
                if self.frame_processed_callback:
                    # 构建完整的帧数据包含frame原始帧和总帧数                    
                    self.frame_processed_callback(
                        frame_index, 
                        output_frame,
                        behaviors,
                        frame  # 添加原始帧作为第四个参数
                    )
                
                frame_index += 1
                
            # 处理所有帧后，计算行为统计
            behavior_stats = []
            for behavior_type in self.behavior_types:
                count = behavior_count[behavior_type]
                if count > 0:
                    # 计算该行为的发生时间
                    time_points = []
                    for frame_idx, frame_behaviors in all_behaviors:
                        for behavior in frame_behaviors:
                            if behavior['type'] == behavior_type:
                                time_sec = frame_idx / fps if fps > 0 else 0
                                time_points.append(time_sec)
                    
                    # 使用中文行为类型（如果有映射）                    
                    display_type = self.behavior_type_map.get(behavior_type, behavior_type)
                    
                    behavior_stats.append({
                        'type': behavior_type,
                        'display_type': display_type,
                        'count': count,
                        'time_points': time_points
                    })
            
            behavior_stats.sort(key=lambda x: x['count'], reverse=True)
            
            self.logger.info(f"行为统计: {[(b['display_type'], b['count']) for b in behavior_stats]}")
            
            cap.release()
            out.release()
            
            return output_path, suspicious_frames, all_behaviors
            
        except Exception as e:
            self.logger.error(f"视频处理错误: {str(e)}")
            traceback.print_exc()
            
            # 清理资源
            cap.release()
            out.release()
            
            return None, [], []

    def _detect_motion_behaviors(self, frame, flow_data=None):
        """检测基于运动的可疑行为
        
        Args:
            frame: 当前帧
            flow_data: 光流数据（可选）
            
        Returns:
            behaviors: 检测到的行为列表
        """
        behaviors = []
        
        # 如果帧无效，直接返回
        if frame is None:
            return behaviors
            
        try:
            # 将当前帧转换为灰度
            current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 如果没有前一帧，存储当前帧并返回
            if self.prev_gray_frame is None:
                self.prev_gray_frame = current_gray
                return behaviors
                
            # 计算帧差
            frame_diff = cv2.absdiff(current_gray, self.prev_gray_frame)
            
            # 更新前一帧灰度图像（确保即使发生错误，下一次调用也能正常工作）
            self.prev_gray_frame = current_gray.copy()
            
            # 应用阈值，将差异二值化
            _, threshold = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
            
            # 应用形态学操作去除噪点
            kernel = np.ones((5, 5), np.uint8)
            threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
            threshold = cv2.dilate(threshold, kernel, iterations=2)
            
            # 计算运动区域的比率
            motion_ratio = np.sum(threshold > 0) / (threshold.shape[0] * threshold.shape[1])
            
            # 获取检测参数
            motion_threshold = self.config.get('detection_parameters', {}).get('motion_threshold', 0.01)
            rapid_motion_threshold = self.config.get('detection_parameters', {}).get('rapid_motion_threshold', 0.03)
            concentrated_motion_ratio = self.config.get('detection_parameters', {}).get('concentrated_motion_ratio', 0.2)
            
            # 检测是否有显著运动
            significant_motion = motion_ratio > motion_threshold
            rapid_movement = motion_ratio > rapid_motion_threshold
            
            # 查找轮廓以确定运动区域
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 计算帧尺度
            frame_height, frame_width = frame.shape[:2]
            
            # 检查是否有集中的运动区域（可能表示撕标签行为）
            concentrated_motion = False
            largest_contour_area = 0
            largest_contour_box = None
            
            if contours:
                # 找出最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                largest_contour_area = cv2.contourArea(largest_contour)
                
                # 如果最大轮廓面积占总运动区域的比例超过阈值，认为是集中运动
                if largest_contour_area > 0:
                    concentrated_motion = (largest_contour_area / np.sum(threshold > 0)) > concentrated_motion_ratio
                    
                    # 获取轮廓的边界框
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    largest_contour_box = [x, y, x+w, y+h]
            
            # 检测行为：遮挡商品区域
            if significant_motion and not concentrated_motion and self.prev_landmarks:
                behavior_type = "Covering Product Area"
                confidence = self.get_behavior_weight(behavior_type) * motion_ratio * 5
                confidence = min(0.85, max(0.4, confidence))
                behaviors.append({
                    'type': behavior_type,
                    'description': self.localization["behavior_detected"].format(
                        self.behavior_type_map.get(behavior_type, behavior_type)),
                    'confidence': confidence,
                    'bbox': [int(frame_width*0.3), int(frame_height*0.3), 
                           int(frame_width*0.7), int(frame_height*0.7)],
                    'color': self.behavior_color_map[behavior_type]
                })
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"检测基于运动的可疑行为出错: {e}")
            return []
    
    def _extract_pose_landmarks(self, image):
        """
        提取姿态关键点
        视频处理专用版本 - 增强检测稳定性，降低关键点可见度要求
        
        Args:
            image: 输入图像
            
        Returns:
            pose_results: 姿态关键点结果
        """
        if self.pose_detector is None:
            self.initialize_pose_detector()
            
        if self.pose_detector is None:
            self.logger.error("姿态检测器初始化失败")
            return None
            
        try:
            # 尝试使用当前姿态检测器
            if hasattr(self.pose_detector, 'process') and callable(getattr(self.pose_detector, 'process')):
                # MediaPipe风格的姿态检测器
                results = self.pose_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                if not hasattr(results, 'pose_landmarks') or results.pose_landmarks is None:
                    # 视频处理中，尝试对图像进行预处理以提高检测率
                    # 调整图像对比度和亮度
                    enhanced = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
                    results = self.pose_detector.process(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
                
                if hasattr(results, 'pose_landmarks') and results.pose_landmarks is not None:
                    landmarks = []
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                        visibility = landmark.visibility if hasattr(landmark, 'visibility') else 0.5
                        landmarks.append([x, y, visibility])
                    
                    # 为视频处理增强关键点数据
                    if not landmarks or len(landmarks) < 5:
                        return None
                    
                    pose_results = {
                        'landmarks': landmarks,
                        'bbox': self._calculate_pose_bbox(landmarks, image.shape),
                        'mp_results': results  # 保留原始MediaPipe结果用于绘制
                    }
                    return pose_results
            
            elif hasattr(self.pose_detector, 'detect') and callable(getattr(self.pose_detector, 'detect')):
                # MoveNet类似的检测器
                keypoints = self.pose_detector.detect(image)
                
                if keypoints is not None and len(keypoints) > 0:
                    # 转换为统一格式
                    landmarks = []
                    for kp in keypoints[0]:
                        x, y, score = int(kp[1]), int(kp[0]), kp[2]
                        landmarks.append([x, y, score])
                    
                    pose_results = {
                        'landmarks': landmarks,
                        'bbox': self._calculate_pose_bbox(landmarks, image.shape)
                    }
                    return pose_results
            
            else:
                # 未知类型的检测器
                self.logger.warning("未知类型的姿态检测器")
            
            # 如果都失败了，返回None
            return None
            
        except Exception as e:
            self.logger.error(f"姿态关键点提取错误: {str(e)}")
            traceback.print_exc()
            
            # 尝试重新初始化检测器
            try:
                self.logger.info("尝试重新初始化姿态检测器")
                self.initialize_pose_detector()
            except Exception as reinit_error:
                self.logger.error(f"重新初始化姿态检测器失败: {str(reinit_error)}")
            
            return None
    
    def _convert_to_landmarks_dict(self, landmarks_list):
        """把 _extract_pose_landmarks 返回的 landmarks list 转换成 dict with integer keys
        
        输入: [[x, y, visibility], ...]  (MediaPipe 33点)
        输出: {0: [x, y, visibility], 1: [x, y, visibility], ...}
        
        注意: 值用 list 格式 [x, y, vis]，不是 dict {'x':..., 'y':...}
        因为子检测器 (_detect_abnormal_arm_positions 等) 用 a[0], a[1] 索引访问
        """
        if landmarks_list is None:
            return None
        result = {}
        for i, pt in enumerate(landmarks_list):
            if len(pt) >= 3:
                result[i] = [float(pt[0]), float(pt[1]), float(pt[2])]
            elif len(pt) >= 2:
                result[i] = [float(pt[0]), float(pt[1]), 0.5]
        return result
    
    def _draw_pose_landmarks(self, image, pose_results):
        """
        在图像上绘制姿态关键点
        
        Args:
            image: 输入图像
            pose_results: 姿态关键点结果
            
        Returns:
            image: 添加了关键点的图像
        """
        try:
            if pose_results is None:
                return image
                
            output_image = image.copy()
            
            # 获取关键点
            landmarks = None
            mp_results = None
            
            if 'landmarks' in pose_results:
                landmarks = pose_results['landmarks']
            
            if 'mp_results' in pose_results and hasattr(pose_results['mp_results'], 'pose_landmarks'):
                mp_results = pose_results['mp_results']
            
            # 如果有MediaPipe结果，使用其内置绘图工具
            if mp_results and hasattr(mp_results, 'pose_landmarks'):
                try:
                    # 使用MediaPipe的绘图工具
                    import mediapipe as mp
                    mp_drawing = mp.solutions.drawing_utils
                    mp_pose = mp.solutions.pose
                    
                    # 转换回RGB用于绘制
                    rgb_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
                    
                    # 绘制姿态关键点
                    mp_drawing.draw_landmarks(
                        rgb_image,
                        mp_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                    )
                    
                    # 转回BGR
                    output_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                    return output_image
                except ImportError:
                    self.logger.warning("MediaPipe绘图工具不可用，使用基本绘图")
            
            # 如果MediaPipe绘图失败或不可用，使用基本OpenCV绘图
            if landmarks and isinstance(landmarks, list) and len(landmarks) > 0:
                # 绘制关键点和连接线
                # 定义连接
                connections = [
                    # 面部连接
                    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
                    # 上半身连接
                    (5, 11), (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
                    # 下半身连接
                    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28)
                ]
                
                # 根据实际关键点数量调整连接
                valid_connections = []
                for conn in connections:
                    if conn[0] < len(landmarks) and conn[1] < len(landmarks):
                        valid_connections.append(conn)
                
                # 绘制连接线
                for connection in valid_connections:
                    start_idx, end_idx = connection
                    if len(landmarks[start_idx]) >= 2 and len(landmarks[end_idx]) >= 2:
                        start_point = (int(landmarks[start_idx][0]), int(landmarks[start_idx][1]))
                        end_point = (int(landmarks[end_idx][0]), int(landmarks[end_idx][1]))
                        
                        # 检查点是否在图像范围内
                        if (0 <= start_point[0] < output_image.shape[1] and 
                            0 <= start_point[1] < output_image.shape[0] and
                            0 <= end_point[0] < output_image.shape[1] and 
                            0 <= end_point[1] < output_image.shape[0]):
                            cv2.line(output_image, start_point, end_point, (0, 255, 0), 2)
                
                # 绘制关键点
                for i, landmark in enumerate(landmarks):
                    if len(landmark) >= 2:
                        x, y = int(landmark[0]), int(landmark[1])
                        
                        # 检查点是否在图像范围内
                        if 0 <= x < output_image.shape[1] and 0 <= y < output_image.shape[0]:
                            # 使用不同颜色标识不同部位
                            if i <= 10:  # 面部关键点
                                color = (255, 0, 0)  # 蓝色
                            elif i <= 22:  # 上半身关键点
                                color = (0, 255, 0)  # 绿色
                            else:  # 下半身关键点
                                color = (0, 0, 255)  # 红色
                                
                            cv2.circle(output_image, (x, y), 4, color, -1)
            
            return output_image
            
        except Exception as e:
            self.logger.error(f"绘制姿态关键点错误: {str(e)}")
            traceback.print_exc()
            return image

    def _detect_pose_based_behaviors(self, image, landmarks, person_bbox, object_detections=None, person_id=None):
        """基于姿态估计检测可疑行为
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            person_bbox: 人物边界框
            object_detections: 物体检测结果
            person_id: 人物ID，用于多人跟踪
            
        Returns:
            behaviors: 检测到的行为列表
        """
        if not landmarks:
            return []
            
        behaviors = []
        
        # 检测异常手臂位置
        arm_behavior = self._detect_abnormal_arm_positions(image, landmarks, person_bbox, person_id)
        if arm_behavior:
            behaviors.extend(arm_behavior)
        
        # 检测可疑蹲姿
        crouch_behavior = self._detect_suspicious_crouching(image, landmarks, person_bbox)
        if crouch_behavior:
            behaviors.extend(crouch_behavior)
        
        # 检测不自然伸手姿势
        if object_detections:
            reach_behavior = self._detect_unusual_reaching(image, landmarks, person_bbox, object_detections)
            if reach_behavior:
                behaviors.extend(reach_behavior)
            
        # 检测身体屏蔽姿势
        if object_detections:
            shield_behavior = self._detect_body_shielding(image, landmarks, person_bbox, object_detections)
            if shield_behavior:
                behaviors.extend(shield_behavior)
        
        # 检测头部异常转动
        head_behavior = self._detect_abnormal_head_movement(image, landmarks, person_bbox)
        if head_behavior:
            behaviors.extend(head_behavior)
        
        # 检测双手背后姿势
        hands_behind_behavior = self._detect_hands_behind_back(image, landmarks, person_bbox)
        if hands_behind_behavior:
            behaviors.extend(hands_behind_behavior)
            
        # 检测分散注意力行为
        if person_id is not None:
            distraction_behavior = self._detect_distraction_behavior(image, landmarks, person_bbox, person_id)
            if distraction_behavior:
                behaviors.extend(distraction_behavior)
                
        # 记录该人的行为历史
        if person_id is not None and behaviors:
            self._update_person_history(person_id, behaviors, landmarks)
        
        return behaviors
    
    def _detect_distraction_behavior(self, image, landmarks, person_bbox, person_id):
        """检测分散注意力行为 - 一只手吸引注意力，另一只手进行盗窃
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            person_bbox: 人物边界框
            person_id: 人物ID
            
        Returns:
            behavior: 检测到的行为或None
        """
        # 检查是否有手臂、手腕和头部关键点
        required_points = [11, 12, 13, 14, 15, 16, 0]  # 左右肩膀，左右肘，左右手腕，头部
        
        if not all(i in landmarks and landmarks[i]['visibility'] > 0.5 for i in required_points):
            return None
            
        # 获取左右手腕位置和移动
        left_wrist = (landmarks[15][0], landmarks[15][1])
        right_wrist = (landmarks[16][0], landmarks[16][1])
        head = (landmarks[0][0], landmarks[0][1])
        
        # 获取前一帧的关键点记录
        prev_landmarks = None
        if person_id in self.person_history and self.person_history[person_id]:
            for record in reversed(self.person_history[person_id]):
                if 'landmarks' in record:
                    prev_landmarks = record['landmarks']
                    break
        
        if prev_landmarks is None:
            return None
            
        # 计算手腕移动
        try:
            prev_left_wrist = (prev_landmarks[15][0], prev_landmarks[15][1])
            prev_right_wrist = (prev_landmarks[16][0], prev_landmarks[16][1])
            
            # 计算左右手腕位移
            left_movement = np.sqrt((left_wrist[0] - prev_left_wrist[0])**2 + 
                                    (left_wrist[1] - prev_left_wrist[1])**2)
            right_movement = np.sqrt((right_wrist[0] - prev_right_wrist[0])**2 + 
                                     (right_wrist[1] - prev_right_wrist[1])**2)
            
            # 获取阈值
            distraction_threshold = self.config.get('detection_parameters', {}).get('distraction_movement_threshold', 0.05)
            
            # 检测一只手快速移动（吸引注意力），另一只手较慢移动（可能盗窃）
            # 图像尺寸标准
            height, width = image.shape[:2]
            image_diag = np.sqrt(height**2 + width**2)
            left_movement_norm = left_movement / image_diag
            right_movement_norm = right_movement / image_diag
            
            is_distraction = False
            
            if (left_movement_norm > distraction_threshold and 
                right_movement_norm < distraction_threshold * 0.3):
                # 左手移动快，右手移动慢
                is_distraction = True
                active_hand = "left"
            elif (right_movement_norm > distraction_threshold and 
                  left_movement_norm < distraction_threshold * 0.3):
                # 右手移动快，左手移动慢
                is_distraction = True
                active_hand = "right"
                
            if is_distraction:
                # 检查手与头部的关系，手是否在视线范围外
                slow_hand = right_wrist if active_hand == "left" else left_wrist
                head_to_hand_angle = np.arctan2(slow_hand[1] - head[1], 
                                               slow_hand[0] - head[0])
                head_facing_angle = 0  # 假设头部朝前，这里可以用头部姿态估计来改进
                
                angle_diff = abs(head_to_hand_angle - head_facing_angle)
                
                # 如果慢手在视线外或身体后
                if angle_diff > np.pi/2:
                    behavior_type = "Distraction Behavior"
                    confidence = self.get_behavior_weight(behavior_type)
                    
                    return {
                        'type': behavior_type,
                        'description': self.localization["behavior_detected"].format(
                            self.behavior_type_map.get(behavior_type, behavior_type)),
                        'confidence': confidence,
                        'bbox': person_bbox,
                        'color': self.behavior_color_map[behavior_type],
                        'details': {
                            'active_hand': active_hand,
                            'movement_ratio': left_movement_norm / (right_movement_norm + 1e-5)
                                              if active_hand == "left" else 
                                              right_movement_norm / (left_movement_norm + 1e-5)
                        }
                    }
        except Exception as e:
            self.logger.error(f"分散注意力行为检测错误: {e}")
            
        return None

    def _update_person_history(self, person_id, behaviors, landmarks):
        """更新人物行为历史
        
        Args:
            person_id: 人物ID
            behaviors: 当前帧检测到的行为
            landmarks: 当前帧的姿态关键点
        """
        # 记录时间戳、关键点和行为
        record = {
            'timestamp': time.time(),
            'behaviors': behaviors,
            'landmarks': landmarks
        }
        
        # 添加到历史记录
        self.person_history[person_id].append(record)
        
        # 限制历史记录大小
        if len(self.person_history[person_id]) > self.history_size:
            self.person_history[person_id].pop(0)

    def detect_group_behaviors(self, frame, detections):
        """检测团队盗窃行为
        
        Args:
            frame: 当前帧
            detections: 检测结果（包含多个人物的边界框和ID）
            
        Returns:
            behaviors: 检测到的团队行为列表
        """
        if not detections or 'tracks' not in detections:
            return []
            
        behaviors = []
        
        try:
            # 获取当前所有人物的位置
            persons = {}
            for track_id, track_info in detections['tracks'].items():
                if track_info.get('class_name', '') == 'person':
                    box = track_info.get('box', [0, 0, 0, 0])
                    persons[track_id] = {
                        'box': box,
                        'center': ((box[0] + box[2])/2, (box[1] + box[3])/2)
                    }
            
            # 检查人物之间的距离和互动
            if len(persons) >= 2:  # 至少需要两个人
                # 获取配置参数
                group_distance_threshold = self.config.get('detection_parameters', {}).get('group_distance_threshold', 150)
                interaction_time_threshold = self.config.get('detection_parameters', {}).get('interaction_time_threshold', 3)
                
                # 检查每对人
                for id1, p1 in persons.items():
                    for id2, p2 in persons.items():
                        if id1 != id2:
                            # 计算距离
                            distance = np.sqrt((p1['center'][0] - p2['center'][0])**2 + 
                                              (p1['center'][1] - p2['center'][1])**2)
                            
                            # 如果距离足够近，记录互动
                            pair_id = tuple(sorted([id1, id2]))
                            if distance < group_distance_threshold:
                                current_time = time.time()
                                
                                if pair_id not in self.group_interactions:
                                    self.group_interactions[pair_id] = {
                                        'start_time': current_time,
                                        'locations': [(p1['center'], p2['center'])],
                                        'suspicious_count': 0
                                    }
                                else:
                                    # 更新互动信息
                                    self.group_interactions[pair_id]['locations'].append((p1['center'], p2['center']))
                                    
                                    # 限制历史位置记录
                                    if len(self.group_interactions[pair_id]['locations']) > 10:
                                        self.group_interactions[pair_id]['locations'].pop(0)
                                    
                                    # 检查移动模式，是否呈现分散和观望态势
                                    if len(self.group_interactions[pair_id]['locations']) >= 3:
                                        positions = self.group_interactions[pair_id]['locations']
                                        # 一人移动，一人静止的模式
                                        p1_movement = np.mean([np.sqrt((pos[0][0] - prev[0][0])**2 + 
                                                                      (pos[0][1] - prev[0][1])**2) 
                                                              for pos, prev in zip(positions[1:], positions[:-1])])
                                        p2_movement = np.mean([np.sqrt((pos[1][0] - prev[1][0])**2 + 
                                                                      (pos[1][1] - prev[1][1])**2) 
                                                              for pos, prev in zip(positions[1:], positions[:-1])])
                                        
                                        # 检查是否一人移动，一人相对静止
                                        if (p1_movement > 3 * p2_movement) or (p2_movement > 3 * p1_movement):
                                            self.group_interactions[pair_id]['suspicious_count'] += 1
                                    
                                    # 检查互动时间
                                    interaction_time = current_time - self.group_interactions[pair_id]['start_time']
                                    
                                    # 检查是否已经互动足够长时间且有可疑行为
                                    if (interaction_time >= interaction_time_threshold and 
                                        self.group_interactions[pair_id]['suspicious_count'] >= 2):
                                        
                                        # 团伙盗窃行为
                                        behavior_type = "Group Theft Coordination"
                                        confidence = self.get_behavior_weight(behavior_type)
                                        
                                        # 创建边界框包含两个人
                                        group_box = [
                                            min(p1['box'][0], p2['box'][0]),
                                            min(p1['box'][1], p2['box'][1]),
                                            max(p1['box'][2], p2['box'][2]),
                                            max(p1['box'][3], p2['box'][3])
                                        ]
                                        
                                        behaviors.append({
                                            'type': behavior_type,
                                            'description': self.localization["behavior_detected"].format(
                                                self.behavior_type_map.get(behavior_type, behavior_type)),
                                            'confidence': confidence,
                                            'bbox': group_box,
                                            'color': self.behavior_color_map[behavior_type],
                                            'details': {
                                                'persons': [id1, id2],
                                                'interaction_time': interaction_time,
                                                'suspicious_score': self.group_interactions[pair_id]['suspicious_count']
                                            }
                                        })
                            else:
                                # 如果距离变远，重置互动记录
                                if pair_id in self.group_interactions:
                                    del self.group_interactions[pair_id]
                
                # 清理过期的互动记录
                current_time = time.time()
                for pair_id in list(self.group_interactions.keys()):
                    if current_time - self.group_interactions[pair_id]['start_time'] > 60:  # 1分钟过期
                        del self.group_interactions[pair_id]
        
        except Exception as e:
            self.logger.error(f"团队行为检测错误: {e}")
            
        return behaviors
        
    def _detect_abnormal_arm_positions(self, frame, landmarks, person_bbox, person_id=None):
        """检测异常的手臂位置，可能表示隐藏行为"""
        behaviors = []
        
        try:
            if not landmarks or len(landmarks) < 11:  # 至少需要包含肩膀和手腕的关键点
                return behaviors
            
            # 获取身体关键点
            left_shoulder = landmarks[5] if len(landmarks) > 5 else None
            right_shoulder = landmarks[6] if len(landmarks) > 6 else None
            left_elbow = landmarks[7] if len(landmarks) > 7 else None
            right_elbow = landmarks[8] if len(landmarks) > 8 else None
            left_wrist = landmarks[9] if len(landmarks) > 9 else None
            right_wrist = landmarks[10] if len(landmarks) > 10 else None
            
            if left_shoulder is None or right_shoulder is None or \
               left_elbow is None or right_elbow is None or \
               left_wrist is None or right_wrist is None:
                return behaviors
            
            # 计算手臂角度
            left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # 异常手臂姿势的检测规则
            unusual_arm_threshold = 30  # 异常角度阈值
            
            # 检测非常弯曲的手臂 - 可能表示隐藏物品
            if left_arm_angle < unusual_arm_threshold or right_arm_angle < unusual_arm_threshold:
                behavior = {
                    'type': 'unusual_arm_position',
                    'confidence': 0.7,
                    'details': '检测到异常手臂姿势',
                    'bbox': person_bbox,
                    'color': (0, 165, 255)  # 橙色
                }
                behaviors.append(behavior)
                self.logger.info(f"检测到异常手臂姿势 (左手臂角度: {left_arm_angle:.1f}, 右手臂角度: {right_arm_angle:.1f})")
            
            # 检测手臂交叉 - 紧张或防御性姿势
            left_wrist_x, right_wrist_x = left_wrist[0], right_wrist[0]
            left_shoulder_x, right_shoulder_x = left_shoulder[0], right_shoulder[0]
            
            if (left_wrist_x > right_shoulder_x and right_wrist_x < left_shoulder_x):
                behavior = {
                    'type': 'arms_crossed',
                    'confidence': 0.65,
                    'details': '检测到交叉手臂姿势',
                    'bbox': person_bbox,
                    'color': (0, 165, 255)  # 橙色
                }
                behaviors.append(behavior)
                self.logger.info("检测到交叉手臂姿势")
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"异常手臂姿势检测错误: {str(e)}")
            traceback.print_exc()
            return behaviors
    
    def _detect_suspicious_crouching(self, image, landmarks, person_bbox):
        """检测可疑的蹲姿行为
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            person_bbox: 人物边界框
            
        Returns:
            behaviors: 检测到的行为列表
        """
        behaviors = []
        
        try:
            if not landmarks or len(landmarks) < 13:  # 需要包含臀部和膝盖的关键点
                return behaviors
            
            # 获取身体关键点
            left_shoulder = landmarks[5] if len(landmarks) > 5 else None
            right_shoulder = landmarks[6] if len(landmarks) > 6 else None
            left_hip = landmarks[11] if len(landmarks) > 11 else None
            right_hip = landmarks[12] if len(landmarks) > 12 else None
            left_knee = landmarks[13] if len(landmarks) > 13 else None
            right_knee = landmarks[14] if len(landmarks) > 14 else None
            
            if left_hip is None or right_hip is None or \
               left_knee is None or right_knee is None or \
               left_shoulder is None or right_shoulder is None:
                return behaviors
            
            # 计算臀部与肩膀之间的距离
            hip_y = (left_hip[1] + right_hip[1]) / 2
            shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            vertical_distance = hip_y - shoulder_y
            
            # 计算臀部与膝盖之间的距离
            knee_y = (left_knee[1] + right_knee[1]) / 2
            hip_knee_distance = knee_y - hip_y
            
            # 估计人物高度（肩膀到膝盖的距离）
            estimated_height = knee_y - shoulder_y
            
            # 计算蹲姿比例 - 臀部位置相对于身高的比例
            if estimated_height > 0:
                crouch_ratio = vertical_distance / estimated_height
                
                # 检测蹲姿 - 当臀部接近膝盖
                if crouch_ratio > 0.4 and hip_knee_distance < estimated_height * 0.25:
                    behavior = {
                        'type': 'suspicious_crouching',
                        'confidence': min(0.5 + crouch_ratio * 0.5, 0.9),
                        'details': '检测到可疑蹲姿',
                        'bbox': person_bbox,
                        'color': (0, 0, 255)  # 红色
                    }
                    behaviors.append(behavior)
                    self.logger.info(f"检测到可疑蹲姿 (比例: {crouch_ratio:.2f})")
            
            return behaviors
        
        except Exception as e:
            self.logger.error(f"可疑蹲姿检测错误: {str(e)}")
            traceback.print_exc()
            return behaviors
            
    def _detect_hands_behind_back(self, image, landmarks, person_bbox):
        """
        检测双手放在背后的姿势 - 优化版，降低假阳性率
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            person_bbox: 人物边界框
            
        Returns:
            behaviors: 检测到的行为列表
        """
        behaviors = []
        
        try:
            if not landmarks or len(landmarks) < 15:  # 需要包含足够的关键点
                return behaviors
            
            # 获取身体关键点
            nose = landmarks[0] if len(landmarks) > 0 else None
            left_shoulder = landmarks[5] if len(landmarks) > 5 else None
            right_shoulder = landmarks[6] if len(landmarks) > 6 else None
            left_elbow = landmarks[7] if len(landmarks) > 7 else None
            right_elbow = landmarks[8] if len(landmarks) > 8 else None
            left_wrist = landmarks[9] if len(landmarks) > 9 else None
            right_wrist = landmarks[10] if len(landmarks) > 10 else None
            left_hip = landmarks[11] if len(landmarks) > 11 else None
            right_hip = landmarks[12] if len(landmarks) > 12 else None
            
            if left_shoulder is None or right_shoulder is None or \
               left_wrist is None or right_wrist is None:
                return behaviors
            
            # 1. 计算肩膀中心线
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            
            # 创建严格的手在背后判定条件
            
            # 2. 检查左手腕是否在右肩后面 - 更严格的条件
            left_wrist_behind = (left_wrist[0] > right_shoulder[0] + shoulder_width * 0.1 and
                                abs(left_wrist[1] - right_shoulder[1]) < shoulder_width * 0.8)
            
            # 3. 检查右手腕是否在左肩后面 - 更严格的条件
            right_wrist_behind = (right_wrist[0] < left_shoulder[0] - shoulder_width * 0.1 and
                                 abs(right_wrist[1] - left_shoulder[1]) < shoulder_width * 0.8)
            
            # 4. 检查手腕可见度 - 手在背后时通常可见度较低
            left_wrist_low_visibility = False
            right_wrist_low_visibility = False
            
            if len(left_wrist) > 2 and left_wrist[2] < 0.4:  # 第三个元素是可见度
                left_wrist_low_visibility = True
                
            if len(right_wrist) > 2 and right_wrist[2] < 0.4:
                right_wrist_low_visibility = True
                
            # 5. 检查手肘弯曲角度 - 手在背后时手肘通常有明显弯曲
            left_arm_angle = None
            right_arm_angle = None
            
            if left_elbow is not None:
                left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
                
            if right_elbow is not None:
                right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
                
            left_elbow_bent = left_arm_angle is not None and left_arm_angle < 140
            right_elbow_bent = right_arm_angle is not None and right_arm_angle < 140
            
            # 6. 增加证据累积 - 同时满足多个条件才判定为手在背后
            left_evidence = 0
            right_evidence = 0
            
            if left_wrist_behind:
                left_evidence += 1
            if left_wrist_low_visibility:
                left_evidence += 1
            if left_elbow_bent:
                left_evidence += 1
                
            if right_wrist_behind:
                right_evidence += 1
            if right_wrist_low_visibility:
                right_evidence += 1
            if right_elbow_bent:
                right_evidence += 1
            
            # 7. 减少连续检测假阳性 - 使用历史信息
            if not hasattr(self, '_hands_behind_history'):
                self._hands_behind_history = {
                    'left_count': 0,
                    'right_count': 0,
                    'last_left_detection': 0,
                    'last_right_detection': 0,
                    'frame_count': 0
                }
                
            # 更新帧计数
            self._hands_behind_history['frame_count'] += 1
            
            # 8. 只有连续多帧检测到才认为是真实的手背后行为
            left_confirmed = False
            right_confirmed = False
            
            # 对左手判断
            if left_evidence >= 2:  # 需要至少2个证据
                self._hands_behind_history['left_count'] += 1
                self._hands_behind_history['last_left_detection'] = self._hands_behind_history['frame_count']
                
                if self._hands_behind_history['left_count'] >= 3:  # 需要连续3帧以上
                    left_confirmed = True
            else:
                # 如果当前帧没有检测到，计数器递减但不低于0
                self._hands_behind_history['left_count'] = max(0, self._hands_behind_history['left_count'] - 1)
                
                # 如果长时间未检测到，重置计数器
                frames_since_last = self._hands_behind_history['frame_count'] - self._hands_behind_history['last_left_detection']
                if frames_since_last > 10:  # 超过10帧未检测到
                    self._hands_behind_history['left_count'] = 0
            
            # 对右手判断
            if right_evidence >= 2:
                self._hands_behind_history['right_count'] += 1
                self._hands_behind_history['last_right_detection'] = self._hands_behind_history['frame_count']
                
                if self._hands_behind_history['right_count'] >= 3:
                    right_confirmed = True
            else:
                self._hands_behind_history['right_count'] = max(0, self._hands_behind_history['right_count'] - 1)
                
                frames_since_last = self._hands_behind_history['frame_count'] - self._hands_behind_history['last_right_detection']
                if frames_since_last > 10:
                    self._hands_behind_history['right_count'] = 0
            
            # 9. 添加行为检测结果
            # 两只手都在背后 - 只有在非常确定的情况下才报告
            if left_confirmed and right_confirmed:
                behavior = {
                    'type': 'hands_behind_back',
                    'confidence': 0.85,
                    'details': '检测到双手放在背后',
                    'bbox': person_bbox,
                    'color': (255, 0, 0)  # 蓝色
                }
                behaviors.append(behavior)
                self.logger.info("检测到双手放在背后")
            # 单手在背后 - 需要更多证据
            elif left_confirmed or right_confirmed:
                hand_side = "左手" if left_confirmed else "右手"
                behavior = {
                    'type': 'one_hand_behind_back',
                    'confidence': 0.75,
                    'details': f'检测到{hand_side}放在背后',
                    'bbox': person_bbox,
                    'color': (255, 0, 0)  # 蓝色
                }
                behaviors.append(behavior)
                self.logger.info(f"检测到{hand_side}放在背后")
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"手放在背后检测错误: {str(e)}")
            traceback.print_exc()
            return behaviors
            
    def _detect_unusual_reaching(self, image, landmarks, person_bbox, object_detections):
        """
        检测异常伸手行为，如快速伸向物品或伸到不自然的位置
        视频处理增强检测敏感度
        """
        behaviors = []
        
        try:
            if not landmarks or len(landmarks) < 10:
                return behaviors
            
            # 获取手腕和肘部关键点
            left_shoulder = landmarks[5] if len(landmarks) > 5 else None
            right_shoulder = landmarks[6] if len(landmarks) > 6 else None
            left_elbow = landmarks[7] if len(landmarks) > 7 else None
            right_elbow = landmarks[8] if len(landmarks) > 8 else None
            left_wrist = landmarks[9] if len(landmarks) > 9 else None
            right_wrist = landmarks[10] if len(landmarks) > 10 else None
            
            if left_shoulder is None or right_shoulder is None or \
               left_elbow is None or right_elbow is None or \
               left_wrist is None or right_wrist is None:
                return behaviors
            
            # 计算手臂与水平线的角度
            left_arm_angle = self._calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_arm_angle = self._calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            # 视频处理降低检测阈值，使检测更敏感
            reaching_threshold = 60  # 降低阈值
            
            # 判断是否存在伸手行为（基于手臂角度）
            left_reaching = left_arm_angle < reaching_threshold
            right_reaching = right_arm_angle < reaching_threshold
            
            # 检查手是否接近物体
            hand_near_object = False
            reached_object = None
            min_distance = float('inf')
            
            for obj in object_detections:
                if 'bbox' in obj and obj.get('class') != 'person':
                    obj_bbox = obj['bbox']
                    obj_center_x = (obj_bbox[0] + obj_bbox[2]) / 2
                    obj_center_y = (obj_bbox[1] + obj_bbox[3]) / 2
                    
                    # 检查左手与物体的距离
                    if left_wrist is not None:
                        left_distance = math.sqrt((left_wrist[0] - obj_center_x)**2 + (left_wrist[1] - obj_center_y)**2)
                        if left_distance < min_distance:
                            min_distance = left_distance
                            reached_object = obj
                    
                    # 检查右手与物体的距离
                    if right_wrist is not None:
                        right_distance = math.sqrt((right_wrist[0] - obj_center_x)**2 + (right_wrist[1] - obj_center_y)**2)
                        if right_distance < min_distance:
                            min_distance = right_distance
                            reached_object = obj
            
            # 视频处理降低距离阈值，增强检测敏感度
            distance_threshold = 100  # 距离阈值
            
            if min_distance < distance_threshold:
                hand_near_object = True
            
            # 如果检测到伸手行为并且手接近物体，记录为可疑行为
            if (left_reaching or right_reaching) and hand_near_object and reached_object:
                # 动态调整置信度
                distance_factor = 1.0 - (min_distance / distance_threshold)
                angle_factor = 1.0 - (min(left_arm_angle, right_arm_angle) / reaching_threshold)
                confidence = min(0.6 + 0.2 * distance_factor + 0.2 * angle_factor, 0.95)
                
                behavior = {
                    'type': 'reaching_for_item',
                    'confidence': confidence,
                    'details': f"伸手拿取物品: {reached_object.get('class', 'object')}",
                    'bbox': person_bbox,
                    'color': (0, 255, 255)  # 黄色
                }
                behaviors.append(behavior)
                self.logger.info(f"检测到伸手拿取物品行为 (物品: {reached_object.get('class', 'object')}, 距离: {min_distance:.2f}, 置信度: {confidence:.2f})")
            
            # 检测手移动到身体中心或口袋位置的行为 - 视频特有检测
            if hasattr(self, '_prev_wrist_positions'):
                left_hip = landmarks[11] if len(landmarks) > 11 else None
                right_hip = landmarks[12] if len(landmarks) > 12 else None
                
                current_left_wrist = left_wrist
                current_right_wrist = right_wrist
                
                # 更新历史位置
                self._left_wrist_history.append(current_left_wrist)
                self._right_wrist_history.append(current_right_wrist)
                
                if len(self._left_wrist_history) > 10:
                    self._left_wrist_history.pop(0)
                if len(self._right_wrist_history) > 10:
                    self._right_wrist_history.pop(0)
                
                # 检测手是否向身体中心移动
                if len(self._left_wrist_history) >= 3 and left_hip is not None:
                    prev_left = self._left_wrist_history[-3]
                    curr_left = self._left_wrist_history[-1]
                    
                    # 计算手移动的方向
                    if prev_left and curr_left:
                        # 检查是否向腰部/口袋移动
                        moving_to_pocket = False
                        if curr_left[1] > prev_left[1] and curr_left[0] > prev_left[0]:  # 向右下方移动
                            if left_hip and curr_left[0] < left_hip[0] and abs(curr_left[1] - left_hip[1]) < 50:
                                moving_to_pocket = True
                        
                        if moving_to_pocket:
                            behavior = {
                                'type': 'item_concealment',
                                'confidence': 0.75,
                                'details': '手向口袋/身体移动，可能在隐藏物品',
                                'bbox': person_bbox,
                                'color': (0, 0, 255)  # 红色
                            }
                            behaviors.append(behavior)
                            self.logger.info("检测到手向口袋/身体移动，可能在隐藏物品")
                
                # 同样检查右手
                if len(self._right_wrist_history) >= 3 and right_hip is not None:
                    prev_right = self._right_wrist_history[-3]
                    curr_right = self._right_wrist_history[-1]
                    
                    # 计算手移动的方向
                    if prev_right and curr_right:
                        # 检查是否向腰部/口袋移动
                        moving_to_pocket = False
                        if curr_right[1] > prev_right[1] and curr_right[0] < prev_right[0]:  # 向左下方移动
                            if right_hip and curr_right[0] > right_hip[0] and abs(curr_right[1] - right_hip[1]) < 50:
                                moving_to_pocket = True
                        
                        if moving_to_pocket:
                            behavior = {
                                'type': 'item_concealment',
                                'confidence': 0.75,
                                'details': '手向口袋/身体移动，可能在隐藏物品',
                                'bbox': person_bbox,
                                'color': (0, 0, 255)  # 红色
                            }
                            behaviors.append(behavior)
                            self.logger.info("检测到手向口袋/身体移动，可能在隐藏物品")
            else:
                # 初始化手腕位置历史
                self._left_wrist_history = [left_wrist]
                self._right_wrist_history = [right_wrist]
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"异常伸手行为检测错误: {str(e)}")
            return behaviors
    
    def _detect_body_shielding(self, image, landmarks, person_bbox, object_detections):
        """
        检测身体遮挡行为
        
        Args:
            image: 输入图像
            landmarks: 姿态关键点
            person_bbox: 人物边界框
            object_detections: 物体检测结果
            
        Returns:
            behaviors: 检测到的行为列表
        """
        behaviors = []
        
        try:
            if not landmarks or len(landmarks) < 11:  # 至少需要包含躯干的关键点
                return behaviors
            
            # 如果没有物体检测结果，不进行身体遮挡检测
            if not object_detections or len(object_detections) == 0:
                return behaviors
            
            # 获取身体中心
            left_shoulder = landmarks[5] if len(landmarks) > 5 else None
            right_shoulder = landmarks[6] if len(landmarks) > 6 else None
            left_hip = landmarks[11] if len(landmarks) > 11 else None
            right_hip = landmarks[12] if len(landmarks) > 12 else None
            
            if left_shoulder is None or right_shoulder is None or \
               left_hip is None or right_hip is None:
                return behaviors
            
            # 计算躯干中心
            torso_center_x = (left_shoulder[0] + right_shoulder[0] + left_hip[0] + right_hip[0]) / 4
            torso_center_y = (left_shoulder[1] + right_shoulder[1] + left_hip[1] + right_hip[1]) / 4
            
            # 计算躯干宽度
            torso_width = max(abs(left_shoulder[0] - right_shoulder[0]), abs(left_hip[0] - right_hip[0]))
            
            # 识别位于人物躯干前方的物体
            for obj in object_detections:
                if 'bbox' not in obj:
                    continue
                
                obj_bbox = obj['bbox']
                obj_center_x = (obj_bbox[0] + obj_bbox[2]) / 2
                obj_center_y = (obj_bbox[1] + obj_bbox[3]) / 2
                
                # 计算物体到躯干中心的距离
                distance = math.sqrt((obj_center_x - torso_center_x)**2 + (obj_center_y - torso_center_y)**2)
                
                # 如果物体靠近躯干，检测是否有遮挡行为
                if distance < torso_width * 1.2:
                    # 检查物体是否被躯干遮挡（在躯干前方）
                    is_shielded = abs(obj_center_x - torso_center_x) < torso_width * 0.5
                    
                    if is_shielded:
                        behavior = {
                            'type': 'body_shielding',
                            'confidence': 0.7,
                            'details': f"使用身体遮挡物品: {obj.get('class', 'object')}",
                            'bbox': person_bbox,
                            'color': (0, 0, 255)  # 红色
                        }
                        behaviors.append(behavior)
                        self.logger.info(f"检测到身体遮挡行为 (物体: {obj.get('class', 'object')})")
                        break
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"身体遮挡行为检测错误: {str(e)}")
            traceback.print_exc()
            return behaviors
        
    def _detect_abnormal_head_movement(self, image, landmarks, person_bbox):
        """
        检测异常头部运动，如快速左右环顾
        视频处理中降低检测阈值，增强敏感度
        """
        behaviors = []
        
        try:
            if not landmarks or len(landmarks) < 1:
                return behaviors
            
            # 获取头部关键点
            nose = landmarks[0]
            left_ear = landmarks[3] if len(landmarks) > 3 else None
            right_ear = landmarks[4] if len(landmarks) > 4 else None
            
            if nose is None or left_ear is None or right_ear is None:
                return behaviors
            
            # 计算头部中心点
            head_center_x = (left_ear[0] + right_ear[0]) / 2
            head_center_y = (left_ear[1] + right_ear[1]) / 2
            
            # 计算鼻子与头部中心的水平偏移量
            horizontal_offset = nose[0] - head_center_x
            
            # 计算偏移量相对于头部宽度的比例
            head_width = abs(right_ear[0] - left_ear[0])
            if head_width > 0:
                offset_ratio = horizontal_offset / head_width
            else:
                offset_ratio = 0
            
            # 视频处理中使用更敏感的阈值
            head_turn_threshold = 0.15  # 降低阈值，使检测更敏感
            
            # 根据偏移量判断头部转向
            head_direction = None
            confidence = 0.0
            
            if offset_ratio > head_turn_threshold:
                head_direction = "右侧"
                confidence = min(0.5 + abs(offset_ratio), 0.9)
            elif offset_ratio < -head_turn_threshold:
                head_direction = "左侧"
                confidence = min(0.5 + abs(offset_ratio), 0.9)
            
            # 如果检测到明显的头部转向，添加行为
            if head_direction:
                behavior = {
                    'type': 'looking_around',
                    'confidence': confidence,
                    'details': f'头部转向{head_direction}',
                    'bbox': person_bbox,
                    'color': (0, 165, 255)  # 橙色
                }
                behaviors.append(behavior)
                self.logger.info(f"检测到异常头部运动: 向{head_direction}看 (置信度: {confidence:.2f})")
            
            # 检测频繁头部移动 - 视频特有的检测
            if hasattr(self, '_prev_head_positions'):
                # 更新头部位置历史
                self._head_position_history.append((head_center_x, head_center_y))
                if len(self._head_position_history) > 10:  # 保留最近10帧的头部位置
                    self._head_position_history.pop(0)
                
                # 计算头部移动的变化量
                if len(self._head_position_history) >= 3:
                    movements = []
                    for i in range(1, len(self._head_position_history)):
                        prev_pos = self._head_position_history[i-1]
                        curr_pos = self._head_position_history[i]
                        movement = math.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                        movements.append(movement)
                    
                    # 检测头部移动的频率和幅度
                    avg_movement = sum(movements) / len(movements)
                    direction_changes = 0
                    
                    # 检测方向变化
                    for i in range(1, len(movements)):
                        if (movements[i] - movements[i-1]) * (movements[i-1] - (movements[i-2] if i > 1 else 0)) < 0:
                            direction_changes += 1
                    
                    # 如果检测到频繁的头部移动，添加可疑行为
                    if (avg_movement > 5 and direction_changes >= 1) or direction_changes >= 2:
                        behavior = {
                            'type': 'suspicious_head_movement',
                            'confidence': min(0.5 + 0.1 * direction_changes, 0.9),
                            'details': '频繁头部移动，可能在观察周围环境',
                            'bbox': person_bbox,
                            'color': (0, 165, 255)  # 橙色
                        }
                        behaviors.append(behavior)
                        self.logger.info(f"检测到频繁头部移动 (方向变化: {direction_changes}, 平均移动: {avg_movement:.2f})")
            else:
                # 初始化头部位置历史
                self._head_position_history = [(head_center_x, head_center_y)]
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"头部动作检测错误: {str(e)}")
            return behaviors
    
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
        
        # 避免除零错误
        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0
        
        # 计算夹角的余弦值
        cos_angle = dot_product / (magnitude_ba * magnitude_bc)
        
        # 防止数值误差导致cos_angle超出[-1, 1]范围
        cos_angle = max(-1, min(1, cos_angle))
        
        # 计算角度并转换为度数
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle 

    def process_frame(self, frame, frame_number, prev_frame=None, object_history=None, prev_person_data=None):
        """
        处理单帧视频并检测行为
        
        Args:
            frame: 当前帧图像
            frame_number: 帧号
            prev_frame: 上一帧图像（可选）
            object_history: 物体历史数据（可选）
            prev_person_data: 上一帧的人物数据（可选）
        
        Returns:
            processed_frame: 处理后的帧
            frame_results: 帧处理结果
            behaviors: 检测到的行为列表
        """
        try:
            # 初始化返回值
            frame_results = {'detected_objects': [], 'person_data': None}
            behaviors = []
            objects_of_interest = []
            processed_frame = frame.copy()
            
            # 在图像中检测物体
            objects = self.object_detector.detect_objects(frame)
            if objects:
                frame_results['detected_objects'] = objects
                
                # 绘制检测到的物体
                for obj in objects:
                    x1, y1, x2, y2 = obj['bbox']
                    label = obj['label']
                    conf = obj['confidence']
                    
                    # 保存感兴趣的物体
                    if label in self.ITEMS_OF_INTEREST:
                        objects_of_interest.append(obj)
                    
                    # 绘制边界框和标签
                    color = (0, 255, 0)  # 绿色边界框
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{label}: {conf:.2f}"
                    cv2.putText(processed_frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            # 检测人物姿态和关键点
            pose_results = self._extract_pose_landmarks(frame)
            
            if pose_results and 'landmarks' in pose_results and pose_results['landmarks']:
                frame_results['person_data'] = pose_results
                
                # 提取人物边界框
                person_bbox = pose_results.get('bbox', None)
                if person_bbox:
                    x1, y1, x2, y2 = person_bbox
                    
                    # 绘制人物边界框
                    cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(processed_frame, "Person", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # 绘制姿态关键点
                if 'mp_results' in pose_results:
                    self._draw_pose_landmarks(processed_frame, pose_results['mp_results'])
                
                # 分析服装特征
                clothing_features = self._analyze_clothing(frame, pose_results['landmarks'], person_bbox)
                if clothing_features:
                    frame_results['clothing_features'] = clothing_features
                    
                # 检测行为
                try:
                    # 检测基于物体的行为
                    if objects_of_interest:
                        obj_related_behaviors = self._detect_object_related_behaviors(
                            frame, 
                            pose_results['landmarks'], 
                            objects_of_interest,
                            person_bbox
                        )
                        if obj_related_behaviors:
                            behaviors.extend(obj_related_behaviors)
                    
                    # 即使没有检测到物体，也检测行为
                    landmarks = pose_results['landmarks']
                    
                    # 添加物品拿取和隐藏行为检测
                    item_behaviors = self._detect_item_grabbing_and_concealment(frame, landmarks, person_bbox)
                    if item_behaviors:
                        behaviors.extend(item_behaviors)
                    
                    # 检测不需要物体的可疑行为
                    posture_behaviors = self._detect_suspicious_postures(frame, landmarks, person_bbox)
                    if posture_behaviors:
                        behaviors.extend(posture_behaviors)
                    
                    # 检测注视行为
                    gaze_behaviors = self._detect_gaze_behaviors(frame, landmarks, person_bbox)
                    if gaze_behaviors:
                        behaviors.extend(gaze_behaviors)
                    
                    # 检测手背后行为
                    hands_behind_behaviors = self._detect_hands_behind_back(frame, landmarks, person_bbox)
                    if hands_behind_behaviors:
                        behaviors.extend(hands_behind_behaviors)
                        
                    # 检测交叉手臂行为
                    arms_crossed_behaviors = self._detect_arms_crossed(frame, landmarks, person_bbox)
                    if arms_crossed_behaviors:
                        behaviors.extend(arms_crossed_behaviors)
                    
                    # 检测异常手臂位置行为
                    arm_position_behaviors = self._detect_abnormal_arm_positions(frame, landmarks, person_bbox)
                    if arm_position_behaviors:
                        behaviors.extend(arm_position_behaviors)
                    
                    # 检测蹲伏行为  
                    crouching_behaviors = self._detect_suspicious_crouching(frame, landmarks, person_bbox)
                    if crouching_behaviors:
                        behaviors.extend(crouching_behaviors)
                        
                    # 使用行为分类器预测行为
                    if self.behavior_classifier:
                        predicted_behaviors = self.predict_behaviors(landmarks, person_bbox, prev_person_data)
                        if predicted_behaviors:
                            behaviors.extend(predicted_behaviors)
                except Exception as e:
                    self.logger.error(f"行为检测错误: {str(e)}")
                    traceback.print_exc()
            
            # 绘制检测到的行为
            if behaviors:
                self._draw_behaviors(processed_frame, behaviors)
                
            # 优化标签位置，避免重叠
            self._optimize_labels(processed_frame, frame_results, behaviors)
            
            return processed_frame, frame_results, behaviors
            
        except Exception as e:
            self.logger.error(f"处理帧 {frame_number} 时出错: {str(e)}")
            traceback.print_exc()
            return frame.copy(), {}, []

    def _adjust_theft_probability_in_retail(self, behaviors, is_retail_environment=True):
        """在零售环境中调整特定行为的盗窃概率
        
        Args:
            behaviors: 检测到的行为列表
            is_retail_environment: 是否为零售环境
            
        Returns:
            adjusted_behaviors: 调整后的行为列表
        """
        if not is_retail_environment or not behaviors:
            return behaviors
            
        # 零售环境中更可疑的行为类型
        highly_suspicious_in_retail = [
            "Single Arm Hiding", 
            "Abnormal Arm Position",
            "Body Shielding",
            "Rapid Item Concealment"
        ]
        
        # 检查是否有可疑行为
        has_suspicious_behavior = any(b['type'] in highly_suspicious_in_retail for b in behaviors)
        
        # 如果检测到可疑行为，但置信度不够高，则提高置信度
        if has_suspicious_behavior:
            adjusted_behaviors = []
            for behavior in behaviors:
                if behavior['type'] in highly_suspicious_in_retail:
                    # 在零售环境中，这类行为的置信度至少应达到0.7
                    if behavior['confidence'] < 0.7:
                        behavior = behavior.copy()  # 避免修改原对象
                        behavior['confidence'] = max(behavior['confidence'], 0.7)
                        behavior['description'] += " (零售环境中的可疑行为)"
                adjusted_behaviors.append(behavior)
                
            # 添加额外的综合性盗窃行为提示
            if all(b['type'] != "Retail Theft Suspicion" for b in adjusted_behaviors):
                # 计算综合置信度
                suspicious_behaviors = [b for b in adjusted_behaviors if b['type'] in highly_suspicious_in_retail]
                if suspicious_behaviors:
                    avg_confidence = sum(b['confidence'] for b in suspicious_behaviors) / len(suspicious_behaviors)
                    # 使用最大的边界框
                    bbox = max([b['bbox'] for b in suspicious_behaviors], key=lambda box: (box[2]-box[0])*(box[3]-box[1]))
                    
                    adjusted_behaviors.append({
                        'type': "Retail Theft Suspicion",
                        'description': "零售环境中的可疑盗窃行为",
                        'confidence': min(0.9, avg_confidence + 0.15),  # 额外提升15%置信度
                        'bbox': bbox,
                        'color': (0, 0, 255)  # 红色
                    })
            
            return adjusted_behaviors
        
        return behaviors

    def load_behavior_classifier(self):
        """
        加载行为分类器模型
        
        Returns:
            classifier: 行为分类器对象
        """
        try:
            # 尝试从配置中获取模型路径
            model_path = self.config.get('behavior_classifier_path')
            if not model_path:
                # 使用默认路径
                model_path = os.path.join('models', 'behavior', 'classifier', 'behavior_classifier.pkl')
            
            self.logger.info(f"尝试加载行为分类器: {model_path}")
            
            if not os.path.exists(model_path):
                self.logger.warning(f"行为分类器模型文件不存在: {model_path}")
                return None
            
            # 加载模型
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # 检查加载的对象是否为字典（配置），如果是则尝试初始化实际模型
                if isinstance(model, dict):
                    # 先尝试直接取 model['model'] 里已经训练好的 XGBClassifier
                    if 'model' in model and hasattr(model['model'], 'predict_proba'):
                        self.logger.info("检测到模型配置字典，直接使用内置模型对象")
                        return model['model']
                    
                    self.logger.info("检测到模型配置字典，尝试初始化模型")
                    
                    # 从配置中提取模型参数
                    model_type = model.get('model_type', 'unknown')
                    model_params = model.get('params', {})
                    
                    # 根据模型类型初始化实际模型
                    if model_type == 'random_forest':
                        from sklearn.ensemble import RandomForestClassifier
                        classifier = RandomForestClassifier(**model_params)
                        if 'fitted_model' in model:
                            # 直接使用已拟合的模型数据
                            for key, value in model['fitted_model'].items():
                                setattr(classifier, key, value)
                        # 添加必要的属性
                        if 'classes_' not in model and 'classes' in model:
                            classifier.classes_ = model['classes']
                        return classifier
                    
                    elif model_type == 'xgboost':
                        try:
                            import xgboost as xgb
                            classifier = xgb.XGBClassifier(**model_params)
                            if 'fitted_model' in model:
                                for key, value in model['fitted_model'].items():
                                    setattr(classifier, key, value)
                            # 添加必要的属性
                            if 'classes_' not in model and 'classes' in model:
                                classifier.classes_ = model['classes']
                            return classifier
                        except ImportError:
                            self.logger.error("XGBoost模型需要但未安装")
                            return None
                    
                    elif model_type == 'svm':
                        from sklearn.svm import SVC
                        classifier = SVC(**model_params, probability=True)
                        if 'fitted_model' in model:
                            for key, value in model['fitted_model'].items():
                                setattr(classifier, key, value)
                        # 添加必要的属性
                        if 'classes_' not in model and 'classes' in model:
                            classifier.classes_ = model['classes']
                        return classifier
                        
                    else:
                        # 如果是未知模型类型，但包含必要的predict_proba方法和classes_属性
                        # 创建一个简单的包装器类
                        class ModelWrapper:
                            def __init__(self, config):
                                self.config = config
                                self.classes_ = config.get('classes', ['normal', 'suspicious'])
                            
                            def predict_proba(self, X):
                                # 返回默认概率
                                return np.array([[0.7, 0.3]] * len(X))
                        
                        self.logger.warning(f"使用通用模型包装器代替未知类型: {model_type}")
                        return ModelWrapper(model)
                        
                else:
                    # 如果加载的不是字典，直接返回模型对象
                    return model
                    
            except Exception as e:
                self.logger.error(f"加载行为分类器时出错: {str(e)}")
                traceback.print_exc()
                return None
                
        except Exception as e:
            self.logger.error(f"初始化行为分类器时出错: {str(e)}")
            traceback.print_exc()
            return None
            
    def predict_behaviors(self, features, prev_features=None):
        """
        使用行为分类器预测行为
        
        Args:
            features: 特征向量
            prev_features: 前一帧的特征向量（可选）
            
        Returns:
            behaviors: 预测的行为列表
        """
        behaviors = []
        
        try:
            if self.behavior_classifier is None:
                return []
            
            try:
                # 检查行为分类器类型
                if isinstance(self.behavior_classifier, dict):
                    self.logger.warning("行为分类器是字典类型，尝试重新加载模型")
                    self.behavior_classifier = self.load_behavior_classifier()
                
                if not hasattr(self.behavior_classifier, 'predict_proba'):
                    self.logger.error(f"行为分类器不是有效的模型对象: {type(self.behavior_classifier)}")
                    
                    # 创建临时分类器
                    class TempClassifier:
                        def __init__(self):
                            self.classes_ = ['normal', 'suspicious']
                        
                        def predict_proba(self, X):
                            # 返回安全的默认值
                            return np.array([[0.7, 0.3]] * len(X))
                    
                    self.behavior_classifier = TempClassifier()
                    self.logger.info("已创建临时分类器代替无效模型")
                
                # 准备输入特征 - 可能需要组合当前和前一帧特征
                if prev_features is not None:
                    # 计算变化特征
                    feature_changes = [curr - prev for curr, prev in zip(features, prev_features)]
                    # 合并静态特征和变化特征
                    combined_features = features + feature_changes
                else:
                    combined_features = features

                # 确保特征格式正确
                X = np.array([combined_features])

                # 预测行为
                probabilities = self.behavior_classifier.predict_proba(X)[0]
                class_names = self.behavior_classifier.classes_

                # 获取配置中的行为阈值
                behavior_thresholds = self.config.get('behavior_thresholds', {})
                default_threshold = self.config.get('default_behavior_threshold', 0.5)

                # 提取有意义的行为预测（概率超过阈值）
                behaviors = []
                for class_idx, class_name in enumerate(class_names):
                    prob = probabilities[class_idx]
                    # 获取该行为的特定阈值，如果没有则使用默认阈值
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
                self.logger.error(f"行为分类器预测出错: {str(e)}")
                traceback.print_exc()
                return []

        except Exception as e:
            self.logger.error(f"预测行为时出错: {str(e)}")
            traceback.print_exc()
            return []

    def _detect_item_grabbing_and_concealment(self, frame, landmarks, person_bbox):
        """
        检测拿取物品和隐藏行为 - 增强版
        此方法无需依赖物体检测结果，而是通过分析手部动作来检测可能的拿取和隐藏行为
        
        Args:
            frame: 输入图像帧
            landmarks: 姿态关键点
            person_bbox: 人物边界框 [x1, y1, x2, y2]
            
        Returns:
            检测到的行为列表
        """
        if not landmarks or not isinstance(landmarks, (list, np.ndarray)) or len(landmarks) < 15:
            return []
            
        try:
            behaviors = []
            
            # 获取左右手腕关键点
            left_wrist = landmarks[9]
            right_wrist = landmarks[10]
            
            # 获取关键点
            nose = landmarks[0]
            left_shoulder = landmarks[5]
            right_shoulder = landmarks[6]
            left_elbow = landmarks[7]
            right_elbow = landmarks[8]
            left_hip = landmarks[11]
            right_hip = landmarks[12]
            
            # 计算身体中心线
            mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
            mid_hip_x = (left_hip[0] + right_hip[0]) / 2
            body_center_x = (mid_shoulder_x + mid_hip_x) / 2
            
            # 计算身体高度和宽度 - 用于确定阈值
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            body_height = max(abs(left_shoulder[1] - left_hip[1]), abs(right_shoulder[1] - right_hip[1]))
            
            # 创建或更新历史记录
            if not hasattr(self, '_item_grab_history'):
                self._item_grab_history = {
                    'positions': {
                        'left_wrist': [],
                        'right_wrist': []
                    },
                    'detected_phases': {
                        'left_reach': False,
                        'left_grab': False,
                        'left_concealment': False,
                        'right_reach': False,
                        'right_grab': False,
                        'right_concealment': False
                    },
                    'grab_time': {
                        'left': 0,
                        'right': 0
                    },
                    'frame_count': 0,
                    'events': []
                }
            
            # 更新帧计数
            self._item_grab_history['frame_count'] += 1
            
            # 更新手腕位置历史 - 使用相对于身体的位置，这样更稳定
            left_rel_x = left_wrist[0] - body_center_x
            left_rel_y = left_wrist[1] - left_shoulder[1]
            right_rel_x = right_wrist[0] - body_center_x
            right_rel_y = right_wrist[1] - right_shoulder[1]
            
            self._item_grab_history['positions']['left_wrist'].append((left_rel_x, left_rel_y))
            self._item_grab_history['positions']['right_wrist'].append((right_rel_x, right_rel_y))
            
            # 保持历史长度适中 - 大约2秒的帧数
            max_history = 40  # 假设视频是20fps
            if len(self._item_grab_history['positions']['left_wrist']) > max_history:
                self._item_grab_history['positions']['left_wrist'] = self._item_grab_history['positions']['left_wrist'][-max_history:]
                self._item_grab_history['positions']['right_wrist'] = self._item_grab_history['positions']['right_wrist'][-max_history:]
            
            # 只有当历史足够长时才进行分析
            if len(self._item_grab_history['positions']['left_wrist']) < 10:
                return behaviors
                
            # 分析左手动作序列
            self._analyze_hand_sequence(
                self._item_grab_history['positions']['left_wrist'], 
                'left',
                shoulder_width,
                body_height,
                behaviors,
                person_bbox
            )
            
            # 分析右手动作序列
            self._analyze_hand_sequence(
                self._item_grab_history['positions']['right_wrist'], 
                'right',
                shoulder_width,
                body_height,
                behaviors,
                person_bbox
            )
            
            # 检测拿取后立即隐藏的组合行为
            self._detect_grab_conceal_sequence(behaviors, person_bbox)
            
            return behaviors
            
        except Exception as e:
            self.logger.error(f"检测物品拿取和隐藏行为时出错: {str(e)}")
            traceback.print_exc()
            return []
            
    def _analyze_hand_sequence(self, positions, hand, shoulder_width, body_height, behaviors, person_bbox):
        """
        分析手部动作序列，检测拿取、隐藏等行为
        
        Args:
            positions: 手腕位置历史
            hand: 'left'或'right'
            shoulder_width: 肩宽
            body_height: 身高
            behaviors: 输出行为列表
            person_bbox: 人物边界框
        """
        try:
            # 计算手部运动情况
            movement_x = []
            movement_y = []
            
            for i in range(1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                movement_x.append(dx)
                movement_y.append(dy)
            
            # 至少需要8帧的运动数据
            if len(movement_x) < 8:
                return
                
            # 1. 分析手部运动的总幅度 - 排除小幅抖动
            total_movement_x = abs(sum(movement_x))
            total_movement_y = abs(sum(movement_y))
            
            # 如果总体运动过小，认为是静止或轻微抖动，不进行行为检测
            if total_movement_x < shoulder_width * 0.2 and total_movement_y < body_height * 0.15:
                # 重置检测状态 - 清除可能的误检
                if self._item_grab_history['frame_count'] % 5 == 0:  # 每5帧检查一次
                    self._item_grab_history['detected_phases'][f'{hand}_reach'] = False
                return
                
            # 2. 分段分析手部运动 - 更准确地识别伸出动作
            # 划分为前期、中期和后期
            segment_size = len(movement_x) // 3
            first_segment_x = sum(movement_x[:segment_size])
            middle_segment_x = sum(movement_x[segment_size:2*segment_size])
            last_segment_x = sum(movement_x[2*segment_size:])
            
            # 计算起点和终点的位置
            start_pos = positions[0]
            mid_pos = positions[len(positions)//2]
            end_pos = positions[-1]
            
            # 计算运动矢量的方向变化
            directions = []
            for i in range(len(movement_x)-1):
                if movement_x[i] * movement_x[i+1] < 0:  # 方向改变
                    directions.append(1)
                else:
                    directions.append(0)
                    
            direction_changes = sum(directions)
            
            # 3. 检测伸手行为 - 需要明显的伸出动作和较少的方向变化
            reach_threshold = shoulder_width * 0.3  # 提高伸展阈值
            reach_detected = False
            
            # 左右手各自的伸展检测逻辑
            if hand == 'left':
                # 左手向右伸出 - 要求前期向右运动大，方向变化少
                if first_segment_x > reach_threshold and direction_changes < len(movement_x) // 3:
                    # 添加额外约束：起点位置在左侧，中点位置更靠右
                    if start_pos[0] < -shoulder_width * 0.2 and mid_pos[0] > start_pos[0] + shoulder_width * 0.2:
                        reach_detected = True
            elif hand == 'right':
                # 右手向左伸出 - 要求前期向左运动大，方向变化少
                if first_segment_x < -reach_threshold and direction_changes < len(movement_x) // 3:
                    # 添加额外约束：起点位置在右侧，中点位置更靠左
                    if start_pos[0] > shoulder_width * 0.2 and mid_pos[0] < start_pos[0] - shoulder_width * 0.2:
                        reach_detected = True
            
            # 4. 连续性检测 - 避免误判单个动作
            if not hasattr(self, '_reach_detection_history'):
                self._reach_detection_history = {
                    'left_count': 0,
                    'left_confirmed': False,
                    'right_count': 0,
                    'right_confirmed': False,
                    'frame_gap': 0
                }
            
            # 更新连续性检测
            if reach_detected:
                self._reach_detection_history[f'{hand}_count'] += 1
                # 需要连续3帧以上检测到才确认
                if self._reach_detection_history[f'{hand}_count'] >= 3:
                    self._reach_detection_history[f'{hand}_confirmed'] = True
            else:
                # 如果当前帧没有检测到，计数器减少
                self._reach_detection_history[f'{hand}_count'] = max(0, self._reach_detection_history[f'{hand}_count'] - 1)
                if self._reach_detection_history[f'{hand}_count'] == 0:
                    self._reach_detection_history[f'{hand}_confirmed'] = False
            
            # 只有当确认伸手动作时才记录
            if self._reach_detection_history[f'{hand}_confirmed'] and not self._item_grab_history['detected_phases'][f'{hand}_reach']:
                self._item_grab_history['detected_phases'][f'{hand}_reach'] = True
                
                # 记录伸展行为 - 只记录一次
                behavior = {
                    'type': 'reaching_motion',
                    'confidence': 0.7,
                    'details': f'检测到{hand}手伸展动作',
                    'bbox': person_bbox,
                    'color': (255, 215, 0)  # 金色
                }
                behaviors.append(behavior)
                self.logger.info(f"检测到{hand}手确认的伸展动作")
                
                # 重置帧间隔计数器
                self._reach_detection_history['frame_gap'] = 0
            
            # 5. 检测抓取动作 - 伸展后向内收回
            grab_detected = False
            
            # 只有在先前确认有伸手动作的情况下才检测抓取
            if self._item_grab_history['detected_phases'][f'{hand}_reach']:
                # 更新帧间隔
                self._reach_detection_history['frame_gap'] += 1
                
                # 检查后半段运动 - 方向与伸出相反
                if hand == 'left':
                    # 左手向左收回
                    if last_segment_x < -reach_threshold * 0.8:
                        grab_detected = True
                elif hand == 'right':
                    # 右手向右收回
                    if last_segment_x > reach_threshold * 0.8:
                        grab_detected = True
                
                # 超过一定帧数没有检测到抓取，重置伸手检测状态
                if self._reach_detection_history['frame_gap'] > 20:  # 约1秒
                    self._item_grab_history['detected_phases'][f'{hand}_reach'] = False
            
            # 6. 连续性检测抓取行为
            if not hasattr(self, '_grab_detection_history'):
                self._grab_detection_history = {
                    'left_count': 0,
                    'left_confirmed': False,
                    'right_count': 0,
                    'right_confirmed': False
                }
            
            # 更新连续性检测
            if grab_detected:
                self._grab_detection_history[f'{hand}_count'] += 1
                # 需要连续2帧以上检测到才确认
                if self._grab_detection_history[f'{hand}_count'] >= 2:
                    self._grab_detection_history[f'{hand}_confirmed'] = True
            else:
                # 如果当前帧没有检测到，计数器减少
                self._grab_detection_history[f'{hand}_count'] = max(0, self._grab_detection_history[f'{hand}_count'] - 1)
                if self._grab_detection_history[f'{hand}_count'] == 0:
                    self._grab_detection_history[f'{hand}_confirmed'] = False
            
            # 只有当确认抓取动作时才记录
            if self._grab_detection_history[f'{hand}_confirmed'] and not self._item_grab_history['detected_phases'][f'{hand}_grab']:
                self._item_grab_history['detected_phases'][f'{hand}_grab'] = True
                self._item_grab_history['grab_time'][hand] = self._item_grab_history['frame_count']
                
                # 记录抓取行为 - 只记录一次
                behavior = {
                    'type': 'item_grabbing',
                    'confidence': 0.8,
                    'details': f'检测到{hand}手抓取物品',
                    'bbox': person_bbox,
                    'color': (0, 255, 255)  # 黄色
                }
                behaviors.append(behavior)
                self.logger.info(f"检测到{hand}手确认的抓取物品行为")
                
                # 重置伸手状态 - 抓取后重新开始检测新的伸手动作
                self._item_grab_history['detected_phases'][f'{hand}_reach'] = False
            
            # 7. 检测隐藏动作 - 向身体中心移动
            conceal_detected = False
            
            # 只有在先前确认有抓取动作的情况下才检测隐藏
            if self._item_grab_history['detected_phases'][f'{hand}_grab']:
                # 计算接近中心线的运动
                recent_x_movement = sum(movement_x[-4:])  # 最近4帧的水平运动
                recent_y_movement = sum(movement_y[-4:])  # 最近4帧的垂直运动
                
                # 手腕当前位置和起始位置
                current_pos = positions[-1]
                
                # 检查是否是向中心移动的模式
                if hand == 'left':
                    # 左手向内移动并靠近身体中心
                    if recent_x_movement < -shoulder_width * 0.15 and abs(current_pos[0]) < shoulder_width * 0.3:
                        conceal_detected = True
                else:
                    # 右手向内移动并靠近身体中心
                    if recent_x_movement > shoulder_width * 0.15 and abs(current_pos[0]) < shoulder_width * 0.3:
                        conceal_detected = True
                        
                # 也检查向下移动 - 可能是放入口袋
                if (recent_y_movement > body_height * 0.12 and 
                    abs(current_pos[0]) < shoulder_width * 0.4 and 
                    current_pos[1] > body_height * 0.3):  # 手位置较低
                    conceal_detected = True
                
                # 检查抓取后经过的帧数 - 隐藏通常发生在抓取后不久
                frames_since_grab = self._item_grab_history['frame_count'] - self._item_grab_history['grab_time'][hand]
                if frames_since_grab > 60:  # 超过3秒
                    # 重置抓取状态 - 太久没有隐藏动作，可能是误检
                    self._item_grab_history['detected_phases'][f'{hand}_grab'] = False
            
            # 8. 连续性检测隐藏行为
            if not hasattr(self, '_conceal_detection_history'):
                self._conceal_detection_history = {
                    'left_count': 0,
                    'left_confirmed': False,
                    'right_count': 0,
                    'right_confirmed': False
                }
            
            # 更新连续性检测
            if conceal_detected:
                self._conceal_detection_history[f'{hand}_count'] += 1
                # 需要连续2帧以上检测到才确认
                if self._conceal_detection_history[f'{hand}_count'] >= 2:
                    self._conceal_detection_history[f'{hand}_confirmed'] = True
            else:
                # 如果当前帧没有检测到，计数器减少
                self._conceal_detection_history[f'{hand}_count'] = max(0, self._conceal_detection_history[f'{hand}_count'] - 1)
                if self._conceal_detection_history[f'{hand}_count'] == 0:
                    self._conceal_detection_history[f'{hand}_confirmed'] = False
            
            # 只有当确认隐藏动作时才记录
            if self._conceal_detection_history[f'{hand}_confirmed'] and not self._item_grab_history['detected_phases'][f'{hand}_concealment']:
                self._item_grab_history['detected_phases'][f'{hand}_concealment'] = True
                
                # 检查是否在抓取后短时间内发生隐藏动作
                grab_frame = self._item_grab_history['grab_time'][hand]
                frames_since_grab = self._item_grab_history['frame_count'] - grab_frame
                
                # 只在合理的时间范围内报告隐藏行为
                if 0 < frames_since_grab < 45:  # 约2秒内
                    # 计算信心值提升
                    confidence_boost = 0.15 if frames_since_grab < 20 else 0.05
                    
                    # 记录隐藏行为
                    behavior = {
                        'type': 'item_concealment',
                        'confidence': 0.75 + confidence_boost,
                        'details': f'检测到{hand}手隐藏物品',
                        'bbox': person_bbox,
                        'color': (0, 0, 255)  # 红色
                    }
                    behaviors.append(behavior)
                    self.logger.info(f"检测到{hand}手确认的隐藏物品行为")
                
            # 9. 防止检测状态长时间保持 - 定期重置
            # 每100帧(约5秒)重置所有检测状态，防止长时间误检
            if self._item_grab_history['frame_count'] % 100 == 0:
                self._reach_detection_history[f'{hand}_count'] = 0
                self._reach_detection_history[f'{hand}_confirmed'] = False
                self._grab_detection_history[f'{hand}_count'] = 0
                self._grab_detection_history[f'{hand}_confirmed'] = False
                self._conceal_detection_history[f'{hand}_count'] = 0
                self._conceal_detection_history[f'{hand}_confirmed'] = False
                self._item_grab_history['detected_phases'][f'{hand}_reach'] = False
                self._item_grab_history['detected_phases'][f'{hand}_grab'] = False
                self._item_grab_history['detected_phases'][f'{hand}_concealment'] = False
                
        except Exception as e:
            self.logger.error(f"分析手部动作序列时出错: {str(e)}")
            traceback.print_exc()
            
    def _detect_grab_conceal_sequence(self, behaviors, person_bbox):
        """
        检测抓取后隐藏的完整行为序列
        
        Args:
            behaviors: 当前已检测的行为
            person_bbox: 人物边界框
        """
        try:
            # 检查是否有完整的抓取+隐藏序列
            left_complete = (self._item_grab_history['detected_phases']['left_grab'] and 
                           self._item_grab_history['detected_phases']['left_concealment'])
                            
            right_complete = (self._item_grab_history['detected_phases']['right_grab'] and 
                            self._item_grab_history['detected_phases']['right_concealment'])
            
            if left_complete or right_complete:
                hand = '左手' if left_complete else '右手'
                
                # 检查时间差 - 只有在抓取后不久发生隐藏才视为组合行为
                grab_frame = self._item_grab_history['grab_time']['left' if left_complete else 'right']
                frames_since_grab = self._item_grab_history['frame_count'] - grab_frame
                
                # 检查是否已经报告过这个组合行为
                event_key = f"{hand}_grab_conceal_{grab_frame}"
                already_reported = event_key in self._item_grab_history['events']
                
                # 只在合适的时间范围内且未报告过的情况下添加组合行为
                if 0 < frames_since_grab < 45 and not already_reported:  # 约2秒内
                    # 添加组合行为 - 高置信度
                    behavior = {
                        'type': 'grab_and_conceal',
                        'confidence': 0.92,
                        'details': f'检测到{hand}抓取并隐藏物品行为序列',
                        'bbox': person_bbox,
                        'color': (0, 0, 255)  # 红色
                    }
                    behaviors.append(behavior)
                    
                    # 记录此事件（防止重复报告）
                    self._item_grab_history['events'].append(event_key)
                    self.logger.info(f"检测到完整的{hand}抓取并隐藏物品行为序列")
                    
                    # 重置状态
                    side = 'left' if left_complete else 'right'
                    self._item_grab_history['detected_phases'][f'{side}_grab'] = False
                    self._item_grab_history['detected_phases'][f'{side}_concealment'] = False
                
                # 超过一定时间未报告，也重置状态
                elif frames_since_grab >= 45:
                    side = 'left' if left_complete else 'right'
                    self._item_grab_history['detected_phases'][f'{side}_grab'] = False
                    self._item_grab_history['detected_phases'][f'{side}_concealment'] = False
                    
        except Exception as e:
            self.logger.error(f"检测抓取并隐藏序列时出错: {str(e)}")
            traceback.print_exc()

    def _draw_behaviors(self, frame, behaviors):
        """在帧上绘制检测到的行为"""
        for behavior in behaviors:
            if 'bbox' in behavior:
                bbox = behavior['bbox']
                color = behavior.get('color', (0, 0, 255))  # 默认红色
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                           (int(bbox[2]), int(bbox[3])), color, 2)
                
                behavior_type = behavior['type']
                confidence = behavior.get('confidence', 0)
                label = f"{behavior_type} ({confidence:.2f})"
                
                # 添加标签
                cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                           
    def _optimize_labels(self, frame, frame_results, behaviors):
        """优化标签位置以避免重叠"""
        # ... existing code ...

    def _calculate_pose_bbox(self, landmarks, image_shape):
        """
        根据姿态关键点计算边界框
        
        Args:
            landmarks: 姿态关键点列表
            image_shape: 图像尺寸
            
        Returns:
            bbox: [x1, y1, x2, y2] 格式的边界框
        """
        if not landmarks or len(landmarks) < 5:
            # 返回整个图像范围
            return [0, 0, image_shape[1], image_shape[0]]
        
        # 提取所有有效的x,y坐标
        valid_points = []
        for lm in landmarks:
            if len(lm) >= 2:
                x, y = lm[0], lm[1]
                if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                    valid_points.append((x, y))
        
        if not valid_points:
            return [0, 0, image_shape[1], image_shape[0]]
        
        # 计算边界框
        x_coords = [p[0] for p in valid_points]
        y_coords = [p[1] for p in valid_points]
        
        x1 = max(0, min(x_coords) - 20)
        y1 = max(0, min(y_coords) - 20)
        x2 = min(image_shape[1], max(x_coords) + 20)
        y2 = min(image_shape[0], max(y_coords) + 20)
        
        return [int(x1), int(y1), int(x2), int(y2)]
