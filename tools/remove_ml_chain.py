#!/usr/bin/env python3
"""Remove the ML prediction chain from video_behavior.py."""

import re

path = "detection_engine/models/behavior/video_behavior.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Remove import pickle
content = content.replace("import pickle  # 添加pickle模块导入用于加载模型\n", "")

# 2. Remove ML attributes and calls from __init__
old_init_ml = """        # 初始化行为检测模型
        self.standard_model = None
        self.enhanced_model = None
        self.model_type = "enhanced"  # 默认使用增强模型
        self.load_xgboost_models()
        
        # 可能的行为分类模型
        self.behavior_classifier = None
        self.load_behavior_classifier()
        
        # 初始化自适应权重系统"""
new_init_ml = """        # 初始化自适应权重系统"""
content = content.replace(old_init_ml, new_init_ml)

# 3. Remove prediction_cache and model_stats from __init__
old_stats = """        # 用于跟踪行为连续性
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
        self.person_history = {}"""
new_stats = """        # 用于跟踪行为连续性
        self.consecutive_behaviors = defaultdict(int)
        
        # 存储过去的帧，用于计算动态特征
        self.frame_history = []
        self.max_frame_history = 5
        
        # 人物历史记录
        self.person_history = {}"""
content = content.replace(old_stats, new_stats)

# 4. Update comment in detect_behaviors_in_image
content = content.replace(
    "         # 使用姿态估计和ML模型检测行为",
    "         # 使用姿态估计和规则引擎检测行为"
)

# 5. Remove ML block from detect_behaviors_in_image
old_ml_block = """                if landmarks_dict and len(landmarks_dict) >= 12:
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
                        behaviors.extend(ml_behaviors)"""
new_ml_block = """                if landmarks_dict and len(landmarks_dict) >= 12:
                    # 保存当前帧的pose_results用于下一帧（保留原始格式）
                    self.prev_landmarks = pose_results
                    
                    # 使用基于规则的姿态行为检测（传 dict 格式）
                    ml_behaviors = self._detect_pose_based_behaviors(image, landmarks_dict, None, detections)
                    if ml_behaviors:
                        behaviors.extend(ml_behaviors)"""
content = content.replace(old_ml_block, new_ml_block)

# 6. Remove load_xgboost_models method
pattern = r"    def load_xgboost_models\(self\):.*?        self\.model_type = None\n"
content = re.sub(pattern, "", content, flags=re.DOTALL)

# 7. Remove predict_with_xgboost method
pattern = r"    def predict_with_xgboost\(self, features, behavior_context=None\):.*?        except Exception as e:\n            self\.logger\.error\(f\"XGBoost预测过程发生错误: \{e\}\"\)\n            return 0, 0\.0\n"
content = re.sub(pattern, "", content, flags=re.DOTALL)

# 8. Remove _convert_to_standard_features method
pattern = r"    def _convert_to_standard_features\(self, enhanced_features\):.*?        return enhanced_features\n\n"
content = re.sub(pattern, "", content, flags=re.DOTALL)

# 9. Remove combine_predictions method
pattern = r"    def combine_predictions\(self, ml_prediction, rule_prediction, context=None\):.*?        return final_pred, weighted_prob\n\n"
content = re.sub(pattern, "", content, flags=re.DOTALL)

# 10. Remove _extract_standard_features method
pattern = r"    def _extract_standard_features\(self, landmarks\):.*?        return None\n\n    def _extract_enhanced_features"
content = re.sub(pattern, "    def _extract_enhanced_features", content, flags=re.DOTALL)

# 11. Remove _extract_enhanced_features method
pattern = r"    def _extract_enhanced_features\(self, landmarks, prev_landmarks=None\):.*?        return std_features \+ \[0\.0\] \* 6  # 出错时返回标准特征，默认动态特征\n\n    def initialize_pose_detector"
content = re.sub(pattern, "    def initialize_pose_detector", content, flags=re.DOTALL)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("video_behavior.py updated")
