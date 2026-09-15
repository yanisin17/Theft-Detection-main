import os
import cv2
import numpy as np
import torch
import logging
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('object_detection')

class ObjectDetector:
    """物体检测类，用于YOLO模型加载和推理，优化零售场景检测"""
    
    def __init__(self, model_path='models/yolov11n.pt'):
        """初始化检测器
        
        Args:
            model_path: YOLO模型路径
        """
        # 确保有有效的模型路径
        if model_path is None:
            # 如果没有提供模型路径，使用默认路径
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
            model_path = os.path.join(base_dir, "models/yolov8n.pt")
            logger.info(f"未提供模型路径，使用默认路径: {model_path}")
            
            # 检查默认模型是否存在
            if not os.path.exists(model_path):
                # 尝试使用工作目录下的模型
                model_path = os.path.join(os.getcwd(), "models", "yolov8n.pt")
                logger.info(f"默认模型不存在，尝试使用工作目录下的模型: {model_path}")
            
        self.model_path = model_path
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 优化检测参数
        self.conf_threshold = 0.15  # 降低置信度阈值
        self.iou_threshold = 0.45  # 增加IoU阈值，更好处理重叠物体
        self.max_det = 300  # 增加最大检测数，确保不遗漏物体
        
        # 零售物品类别 - 扩展物品类别以提高检测能力
        self.retail_items = [
            'bottle', 'book', 'cell phone', 'laptop', 'backpack', 
            'handbag', 'suitcase', 'umbrella', 'wallet', 'bag', 
            'box', 'package', 'item', 'toy', 'food'
        ]
        
        # 商品相关类别 - 用于检测商品区域
        self.retail_furniture = [
            'chair', 'couch', 'bed', 'dining table', 'toilet',
            'tv', 'refrigerator', 'oven', 'sink', 'microwave'
        ]
        
        # 创建映射，用于处理不同模型格式的结果
        self.class_mapping = {
            'book': 'book',
            'cell phone': 'cell phone',
            'bottle': 'bottle',
            'tvmonitor': 'tv',
            'remote': 'remote',
            'laptop': 'laptop',
            'mouse': 'mouse',
            'keyboard': 'keyboard',
            'fork': 'fork',
            'knife': 'knife',
            'spoon': 'spoon',
            'bowl': 'bowl',
            'banana': 'food',
            'apple': 'food',
            'orange': 'food',
            'backpack': 'backpack',
            'handbag': 'handbag',
            'shopping-basket': 'shopping-basket',
            'package': 'package',
            'vase': 'bottle',
            'cup': 'bottle'
        }
        
        # 启用图像增强
        self.enable_enhancement = True
        
        logger.info(f"初始化增强型零售检测器，使用设备: {self.device}")
        self.load_yolo_model()
    
    def load_yolo_model(self):
        """加载YOLO模型，尝试多种加载方式"""
        logger.info(f"尝试加载YOLO模型: {self.model_path}")
        
        if self.model_path is None:
            logger.error("模型路径为空，无法加载模型")
            raise ValueError("模型路径不能为空")
        
        try:
            # 尝试使用ultralytics YOLO类加载
            try:
                from ultralytics import YOLO
                logger.info("使用ultralytics.YOLO加载模型")
                
                # 检查路径是否存在
                if os.path.exists(self.model_path):
                    self.model = YOLO(self.model_path)
                else:
                    # 如果文件不存在，尝试使用预训练模型
                    logger.info(f"模型文件不存在: {self.model_path}，尝试加载预训练模型")
                    model_name = os.path.basename(self.model_path).split(".")[0]
                    # 尝试加载预训练模型 (例如yolov8n, yolov8s等)
                    try:
                        self.model = YOLO(model_name)
                        logger.info(f"成功加载预训练模型: {model_name}")
                    except Exception:
                        # 最后尝试加载默认模型
                        logger.info("尝试加载默认的yolov8n模型")
                        self.model = YOLO("yolov8n.pt")
                
                # 确保模型处于半精度模式以提高性能
                if hasattr(self.model, 'to') and callable(getattr(self.model, 'to')):
                    self.model.to(self.device)
                    
                logger.info("成功使用ultralytics YOLO加载模型")
                return
            except (ImportError, Exception) as e:
                logger.warning(f"使用ultralytics加载失败: {e}")
            
            # 尝试从Hugging Face加载
            try:
                logger.info("尝试从Hugging Face加载模型")
                from transformers import AutoImageProcessor, AutoModelForObjectDetection
                
                # 使用COCO预训练模型
                processor = AutoImageProcessor.from_pretrained("hustvl/yolos-tiny")
                model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-tiny")
                
                self.model = {"processor": processor, "model": model, "type": "huggingface"}
                logger.info("成功从Hugging Face加载YOLOS模型")
                return
            except Exception as e:
                logger.warning(f"从Hugging Face加载模型失败: {e}")
            
            # 尝试直接从文件加载模型
            if os.path.exists(self.model_path):
                logger.info(f"尝试从本地文件加载: {self.model_path}")
                self.model = torch.load(self.model_path, map_location=self.device)
                
                # 确保模型处于评估模式
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                
                logger.info("成功从本地文件加载模型")
                return
            
            # 最后尝试从torch hub加载
            logger.info("尝试从torch hub加载模型")
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                self.model.to(self.device)
                logger.info("成功从torch hub加载模型")
                return
            except Exception as e:
                logger.warning(f"从torch hub加载失败: {e}")
                
            # 如果所有尝试都失败，提示下载模型
            logger.error("所有加载模型的尝试均失败")
            logger.info("请手动下载模型文件到models目录: https://github.com/ultralytics/yolov5/releases")
            raise RuntimeError("无法加载任何可用的YOLO模型，请检查模型文件")
            
        except Exception as e:
            logger.error(f"加载YOLO模型失败: {e}")
            raise RuntimeError(f"无法加载YOLO模型: {e}")
    
    def enhance_image(self, img):
        """增强图像，提高对小物体和低对比度物体的检测能力
        
        Args:
            img: 输入图像 (numpy ndarray)
            
        Returns:
            enhanced_img: 增强后的图像
        """
        if not self.enable_enhancement:
            return img
            
        try:
            # 转换为RGB（如果是BGR）
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img.copy()
            
            # 1. 自适应直方图均衡化 - 提高对比度
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            
            # 转换为LAB颜色空间并对L通道应用CLAHE
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            lab_planes = list(cv2.split(lab))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            
            # 转回RGB
            enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            # 2. 锐化 - 提高边缘细节
            kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
            enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)
            
            # 3. 轻微提高饱和度 - 使物体更容易区分
            hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # 增加饱和度
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            enhanced_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            return enhanced_img
            
        except Exception as e:
            logger.warning(f"图像增强失败: {e}")
            return img
    
    def detect_hand_regions(self, img, results):
        """检测图像中的手部区域，重点分析可能拿取物品的行为
        
        Args:
            img: 输入图像
            results: YOLO检测结果
            
        Returns:
            hand_regions: 手部区域列表 [x1, y1, x2, y2]
        """
        hand_regions = []
        
        try:
            # 获取所有人物检测结果
            persons = []
            
            # 处理不同格式的检测结果
            if hasattr(results, 'pandas') and callable(getattr(results, 'pandas')):
                df = results.pandas().xyxy[0]
                for _, row in df.iterrows():
                    if row['name'].lower() == 'person':
                        persons.append({
                            'box': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                            'conf': row['confidence']
                        })
            elif hasattr(results, 'boxes'):
                # ultralytics YOLOv8
                for box in results.boxes:
                    cls_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else -1
                    class_name = results.names.get(cls_id, "")
                    if class_name.lower() == 'person':
                        persons.append({
                            'box': box.xyxy[0].cpu().numpy(),
                            'conf': float(box.conf[0])
                        })
            
            # 对于每个人物，估计手部区域
            for person in persons:
                box = person['box']
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                
                # 估计左手区域 (左1/4躯干区域)
                left_hand_x1 = max(0, x1)
                left_hand_y1 = max(0, y1 + height * 0.35)  # 从躯干上部开始
                left_hand_x2 = x1 + width * 0.35
                left_hand_y2 = min(img.shape[0], y1 + height * 0.7)  # 到躯干中部结束
                
                # 估计右手区域 (右1/4躯干区域)
                right_hand_x1 = x2 - width * 0.35
                right_hand_y1 = max(0, y1 + height * 0.35)
                right_hand_x2 = min(img.shape[1], x2)
                right_hand_y2 = min(img.shape[0], y1 + height * 0.7)
                
                # 将手部区域扩大20%，确保能捕获手持物品
                def expand_box(x1, y1, x2, y2, factor=0.3):
                    w, h = x2 - x1, y2 - y1
                    ex, ey = w * factor, h * factor
                    return max(0, x1 - ex), max(0, y1 - ey), min(img.shape[1], x2 + ex), min(img.shape[0], y2 + ey)
                
                left_hand_region = expand_box(left_hand_x1, left_hand_y1, left_hand_x2, left_hand_y2)
                right_hand_region = expand_box(right_hand_x1, right_hand_y1, right_hand_x2, right_hand_y2)
                
                hand_regions.append(left_hand_region)
                hand_regions.append(right_hand_region)
            
            return hand_regions
            
        except Exception as e:
            logger.warning(f"手部区域检测失败: {e}")
            return []
    
    def detect_objects(self, img):
        """检测图像中的物体，特别优化零售场景
        
        Args:
            img: 输入图像 (可以是PIL图像、文件路径或opencv图像)
            
        Returns:
            results: 检测结果
            enhanced_results: 增强检测的结果
        """
        if self.model is None:
            logger.error("模型未加载，无法执行检测")
            return {"original": None, "enhanced": None, "hand_objects": []}
        
        try:
            # 确保图像格式正确
            if img is None:
                logger.error("输入图像为空")
                return {"original": None, "enhanced": None, "hand_objects": []}
                
            if isinstance(img, str):
                # 如果是文件路径
                try:
                    img = cv2.imread(img)
                    if img is None:
                        logger.error(f"无法读取图像文件: {img}")
                        return {"original": None, "enhanced": None, "hand_objects": []}
                except Exception as e:
                    logger.error(f"读取图像文件失败: {e}")
                    return {"original": None, "enhanced": None, "hand_objects": []}
            
            if isinstance(img, np.ndarray):
                # 检查图像是否有效
                if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                    logger.error("图像尺寸无效")
                    return {"original": None, "enhanced": None, "hand_objects": []}
                    
                # 如果是OpenCV图像，转换为RGB格式
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = img
            else:
                logger.error(f"不支持的图像类型: {type(img)}")
                return {"original": None, "enhanced": None, "hand_objects": []}
            
            # 保存原始图像尺寸
            original_img = img_rgb.copy()
            
            # 检查是否使用Hugging Face模型
            if isinstance(self.model, dict) and self.model.get('type') == 'huggingface':
                return self._detect_with_huggingface(original_img)
            
            # 1. 首先对原始图像执行标准检测
            try:
                if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
                    # 使用ultralytics YOLO
                    logger.info("使用ultralytics YOLO.predict方法执行检测")
                    results = self.model.predict(
                        source=img_rgb, 
                        conf=self.conf_threshold,
                        iou=self.iou_threshold,
                        max_det=self.max_det
                    )
                else:
                    # 使用torch模型
                    logger.info("使用torch模型执行检测")
                    results = self.model(img_rgb, conf=self.conf_threshold, iou=self.iou_threshold)
            except Exception as e:
                logger.error(f"执行检测失败: {e}")
                results = None
            
            # 2. 增强图像并再次检测，特别关注小物体
            try:
                enhanced_img = self.enhance_image(original_img)
                
                if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
                    enhanced_results = self.model.predict(
                        source=enhanced_img, 
                        conf=self.conf_threshold * 0.8,  # 对增强图像使用更低的置信度阈值
                        iou=self.iou_threshold,
                        max_det=self.max_det
                    )
                else:
                    enhanced_results = self.model(enhanced_img, conf=self.conf_threshold * 0.8, iou=self.iou_threshold)
            except Exception as e:
                logger.error(f"增强检测失败: {e}")
                enhanced_results = None
            
            # 如果两次检测都失败，返回空结果
            if results is None and enhanced_results is None:
                logger.error("标准检测和增强检测均失败")
                return {"original": None, "enhanced": None, "hand_objects": []}
            
            # 3. 检测手部区域
            try:
                if results is not None:
                    hand_regions = self.detect_hand_regions(original_img, results)
                else:
                    hand_regions = []
            except Exception as e:
                logger.error(f"手部区域检测失败: {e}")
                hand_regions = []
            
            # 4. 对每个手部区域进行特别分析
            hand_objects = []
            
            for region in hand_regions:
                try:
                    x1, y1, x2, y2 = [int(c) for c in region]
                    if x2 <= x1 or y2 <= y1:
                        continue  # 忽略无效区域
                    
                    # 截取手部区域
                    hand_img = original_img[y1:y2, x1:x2]
                    if hand_img.size == 0:
                        continue
                        
                    # 对手部区域执行特定检测，使用更低的置信度
                    if hasattr(self.model, 'predict') and callable(getattr(self.model, 'predict')):
                        hand_results = self.model.predict(
                            source=hand_img, 
                            conf=self.conf_threshold * 0.7,  # 手部区域使用更低的置信度
                            iou=self.iou_threshold,
                            max_det=50
                        )
                    else:
                        hand_results = self.model(hand_img, conf=self.conf_threshold * 0.7)
                    
                    # 将手部区域的检测结果转换为完整图像坐标
                    if hasattr(hand_results, 'pandas') and callable(getattr(hand_results, 'pandas')):
                        df = hand_results.pandas().xyxy[0]
                        for _, row in df.iterrows():
                            # 检查是否是商品类别
                            item_class = row['name'].lower()
                            if item_class in self.retail_items or item_class in self.class_mapping:
                                # 转换坐标回完整图像
                                x1_item = x1 + row['xmin']
                                y1_item = y1 + row['ymin']
                                x2_item = x1 + row['xmax']
                                y2_item = y1 + row['ymax']
                                
                                hand_objects.append({
                                    'bbox': [x1_item, y1_item, x2_item, y2_item],
                                    'class': self.class_mapping.get(item_class, item_class),
                                    'confidence': row['confidence'] * 0.95,  # 轻微降低置信度
                                    'hand_region': True
                                })
                    elif hasattr(hand_results, 'boxes'):
                        for box in hand_results.boxes:
                            cls_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else -1
                            class_name = hand_results.names.get(cls_id, "").lower()
                            
                            if class_name in self.retail_items or class_name in self.class_mapping:
                                box_coords = box.xyxy[0].cpu().numpy()
                                x1_item = x1 + box_coords[0]
                                y1_item = y1 + box_coords[1]
                                x2_item = x1 + box_coords[2]
                                y2_item = y1 + box_coords[3]
                                
                                hand_objects.append({
                                    'bbox': [x1_item, y1_item, x2_item, y2_item],
                                    'class': self.class_mapping.get(class_name, class_name),
                                    'confidence': float(box.conf[0]) * 0.95,
                                    'hand_region': True
                                })
                except Exception as e:
                    logger.warning(f"手部区域物体检测失败: {e}")
                    continue
            
            # 合并标准检测结果、增强检测结果和手部区域检测结果
            merged_results = {
                'original': results,
                'enhanced': enhanced_results,
                'hand_objects': hand_objects
            }
            
            return merged_results
            
        except Exception as e:
            logger.error(f"检测过程中出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"original": None, "enhanced": None, "hand_objects": []}
            
    def _detect_with_huggingface(self, image):
        """使用Hugging Face模型执行检测
        
        Args:
            image: RGB格式的输入图像
            
        Returns:
            结果字典，包含标准格式的检测结果
        """
        try:
            import torch
            from PIL import Image
            
            # 将numpy图像转换为PIL格式
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
                
            # 获取模型和处理器
            processor = self.model.get("processor")
            model = self.model.get("model")
            
            if processor is None or model is None:
                logger.error("Hugging Face模型或处理器未正确加载")
                return {"original": None, "enhanced": None, "hand_objects": []}
                
            # 处理图像
            inputs = processor(images=pil_image, return_tensors="pt")
            
            # 执行推理
            with torch.no_grad():
                outputs = model(**inputs)
                
            # 后处理检测结果
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = processor.post_process_object_detection(
                outputs, threshold=self.conf_threshold, target_sizes=target_sizes
            )[0]
            
            # 转换为标准格式
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box_coords = box.tolist()
                confidence = score.item()
                class_id = label.item()
                class_name = model.config.id2label[class_id]
                
                detections.append({
                    'bbox': [box_coords[0], box_coords[1], box_coords[2], box_coords[3]],
                    'class': class_name,
                    'confidence': confidence,
                    'source': 'huggingface'
                })
            
            # 创建自定义结果对象以匹配标准格式
            original_results = {
                "boxes": detections,
                "names": {i: model.config.id2label[i] for i in model.config.id2label},
                "huggingface": True
            }
            
            # 增强图像检测
            enhanced_image = self.enhance_image(image)
            enhanced_pil = Image.fromarray(enhanced_image)
            
            # 对增强图像执行检测
            enhanced_inputs = processor(images=enhanced_pil, return_tensors="pt")
            with torch.no_grad():
                enhanced_outputs = model(**enhanced_inputs)
                
            # 后处理增强图像的检测结果
            enhanced_results = processor.post_process_object_detection(
                enhanced_outputs, 
                threshold=self.conf_threshold * 0.8,  # 降低阈值提高召回率
                target_sizes=target_sizes
            )[0]
            
            # 转换为标准格式
            enhanced_detections = []
            for score, label, box in zip(enhanced_results["scores"], enhanced_results["labels"], enhanced_results["boxes"]):
                box_coords = box.tolist()
                confidence = score.item()
                class_id = label.item()
                class_name = model.config.id2label[class_id]
                
                enhanced_detections.append({
                    'bbox': [box_coords[0], box_coords[1], box_coords[2], box_coords[3]],
                    'class': class_name,
                    'confidence': confidence,
                    'source': 'huggingface_enhanced'
                })
            
            # 创建自定义结果对象以匹配标准格式
            enhanced_results_obj = {
                "boxes": enhanced_detections,
                "names": {i: model.config.id2label[i] for i in model.config.id2label},
                "huggingface": True
            }
            
            # 合并结果
            return {
                'original': original_results,
                'enhanced': enhanced_results_obj,
                'hand_objects': []  # 暂不支持手部检测
            }
            
        except Exception as e:
            logger.error(f"Hugging Face检测失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"original": None, "enhanced": None, "hand_objects": []}
            
    def combine_detections(self, merged_results):
        """将多种检测结果合并成统一格式
        
        Args:
            merged_results: 包含标准检测、增强检测和手部区域检测的结果
            
        Returns:
            combined_detections: 合并后的检测结果，格式为列表[{bbox, class, confidence}, ...]
        """
        combined_detections = []
        
        try:
            # 处理标准检测结果
            original_results = merged_results.get('original')
            if original_results is not None:
                if hasattr(original_results, 'pandas') and callable(getattr(original_results, 'pandas')):
                    df = original_results.pandas().xyxy[0]
                    for _, row in df.iterrows():
                        combined_detections.append({
                            'bbox': [row['xmin'], row['ymin'], row['xmax'], row['ymax']],
                            'class': row['name'],
                            'confidence': row['confidence'],
                            'source': 'original'
                        })
                elif hasattr(original_results, 'boxes'):
                    for box in original_results.boxes:
                        cls_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else -1
                        class_name = original_results.names.get(cls_id, "")
                        box_coords = box.xyxy[0].cpu().numpy()
                        
                        combined_detections.append({
                            'bbox': [box_coords[0], box_coords[1], box_coords[2], box_coords[3]],
                            'class': class_name,
                            'confidence': float(box.conf[0]),
                            'source': 'original'
                        })
            
            # 处理增强检测结果
            enhanced_results = merged_results.get('enhanced')
            if enhanced_results is not None:
                if hasattr(enhanced_results, 'pandas') and callable(getattr(enhanced_results, 'pandas')):
                    df = enhanced_results.pandas().xyxy[0]
                    for _, row in df.iterrows():
                        # 检查是否与已有检测结果重叠过高
                        new_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                        if not self._is_duplicate(new_box, combined_detections):
                            combined_detections.append({
                                'bbox': new_box,
                                'class': row['name'],
                                'confidence': row['confidence'] * 0.95,  # 轻微降低增强检测的置信度
                                'source': 'enhanced'
                            })
                elif hasattr(enhanced_results, 'boxes'):
                    for box in enhanced_results.boxes:
                        cls_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else -1
                        class_name = enhanced_results.names.get(cls_id, "")
                        box_coords = box.xyxy[0].cpu().numpy()
                        
                        # 检查是否与已有检测结果重叠过高
                        new_box = [box_coords[0], box_coords[1], box_coords[2], box_coords[3]]
                        if not self._is_duplicate(new_box, combined_detections):
                            combined_detections.append({
                                'bbox': new_box,
                                'class': class_name,
                                'confidence': float(box.conf[0]) * 0.95,
                                'source': 'enhanced'
                            })
            
            # 添加手部区域检测结果
            hand_objects = merged_results.get('hand_objects', [])
            for obj in hand_objects:
                # 检查是否与已有检测结果重叠过高
                if not self._is_duplicate(obj['bbox'], combined_detections):
                    combined_detections.append({
                        'bbox': obj['bbox'],
                        'class': obj['class'],
                        'confidence': obj['confidence'],
                        'source': 'hand_region'
                    })
            
            return combined_detections
            
        except Exception as e:
            logger.error(f"合并检测结果出错: {e}")
            return []
    
    def _is_duplicate(self, new_box, existing_detections, iou_threshold=0.7):
        """检查新检测框是否与现有检测结果有显著重叠
        
        Args:
            new_box: 新检测框 [x1, y1, x2, y2]
            existing_detections: 现有检测结果列表
            iou_threshold: IoU阈值，超过此值认为是重复
            
        Returns:
            is_duplicate: 是否重复
        """
        for det in existing_detections:
            existing_box = det['bbox']
            
            # 计算IoU
            x1 = max(new_box[0], existing_box[0])
            y1 = max(new_box[1], existing_box[1])
            x2 = min(new_box[2], existing_box[2])
            y2 = min(new_box[3], existing_box[3])
            
            # 检查是否有交集
            if x2 < x1 or y2 < y1:
                continue
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
            area2 = (existing_box[2] - existing_box[0]) * (existing_box[3] - existing_box[1])
            union = area1 + area2 - intersection
            
            if intersection / union > iou_threshold:
                return True
                
        return False 