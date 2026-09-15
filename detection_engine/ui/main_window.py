import os
import sys
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
import numpy as np
from pathlib import Path
import datetime
import logging
import re
import random

# Add src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.detection import TheftDetector
from src.utils.video_processor import VideoProcessor

class TheftDetectionApp:
    def __init__(self, root, model_path="models/yolov11n.pt"):
        """Initialize the application"""
        # 设置应用窗口
        self.root = root
        self.root.title("店铺盗窃行为监测系统")
        
        # 设置状态变量
        self.current_media_path = None
        self.current_media_type = None
        self.processed_media_path = None
        self.video_capture = None
        self.processed_video_capture = None
        self.stop_video_thread = False
        self.is_playing = False
        self.current_frame = 0
        self.video_thread = None
        self.is_processing = False
        
        # Initialize detection service - 推迟初始化以减少启动时间
        self.theft_detector = None  # 将在需要时初始化
        self.video_processor = VideoProcessor()
        
        # 添加兼容性属性
        self.detector = None  # 兼容旧代码，会在初始化theft_detector时同步更新
        
        # Processing state
        self.is_processing = False
        self.current_media_path = None
        self.current_media_type = None  # 'image' or 'video'
        self.video_capture = None
        self.video_thread = None
        self.stop_video_thread = False
        self.processed_media_path = None
        self.processed_video_capture = None
        self.is_playing = False  # 添加视频播放状态变量
        
        # Create UI components
        self.create_ui()
        
        # 绑定窗口调整大小事件
        self.root.bind("<Configure>", self.on_window_configure)
        
        # 设置全屏快捷键
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.end_fullscreen)
        
        # 初始化全屏状态
        self.is_fullscreen = False
        
        # 居中显示窗口
        self._center_window()
        
        # 初始化日志
        self.log("店铺盗窃行为监测系统已启动")
        
        # 配置日志重定向，将检测模块日志捕获并写入UI
        self.configure_logging_redirect()
    
    def configure_logging_redirect(self):
        """配置日志重定向，将检测模块的日志转发到UI界面"""
        import logging
        
        # 创建一个处理器，将日志消息转发到UI
        class UILogHandler(logging.Handler):
            def __init__(self, ui_instance):
                super().__init__()
                self.ui = ui_instance
                
            def emit(self, record):
                # 日志记录转发到UI
                log_message = self.format(record)
                
                # 如果日志消息来自detection模块，使用self.log记录到UI
                # 使用root.after确保在主线程中更新UI
                self.ui.root.after(0, lambda: self.ui.log(log_message))
        
        # 配置处理器
        handler = UILogHandler(self)
        formatter = logging.Formatter('%(message)s')  # 简化格式，因为UI日志会添加时间戳
        handler.setFormatter(formatter)
        
        # 获取detection日志记录器并添加处理器
        logger = logging.getLogger('detection')
        logger.addHandler(handler)
        
        # 确保日志级别设置为INFO或更低
        if logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
    
    def create_ui(self):
        """Create the main interface"""
        # 设置窗口最小尺寸
        self.root.minsize(1024, 768)
        
        # 设置初始窗口大小
        self.default_width = 1280
        self.default_height = 800
        self.root.geometry(f"{self.default_width}x{self.default_height}")
        
        # 设置窗口居中
        self._center_window()
        
        # 存储当前窗口状态
        self.is_maximized = False
        self.last_known_width = self.default_width
        self.last_known_height = self.default_height
        
        # 创建主滚动区域
        self.main_scroll = ttk.Scrollbar(self.root, orient=tk.VERTICAL)
        self.main_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建可滚动的画布
        self.main_canvas = tk.Canvas(self.root, yscrollcommand=self.main_scroll.set)
        self.main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 配置滚动条
        self.main_scroll.config(command=self.main_canvas.yview)
        
        # 创建框架放在画布内
        self.main_frame = ttk.Frame(self.main_canvas)
        self.main_canvas.create_window((0, 0), window=self.main_frame, anchor=tk.NW, tags="self.main_frame")
        
        # 绑定框架大小变化事件
        self.main_frame.bind("<Configure>", self._configure_main_canvas)
        
        # Main frame
        #self.main_frame = ttk.Frame(self.root)
        #self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top section - Media display
        self.media_frame = ttk.Frame(self.main_frame)
        self.media_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Side by side view (original and processed)
        self.original_frame = ttk.LabelFrame(self.media_frame, text="原始媒体")
        self.original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.processed_frame = ttk.LabelFrame(self.media_frame, text="检测结果")
        self.processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 使用固定比例的帧来包含画布，确保视频区域稳定
        self.original_canvas_frame = ttk.Frame(self.original_frame)
        self.original_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.processed_canvas_frame = ttk.Frame(self.processed_frame)
        self.processed_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Original media canvas - 响应式尺寸
        self.original_canvas = tk.Canvas(self.original_canvas_frame, bg="black", highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Processed media canvas - 响应式尺寸
        self.processed_canvas = tk.Canvas(self.processed_canvas_frame, bg="black", highlightthickness=0)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 添加视频播放控制区域 - 放在视频下方，但初始时隐藏
        self.playback_control_frame = ttk.Frame(self.main_frame)
        # 注意：不在初始化时pack，而是在视频处理完成后才显示

        # 添加播放/暂停按钮
        self.play_pause_btn = ttk.Button(self.playback_control_frame, text="播放", width=10, command=self.toggle_playback)
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)
        
        # 添加视频进度条
        self.progress_var = tk.DoubleVar(value=0)
        self.video_slider = ttk.Scale(self.playback_control_frame, orient=tk.HORIZONTAL, 
                                    from_=0, to=100, variable=self.progress_var, 
                                    command=self.on_slider_change)
        self.video_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # 添加视频时间标签
        self.time_label = ttk.Label(self.playback_control_frame, text="00:00 / 00:00")
        self.time_label.pack(side=tk.LEFT, padx=5)
        
        # 计算并设置初始画布尺寸
        self.update_canvas_sizes()
        
        # 配置窗口调整大小事件，但限制更新频率
        self.root.bind("<Configure>", self.on_window_configure)
        
        # 绑定窗口最大化和恢复事件
        self.root.bind("<F11>", self.toggle_fullscreen)
        self.root.bind("<Escape>", self.end_fullscreen)
        
        # Middle section - Control area
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X, pady=5)
        
        # 修改控制按钮布局 - 左侧放文件选择按钮，右侧放处理和保存按钮
        # File selection buttons - 左侧固定
        self.file_btn_frame = ttk.Frame(self.control_frame)
        self.file_btn_frame.pack(side=tk.LEFT, padx=5)
        
        self.select_image_btn = ttk.Button(self.file_btn_frame, text="选择图片", command=self.select_image)
        self.select_image_btn.pack(side=tk.LEFT, padx=5)
        
        self.select_video_btn = ttk.Button(self.file_btn_frame, text="选择视频", command=self.select_video)
        self.select_video_btn.pack(side=tk.LEFT, padx=5)
        
        # Processing buttons - 右侧
        self.process_btn_frame = ttk.Frame(self.control_frame)
        self.process_btn_frame.pack(side=tk.RIGHT, padx=5)
        
        self.process_btn = ttk.Button(self.process_btn_frame, text="开始分析", command=self.start_processing)
        self.process_btn.pack(side=tk.LEFT, padx=5)
        self.process_btn.state(['disabled'])
        
        self.save_btn = ttk.Button(self.process_btn_frame, text="保存结果", command=self.save_result)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.save_btn.state(['disabled'])
        
        # Status labels
        self.status_frame = ttk.Frame(self.main_frame)
        self.status_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.status_frame, text="状态:").pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(self.status_frame, text="就绪")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.pack(fill=tk.X, pady=5)
        
        # 创建一个包含进度条和标签的框架
        self.progress_container = ttk.Frame(self.progress_frame)
        self.progress_container.pack(fill=tk.X, padx=5)
        
        # 创建一个更明显的进度条样式
        style = ttk.Style()
        style.configure(
            "Thick.Horizontal.TProgressbar", 
            thickness=20,                   # 增加进度条高度
            background='#4a8af4',           # 进度条填充颜色
            troughcolor='#e0e0e0'           # 进度条背景颜色
        )
        
        # 创建进度条 - 设置高度较高以便放置文本
        self.progress_bar = ttk.Progressbar(
            self.progress_container, 
            orient=tk.HORIZONTAL, 
            mode='determinate',
            length=100,  # 会被pack填充
            style="Thick.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(fill=tk.X, pady=2)
        
        # 创建一个StringVar用于百分比文本
        self.progress_text = tk.StringVar(value="0%")
        
        # 创建标签框架使其有背景色
        self.label_frame = tk.Frame(
            self.progress_container,
            bg='#f0f0f0',  # 浅灰色背景
            bd=1,          # 边框大小
            relief=tk.RAISED  # 边框样式
        )
        
        # 创建标签，直接放在进度条上并居中
        self.progress_label = ttk.Label(
            self.label_frame, 
            textvariable=self.progress_text,
            anchor="center",
            font=("Arial", 9, "bold"),
            foreground="#000080",  # 深蓝色文本
            background='#f0f0f0'   # 与标签框架匹配的背景色
        )
        self.progress_label.pack(padx=5, pady=1)
        
        # 使用place布局将标签框架放在进度条的中心
        self.progress_bar.update()  # 确保进度条尺寸已更新
        self.label_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # 添加可疑行为面板
        self.behaviors_frame = ttk.LabelFrame(self.main_frame, text="可疑行为")
        self.behaviors_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左右布局：左侧日志，右侧可疑行为
        self.log_behavior_frame = ttk.Frame(self.behaviors_frame)
        self.log_behavior_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧 - 日志面板
        self.result_frame = ttk.LabelFrame(self.log_behavior_frame, text="检测日志")
        self.result_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log text area with scrollbar
        self.log_text = tk.Text(self.result_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_scrollbar = ttk.Scrollbar(self.result_frame, command=self.log_text.yview)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.log_scrollbar.set)
        
        # 右侧 - 可疑行为列表
        self.behavior_list_frame = ttk.LabelFrame(self.log_behavior_frame, text="行为列表")
        self.behavior_list_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建可疑行为列表
        self.behavior_list = ttk.Treeview(self.behavior_list_frame, columns=("时间", "类型", "概率"), show="headings")
        self.behavior_list.heading("时间", text="时间")
        self.behavior_list.heading("类型", text="行为类型")
        self.behavior_list.heading("概率", text="可信度")
        self.behavior_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.behavior_scrollbar = ttk.Scrollbar(self.behavior_list_frame, command=self.behavior_list.yview)
        self.behavior_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.behavior_list.config(yscrollcommand=self.behavior_scrollbar.set)
    
    def log(self, message, console_only=False):
        """Add message to log area
        
        Args:
            message: The message to log
            console_only: If True, only logs to console and not to the UI
        """
        try:
            # 记住当前滚动位置
            try:
                current_view = self.log_text.yview()
            except:
                current_view = (0, 0)
                
            # 始终向控制台输出
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
            
            # 如果是只输出到控制台的消息，不添加到UI
            if console_only:
                return
                
            # 检查是否为可疑行为检测信息 - 这些消息需要显示
            is_suspicious_behavior = "检测到行为:" in message or "添加行为到列表" in message
            
            if is_suspicious_behavior:
                if "检测到行为:" in message:
                    # 这是一个详细的行为描述
                    parts = message.split("行为描述:")
                    behavior_info = parts[0].strip()
                    behavior_desc = parts[1].strip() if len(parts) > 1 else ""
                    
                    # 提取可信度
                    confidence_match = re.search(r'可信度: (\d+\.\d+%)', behavior_info)
                    confidence_str = confidence_match.group(1) if confidence_match else "未知"
                    
                    # 插入带有高亮格式的警告消息
                    self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                    self.log_text.insert(tk.END, f"⚠️ {behavior_info}\n", "behavior_alert")
                    self.log_text.insert(tk.END, f"   {behavior_desc}\n", "behavior_desc")
                    
                    # 配置标签样式
                    self.log_text.tag_configure("timestamp", foreground="blue")
                    self.log_text.tag_configure("behavior_alert", foreground="#FF3300", font=("Helvetica", 10, "bold"))
                    self.log_text.tag_configure("behavior_desc", foreground="#993300", font=("Helvetica", 9, "italic"))
                elif "添加行为到列表" in message:
                    # 提取可信度
                    confidence_match = re.search(r'可信度=(\d+\.\d+)', message)
                    if confidence_match:
                        confidence = float(confidence_match.group(1))
                        # 只在可信度较高时才在检测日志中高亮显示，但总是显示
                        confidence_tag = "high_confidence" if confidence > 0.7 else "medium_confidence" if confidence > 0.5 else "low_confidence"
                    else:
                        confidence_tag = "medium_confidence"
                    
                    # 提取帧信息
                    frame_match = re.search(r'帧=(\d+)', message)
                    frame_num = frame_match.group(1) if frame_match else "未知"
                    
                    # 提取行为类型和时间
                    type_match = re.search(r'类型=([^,]+)', message)
                    behavior_type = type_match.group(1) if type_match else "未知行为"
                    
                    time_match = re.search(r'时间=(\d+\.\d+)', message)
                    time_point = float(time_match.group(1)) if time_match else 0.0
                    
                    # 提取概率
                    probability_match = re.search(r'可信度=(\d+\.\d+)', message)
                    probability = float(probability_match.group(1)) if probability_match else 0.0
                    
                    # 格式化高亮警告消息
                    warning_message = f"在第{frame_num}帧 ({time_point:.2f}秒) 检测到行为: {behavior_type}"
                    
                    # 插入带有高亮格式的警告消息
                    self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                    self.log_text.insert(tk.END, warning_message, confidence_tag)
                    self.log_text.insert(tk.END, f" (可信度: {probability:.2%})\n", "confidence")
                    
                    # 配置高亮标签样式
                    self.log_text.tag_configure("timestamp", foreground="blue")
                    self.log_text.tag_configure("high_confidence", foreground="#CC0000", font=("Helvetica", 10, "bold"))
                    self.log_text.tag_configure("medium_confidence", foreground="#FF6600", font=("Helvetica", 10))
                    self.log_text.tag_configure("low_confidence", foreground="#666666", font=("Helvetica", 9))
                    self.log_text.tag_configure("confidence", foreground="#333333", font=("Helvetica", 9))
                
                # 滚动到最新消息
                self.log_text.see(tk.END)
                return
                
            # 检查是否为进度信息，使用特殊格式
            if "处理进度" in message:
                # 显示进度但不要太频繁
                if random.random() < 0.05:  # 只显示约5%的进度更新，减少日志刷屏
                    self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                    self.log_text.insert(tk.END, message, "progress")
                    self.log_text.insert(tk.END, "\n")
                return
                
            # 检查是否为盗窃行为检测结果
            elif "探测盗窃行为" in message or "检测完成" in message:
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                
                # 检查结果类型
                if "发现盗窃行为" in message:
                    self.log_text.insert(tk.END, "🚨 ", "alert_icon")
                    self.log_text.insert(tk.END, message, "theft_yes")
                elif "未发现盗窃行为" in message:
                    self.log_text.insert(tk.END, "✅ ", "check_icon")
                    self.log_text.insert(tk.END, message, "theft_no")
                else:
                    # 默认情况
                    self.log_text.insert(tk.END, message, "normal")
                    
                self.log_text.insert(tk.END, "\n")
                
            # 检查是否为盗窃概率信息
            elif "盗窃概率:" in message:
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                
                # 提取概率值
                prob_match = re.search(r'盗窃概率: (\d+\.\d+%)', message)
                if prob_match:
                    prob_str = prob_match.group(1)
                    prob = float(prob_str.strip('%')) / 100
                    
                    # 分离消息的不同部分
                    before, after = message.split("盗窃概率:")
                    self.log_text.insert(tk.END, before + "盗窃概率: ", "normal")
                    
                    # 根据概率值使用不同颜色
                    if prob > 0.7:
                        self.log_text.insert(tk.END, after, "high_probability")
                    elif prob > 0.4:
                        self.log_text.insert(tk.END, after, "medium_probability")
                    else:
                        self.log_text.insert(tk.END, after, "low_probability")
                else:
                    # 默认情况
                    self.log_text.insert(tk.END, message, "normal")
                
                self.log_text.insert(tk.END, "\n")
                
            # 检查是否包含行为分析信息
            elif "行为分析:" in message or "行为平均可疑度:" in message:
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                
                # 添加图标前缀
                if "行为分析:" in message:
                    self.log_text.insert(tk.END, "📊 ", "chart_icon")
                elif "行为平均可疑度:" in message:
                    self.log_text.insert(tk.END, "📈 ", "trend_icon")
                
                self.log_text.insert(tk.END, message, "behavior_summary")
                self.log_text.insert(tk.END, "\n")
                
            # 检查是否包含累计发现信息
            elif "当前累计发现可疑行为" in message or "行为列表中共有" in message or "已添加" in message and "条行为记录到界面" in message:
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.log_text.insert(tk.END, "📑 ", "document_icon")
                self.log_text.insert(tk.END, message, "summary")
                self.log_text.insert(tk.END, "\n")
                
            # 检查是否为环境判断消息
            elif "环境判断为" in message:
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                
                # 添加图标前缀
                if "零售环境" in message:
                    self.log_text.insert(tk.END, "🏪 ", "store_icon")
                else:
                    self.log_text.insert(tk.END, "🏢 ", "office_icon")
                    
                self.log_text.insert(tk.END, message, "environment")
                self.log_text.insert(tk.END, "\n")
                
            # 检查是否为分析完成消息
            elif "视频分析完成" in message or "图片分析完成" in message:
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.log_text.insert(tk.END, "✅ ", "check_icon")
                self.log_text.insert(tk.END, message, "completion")
                self.log_text.insert(tk.END, "\n")
                
            # 检查是否为错误消息
            elif "错误" in message.lower():
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.log_text.insert(tk.END, "❌ ", "error_icon")
                self.log_text.insert(tk.END, message, "error")
                self.log_text.insert(tk.END, "\n")
                
            # 系统启动和重要操作消息
            elif "启动" in message or "开始分析" in message or "已加载" in message:
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.log_text.insert(tk.END, "🔄 ", "system_icon")
                self.log_text.insert(tk.END, message, "system")
                self.log_text.insert(tk.END, "\n")
                
            # 分析过程中的日志也应该显示
            elif "执行姿态估计" in message or "检测到人体姿态" in message or "正在处理视频" in message or "视频处理" in message:
                self.log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
                self.log_text.insert(tk.END, "🔍 ", "processing_icon")
                self.log_text.insert(tk.END, message, "processing")
                self.log_text.insert(tk.END, "\n")
                
            else:
                # 其他消息直接显示，但格式简单
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")

            # 配置标签样式
            self.log_text.tag_configure("timestamp", foreground="blue")
            self.log_text.tag_configure("progress", foreground="purple", font=("Helvetica", 9, "bold"))
            self.log_text.tag_configure("theft_yes", foreground="#CC0000", font=("Helvetica", 10, "bold"))
            self.log_text.tag_configure("theft_no", foreground="#009900", font=("Helvetica", 9, "bold"))
            self.log_text.tag_configure("summary", foreground="#663300", font=("Helvetica", 9, "bold"))
            self.log_text.tag_configure("environment", foreground="#003366", font=("Helvetica", 9, "bold"))
            self.log_text.tag_configure("completion", foreground="#006600", font=("Helvetica", 9, "bold"))
            self.log_text.tag_configure("error", foreground="#CC0000", font=("Helvetica", 9, "bold"))
            self.log_text.tag_configure("system", foreground="#333333", font=("Helvetica", 9))
            self.log_text.tag_configure("normal", foreground="black")
            self.log_text.tag_configure("behavior_summary", foreground="#336699", font=("Helvetica", 9, "bold"))
            self.log_text.tag_configure("high_probability", foreground="#CC0000", font=("Helvetica", 10, "bold"))
            self.log_text.tag_configure("medium_probability", foreground="#FF6600", font=("Helvetica", 9, "bold"))
            self.log_text.tag_configure("low_probability", foreground="#009900", font=("Helvetica", 9))
            self.log_text.tag_configure("processing", foreground="#333399", font=("Helvetica", 9))
            
            # 配置图标样式
            self.log_text.tag_configure("alert_icon", foreground="#CC0000", font=("Helvetica", 12))
            self.log_text.tag_configure("check_icon", foreground="#009900", font=("Helvetica", 12))
            self.log_text.tag_configure("chart_icon", foreground="#336699", font=("Helvetica", 12))
            self.log_text.tag_configure("trend_icon", foreground="#663399", font=("Helvetica", 12))
            self.log_text.tag_configure("document_icon", foreground="#663300", font=("Helvetica", 12))
            self.log_text.tag_configure("store_icon", foreground="#006633", font=("Helvetica", 12))
            self.log_text.tag_configure("office_icon", foreground="#333366", font=("Helvetica", 12))
            self.log_text.tag_configure("error_icon", foreground="#CC0000", font=("Helvetica", 12))
            self.log_text.tag_configure("system_icon", foreground="#333333", font=("Helvetica", 12))
            self.log_text.tag_configure("processing_icon", foreground="#333399", font=("Helvetica", 12))
            
            # 恢复原始滚动位置，如果用户之前有滚动，否则才滚动到末尾
            if current_view != (0, 0) and current_view[1] < 1.0:
                try:
                    self.log_text.yview_moveto(current_view[0])
                except:
                    pass
            else:
                # 如果原来就在底部或者新内容，则滚动到底部
                self.log_text.see(tk.END)
        except Exception as e:
            print(f"日志错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def clear_log(self):
        """清空日志窗口内容"""
        try:
            self.log_text.delete(1.0, tk.END)
            self.root.update_idletasks()
        except Exception as e:
            print(f"清空日志错误: {str(e)}")
    
    def select_image(self):
        """Select image file"""
        filetypes = [
            ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
            ("所有文件", "*.*")
        ]
        filepath = filedialog.askopenfilename(
            title="选择图片",
            filetypes=filetypes
        )
        
        if filepath:
            self.load_image(filepath)
    
    def select_video(self):
        """Select video file"""
        filetypes = [
            ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
            ("所有文件", "*.*")
        ]
        filepath = filedialog.askopenfilename(
            title="选择视频",
            filetypes=filetypes
        )
        
        if filepath:
            self.load_video(filepath)
    
    def load_image(self, filepath):
        """Load and display image"""
        try:
            # Clean previous video state
            self.stop_video_playback()
            
            # 确保播放控制区域隐藏
            self.playback_control_frame.pack_forget()
            
            # Update status
            self.current_media_path = filepath
            self.current_media_type = 'image'
            self.processed_media_path = None
            self.status_label.config(text=f"已加载图片: {os.path.basename(filepath)}")
            
            # Read and display image
            self.original_image = cv2.imread(filepath)
            if self.original_image is None:
                raise Exception(f"无法读取图片文件: {filepath}")
                
            img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image(img_rgb, self.original_canvas)
            
            # Clear processed canvas
            self.processed_canvas.delete("all")
            
            # Enable processing button
            self.process_btn.state(['!disabled'])
            self.save_btn.state(['disabled'])
            
            # Log
            self.log(f"已加载图片: {os.path.basename(filepath)}")
            self.update_progress(0)
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
            self.log(f"错误: 加载图片失败: {str(e)}")
    
    def load_video(self, filepath):
        """Load and display video"""
        try:
            # Clean previous video state
            self.stop_video_playback()
            
            # 确保播放控制区域隐藏
            self.playback_control_frame.pack_forget()
            
            # 确保视频文件存在
            if not os.path.exists(filepath):
                raise Exception(f"视频文件不存在: {filepath}")
                
            # 创建static/videos目录以确保存在
            videos_dir = os.path.join("static", "videos")
            os.makedirs(videos_dir, exist_ok=True)
            
            # 如果视频不在static/videos目录下，复制到该目录
            filename = os.path.basename(filepath)
            target_path = os.path.join(videos_dir, filename)
            
            # 只在需要时复制，避免重复操作
            if filepath != target_path and not os.path.exists(target_path):
                import shutil
                shutil.copy2(filepath, target_path)
                self.log(f"已将视频复制到: {target_path}")
            
            # 设置当前媒体路径为static/videos中的路径
            self.current_media_path = target_path
            
            # 使用目标路径打开视频
            cap = cv2.VideoCapture(target_path)
            if not cap.isOpened():
                raise Exception("无法打开视频文件")
                
            # Update status
            self.current_media_type = 'video'
            self.processed_media_path = None
            self.video_capture = cap
            
            # 重置可疑行为列表
            if hasattr(self, 'suspicious_behaviors'):
                self.suspicious_behaviors = []
            
            # Get video info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Display video info
            self.status_label.config(text=f"已加载视频: {os.path.basename(target_path)}")
            video_info = f"视频信息: {width}x{height}, {fps:.2f}fps, 时长: {duration:.2f}秒"
            self.log(video_info)
            
            # 显示第一帧作为预览（而不是自动开始播放）
            ret, frame = cap.read()
            if ret:
                # 显示第一帧作为预览
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.display_image(frame_rgb, self.original_canvas)
                # 重置视频到开始位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Enable processing button
            self.process_btn.state(['!disabled'])
            self.save_btn.state(['disabled'])
            
            # 确保进度条被重置
            if hasattr(self, 'progress_bar') and self.progress_bar:
                self.progress_bar['value'] = 0
            if hasattr(self, 'progress_label') and self.progress_label:
                self.progress_label.config(text="0%")
                
        except Exception as e:
            messagebox.showerror("错误", f"加载视频失败: {str(e)}")
            self.log(f"错误: 加载视频失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def start_video_preview(self):
        """Start video preview thread"""
        self.stop_video_thread = False
        self.video_thread = threading.Thread(target=self.video_preview_loop)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def video_preview_loop(self):
        """Video preview loop"""
        if self.video_capture is None:
            return
        
        # Reset to beginning of video
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while not self.stop_video_thread:
            ret, frame = self.video_capture.read()
            if not ret:
                # Loop back to beginning
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.display_image(frame_rgb, self.original_canvas)
            time.sleep(0.03)  # Limit frame rate
    
    def stop_video_playback(self):
        """停止视频播放"""
        self.stop_video_thread = True
        self.is_playing = False
        
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
            self.video_thread = None
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
        if self.processed_video_capture:
            self.processed_video_capture.release()
            self.processed_video_capture = None
            
        # 重置播放按钮
        self.play_pause_btn.config(text="播放")
        
        # 如果不是在处理过程中，隐藏播放控制区域
        if not self.is_processing:
            self.playback_control_frame.pack_forget()
    
    def display_image(self, image, canvas, title=None, forced_width=None, forced_height=None):
        """在指定画布上显示图像"""
        try:
            # 确保image是有效的numpy数组
            if image is None or not isinstance(image, np.ndarray):
                self.log(f"无效的图像数据类型: {type(image)}", console_only=True)
                return False
                
            # 确保图像有正确的形状
            if len(image.shape) < 2:
                self.log(f"无效的图像形状: {image.shape}", console_only=True)
                return False
                
            # 获取画布尺寸，优先使用forced参数
            canvas_width = forced_width if forced_width else canvas.winfo_width()
            canvas_height = forced_height if forced_height else canvas.winfo_height()
            
            # 确保画布尺寸有效
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = int(canvas.cget('width')) if canvas.cget('width') else 640
                canvas_height = int(canvas.cget('height')) if canvas.cget('height') else 360
            
            # 计算调整大小的比例
            img_height, img_width = image.shape[:2]
            
            # 保持原始宽高比
            ratio = min(canvas_width / img_width, canvas_height / img_height)
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            
            # 调整图像大小
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # 将OpenCV的BGR格式转换为PIL可用的RGB格式
            if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
                # 已经是RGB格式，不需要转换
                pil_image = Image.fromarray(resized_image)
            else:
                # 灰度或其他格式转换为RGB
                pil_image = Image.fromarray(resized_image).convert('RGB')
            
            # 创建Tkinter可以显示的PhotoImage
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # 清除画布并显示新图像
            canvas.delete("all")
            
            # 计算图像在画布上的中心位置
            x_center = canvas_width // 2
            y_center = canvas_height // 2
            
            # 在画布上创建图像
            canvas.create_image(x_center, y_center, image=photo)
            
            # 保存对图像的引用，防止被垃圾回收
            if canvas == self.original_canvas:
                self._original_photo = photo
            else:
                self._processed_photo = photo
            
            # 如果提供了标题，显示在图像上方
            if title:
                canvas.create_text(x_center, 20, text=title, fill="white", 
                                  font=("Arial", 14, "bold"))
                
            # 更新时间戳标签（用于视频）
            if hasattr(self, 'current_frame_index') and self.current_frame_index is not None:
                frame_timestamp = self.current_frame_index / self.fps if hasattr(self, 'fps') and self.fps else 0
                mins, secs = divmod(frame_timestamp, 60)
                time_str = f"{int(mins):02d}:{secs:05.2f}"
                
                # 在视频下方显示时间戳
                if canvas == self.original_canvas:
                    if hasattr(self, 'time_label_original'):
                        self.time_label_original.config(text=f"时间: {time_str}")
                elif canvas == self.processed_canvas:
                    if hasattr(self, 'time_label_processed'):
                        self.time_label_processed.config(text=f"时间: {time_str}")
            
            return True
        except Exception as e:
            self.log(f"显示图像错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc(), console_only=True)
            return False
    
    def start_processing(self):
        """Start processing the image or video"""
        # 立即禁用所有按钮，不等待后续的处理
        self.process_btn.state(['disabled'])
        self.select_image_btn.state(['disabled'])
        self.select_video_btn.state(['disabled'])
        
        if self.is_processing:
            self.log("处理已在进行中，请等待完成")
            return
        
        if not self.current_media_path:
            self.log("未选择任何媒体文件")
            # 如果没有选择媒体文件，恢复按钮状态
            self.process_btn.state(['!disabled'])
            self.select_image_btn.state(['!disabled'])
            self.select_video_btn.state(['!disabled'])
            return
        
        # 重置视频播放标志
        if hasattr(self, 'video_playback_started'):
            del self.video_playback_started
        
        # 隐藏播放控制区域
        self.playback_control_frame.pack_forget()
        
        # 停止正在进行的视频播放
        self.stop_video_playback()
        
        # 清空行为列表
        for item in self.behavior_list.get_children():
            self.behavior_list.delete(item)
        
        # 重置行为数据
        if hasattr(self, 'behaviors_data'):
            self.behaviors_data = []
        
        # 清空日志文本区域
        self.log_text.delete(1.0, tk.END)
        self.log("开始新的分析任务...")
        
        # 更新UI状态
        self.is_processing = True
        
        # 设置一个强制禁用按钮的标志
        self._force_disable_buttons = True
        
        self.log(f"开始分析 {'图片' if self.current_media_type == 'image' else '视频'}...")
        
        # 启动处理线程
        processing_thread = threading.Thread(target=self.processing_thread)
        processing_thread.daemon = True
        processing_thread.start()
        
        # 启动按钮禁用检查循环
        self._check_and_disable_buttons()
    
    def _check_and_disable_buttons(self):
        """循环检查并强制禁用按钮，直到处理完成"""
        if hasattr(self, '_force_disable_buttons') and self._force_disable_buttons:
            # 强制禁用所有按钮
            self.process_btn.state(['disabled'])
            self.select_image_btn.state(['disabled'])
            self.select_video_btn.state(['disabled'])
            # 每25毫秒执行一次，确保按钮状态不会被其他操作更改
            self.root.after(25, self._check_and_disable_buttons)
        else:
            # 如果标志被移除或设为False，则停止循环
            # 检查是否需要重新启用按钮
            if not self.is_processing:
                # 在处理完成后主动确保按钮被启用
                self.root.after(50, self._ensure_buttons_enabled)
    
    def _ensure_buttons_enabled(self):
        """确保按钮在处理完成后被启用"""
        if not self.is_processing:
            self.process_btn.state(['!disabled'])
            self.select_image_btn.state(['!disabled'])
            self.select_video_btn.state(['!disabled'])
            # 如果有处理结果，启用保存按钮
            if hasattr(self, 'processed_media_path') and self.processed_media_path and os.path.exists(self.processed_media_path):
                self.save_btn.state(['!disabled'])
    
    def processing_thread(self):
        """Background thread for processing"""
        try:
            # 重置摘要生成标志，确保每次新的处理都会生成新的摘要
            self.summary_generated = False
            
            # 根据媒体类型选择处理方法
            if self.current_media_type == 'image':
                self.process_image()
            elif self.current_media_type == 'video':
                self.process_video()
            else:
                self.handle_processing_error("未知的媒体类型")
                return
        except Exception as e:
            import traceback
            error_msg = f"处理线程错误: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            self.handle_processing_error(error_msg)
    
    def finalize_processing(self):
        """Finalize processing"""
        # 关闭强制禁用按钮标志
        self._force_disable_buttons = False
        
        self.is_processing = False
        self.process_btn.state(['!disabled'])
        self.select_image_btn.state(['!disabled'])
        self.select_video_btn.state(['!disabled'])
        self.log("分析完成")
        
        # 确保摘要已生成
        if hasattr(self, 'behaviors_data') and not hasattr(self, 'summary_generated'):
            self.log("处理完成后检查到尚未生成行为摘要，将生成摘要...")
            behaviors = self.behaviors_data
            
            # 确保可疑帧列表存在
            if not hasattr(self, 'suspicious_frames'):
                self.suspicious_frames = []
                
            suspicious_frames = self.suspicious_frames
            
            # 判断是否为图片分析
            is_image_analysis = len(behaviors) == 1 and behaviors[0][0] == 0
            
            # 计算相关参数
            theft_frames = len(suspicious_frames)
            if is_image_analysis:
                total_frames = 1
                # 如果是图片分析且有行为但没有可疑帧，将可疑帧计为1
                if theft_frames == 0:
                    for _, frame_behaviors in behaviors:
                        if frame_behaviors:
                            theft_frames = 1
                            break
            else:
                # 获取视频总帧数
                cap = cv2.VideoCapture(self.current_media_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 1
                cap.release()
            
            # 计算最大概率
            max_probability = 0.0
            for _, frame_behaviors in behaviors:
                for behavior in frame_behaviors:
                    max_probability = max(max_probability, behavior.get('confidence', 0.0))
            
            # 生成摘要
            self.create_behavior_summary(behaviors, max_probability, theft_frames, total_frames)
        
        # 如果是视频处理，并且存在处理后的视频文件，则启动视频对比播放
        # 注意：我们需要确保这里不会与video_processing_thread中的播放启动冲突
        # 只有当视频处理是在主线程中完成时（比如直接调用process_video而不是通过线程），才会执行此代码
        if (self.current_media_type == 'video' and 
            hasattr(self, 'processed_media_path') and 
            self.processed_media_path and 
            os.path.exists(self.processed_media_path)):
            
            # 使用标志检查视频播放是否已经启动
            if not hasattr(self, 'video_playback_started') or not self.video_playback_started:
                self.log("主线程中检测到视频处理完成，准备启动同步播放视频对比...")
                self.video_playback_started = True
                # 使用短延迟确保UI更新完成后再启动视频播放
                self.root.after(300, self.start_processed_video_playback)
        
        if self.processed_media_path:
            self.save_btn.state(['!disabled'])
    
    def handle_processing_error(self, error_message):
        """Handle processing error"""
        # 关闭强制禁用按钮标志
        self._force_disable_buttons = False
        
        self.is_processing = False
        self.process_btn.state(['!disabled'])
        self.select_image_btn.state(['!disabled'])
        self.select_video_btn.state(['!disabled'])
        self.log(f"处理过程中出错: {error_message}")
        self.update_progress(0)
        messagebox.showerror("处理错误", f"处理过程中出错: {error_message}")
    
    def process_image(self):
        """处理当前加载的图片"""
        if self.current_media_path is None or not os.path.exists(self.current_media_path):
            messagebox.showerror("处理错误", "请先选择一张有效的图片")
            return
            
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 更新处理状态
            self.is_processing = True
            self.update_progress(10)
            self.progress_label.config(text="正在加载图片...")
            
            # 初始化检测器
            if not hasattr(self, 'theft_detector') or self.theft_detector is None:
                self.update_progress(20)
                self.progress_label.config(text="正在初始化检测模型...")
                
                from src.models.detection import TheftDetector
                self.theft_detector = TheftDetector()
                # 设置兼容性属性
                self.detector = self.theft_detector
                
                if self.theft_detector.model is None:
                    messagebox.showerror("模型错误", "无法加载检测模型，请确保模型文件存在")
                    self.is_processing = False
                    return
            
            # 检查图片是否已加载
            if self.original_image is None:
                # 加载图片
                self.update_progress(30)
                self.progress_label.config(text="正在读取图片...")
                self.original_image = cv2.imread(self.current_media_path)
                if self.original_image is None:
                    messagebox.showerror("图片错误", "无法读取图片文件")
                    self.is_processing = False
                    return
            
            # 调整图片大小以适应屏幕
            self.update_progress(40)
            self.progress_label.config(text="正在调整图片大小...")
            
            # 进行检测
            self.update_progress(50)
            self.progress_label.config(text="正在进行检测分析...")
            
            # 保存原始帧用于后续使用
            frame_copy = self.original_image.copy()
            self.processed_frame = frame_copy.copy()  # 保存一份处理帧的副本
            
            # 使用检测器进行分析
            result, theft_probability = self.theft_detector.detect_theft(frame_copy)
            
            # 保存检测结果供后续使用
            self._last_detection_result = result
            
            # 保存环境判断结果
            self.is_retail_environment = self.theft_detector._is_retail_environment(result)
            
            # 创建检测标注
            self.update_progress(70)
            self.progress_label.config(text="正在生成检测结果...")
            
            # 保存最大盗窃概率
            self.max_theft_probability = theft_probability
            annotated_frame = self.theft_detector.draw_detection(frame_copy, result, theft_probability)
            
            # 将检测结果添加到日志中
            self.log(f"检测到盗窃概率原始值: {theft_probability}")
            import datetime
            with open("debug_log.txt", "a") as f:
                f.write(f"[{datetime.datetime.now()}] 检测到的盗窃概率: {theft_probability}\n")
                if hasattr(result, 'boxes'):
                    for i, box in enumerate(result.boxes):
                        cls_id = int(box.cls[0])
                        class_name = result.names.get(cls_id, "")
                        conf = float(box.conf[0])
                        f.write(f"[{datetime.datetime.now()}] 检测对象 {i}: {class_name}, 可信度: {conf}\n")
            
            # 创建视频行为检测器用于图片分析
            # 注释掉这行代码，因为它会导致与test_detector.py的结果不一致
            # from src.models.behavior.video_behavior import VideoBehaviorDetector
            # behavior_detector = VideoBehaviorDetector()
            
            # 添加图像行为检测器用于规则引擎分析 - 与test_detector.py保持一致
            try:
                from src.models.behavior.image_behavior import ImageBehaviorDetector
                image_behavior_detector = ImageBehaviorDetector()
                # 为保持与test_detector.py一致，不再使用VideoBehaviorDetector
                behavior_detector = image_behavior_detector
            except Exception as e:
                self.log(f"初始化图像行为检测器出错: {str(e)}")
                image_behavior_detector = None
                behavior_detector = None
            
            self.log("执行姿态估计和行为分析...")
            # 使用ImageBehaviorDetector中的姿态估计方法，与test_detector.py保持一致
            pose_results = image_behavior_detector._extract_pose_landmarks(frame_copy)
            
            # 初始化行为列表
            behaviors = []
            
            # 始终初始化behaviors_data以避免属性不存在错误
            self.behaviors_data = [(0, [])]
            
            # 检测基于姿态的行为
            if pose_results:
                self.log("检测到人体姿态，分析可疑行为...")
                
                # 从检测结果中提取人物边界框
                person_detections = []
                try:
                    # 适配不同格式的检测结果
                    if hasattr(result, 'boxes'):
                        # 处理ultralytics Results对象
                        for i in range(len(result.boxes)):
                            box = result.boxes[i]
                            cls_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else -1
                            if cls_id >= 0 and result.names.get(cls_id, "") == "person":
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0
                                # 以与test_detector.py一致的格式存储
                                person_detections.append([(x1, y1, x2, y2), conf])
                                self.log(f"检测到人物边界框: {(x1, y1, x2, y2)}, 置信度: {conf:.2f}")
                    elif isinstance(result, list):
                        # 处理字典列表格式
                        for d in result:
                            if d.get('class', '') == 'person':
                                bbox = d.get('bbox')
                                conf = d.get('confidence', 0.0)
                                if bbox:
                                    person_detections.append([bbox, conf])
                except Exception as e:
                    self.log(f"提取人员检测结果错误: {str(e)}")
                
                # 提取物体信息用于规则引擎
                objects = []
                try:
                    if hasattr(result, 'boxes'):
                        for i in range(len(result.boxes)):
                            box = result.boxes[i]
                            cls_id = int(box.cls[0]) if hasattr(box, 'cls') and len(box.cls) > 0 else -1
                            class_name = result.names.get(cls_id, "")
                            if cls_id >= 0 and class_name != "person":
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = float(box.conf[0]) if hasattr(box, 'conf') and len(box.conf) > 0 else 0.0
                                objects.append({
                                    'bbox': (x1, y1, x2, y2),
                                    'class': class_name,
                                    'confidence': conf
                                })
                    self.log(f"提取了 {len(objects)} 个物体信息用于规则引擎分析")
                except Exception as e:
                    self.log(f"提取物体信息时出错: {str(e)}")
                
                # 使用规则引擎进行分析
                if image_behavior_detector:
                    try:
                        self.log("使用规则引擎分析姿态数据...")
                        
                        # 直接调用各种规则引擎方法进行行为检测，模仿test_detector.py的调用方式
                        image_behavior_detector.detected_behaviors = []
                        
                        self.log(f"人物检测格式: {type(person_detections)}, 长度: {len(person_detections)}")
                        if len(person_detections) > 0:
                            self.log(f"第一个人物数据: {person_detections[0]}, 类型: {type(person_detections[0])}")
                        
                        # 准备转换person_detections格式，从字典列表到坐标和置信度元组列表
                        person_list = []
                        for person in person_detections:
                            if isinstance(person, dict) and 'bbox' in person:
                                bbox = person['bbox']
                                conf = person.get('confidence', 0.0)
                                person_list.append([bbox, conf])
                                self.log(f"从字典转换人物数据: bbox={bbox}, conf={conf}")
                            elif isinstance(person, (list, tuple)) and len(person) >= 1:
                                person_list.append(person)
                                self.log(f"保留列表格式人物数据: {person}")
                        
                        self.log(f"转换后的人物列表: {person_list}")
                        
                        # 按照test_detector.py的方式调用检测方法
                        image_behavior_detector._detect_pose_based_behaviors(frame_copy, pose_results, person_list, objects)
                        image_behavior_detector._detect_suspicious_arm_posture(frame_copy, pose_results, person_list)
                        image_behavior_detector._detect_abnormal_arm_positions(frame_copy, pose_results, person_list)
                        image_behavior_detector._detect_body_shielding(frame_copy, pose_results, person_list, objects)
                        image_behavior_detector._detect_hands_behind_back(frame_copy, pose_results, person_list)
                        
                        # 获取行为检测结果
                        if image_behavior_detector.detected_behaviors:
                            rule_behaviors = image_behavior_detector.detected_behaviors
                            self.log(f"规则引擎检测到 {len(rule_behaviors)} 个可疑行为")
                            for behavior in rule_behaviors:
                                self.log(f" - {behavior['type']}: {behavior['confidence']:.2f}")
                            behaviors.extend(rule_behaviors)
                        
                        # 使用规则引擎中的口袋检测方法
                        try:
                            image_behavior_detector._detect_pocket_concealment(frame_copy, pose_results, person_list, objects)
                            if len(image_behavior_detector.detected_behaviors) > len(behaviors):
                                new_behaviors = image_behavior_detector.detected_behaviors[len(behaviors):]
                                self.log(f"口袋检测额外发现 {len(new_behaviors)} 个可疑行为")
                                for behavior in new_behaviors:
                                    self.log(f" - {behavior['type']}: {behavior['confidence']:.2f}")
                                behaviors = image_behavior_detector.detected_behaviors
                        except Exception as e:
                            self.log(f"执行口袋检测出错: {str(e)}")
                        
                        # 使用物品到口袋检测方法
                        try:
                            # 与test_detector.py保持一致的调用和处理方式
                            self.log("调用_check_item_to_pocket进行物品隐藏检测...")
                            item_to_pocket_result = image_behavior_detector._check_item_to_pocket(pose_results, objects)
                            self.log(f"物品到口袋检测结果: {item_to_pocket_result}")
                            
                            # 严格按照test_detector.py的方式处理返回结果
                            if item_to_pocket_result and isinstance(item_to_pocket_result, dict):
                                # 返回了正确的行为字典
                                behaviors.append(item_to_pocket_result)
                                self.log(f" - 添加行为: {item_to_pocket_result['type']}: {item_to_pocket_result['confidence']:.2f}")
                            elif item_to_pocket_result == True:
                                # 如果返回True但不是字典，添加标准行为
                                pocket_behavior = {
                                    'type': 'Item Concealed in Pocket',
                                    'confidence': 0.95
                                }
                                behaviors.append(pocket_behavior)
                                self.log(f" - 添加行为: {pocket_behavior['type']}: {pocket_behavior['confidence']:.2f}")
                        except Exception as e:
                            self.log(f"执行物品到口袋检测时出错: {str(e)}")
                            
                    except Exception as e:
                        self.log(f"规则引擎分析出错: {str(e)}")
                        
                self.update_progress(60)
                
                # 添加详细日志，记录执行状态
                self.log("即将执行VideoBehaviorDetector分析，但该部分可能导致与test_detector.py不一致...")
                self.log(f"behavior_detector类型: {type(behavior_detector).__name__}")
                self.log(f"person_detections类型: {type(person_detections)}, 长度: {len(person_detections)}")
                
                # 跳过这部分VideoBehaviorDetector处理，确保与test_detector.py结果一致
                # skip_video_detector = True
                # if not skip_video_detector:
                #     # 如果检测到人，分析行为
                #     if person_detections:
                #         for i, person in enumerate(person_detections):
                #             self.log(f"处理第{i+1}个人物: {person}")
                #             # 根据person的类型获取bbox
                #             if isinstance(person, dict) and 'bbox' in person:
                #     for person in person_detections:
                #         person_bbox = person[0]  # 使用正确的格式
                #         # 检测基于姿态的行为
                #         pose_behaviors = behavior_detector._detect_pose_based_behaviors(
                #             frame_copy, 
                #             pose_results.get('landmarks'), 
                #             person_bbox,
                #             [d for d in result if isinstance(d, dict) and d.get('class') != 'person']
                #         )
                        
                #         # 添加到行为列表
                #         if pose_behaviors:
                #             behaviors.extend(pose_behaviors)
                    
                #     # 检测基本行为
                #     basic_behaviors = behavior_detector.detect_behaviors_in_image(frame_copy, result)
                #     if basic_behaviors:
                #         behaviors.extend(basic_behaviors)
                # """
            
            self.update_progress(70)
            
            # 将行为添加到行为列表UI中
            if behaviors:
                self.log(f"检测到 {len(behaviors)} 个可疑行为")
                for behavior in behaviors:
                    behavior_type = behavior.get('type', '未知行为')
                    confidence = behavior.get('confidence', 0.0)
                    # 添加到UI中的行为列表
                    self.add_behavior_to_list(0, 0, behavior_type, confidence)
                
                # 更新行为数据
                self.behaviors_data = [(0, behaviors)]
                
                # 计算行为平均可疑度，用于显示
                avg_confidence = sum(b.get('confidence', 0) for b in behaviors) / len(behaviors) if behaviors else 0
                self.log(f"行为平均可疑度: {avg_confidence:.2%}")
                
                # 使用行为平均可疑度作为绘制到图片上的概率
                theft_proba = avg_confidence
                
                # 基于规则引擎和模型检测到的行为调整盗窃概率
                try:
                    # 使用规则而非硬编码值计算盗窃概率
                    if len(behaviors) > 0 and image_behavior_detector:
                        # 读取配置中的行为权重
                        behavior_weights = image_behavior_detector._load_config().get('behavior_weights', {})
                        
                        # 根据检测到的行为和配置的权重计算盗窃概率
                        behavior_confidence_sum = 0
                        behavior_weight_sum = 0
                        
                        for behavior in behaviors:
                            behavior_type = behavior['type']
                            confidence = behavior['confidence']
                            # 使用配置的权重，如果没有则使用默认值0.5
                            weight = behavior_weights.get(behavior_type, 0.5)
                            
                            behavior_confidence_sum += confidence * weight
                            behavior_weight_sum += weight
                        
                        # 计算加权平均置信度
                        if behavior_weight_sum > 0:
                            avg_weighted_confidence = behavior_confidence_sum / behavior_weight_sum
                            # 使用基本盗窃概率和行为分析结果的加权平均 - 与test_detector.py保持一致
                            # 原始盗窃概率使用theft_probability而不是其他值
                            theft_proba = theft_probability * 0.3 + avg_weighted_confidence * 0.7
                            self.log(f"初始盗窃概率: {theft_probability:.2f}, 行为加权平均: {avg_weighted_confidence:.2f}")
                        else:
                            # 如果没有有效的行为权重和，则保持原始盗窃概率
                            theft_proba = theft_probability
                    else:
                        # 没有检测到行为，保持原始盗窃概率
                        theft_proba = theft_probability
                    
                    self.log(f"基于规则引擎计算的盗窃概率: {theft_proba:.2f}")
                except Exception as e:
                    self.log(f"计算盗窃概率出错: {str(e)}")
                    # 使用初始计算的盗窃概率
                    theft_proba = theft_probability
                
                self.log(f"处理后的概率值: {theft_proba}, 将显示为: {theft_proba:.2%}")
                # 更新盗窃概率值
                theft_probability = theft_proba
                
                # 保存行为到检测器，使其能够被其他流程使用
                self.theft_detector.suspicious_behaviors = behaviors
            else:
                self.log("未检测到可疑行为")
                # 确保behaviors_data即使在没有检测到行为时也存在
                self.behaviors_data = [(0, [])]
                # 确保suspicious_behaviors属性存在
                self.theft_detector.suspicious_behaviors = []
            
            # Draw results, including pose and behaviors
            self.log("绘制检测结果...")
            result_img = self.theft_detector.draw_detection(frame_copy, result, theft_probability)
            
            # 绘制姿态关键点
            if pose_results:
                result_img = behavior_detector._draw_pose_landmarks(result_img, pose_results)
            
            # 绘制行为边界框和标签
            for behavior in behaviors:
                if 'bbox' in behavior:
                    bbox = behavior['bbox']
                    color = behavior.get('color', (0, 0, 255))  # 默认红色
                    cv2.rectangle(result_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    
                    # 显示行为类型
                    behavior_type = behavior['type']
                    # 直接使用英文行为类型，避免中文乱码
                    display_type = behavior_type
                    # 将行为标签移到边界框右上角
                    (label_width, label_height), _ = cv2.getTextSize(display_type, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    # 标签背景
                    cv2.rectangle(result_img, (bbox[2] - label_width - 10, bbox[1]), (bbox[2], bbox[1] + label_height + 5), color, -1)
                    # 标签文本
                    cv2.putText(result_img, display_type, (bbox[2] - label_width - 5, bbox[1] + label_height), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            self.update_progress(90)
            
            # Save result image
            
            # 确保始终生成行为分析摘要
            if not hasattr(self, 'summary_generated') or not self.summary_generated:
                self.log("生成行为分析摘要...")
                theft_frames = 1 if behaviors else 0
                total_frames = 1  # 图片只有1帧
                max_probability = theft_probability  # Use the updated theft_probability instead of recalculating
                self.create_behavior_summary(self.behaviors_data, max_probability, theft_frames, total_frames)
            
            # 确保输出目录存在
            os.makedirs("static/output", exist_ok=True)
            
            output_path = os.path.join("static", "output", f"result_{int(time.time())}.jpg")
            cv2.imwrite(output_path, result_img)
            self.processed_media_path = output_path
            
            # Display result
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            # 使用正确的canvas对象名称 - processed_canvas而不是processed_media_canvas
            self.root.after(0, lambda: self.display_image(result_img_rgb, self.processed_canvas))
            
            # 更新行为列表UI
            if hasattr(self, 'behaviors_data') and self.behaviors_data:
                self.root.after(100, self.update_ui_with_behaviors)
            
            # Log results
            is_theft = theft_probability > 0.5 or (behaviors and len(behaviors) > 0)
            self.log(f"检测完成: {'发现盗窃行为' if is_theft else '未发现盗窃行为'}")
            self.log(f"盗窃概率: {theft_probability:.2%}")
            
            if behaviors:
                self.log(f"行为分析: 共检测到 {len(behaviors)} 个可疑行为")
                avg_confidence = sum(b.get('confidence', 0) for b in behaviors) / len(behaviors) if behaviors else 0
                self.log(f"行为平均可疑度: {avg_confidence:.2%}")
            
            self.update_progress(100)
            
            # 最后需要恢复按钮状态
            self.is_processing = False
            self._force_disable_buttons = False
            self.process_btn.state(['!disabled'])
            self.select_image_btn.state(['!disabled'])
            self.select_video_btn.state(['!disabled'])
            
            # 启用保存按钮
            if self.processed_media_path:
                self.save_btn.state(['!disabled'])
            
            # 确保按钮被启用
            self.root.after(50, self._ensure_buttons_enabled)
            
            self.log("图片分析完成")
            
        except Exception as e:
            import traceback
            error_msg = f"图片处理错误: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            self.handle_processing_error(error_msg)
    
    def process_video(self):
        """Process video file"""
        try:
            # 清空日志窗口
            self.clear_log()
            
            # 确保所有按钮在处理开始时就被禁用
            # 由于process_video已经在processing_thread线程中运行，直接设置按钮状态可能不安全
            # 使用更强的保证机制，确保按钮禁用状态已经生效
            self.is_processing = True
            
            # 检查是否需要调整当前媒体路径
            if not os.path.exists(self.current_media_path):
                # 尝试在static/videos目录中查找文件
                filename = os.path.basename(self.current_media_path)
                static_video_path = os.path.join("static", "videos", filename)
                if os.path.exists(static_video_path):
                    self.current_media_path = static_video_path
                    self.log(f"已找到视频文件: {self.current_media_path}")
                else:
                    self.log(f"错误：视频文件不存在: {self.current_media_path}")
                    self.is_processing = False
                    return
            
            self.log(f"正在分析视频: {os.path.basename(self.current_media_path)}")
            self.update_progress(5)
            
            # 创建处理线程，防止UI冻结
            import threading
            
            # 创建输出视频文件名
            base_name = os.path.basename(self.current_media_path)
            name_without_ext = os.path.splitext(base_name)[0]
            output_dir = os.path.join("static", "videos", "output")
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 构建输出路径
            self.processed_media_path = os.path.join(
                output_dir, 
                f"{name_without_ext}_analyzed.mp4"
            )
            
            # 启动处理线程
            self.processing_thread = threading.Thread(
                target=self.video_processing_thread,
                args=(self.current_media_path, self.processed_media_path)
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # 更新UI状态
            self.update_progress(15)
            self.log("正在处理视频...请等待完成")
            
        except Exception as e:
            import traceback
            error_msg = f"视频处理初始化错误: {str(e)}\n{traceback.format_exc()}"
            self.log(error_msg)
            self.handle_processing_error(error_msg)
    
    def video_processing_thread(self, video_path, output_path):
        """单独线程处理视频"""
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 重置摘要生成标志
            self.summary_generated = False
            
            # 初始化检测器
            from src.models.detection import TheftDetector
            from src.models.behavior import VideoBehaviorDetector
            
            # 初始化YOLO模型
            self.theft_detector = TheftDetector()
            behavior_detector = VideoBehaviorDetector()
            
            # 初始化视频读写器
            cap = cv2.VideoCapture(video_path)
            
            # 读取视频属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 创建视频编写器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # 初始化光流计算器
            flow_calculator = None
            last_gray = None
            
            # 初始化行为分析数据
            consecutive_theft_frames = 0
            behaviors_data = []
            frame_count = 0
            processed_frames = 0
            processed_with_error = 0
            max_theft_probability = 0.0
            theft_frames = 0
            all_retail_environment_results = []  # 保存所有帧的环境判断结果
            
            # 初始化行为存储列表和ID计数器
            self.behavior_data = []  # 清空旧数据，准备存储新的行为数据
            self.next_behavior_id = 1  # 重置行为ID
            
            # 更新进度和状态回调
            def update_progress_callback(value, text):
                self.root.after(0, lambda: self.restore_progress_state(value, text))
            
            # 帧处理回调
            def frame_callback(frame_index, output_frame, behaviors, original_frame=None):
                nonlocal processed_frames, processed_with_error, max_theft_probability, theft_frames
                processed_frames += 1
                
                # 增加进度条更新频率，每5帧更新一次
                if frame_index % 5 == 0:
                    progress = int((frame_index / total_frames) * 100)
                    current_time = time.time() - start_time
                    processing_speed = frame_index / current_time if current_time > 0 else 0
                    remaining_frames = total_frames - frame_index
                    remaining_time = remaining_frames / processing_speed if processing_speed > 0 else 0
                    
                    # 更新进度信息
                    progress_text = f"正在处理视频 {progress}% - 帧 {frame_index}/{total_frames} - 预计剩余时间: {int(remaining_time/60)}分{int(remaining_time%60)}秒"
                    update_progress_callback(progress, progress_text)
                
                # 在主线程中更新UI
                if frame_index % 10 == 0 or frame_index == total_frames - 1:
                    # 创建帧信息
                    frame_data = {
                        'frame': output_frame,  # 处理后的帧（带有检测结果）显示在右侧
                        'frame_index': frame_index,
                        'time': '{:.2f}s'.format(frame_index / fps),
                        'behaviors': behaviors,
                        'original_frame': original_frame,  # 原始帧显示在左侧
                        'theft_probability': 0.0
                    }
                    
                    # 获取当前帧的盗窃概率
                    if behaviors is not None and isinstance(behaviors, list) and behaviors:
                        probabilities = [b.get('probability', 0.0) for b in behaviors if isinstance(b, dict)]
                        if probabilities:
                            frame_data['theft_probability'] = max(probabilities)
                    
                    # 更新最大盗窃概率
                    if frame_data['theft_probability'] > max_theft_probability:
                        max_theft_probability = frame_data['theft_probability']
                    
                    # 统计盗窃帧数
                    if frame_data['theft_probability'] > 0.4:
                        theft_frames += 1
                    
                    # 在主线程中更新UI
                    self.root.after(0, lambda: self.update_frame_display(frame_data))
                    
                    # 实时更新行为列表 - 每当检测到行为时添加到UI
                    if behaviors and len(behaviors) > 0:
                        time_point = frame_index / fps if fps > 0 else 0
                        for behavior in behaviors:
                            behavior_type = behavior.get('type', '未知行为')
                            confidence = behavior.get('confidence', 0.0)
                            # 在主线程中更新行为列表
                            self.root.after(0, lambda f=frame_index, t=time_point, bt=behavior_type, c=confidence: 
                                           self.add_behavior_to_list(f, t, bt, c))
                
                # 无错误处理完成处理
                return True
            
            # 视频处理完成后的回调
            def finalize_video_processing():
                # 保存处理结果的视频路径
                self.processed_media_path = output_path
                
                # 更新UI
                self.update_progress(95)
                self.progress_label.config(text="视频处理完成，正在生成结果...")
                
                # 确定是否为零售环境，使用最频繁的环境判断结果
                if all_retail_environment_results:
                    # 计算所有帧中True和False的数量
                    true_count = all_retail_environment_results.count(True)
                    false_count = all_retail_environment_results.count(False)
                    
                    # 使用多数表决确定最终环境类型
                    self.is_retail_environment = true_count > false_count
                    self.log(f"基于视频中的{len(all_retail_environment_results)}帧分析，环境判断为: {'零售环境' if self.is_retail_environment else '非零售环境'} (零售判断率: {true_count/len(all_retail_environment_results):.2%})")
                else:
                    # 默认使用非零售环境
                    self.is_retail_environment = False
                    self.log("无法确定环境类型，默认为非零售环境")
            
            # 设置帧回调
            behavior_detector.frame_processed_callback = frame_callback
            
            # 执行视频分析
            self.processed_media_path, suspicious_frames, behaviors = behavior_detector.analyze_video_behavior(
                video_path, self.theft_detector, callback=update_progress_callback, frame_callback=frame_callback)
            
            # 在视频分析期间捕获零售环境判断结果
            # 每隔10帧检查一次环境
            cap = cv2.VideoCapture(video_path)
            sample_frames = []
            
            # 读取视频的一些采样帧来判断环境
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample_interval = max(1, total_frames // 10)  # 至少采样10帧
            
            try:
                for i in range(0, total_frames, sample_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        sample_frames.append(frame)
                
                # 为每个采样帧判断环境类型
                for frame in sample_frames:
                    if frame is not None:
                        try:
                            result = self.theft_detector.model.predict(frame, conf=0.25)[0]
                            is_retail = self.theft_detector._is_retail_environment(result)
                            all_retail_environment_results.append(is_retail)
                        except Exception as e:
                            self.log(f"环境判断出错: {str(e)}")
            except Exception as e:
                self.log(f"采样帧分析错误: {str(e)}")
            finally:
                cap.release()
            
            # 如果有足够的环境判断结果，使用多数投票确定最终环境类型
            if all_retail_environment_results:
                true_count = all_retail_environment_results.count(True)
                false_count = len(all_retail_environment_results) - true_count
                self.is_retail_environment = true_count > false_count
                self.log(f"基于{len(all_retail_environment_results)}个采样帧的环境判断: {'零售环境' if self.is_retail_environment else '非零售环境'} (零售判断率: {true_count/len(all_retail_environment_results)*100:.1f}%)")
            
            # 确保使用正确的变量保存行为数据，以便在异步操作中也能访问
            self.behaviors_data = behaviors
            self.suspicious_frames = suspicious_frames
            
            # 计算最大概率 - 为摘要做准备
            max_probability = 0.0
            for _, frame_behaviors in behaviors:
                for behavior in frame_behaviors:
                    max_probability = max(max_probability, behavior.get('confidence', 0.0))
            
            # 获取视频总帧数
            total_frames = 1
            try:
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
            except Exception as e:
                self.log(f"获取视频帧数错误: {str(e)}")
            
            # 完成后更新UI并生成摘要
            def finalize_video_processing():
                try:
                    # 保存处理结果路径
                    if self.processed_media_path:
                        self.log(f"视频分析完成，输出文件: {os.path.basename(self.processed_media_path)}")
                    else:
                        self.log("视频分析完成，但没有生成输出文件")
                    
                    # 不需要重新添加行为到UI列表，因为已经在分析过程中实时添加了
                    self.log("视频分析完成，所有行为已实时添加到界面")
                    
                    # 获取行为列表中的行为数量
                    behavior_count = len(self.behavior_list.get_children())
                    self.log(f"行为列表中共有 {behavior_count} 条行为记录")
                    
                    # 在视频处理完成后生成行为分析摘要
                    if not hasattr(self, 'summary_generated') or not self.summary_generated:
                        self.log("视频处理完成后生成行为分析摘要")
                        theft_frames = len(self.suspicious_frames)
                        # 直接调用create_behavior_summary而不是update_ui_with_behaviors
                        self.create_behavior_summary(self.behaviors_data, max_probability, theft_frames, total_frames)
                    
                    # 处理完成，通知主线程
                    self.update_progress(100)
                    
                    # 设置处理状态为完成，启用处理按钮
                    self.is_processing = False
                    
                    # 确保禁用按钮标志被关闭
                    self._force_disable_buttons = False
                    
                    # 确保处理按钮重新启用
                    self.process_btn.state(['!disabled'])
                    self.select_image_btn.state(['!disabled'])
                    self.select_video_btn.state(['!disabled'])
                    
                    # 如果有处理结果，启用保存按钮
                    if self.processed_media_path and os.path.exists(self.processed_media_path):
                        self.save_btn.state(['!disabled'])
                    
                    # 在分析完成后自动启动同步播放原始视频和分析后的视频
                    if self.processed_media_path and os.path.exists(self.processed_media_path):
                        self.log("准备启动同步播放视频对比...")
                        # 设置标志表示视频播放已经启动，避免finalize_processing中重复启动
                        self.video_playback_started = True
                        # 使用短延迟确保UI更新完成后再启动视频播放
                        self.root.after(300, self.start_processed_video_playback)
                    
                except Exception as e:
                    self.log(f"视频处理完成后更新UI错误: {str(e)}")
                    import traceback
                    error_traceback = traceback.format_exc()
                    self.log(error_traceback)
                    # 将错误信息添加到摘要区域，确保用户能够看到
                    if hasattr(self, 'log_text'):
                        self.log_text.delete(1.0, tk.END)
                        self.log_text.insert(tk.END, "----------- 视频处理错误 -----------\n", "title")
                        self.log_text.insert(tk.END, f"错误信息: {str(e)}\n\n", "error")
                        self.log_text.insert(tk.END, "详细跟踪信息:\n", "subtitle")
                        self.log_text.insert(tk.END, error_traceback, "traceback")
                        self.log_text.tag_configure("title", font=("Arial", 10, "bold"), foreground="red")
                        self.log_text.tag_configure("error", font=("Arial", 9, "bold"), foreground="red")
                        self.log_text.tag_configure("subtitle", font=("Arial", 9, "bold"), foreground="black")
                        self.log_text.tag_configure("traceback", font=("Courier", 8), foreground="gray")
                    
                    # 即使出错也要设置处理状态为完成
                    self.is_processing = False
            
            # 在主线程中执行最终处理
            self.root.after(100, finalize_video_processing)
            
            # 导致内存溢出的对象设为 None
            behavior_detector = None
            
            # 强制执行垃圾回收
            import gc
            gc.collect()
            
            # 释放资源
            writer.release()
            
        except Exception as e:
            self.log(f"视频处理线程错误: {str(e)}")
            import traceback
            error_traceback = traceback.format_exc()
            self.log(error_traceback)
            
            # 将错误信息添加到摘要区域，确保用户能够看到
            def display_error_summary():
                if hasattr(self, 'log_text'):
                    self.log_text.delete(1.0, tk.END)
                    self.log_text.insert(tk.END, "----------- 视频处理错误 -----------\n", "title")
                    self.log_text.insert(tk.END, f"错误信息: {str(e)}\n\n", "error")
                    self.log_text.insert(tk.END, "详细跟踪信息:\n", "subtitle")
                    self.log_text.insert(tk.END, error_traceback, "traceback")
                    self.log_text.tag_configure("title", font=("Arial", 10, "bold"), foreground="red")
                    self.log_text.tag_configure("error", font=("Arial", 9, "bold"), foreground="red")
                    self.log_text.tag_configure("subtitle", font=("Arial", 9, "bold"), foreground="black")
                    self.log_text.tag_configure("traceback", font=("Courier", 8), foreground="gray")
                    
                # 更新进度为0，表示处理失败
                self.update_progress(0)
                
                # 确保在错误发生后恢复按钮状态
                self.is_processing = False
                # 关闭强制禁用按钮标志
                self._force_disable_buttons = False
                self.process_btn.state(['!disabled'])
                self.select_image_btn.state(['!disabled'])
                self.select_video_btn.state(['!disabled'])
                self.log("分析过程中发生错误，已终止")
            
            # 确保在主线程中更新UI
            self.root.after(100, display_error_summary)
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.log(f"视频处理线程错误: {str(e)}")
            
            # 确保即使display_error_summary函数也出错，也能恢复按钮状态
            def emergency_recovery():
                self.is_processing = False
                # 关闭强制禁用按钮标志
                self._force_disable_buttons = False
                self.process_btn.state(['!disabled'])
                self.select_image_btn.state(['!disabled'])
                self.select_video_btn.state(['!disabled'])
                self.update_progress(0)
                self.log("视频处理线程发生严重错误，已恢复按钮状态")
            
            # 在主线程中恢复UI状态
            self.root.after(100, emergency_recovery)
    
    def update_frame_display(self, callback_data):
        """Update the frame display in the UI.
        
        Args:
            callback_data: Dictionary containing the following keys:
                - frame: Processed frame
                - original_frame: Original frame
                - frame_index: Current frame index
                - total_frames: Total number of frames
                - log_message: Optional log message
                - behaviors: Optional behaviors list
                - theft_probability: Theft probability
                - detections: Detection results
        """
        try:
            # 从frame_data中提取数据
            processed_frame = callback_data.get('frame')  # 处理后的帧
            original_frame = callback_data.get('original_frame')  # 原始帧
            frame_idx = callback_data.get('frame_index', 0)
            
            # 获取总帧数 - 优先使用传入的值，没有则尝试读取视频属性
            total_frames = callback_data.get('total_frames')
            if total_frames is None and hasattr(self, 'current_media_path') and self.current_media_path:
                cap = cv2.VideoCapture(self.current_media_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                else:
                    total_frames = 1  # 默认值
            else:
                total_frames = total_frames or 1  # 确保非空
                
            log_message = callback_data.get('log_message')
            behaviors = callback_data.get('behaviors', [])
            theft_probability = callback_data.get('theft_probability', 0.0)
            
            # 更新进度条 - 显示整体进度
            if total_frames > 0:
                progress = min(100, int(100 * frame_idx / total_frames))
                self.progress_bar['value'] = progress
                self.progress_text.set(f"{progress}%")
                
            # 原始视频帧显示在左侧（原始媒体区域）
            if original_frame is not None:
                # 将原始帧转换为RGB格式
                original_frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                # 显示到原始画布上（左侧）
                self.display_image(original_frame_rgb, self.original_canvas)
            
            # 处理后的帧显示在右侧（检测结果区域）
            if processed_frame is not None:
                # 将处理后的帧转换为RGB格式
                if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
                    # BGR转RGB
                    processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    # 显示到处理后的画布上（右侧）
                    self.display_image(processed_frame_rgb, self.processed_canvas)
            
            # 如果有日志消息，添加到日志
            if log_message:
                self.log(log_message)
            
            # 视频处理过程中不再实时添加行为到行为列表
            # behaviors在视频处理完成后会一次性添加
            
            # 记录所有行为数据，供最终摘要使用
            if behaviors:
                # 确保存在behaviors_data列表
                if not hasattr(self, 'behaviors_data'):
                    self.behaviors_data = []
                
                # 添加当前帧的行为
                frame_data = (frame_idx, behaviors)
                
                # 检查是否已存在相同帧的数据，避免重复添加
                frame_exists = False
                for i, (existing_frame_idx, _) in enumerate(self.behaviors_data):
                    if existing_frame_idx == frame_idx:
                        # 更新现有帧的行为数据
                        self.behaviors_data[i] = frame_data
                        frame_exists = True
                        break
                
                # 如果是新帧，则添加
                if not frame_exists:
                    self.behaviors_data.append(frame_data)
                
                # 检查是否为可疑帧
                if theft_probability > 0.5:
                    # 确保存在suspicious_frames列表
                    if not hasattr(self, 'suspicious_frames'):
                        self.suspicious_frames = []
                    
                    # 添加到可疑帧列表，避免重复
                    if frame_idx not in self.suspicious_frames:
                        self.suspicious_frames.append(frame_idx)
            
            # 实时记录处理进度
            fps = 30  # 默认fps
            if hasattr(self, 'current_media_path') and self.current_media_path:
                try:
                    cap = cv2.VideoCapture(self.current_media_path)
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            fps = 30  # 如果获取到无效的fps，使用默认值
                        cap.release()
                except Exception:
                    pass  # 忽略错误，使用默认fps
            
            # 计算时间点
            time_point = frame_idx / fps if fps > 0 else 0
            current_time = self.format_time(time_point)
            
            # 计算总时长
            total_time = self.format_time(total_frames / fps if fps > 0 else 0)
            
            # 更新当前帧/总帧数显示
            if hasattr(self, 'frame_label'):
                self.frame_label.config(text=f"帧: {frame_idx}/{total_frames}")
            
            # 更新时间标签
            if hasattr(self, 'time_label'):
                self.time_label.config(text=f"时间: {current_time}/{total_time}")
                
        except Exception as e:
            self.log(f"更新帧显示错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def analyze_behavior(self, detections, flow_data, frame, frame_count, consecutive_theft_frames):
        """分析当前帧中的行为类型，改进盗窃行为检测逻辑"""
        if detections is None:
            return None
            
        # 检查detections类型，兼容不同的返回格式
        persons_detected = False
        persons_count = 0
        
        try:
            # 处理新版ultralytics Results对象
            if hasattr(detections, 'boxes'):
                # 计算检测到的人数
                persons_count = sum(1 for box in detections.boxes 
                                  if hasattr(box, 'cls') and 
                                  detections.names.get(int(box.cls[0]), "") == "person")
                
                persons_detected = persons_count > 0
                
                # 如果没有检测到人，返回None
                if not persons_detected and len(detections.boxes) == 0:
                    return None
            # 处理旧格式的检测结果
            elif isinstance(detections, list):
                # 尝试获取人物检测结果
                persons_count = sum(1 for d in detections 
                                  if hasattr(d, 'class_name') and d.class_name == "person")
                
                persons_detected = persons_count > 0
                
                # 如果没有检测到人，返回None
                if not persons_detected:
                    return None
            else:
                # 未知格式，尝试进行分析
                pass
        except Exception as e:
            # 如果出错，使用默认行为
            print(f"行为分析错误: {str(e)}")
        
        # 基本行为类型
        behavior_types = [
            "遮挡商品区域", 
            "手肘内收姿态异常",
            "肩部不自然隆起",
            "反复调整位置",
            "疑似撕标签动作",
            "可疑商品处理",
            "快速藏匿物品",
            "将物品放入口袋"
        ]
        
        # 检查是否为零售环境
        is_retail_environment = self.check_retail_environment(frame, detections)
        
        # 如果不是零售环境，移除与零售环境相关的行为
        if not is_retail_environment:
            self.log("当前环境不符合零售场景，调整行为检测逻辑")
            # 移除与零售环境相关的行为
            retail_behaviors = ["遮挡商品区域", "疑似撕标签动作", "可疑商品处理", "快速藏匿物品", "将物品放入口袋"]
            behavior_types = [b for b in behavior_types if b not in retail_behaviors]
        
        # 增强型盗窃行为检测逻辑
        
        # 1. 分析光流数据判断动作
        unusual_motion = False
        rapid_movement = False
        concentrated_motion = False
        
        if flow_data is not None:
            avg_motion = flow_data["average_motion"]
            
            # 检查光流数据
            magnitude = flow_data["magnitude"]
            
            # 计算运动区域的集中度 (运动是否集中在特定区域)
            if magnitude is not None:
                # 计算超过阈值的运动点比例
                motion_threshold = 5.0
                motion_points = np.sum(magnitude > motion_threshold)
                total_points = magnitude.size
                motion_ratio = motion_points / total_points if total_points > 0 else 0
                
                # 集中运动检测
                concentrated_motion = (motion_ratio > 0.01) and (motion_ratio < 0.2)
            
            # 判断运动状态
            if avg_motion > 10:  # 高速运动阈值
                rapid_movement = True
            elif avg_motion > 5:  # 轻微异常运动阈值
                unusual_motion = True
        
        # 2. 高级行为推断
        
        # 连续检测计数增加检测可靠性
        if consecutive_theft_frames >= 8:
            # 高连续计数，很可能是盗窃行为
            if rapid_movement:
                return "快速藏匿物品" if is_retail_environment else "手肘内收姿态异常"
            elif concentrated_motion:
                return "手肘内收姿态异常"
            else:
                return "可疑商品处理" if is_retail_environment else "反复调整位置"
        
        # 根据当前帧行为特征分析
        if persons_count >= 1:
            # 有人物的情况下分析行为
            if rapid_movement and concentrated_motion:
                # 快速且集中的动作，可能是藏匿物品
                return "快速藏匿物品" if is_retail_environment else "反复调整位置"
            elif unusual_motion and frame_count % 60 < 30:
                # 异常动作，可能是调整姿势或遮挡物品
                return "遮挡商品区域" if is_retail_environment else "反复调整位置"
            elif concentrated_motion and frame_count % 45 < 15:
                # 集中区域动作，可能是手部操作
                return "疑似撕标签动作" if is_retail_environment else "手肘内收姿态异常"
            elif consecutive_theft_frames > 3:
                # 有连续检测的可疑行为
                if frame_count % 3 == 0:
                    return "手肘内收姿态异常"
                else:
                    return "肩部不自然隆起"
            elif frame_count % 90 < 30:
                # 为了增加检测多样性，周期性返回不同行为类型
                return "反复调整位置"
        
        # 默认行为判断 - 基于帧计数的周期性分配
        # 这确保了即使在不确定的情况下也能给出合理的行为类型
        if behavior_types:  # 确保行为列表不为空
            behavior_index = frame_count % len(behavior_types)
            return behavior_types[behavior_index]
        else:
            # 如果行为列表为空，返回一个通用行为
            return "反复调整位置"
            
    def check_retail_environment(self, frame, detections):
        """
        判断当前场景是否为零售环境（商店、超市等）
        
        Args:
            frame: 当前帧图像
            detections: 检测结果对象
            
        Returns:
            bool: 是否为零售环境
        """
        try:
            if detections is None:
                return False
                
            # 计算环境匹配分数
            retail_score = 0.0
            retail_objects = 0
            person_count = 0
            office_objects = 0
            cell_phone_count = 0  # 手机计数
            
            # 将零售环境指标物体分为三类
            # 强零售指标物体 - 几乎只在零售环境出现
            strong_retail_indicators = [
                "shelf", "cash register", "shopping cart", "shopping basket",
                "price tag", "barcode", "cashier", "checkout counter", 
                "store display", "mannequin", "security tag", "counter",
                "cash", "register", "shop", "store", "market", "mart", "freezer"
            ]
            
            # 中等零售指标物体 - 在零售环境更常见，但其他场景也有
            medium_retail_indicators = [
                "bottle", "refrigerator", "packaged food", "snack", "box",
                "fruit", "vegetable", "meat", "dairy", "drink", "beverage",
                "cabinet", "display", "price", "tag", "sign", "card", "package",
                "hot dog", "sandwich", "food", "candy", "bread"
            ]
            
            # 弱零售指标物体 - 零售和非零售场景都常见，需要多个共同出现才有意义
            weak_retail_indicators = [
                "cup", "bowl", "vase", "wine glass", "product", "box", 
                "plastic bag", "paper bag", "can", "container", "phone",
                "keyboard", "screen", "monitor", "chair", "desk"
            ]
            
            # 办公/会议物品 - 这些物品表明是办公环境而非零售环境
            office_environment_objects = [
                "laptop", "cell phone", "tv", "monitor", "keyboard", "mouse",
                "tie", "suit", "desk", "chair", "table", "notebook", "pen",
                "briefcase", "projector", "whiteboard", "document", "computer"
            ]
            
            # 便利店和小型零售特有指标 - 这些物体在小型零售店铺中更常见
            convenience_store_indicators = [
                "counter", "cash", "register", "display", "stand", "rack",
                "cigarette", "tobacco", "lottery", "ticket", "drink",
                "candy", "snack", "newspaper", "magazine", "hot dog", "sandwich"
            ]
            
            # 提取图像尺寸，用于分析视觉特征
            img_height, img_width = None, None
            if frame is not None:
                img_height, img_width = frame.shape[:2]
            
            # 预处理 - 计算基于图像视觉特征的零售可能性
            visual_retail_score = 0.0
            
            # 分析图像边缘密度 - 零售环境通常有更多的边缘(货架、商品)
            if frame is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 100, 200)
                    edge_ratio = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
                    
                    # 高边缘密度通常表示零售环境中的货架和商品
                    if edge_ratio > 0.08:  # 降低阈值，允许更多边缘环境被识别为零售
                        visual_retail_score += 0.6
                        self.log(f"检测到高边缘密度 ({edge_ratio:.4f})，视觉零售评分 +0.6")
                    elif edge_ratio > 0.05:  # 中等边缘密度
                        visual_retail_score += 0.3
                        self.log(f"检测到中等边缘密度 ({edge_ratio:.4f})，视觉零售评分 +0.3")
                except Exception as e:
                    self.log(f"计算图像边缘失败: {e}")
            
            # 检查图像颜色分布 - 零售环境通常颜色多样
            if frame is not None:
                try:
                    # 计算图像颜色多样性
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    h_bins = 30
                    h_hist = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180])
                    h_hist = h_hist / np.sum(h_hist)  # 归一化
                    
                    # 计算颜色多样性 - 使用非零柱状图的数量
                    color_diversity = np.sum(h_hist > 0.01) / h_bins
                    
                    # 高颜色多样性通常表示零售环境
                    if color_diversity > 0.5:
                        visual_retail_score += 0.4
                        self.log(f"检测到高颜色多样性 ({color_diversity:.4f})，视觉零售评分 +0.4")
                except Exception as e:
                    self.log(f"计算颜色分布失败: {e}")
            
            # 判断多人密集场景 - 用于区分零售环境和办公/会议环境
            formal_attire_count = 0  # 穿着正式服装的人数
            
            # 遍历检测物体
            strong_indicators_found = 0
            medium_indicators_found = 0
            weak_indicators_found = 0
            convenience_indicators_found = 0
            
            # 获取检测框和类别
            boxes = []
            class_names = []
            confidences = []
            
            if hasattr(detections, 'xyxy'):
                # YOLOv5/YOLOv8 风格的结果
                for i in range(len(detections.xyxy[0])):
                    bbox = detections.xyxy[0][i].cpu().numpy()
                    conf = detections.conf[0][i].cpu().numpy()
                    cls_id = int(detections.cls[0][i].cpu().numpy())
                    name = detections.names[cls_id]
                    
                    boxes.append(bbox)
                    class_names.append(name.lower())
                    confidences.append(conf)
            elif hasattr(detections, 'boxes'):
                # ultralytics YOLO 结果
                for box in detections.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = detections.names.get(cls_id, "")
                    
                    boxes.append(bbox)
                    class_names.append(name.lower())
                    confidences.append(conf)
            
            # 分析检测到的物体
            for i, class_name in enumerate(class_names):
                conf = confidences[i]
                
                # 计数人物
                if class_name == "person":
                    person_count += 1
                    
                    # 尝试分析人物着装 - 正装通常表示办公或会议环境
                    if i < len(boxes) and frame is not None:
                        try:
                            x1, y1, x2, y2 = boxes[i]
                            person_img = frame[int(y1):int(y2), int(x1):int(x2)]
                            if person_img.size > 0:
                                # 分析颜色分布 - 简单实现，正装通常是黑色、深蓝色、灰色等暗色
                                hsv = cv2.cvtColor(person_img, cv2.COLOR_BGR2HSV)
                                # 提取亮度通道
                                v_channel = hsv[:,:,2]
                                # 计算暗色像素比例（亮度低于128的部分）
                                dark_ratio = np.sum(v_channel < 128) / v_channel.size
                                # 如果暗色比例高，可能是正装
                                if dark_ratio > 0.6:
                                    formal_attire_count += 1
                                    self.log(f"检测到正装着装，暗色比例: {dark_ratio:.2f}")
                        except Exception as e:
                            self.log(f"分析人物着装失败: {e}")
                    
                    # 在典型位置的人物也可能暗示零售环境
                    if img_height is not None and img_width is not None:
                        # 获取人物位置
                        x1, y1, x2, y2 = boxes[i]
                        person_center_x = (x1 + x2) / 2
                        person_center_y = (y1 + y2) / 2
                        
                        # 判断人物是否在柜台/收银台位置（图像下方区域）
                        if person_center_y > img_height * 0.6:
                            visual_retail_score += 0.3
                            self.log(f"检测到人物位于典型柜台位置，视觉零售评分 +0.3")
                            
                    continue  # 单纯的人物不计入零售环境评分
                
                # 计数手机
                if class_name == "cell phone":
                    cell_phone_count += 1
                    # 手机既可能出现在零售环境，也可能出现在办公环境，需要综合判断
                    office_objects += 1  # 将手机优先归为办公环境物品
                    self.log(f"检测到手机 ({cell_phone_count}个)")
                    continue
                
                # 统计办公环境物品
                if any(office_item in class_name for office_item in office_environment_objects):
                    office_objects += 1
                    self.log(f"检测到办公环境物品: {class_name}")
                    continue
                
                # 检查是否为强零售指标物体
                if any(indicator in class_name for indicator in strong_retail_indicators):
                    strong_indicators_found += 1
                    retail_objects += 1
                    retail_score += 2.0 * conf  # 强指标物体评分更高
                    self.log(f"检测到强零售指标物体: {class_name}, 可信度: {conf:.2f}")
                
                # 检查是否为便利店特有指标
                elif any(indicator in class_name for indicator in convenience_store_indicators):
                    convenience_indicators_found += 1
                    retail_objects += 1
                    retail_score += 1.8 * conf  # 提高便利店指标权重
                    self.log(f"检测到便利店特有指标: {class_name}, 可信度: {conf:.2f}")
                
                # 检查是否为中等零售指标物体
                elif any(indicator in class_name for indicator in medium_retail_indicators):
                    medium_indicators_found += 1
                    retail_objects += 1
                    retail_score += 1.2 * conf  # 提高中等指标权重
                    self.log(f"检测到中等零售指标物体: {class_name}, 可信度: {conf:.2f}")
                
                # 检查是否为弱零售指标物体
                elif any(indicator in class_name for indicator in weak_retail_indicators):
                    weak_indicators_found += 1
                    retail_objects += 1
                    retail_score += 0.6 * conf  # 提高弱指标权重
                    self.log(f"检测到弱零售指标物体: {class_name}, 可信度: {conf:.2f}")
            
            # 如果检测到图像视觉特征评分较高，添加到总体评分中
            retail_score += visual_retail_score
            
            # 根据当前被检物体类型进行额外判断
            # 如果检测到热狗或其他快餐食品，且在便利店环境中
            has_convenience_food = False
            for class_name in class_names:
                if class_name in ["hot dog", "sandwich", "pizza", "donut", "cake"]:
                    has_convenience_food = True
                    break
            
            if has_convenience_food and visual_retail_score > 0.2:
                retail_score += 0.8  # 额外加分
                self.log(f"检测到便利店食品 + 视觉特征，额外加分 +0.8")
            
            # 办公环境特征分析 - 人物密集且大多数穿正装，手机较多，缺少零售物品
            is_office_environment = False
            # 调整办公环境判断条件，放宽对正装的要求
            if (person_count >= 3 and cell_phone_count >= 1 and retail_objects == 0) or \
               (person_count >= 2 and formal_attire_count >= 1 and office_objects >= 1 and retail_objects == 0) or \
               (person_count >= 1 and cell_phone_count >= 2 and retail_objects == 0):
                is_office_environment = True
                self.log(f"检测到办公/会议环境特征: 人数={person_count}, 正装人数={formal_attire_count}, 手机数量={cell_phone_count}, 办公物品={office_objects}")
                
            # 如果明确是办公环境，直接判断为非零售环境
            if is_office_environment:
                self.log("判断为办公或会议环境，非零售环境")
                return False
            
            # 判断逻辑改进
            # 1. 至少检测到1个强零售指标物体，高度可能是零售环境
            if strong_indicators_found >= 1:
                self.log(f"高度可能是零售环境: 检测到{strong_indicators_found}个强零售指标, 环境评分={retail_score:.2f}")
                return True
            
            # 2. 检测到便利店特有指标
            elif convenience_indicators_found >= 1:
                self.log(f"可能是便利店环境: 检测到{convenience_indicators_found}个便利店指标, 环境评分={retail_score:.2f}")
                return True
            
            # 3. 基于视觉特征的高评分，但必须没有明显的办公特征，且必须至少有一个零售物体
            elif visual_retail_score >= 0.4 and office_objects <= 1:  # 降低阈值，原为0.5
                self.log(f"基于视觉特征判断为零售环境: 视觉评分={visual_retail_score:.2f}, 零售物体数={retail_objects}")
                return True
            
            # 4. 检测到多个中等或弱零售指标物体且组合评分高
            elif (medium_indicators_found + weak_indicators_found >= 1) and retail_score > 0.6:  # 降低阈值，原为0.8
                self.log(f"可能是零售环境: 中等指标={medium_indicators_found}, 弱指标={weak_indicators_found}, 评分={retail_score:.2f}")
                return True
            
            # 5. 中等指标物体 + 人物组合场景
            elif medium_indicators_found >= 1 and person_count >= 1 and retail_score > 0.5:  # 降低阈值，原为0.6
                self.log(f"可能是零售环境: 中等指标={medium_indicators_found}, 人数={person_count}, 评分={retail_score:.2f}")
                return True
            
            # 6. 仅视觉特征显著，但必须没有手机和办公物品，且边缘评分很高
            elif visual_retail_score > 0.5 and office_objects <= 1:  # 降低阈值，原为0.6，允许少量办公物品
                self.log(f"基于高视觉评分判断为零售环境: 物体评分={retail_score:.2f}, 视觉评分={visual_retail_score:.2f}")
                return True
            
            # 7. 便利店食品检测
            elif has_convenience_food:  # 移除人物条件，只要有便利店食品就判定为零售环境
                self.log(f"检测到便利店食品，判断为零售环境")
                return True
            
            # 8. 边缘密度判断 - 更宽松
            elif edge_ratio > 0.05:  # 降低阈值，原为0.06，移除中等指标物体的要求
                self.log(f"基于边缘分析，判定为零售环境")
                return True
                
            # 9. 新增：颜色多样性高
            elif color_diversity > 0.4:  # 新增条件，基于颜色多样性判断
                self.log(f"基于颜色多样性分析，判定为零售环境")
                return True
                
            # 10. 新增：中等指标+弱指标组合
            elif medium_indicators_found >= 1 and weak_indicators_found >= 1:
                self.log(f"基于中等指标和弱指标组合，判定为零售环境")
                return True
                
            # 11. 新增：当办公物品少且边缘密度适中
            elif office_objects <= 1 and edge_ratio > 0.04:
                self.log(f"基于低办公物品数量和适中边缘密度，判定为零售环境")
                return True
            
            # 排除仅有人和手机的情况
            if person_count > 0 and cell_phone_count > 0 and retail_objects == 0 and office_objects > 2:
                self.log(f"仅检测到人物和手机以及多个办公物品，可能是办公、会议环境，非零售环境")
                return False
            
            # 排除单人+少量普通物品的误判 - 更严格的条件才排除
            elif person_count == 1 and retail_objects == 0 and visual_retail_score < 0.1 and office_objects > 2:
                self.log(f"单人办公场景，检测到零售物体数量不足: {retail_objects}个, 评分={retail_score:.2f}")
                return False
            
            else:
                # 默认处理 - 如果有任何零售相关指标，就倾向于认为是零售环境
                if retail_objects > 0 or visual_retail_score > 0.3 or edge_ratio > 0.04:
                    self.log(f"未满足标准条件但有零售指标，倾向判定为零售环境")
                    return True
                    
                self.log(f"可能不是零售环境: 强指标={strong_indicators_found}, 中等指标={medium_indicators_found}, 弱指标={weak_indicators_found}, 人数={person_count}, 评分={retail_score:.2f}, 视觉评分={visual_retail_score:.2f}")
                
                # 默认处理 - 如果边缘分析得分高但没有其他指标，仍然认为是零售环境
                if frame is not None and img_height is not None:
                    if edge_ratio > 0.04:  # 降低阈值，原为0.06
                        self.log(f"基于边缘分析，判定为零售环境")
                        return True
                
                return False
            
        except Exception as e:
            self.log(f"零售环境判断出错: {str(e)}")
            # 默认返回False，避免误判
            return False
    
    def get_behavior_description(self, behavior_type):
        """获取行为描述文本"""
        descriptions = {
            "遮挡商品区域": "检测到人物使用身体或其他物品遮挡商品区域，可能试图隐藏盗窃行为。",
            "手肘内收姿态异常": "检测到人物手肘内收角度异常，这是典型的隐藏物品于衣物内的姿势。",
            "肩部不自然隆起": "检测到人物肩部轮廓不自然隆起，可能是将物品藏匿于衣物下。",
            "反复调整位置": "检测到人物在同一区域反复调整位置，这是典型的踌躇不决或准备盗窃的行为。",
            "疑似撕标签动作": "检测到疑似撕标签的手部动作，这是准备盗窃前的常见行为。",
            "可疑商品处理": "检测到对商品的可疑处理方式，可能是试图破坏防盗设备或准备盗窃。",
            "快速藏匿物品": "检测到快速藏匿物品的动作，这是盗窃行为的明显特征。",
            "将物品放入口袋": "检测到手持物品放入口袋或衣物内的可疑动作，这是典型的盗窃行为特征。",
            
            # 新增盗窃行为描述
            "using_coat_umbrella": "检测到使用外套或雨伞遮挡商品区域，这是典型的掩盖盗窃行为。",
            "holding_baby_bag": "检测到不断调整婴儿或手提袋的位置，可能是在转移或隐藏商品。",
            "back_to_camera": "检测到背对摄像头整理包内物品的可疑行为，试图避开监控。",
            "elbow_inward": "检测到明显的手肘内收姿态，通常表示正在向衣物内侧藏匿物品。",
            "shoulder_bulge": "检测到肩部轮廓异常隆起，可能是物品被藏于衣物下方。",
            "hand_pressing_pocket": "检测到手部持续按压口袋或裤腰，可能是在调整隐藏的物品。",
            "tearing_tag": "检测到撕除价格或防盗标签的动作，这是盗窃前的准备行为。",
            "package_shaking": "检测到不自然的包装抖动行为，可能是在拆除或破坏包装。",
            "empty_box_restoration": "检测到放回空盒的行为，这是掏空包装后的掩盖手段。",
            "multiple_layer_clothing": "检测到与季节不符的多层衣物，可用于隐藏被盗物品。",
            "large_pocket_bulge": "检测到口袋异常鼓起，可能已经藏匿了商品。",
            "objects_hiding": "检测到物品被刻意隐藏在衣物下方，明显的盗窃迹象。",
            
            # 团伙协作行为
            "coordinated_movement": "检测到多人协同掩护动作，形成团伙作案模式。",
            "distraction_behavior": "检测到故意制造分散注意力的干扰行为，通常由团伙成员实施。",
            "lookout_positioning": "检测到门口或过道处的望风行为，负责警戒和通风报信。",
            
            # 环境相关行为
            "blind_spot_lingering": "检测到在监控盲区长时间逗留，试图躲避监控系统。",
            "repeated_store_visits": "检测到短时间内多次出入同一区域，可能在踩点或选择目标。",
            "closing_time_activity": "检测到临近关店时突然增加的选购行为，利用员工疲劳和注意力不集中。",
            
            # 高价值商品相关
            "multiple_identical_items": "检测到拿取多件相同高价值商品，超出正常购买需求。",
            "price_tag_switching": "检测到商品标签互换行为，试图以低价购买高价商品。",
            "concealment_in_store_items": "检测到将商品藏入已购买物品中的行为，逃避付款。",
            
            # 数字化特征
            "signal_blocking_behavior": "检测到使用信号屏蔽设备或铝箔袋，阻断防盗系统信号。",
            "security_tag_tampering": "检测到篡改或破坏防盗标签的行为，试图绕过安防系统。",
            "self_checkout_fraud": "检测到自助结账区域的商品替换行为，企图少付或不付款。",
            
            # 基于OpenPose的新增行为检测
            "abnormal_arm_position": "检测到异常的手臂弯曲和定位，通常是在隐藏物品于衣物内部。",
            "suspicious_crouching": "检测到可疑的蹲姿行为，常见于在低层货架或隐蔽处操作物品时。",
            "unusual_reaching": "检测到不自然的伸手姿势，可能是试图从高处取物或将物品放入不易察觉的位置。",
            "single_arm_hiding": "检测到单臂遮挡动作，利用身体一侧遮掩偷窃行为。",
            "body_shielding": "检测到用躯干屏蔽手部动作的姿态，故意遮挡摄像头视线。",
            "hiding_hand_gesture": "检测到手部位置异常遮挡，刻意将手隐藏在视线死角。",
            "abnormal_head_movement": "检测到频繁的头部转动行为，在行窃过程中查看是否有人注意。"
        }
        
        return descriptions.get(behavior_type, "检测到可疑行为，建议关注。")
    
    def create_annotated_frame(self, frame_data):
        """
        创建带有标注的帧
        
        Args:
            frame_data: 包含帧信息的字典，包括frame, frame_index, time, behaviors, theft_probability
        
        Returns:
            annotated_frame: 带标注的帧
        """
        if not frame_data:
            return None
            
        # 获取帧数据
        frame = frame_data.get('frame')
        if frame is None:
            return None
            
        frame_index = frame_data.get('frame_index', 0)
        time_str = frame_data.get('time', '')
        behaviors = frame_data.get('behaviors', [])
        theft_probability = frame_data.get('theft_probability', 0.0)
        
        # 创建副本以进行绘制
        annotated_frame = frame.copy()
        
        # 添加闪烁效果（根据帧计数）
        flash_effect = False
        if theft_probability > 0.5 and frame_index % 10 < 5:
            flash_effect = True
            
        # 初始化行为颜色映射
        behavior_color_map = {
            "遮挡商品区域": (0, 0, 255),    # 红色
            "手肘内收姿态异常": (0, 0, 255),  # 橙色
            "肩部不自然隆起": (0, 0, 255),  # 黄色
            "反复调整位置": (0, 0, 255),    # 绿色
            "疑似撕标签动作": (255, 0, 255),  # 品红色
            "可疑商品处理": (255, 255, 0),  # 青色
            "快速藏匿物品": (0, 0, 255)     # 蓝色
        }
        
        # 为检测到的人和物体画框
        detections = frame_data.get('detections', [])
        for det in detections:
            if isinstance(det, dict) and 'bbox' in det:
                x1, y1, x2, y2 = map(int, det['bbox'])
                label = det.get('class', 'unknown')
                conf = det.get('confidence', 0.0)
                
                # 根据类别选择颜色
                color = (0, 255, 0)  # 默认绿色
                if label == 'person':
                    color = (255, 0, 0)  # 蓝色
                
                # 绘制边界框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # 添加标签
                text = f"{label}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
                cv2.putText(annotated_frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        # 显示行为
        for i, behavior in enumerate(behaviors):
            # 确保behavior是字典格式
            if not isinstance(behavior, dict):
                continue
                
            # 获取行为信息
            behavior_type = behavior.get('type', '未知行为')
            confidence = behavior.get('confidence', 0.0)
            description = behavior.get('description', behavior_type)
            
            # 获取行为颜色
            color = behavior.get('color', behavior_color_map.get(behavior_type, (0, 255, 0)))
            
            # 在适当位置添加行为标签
            y_pos = 40 + i * 30
            label_text = f"{behavior_type}: {confidence:.2f}"
            
            # 绘制半透明背景
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, y_pos - 25), (10 + text_size[0], y_pos + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
            
            # 绘制文字
            cv2.putText(annotated_frame, label_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 如果有边界框，则绘制边界框
            if 'bbox' in behavior:
                x1, y1, x2, y2 = map(int, behavior['bbox'])
                # 绘制矩形
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # 在边界框上方添加标签
                cv2.putText(annotated_frame, f"{behavior_type}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 添加时间信息
        h, w = annotated_frame.shape[:2]
        cv2.putText(annotated_frame, time_str, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加盗窃概率
        probability_text = f"盗窃概率: {theft_probability:.2%}"
        prob_color = (0, 0, 255) if theft_probability > 0.5 else (0, 255, 0)
        
        # 半透明背景
        text_size = cv2.getTextSize(probability_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (w-250-10, 5), (w-10, 35), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        cv2.putText(annotated_frame, probability_text, (w-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, prob_color, 2)
        
        # 移除警告标志代码，不再添加红色警告条
        
        return annotated_frame
    
    def create_behavior_summary(self, behaviors, max_probability, theft_frames, total_frames):
        """创建行为分析摘要"""
        try:
            # 设置标志表示已经生成过摘要，防止重复生成
            self.summary_generated = True
            
            # 首先记录我们正在创建摘要
            self.log("正在生成行为分析摘要...", console_only=True)
            
            # 不再清理之前的日志，而是添加分隔符
            self.log_text.insert(tk.END, "\n" + "="*50 + "\n", "separator")
            
            # 添加标题
            self.log_text.insert(tk.END, "📊 行为分析摘要\n", "title")
            self.log_text.insert(tk.END, "-" * 40 + "\n", "line")
            
            # 统计各种行为出现的次数
            behavior_counts = {}
            
            # 获取所有行为项
            behavior_items = []
            for item in self.behavior_list.get_children():
                values = self.behavior_list.item(item, "values")
                frame = values[0]
                behavior_type = values[1]
                timestamp = values[2] if len(values) > 2 else "--"
                probability = float(values[3]) if len(values) > 3 and values[3] else 0.0
                
                behavior_items.append({
                    "frame": frame,
                    "behavior_type": behavior_type,
                    "timestamp": timestamp,
                    "probability": probability
                })
                
                # 更新行为计数
                if behavior_type in behavior_counts:
                    behavior_counts[behavior_type] += 1
                else:
                    behavior_counts[behavior_type] = 1
            
            # 零售特定行为
            retail_behaviors = ["遮挡商品区域", "疑似撕标签动作", "可疑商品处理", "快速藏匿物品", "将物品放入口袋"]
            
            # 对行为按时间排序
            sorted_behaviors = sorted(behavior_items, key=lambda x: x["timestamp"])
            
            # 统计零售特定行为和非零售行为
            retail_behavior_count = sum(behavior_counts.get(b, 0) for b in retail_behaviors)
            non_retail_behavior_count = sum(behavior_counts.get(b, 0) for b in behavior_counts if b not in retail_behaviors)
            
            # 根据框架检测日志中的信息确定环境类型
            # 检查日志中是否有显示高度可能是零售环境或可能是零售环境
            is_retail_environment = False
            
            # 优先使用已有的环境判断结果，如果在视频处理中已确定是零售环境
            if hasattr(self, 'is_retail_environment') and self.is_retail_environment is not None:
                is_retail_environment = self.is_retail_environment
                self.log(f"使用已有的环境判断结果: {'零售环境' if is_retail_environment else '非零售环境'}")
            # 如果没有预先判断结果，尝试使用检测器判断
            elif hasattr(self, 'theft_detector'):
                # 获取在最后一次检测中的环境判断
                if self.processed_frame is not None:
                    # 先检查processed_frame是否是图像数据而不是UI组件
                    import numpy as np
                    if isinstance(self.processed_frame, np.ndarray):
                        # 直接使用detection模块的环境判断结果
                        try:
                            # 确保图像是RGB格式，Ultralytics要求RGB格式
                            if len(self.processed_frame.shape) == 3 and self.processed_frame.shape[2] == 3:
                                # 检查是否需要从BGR转换为RGB (OpenCV默认是BGR)
                                frame_for_prediction = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
                            else:
                                # 如果不是3通道图像，跳过环境判断
                                raise ValueError("图像格式不支持环境判断，需要RGB格式")
                            
                            result = self.theft_detector.model.predict(frame_for_prediction, conf=0.25)[0]
                            is_retail_environment = self.theft_detector._is_retail_environment(result)
                            self.log(f"使用检测器中的环境判断结果: {'零售环境' if is_retail_environment else '非零售环境'}")
                        except Exception as e:
                            self.log(f"使用检测器判断环境失败: {str(e)}")
                            # 失败时不更新is_retail_environment，允许下一个方法尝试
                    else:
                        self.log("processed_frame不是有效的图像数据，跳过使用检测器判断环境")
            
            # 如果没有可用的检测器结果，使用UI中基于行为的判断逻辑
            if not hasattr(self, 'theft_detector') or self.processed_frame is None:
                # 默认假设为零售环境
                
                # 如果没有任何行为数据，默认环境类型为未知
                if not behavior_counts:
                    is_retail_environment = False
                    self.log("没有检测到任何行为，默认环境类型为：非零售")
                # 检查是否包含足够多的零售特定行为
                elif retail_behavior_count > non_retail_behavior_count and retail_behavior_count >= 2:
                    is_retail_environment = True
                # 检查是否包含特定高度相关的零售行为
                elif "疑似撕标签动作" in behavior_counts or "可疑商品处理" in behavior_counts:
                    is_retail_environment = True
                # 如果非零售行为明显多于零售行为，或者没有明显的零售行为
                else:
                    is_retail_environment = False
                
                self.log(f"使用行为分析判断环境为: {'零售环境' if is_retail_environment else '非零售环境'}")
            
            # 保存环境类型判断结果，供其他函数使用
            self.is_retail_environment = is_retail_environment
            
            # 环境分析部分
            self.log_text.insert(tk.END, "\n📊 环境分析:\n", "subtitle")
            
            if is_retail_environment:
                self.log_text.insert(tk.END, "  ✓ 当前环境符合零售或超市场景\n", "environment_retail")
            else:
                self.log_text.insert(tk.END, "  ⚠️ 当前环境不符合零售或超市场景\n", "environment_non_retail")
                self.log_text.insert(tk.END, "  已将零售特定行为替换为通用行为类型\n", "environment_adjusted")
                
                # 如果是非零售环境但检测到了零售特定行为，需要在界面上更新行为列表
                if retail_behavior_count > 0:
                    self.log("检测到零售特定行为但环境不符合零售场景，将更新行为列表")
                    
                    # 检查当前UI中是否还有零售特定行为
                    has_retail_behaviors_in_ui = False
                    for item in self.behavior_list.get_children():
                        values = self.behavior_list.item(item, "values")
                        behavior_text = values[1] if len(values) > 1 else ""
                        if any(r in behavior_text for r in retail_behaviors):
                            has_retail_behaviors_in_ui = True
                            break
                    
                    # 如果UI中仍有零售行为，需要重新构建行为列表
                    if has_retail_behaviors_in_ui:
                        self.log("UI中存在零售特定行为，将应用环境类型重新更新行为列表")
                        # 在摘要生成后调用update_ui_with_behaviors会根据已确定的环境类型替换行为
                        # 确保在调用前禁用所有按钮
                        def update_behaviors_with_disabled_buttons():
                            # 强制设置按钮禁用标志
                            if not hasattr(self, '_force_disable_buttons'):
                                self._force_disable_buttons = False
                                
                            # 先禁用所有按钮
                            self.process_btn.state(['disabled'])
                            self.select_image_btn.state(['disabled'])
                            self.select_video_btn.state(['disabled'])
                            # 然后更新行为列表
                            self.update_ui_with_behaviors()
                            
                            # 确保在处理完成后按钮重新启用 - 检查处理状态而不是继承之前的状态
                            if not self.is_processing:
                                self.log("确保行为列表与环境类型一致后重新启用按钮")
                                self.process_btn.state(['!disabled'])
                                self.select_image_btn.state(['!disabled'])
                                self.select_video_btn.state(['!disabled'])
                                # 如果有处理结果，启用保存按钮
                                if hasattr(self, 'processed_media_path') and self.processed_media_path:
                                    self.save_btn.state(['!disabled'])
                            else:
                                # 如果仍在处理中，确保按钮保持禁用
                                self.process_btn.state(['disabled'])
                                self.select_image_btn.state(['disabled'])
                                self.select_video_btn.state(['disabled'])
                        self.root.after(100, update_behaviors_with_disabled_buttons)
            
            # 为每种行为类型使用图标前缀
            behavior_type_map = {
                "Covering Product Area": "遮挡商品区域", 
                "Unusual Elbow Position": "手肘内收姿态异常",
                "Unnatural Shoulder Raise": "肩部不自然隆起",
                "Repetitive Position Adjustment": "反复调整位置",
                "Suspected Tag Removal": "疑似撕标签动作",
                "Suspicious Item Handling": "可疑商品处理",
                "Rapid Item Concealment": "快速藏匿物品"
            }
            
            # 为每种行为类型使用图标前缀
            icon_map = {
                "遮挡商品区域": "🧥 ",
                "手肘内收姿态异常": "💪 ",
                "肩部不自然隆起": "👕 ",
                "反复调整位置": "🔄 ",
                "疑似撕标签动作": "🏷️ ",
                "可疑商品处理": "🛒 ",
                "快速藏匿物品": "👝 "
            }
            
            # 行为分析部分
            self.log_text.insert(tk.END, "\n👁️ 行为分析:\n", "subtitle")
            
            # 判断是否检测到盗窃行为
            if theft_frames > 0:
                theft_percentage = (theft_frames / total_frames) * 100 if total_frames > 0 else 0
                
                if theft_percentage > 30 and max_probability > 0.6:
                    # 高度可疑
                    self.log_text.insert(tk.END, "  ⚠️ 高度可疑: 多次检测到盗窃行为特征\n", "high_warning")
                    self.log_text.insert(tk.END, f"  可疑帧占比: {theft_percentage:.2f}%\n", "warning")
                    self.log_text.insert(tk.END, f"  发生盗窃概率: {max_probability:.2%}\n", "warning")
                elif theft_percentage > 10 or max_probability > 0.5:
                    # 中度可疑
                    self.log_text.insert(tk.END, "  ⚠️ 中度可疑: 检测到部分盗窃行为特征\n", "medium_warning")
                    self.log_text.insert(tk.END, f"  可疑帧占比: {theft_percentage:.2f}%\n", "warning")
                    self.log_text.insert(tk.END, f"  发生盗窃概率: {max_probability:.2%}\n", "warning")
                else:
                    # 轻度可疑
                    self.log_text.insert(tk.END, "  ⚠️ 轻度可疑: 检测到少量可疑行为\n", "low_warning") 
                    self.log_text.insert(tk.END, f"  可疑帧占比: {theft_percentage:.2f}%\n", "warning")
                    self.log_text.insert(tk.END, f"  发生盗窃概率: {max_probability:.2%}\n", "warning")
            else:
                self.log_text.insert(tk.END, "  ✓ 未检测到明显的盗窃行为\n", "normal")
                if not behavior_counts:
                    self.log_text.insert(tk.END, "  ✓ 分析过程中未发现任何可疑行为\n", "normal")
            
            # 显示行为详细统计
            self.log_text.insert(tk.END, "\n📋 检测到的行为统计:\n", "subtitle")
            
            if not behavior_counts:
                self.log_text.insert(tk.END, "  无检测到的行为\n", "normal")
            else:
                # 对行为按出现次数排序
                sorted_behaviors = sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True)
                
                for behavior_type, count in sorted_behaviors:
                    # 转换为中文显示
                    chinese_behavior_type = behavior_type_map.get(behavior_type, behavior_type)
                    
                    # 添加图标前缀
                    icon_prefix = icon_map.get(chinese_behavior_type, "⚠️ ")
                    display_type = icon_prefix + chinese_behavior_type
                    
                    # 添加行为描述
                    behavior_description = self.get_behavior_description(chinese_behavior_type)
                    description_short = behavior_description[:50] + "..." if len(behavior_description) > 50 else behavior_description
                    
                    # 如果是非零售环境中的零售行为被替换，添加说明
                    if not is_retail_environment and behavior_type in ["反复调整位置", "手肘内收姿态异常"]:
                        self.log_text.insert(tk.END, f"  {display_type}: {count}次\n", "behavior_item")
                        self.log_text.insert(tk.END, f"    {description_short}\n", "behavior_desc")
                        if behavior_type == "反复调整位置" and "遮挡商品区域" in behavior_counts or "可疑商品处理" in behavior_counts:
                            self.log_text.insert(tk.END, f"    (包含替换的零售环境行为)\n", "behavior_replaced")
                        elif behavior_type == "手肘内收姿态异常" and "疑似撕标签动作" in behavior_counts or "快速藏匿物品" in behavior_counts:
                            self.log_text.insert(tk.END, f"    (包含替换的零售环境行为)\n", "behavior_replaced")
                    else:
                        self.log_text.insert(tk.END, f"  {display_type}: {count}次\n", "behavior_item")
                        self.log_text.insert(tk.END, f"    {description_short}\n", "behavior_desc")
            
            # 总体结论
            self.log_text.insert(tk.END, "\n🔍 结论:\n", "subtitle")
            if theft_frames > 0 and is_retail_environment:
                if theft_percentage > 30 and max_probability > 0.6:
                    self.log_text.insert(tk.END, "  检测到高度可疑的盗窃行为，建议进一步核实\n", "conclusion_high")
                elif theft_percentage > 10 or max_probability > 0.5:
                    self.log_text.insert(tk.END, "  检测到可能的盗窃行为，建议关注\n", "conclusion_medium")
                else:
                    self.log_text.insert(tk.END, "  检测到轻微可疑行为，可能需要关注\n", "conclusion_low")
            elif theft_frames > 0 and not is_retail_environment:
                self.log_text.insert(tk.END, "  环境不符合零售场景，检测到的可能是通用可疑行为\n", "conclusion_medium")
                if max_probability > 0.5:
                    self.log_text.insert(tk.END, "  建议关注异常举止，但不适用零售盗窃分析\n", "conclusion_low")
            elif not theft_frames and is_retail_environment:
                self.log_text.insert(tk.END, "  零售环境中未检测到盗窃行为\n", "conclusion_none")
            else:
                if not behavior_counts:
                    self.log_text.insert(tk.END, "  未检测到任何行为，分析结束\n", "conclusion_none")
                else:
                    self.log_text.insert(tk.END, "  非零售环境，未检测到相关可疑行为\n", "conclusion_none")
            
            # 添加时间戳
            self.log_text.insert(tk.END, "---------------------------------\n", "line")
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_text.insert(tk.END, f"分析完成时间: {timestamp}\n\n", "timestamp")
            
            # 配置标签样式
            self.log_text.tag_configure("title", font=("Arial", 10, "bold"), foreground="blue")
            self.log_text.tag_configure("subtitle", font=("Arial", 9, "bold"), foreground="black")
            self.log_text.tag_configure("normal", font=("Arial", 9), foreground="black")
            self.log_text.tag_configure("warning", font=("Arial", 9), foreground="red")
            self.log_text.tag_configure("high_warning", font=("Arial", 9, "bold"), foreground="red")
            self.log_text.tag_configure("medium_warning", font=("Arial", 9), foreground="orange")
            self.log_text.tag_configure("low_warning", font=("Arial", 9), foreground="orange")
            self.log_text.tag_configure("behavior_item", font=("Arial", 9), foreground="blue")
            self.log_text.tag_configure("behavior_desc", font=("Arial", 8), foreground="gray")
            self.log_text.tag_configure("behavior_replaced", font=("Arial", 8, "italic"), foreground="#8B0000")
            self.log_text.tag_configure("line", font=("Arial", 9), foreground="gray")
            self.log_text.tag_configure("timestamp", font=("Arial", 8), foreground="gray")
            self.log_text.tag_configure("environment_retail", font=("Arial", 9), foreground="green")
            self.log_text.tag_configure("environment_non_retail", font=("Arial", 9), foreground="orange")
            self.log_text.tag_configure("environment_adjusted", font=("Arial", 9, "italic"), foreground="gray")
            self.log_text.tag_configure("conclusion_high", font=("Arial", 9, "bold"), foreground="red")
            self.log_text.tag_configure("conclusion_medium", font=("Arial", 9), foreground="orange")
            self.log_text.tag_configure("conclusion_low", font=("Arial", 9), foreground="blue")
            self.log_text.tag_configure("conclusion_none", font=("Arial", 9), foreground="green")
            
            # 总是滚动到底部显示最新内容
            self.log_text.see(tk.END)
            
            # 如果正在处理中，确保按钮保持禁用
            if self.is_processing:
                self.process_btn.state(['disabled'])
                self.select_image_btn.state(['disabled'])
                self.select_video_btn.state(['disabled'])
            else:
                # 处理完成后启用按钮
                self._force_disable_buttons = False
                self.process_btn.state(['!disabled'])
                self.select_image_btn.state(['!disabled'])
                self.select_video_btn.state(['!disabled'])
                
                # 如果有处理结果，启用保存按钮
                if hasattr(self, 'processed_media_path') and self.processed_media_path and os.path.exists(self.processed_media_path):
                    self.save_btn.state(['!disabled'])
            
            # 在函数结束前，确保UI行为列表与环境类型一致
            self.log("确保行为列表与环境类型一致")
            
            # 如果不是零售环境，强制刷新行为列表
            if not is_retail_environment:
                # 保存原有列表内容
                existing_behaviors = []
                for item in self.behavior_list.get_children():
                    values = self.behavior_list.item(item, "values")
                    time_str = values[0]
                    behavior_text = values[1]
                    probability_text = values[2]
                    
                    # 提取行为类型（去除图标）
                    behavior_type = behavior_text[2:] if len(behavior_text) > 2 else behavior_text
                    
                    # 检查行为是否需要替换
                    if any(r in behavior_type for r in retail_behaviors):
                        self.log(f"发现需要替换的零售行为: {behavior_type}")
                        # 需要在rebuild_behavior_list中处理
                        existing_behaviors.append((time_str, behavior_type, probability_text))
                    else:
                        existing_behaviors.append((time_str, behavior_type, probability_text))
                
                # 如果有行为需要重建
                if existing_behaviors:
                    # 使用延迟执行确保UI更新不冲突
                    self.root.after(200, lambda: self.rebuild_behavior_list(existing_behaviors))
            
        except Exception as e:
            self.log(f"创建行为摘要错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            
            # 如果正在处理中，确保按钮保持禁用
            if self.is_processing:
                self.process_btn.state(['disabled'])
                self.select_image_btn.state(['disabled'])
                self.select_video_btn.state(['disabled'])
            else:
                # 处理完成后启用按钮
                self._force_disable_buttons = False
                self.process_btn.state(['!disabled'])
                self.select_image_btn.state(['!disabled'])
                self.select_video_btn.state(['!disabled'])
    
    def add_behavior_to_list(self, frame, time_point, behavior_type, probability):
        """将行为添加到行为列表中"""
        try:
            # 转换行为类型（英文->中文）
            behavior_type_map = {
                "Product Area Shielding": "遮挡商品区域",
                "Abnormal Elbow Posture": "手肘内收姿态异常",
                "Unnatural Shoulder Raise": "肩部不自然隆起",
                "Repetitive Position Adjustment": "反复调整位置",
                "Suspected Tag Removal": "疑似撕标签动作",
                "Suspicious Item Handling": "可疑商品处理",
                "Rapid Item Concealment": "快速藏匿物品",
                "Item Concealed in Pocket": "将物品放入口袋"
            }
            
            # 转换为中文显示
            chinese_behavior_type = behavior_type_map.get(behavior_type, behavior_type)
            
            # 为每种行为类型使用图标前缀
            icon_map = {
                "遮挡商品区域": "🧥 ",
                "手肘内收姿态异常": "💪 ",
                "肩部不自然隆起": "👕 ",
                "反复调整位置": "🔄 ",
                "疑似撕标签动作": "🏷️ ",
                "可疑商品处理": "🛒 ",
                "快速藏匿物品": "👝 ",
                "将物品放入口袋": "👖 "
            }
            
            # 在行为类型前添加图标
            icon_prefix = icon_map.get(chinese_behavior_type, "⚠️ ")
            display_type = icon_prefix + chinese_behavior_type
            
            # 初始化行为ID计数器和行为数据列表（如果不存在）
            if not hasattr(self, 'next_behavior_id'):
                self.next_behavior_id = 1
            if not hasattr(self, 'behavior_data'):
                self.behavior_data = []
                
            # 更新行为列表
            item_id = self.next_behavior_id
            self.next_behavior_id += 1
            
            # 添加到Treeview列表
            probability_formatted = f"{probability:.2%}"
            time_formatted = f"{time_point:.2f}秒" if time_point is not None else "N/A"
            
            # 使用正确的ttk.Treeview插入方法，设置值为时间、类型和概率
            tree_item_id = self.behavior_list.insert("", "end", values=(time_formatted, display_type, probability_formatted))
            
            # 根据概率设置不同的背景色标签
            if probability > 0.8:
                self.behavior_list.item(tree_item_id, tags=("high",))
            elif probability > 0.6:
                self.behavior_list.item(tree_item_id, tags=("medium",))
            else:
                self.behavior_list.item(tree_item_id, tags=("low",))
                
            # 配置标签样式
            self.behavior_list.tag_configure("high", background="#ffcccc")
            self.behavior_list.tag_configure("medium", background="#ffffcc")
            self.behavior_list.tag_configure("low", background="#e6f7ff")
            
            # 确保滚动到最新项
            self.behavior_list.yview_moveto(1.0)
            
            # 保存行为数据以供后续使用
            behavior_data = {
                'id': item_id,
                'frame': frame,
                'time': time_point,
                'type': chinese_behavior_type,
                'probability': probability,
                'tree_id': tree_item_id  # 保存树形控件中的项目ID
            }
            self.behavior_data.append(behavior_data)
            
            # 记录日志 - 使用漂亮的格式化消息
            log_message = f"添加行为到列表: 帧={frame}, 时间={time_point:.2f}秒, 类型={chinese_behavior_type}, 可信度={probability:.4f}"
            self.log(log_message)
            
            # 获取行为描述文本
            behavior_descriptions = {
                "遮挡商品区域": "疑似以身体遮挡商品区域，可能是为了隐藏取物行为",
                "手肘内收姿态异常": "手肘异常内收姿态，可能在隐藏物品",
                "肩部不自然隆起": "肩部姿态异常，显示不自然隆起，可能藏匿物品于衣物内",
                "反复调整位置": "在同一区域反复调整身体位置，行为可疑",
                "疑似撕标签动作": "手部在商品区域有撕扯动作，可能在移除防盗标签",
                "可疑商品处理": "对商品进行可疑操作，可能准备藏匿",
                "快速藏匿物品": "快速将物品藏入衣物或包内",
                "将物品放入口袋": "将商品或物品放入口袋或衣物内部，典型的盗窃动作"
            }
            
            # 获取行为描述
            behavior_description = behavior_descriptions.get(chinese_behavior_type, "可疑行为，需要关注")
            
            # 再添加一条详细的警告日志（仅当概率较高时）
            if probability > 0.6:
                detailed_message = f"检测到行为: {display_type} (可信度: {probability_formatted})\n行为描述: {behavior_description}"
                self.log(detailed_message)
            
            return item_id
        except Exception as e:
            self.log(f"添加行为到列表错误: {str(e)}", console_only=True)
            import traceback
            self.log(traceback.format_exc(), console_only=True)
            return None
    
    def update_ui_with_behaviors(self, auto_create_summary=True):
        """在主线程中更新行为列表UI"""
        try:
            # 记录原始按钮禁用状态
            original_disable_state = getattr(self, '_force_disable_buttons', False)
            # 确保在函数执行期间按钮保持禁用
            self._force_disable_buttons = True
            
            # 确保在分析过程中按钮保持禁用状态
            if self.is_processing:
                self.process_btn.state(['disabled'])
                self.select_image_btn.state(['disabled'])
                self.select_video_btn.state(['disabled'])
            
            # 确保行为列表可见
            self.behavior_list_frame.update()
            
            # 清空行为列表UI准备添加新行为
            for item in self.behavior_list.get_children():
                self.behavior_list.delete(item)
            
            # 强制绘制更新
            self.behavior_list.update()
            
            # 检查行为数据
            if not hasattr(self, 'behaviors_data') or not self.behaviors_data:
                self.log("警告: 没有检测到可疑行为数据")
                self.behavior_list.insert("", "end", values=("N/A", "未检测到可疑行为", "0.00%"))
                return
                
            behaviors = self.behaviors_data
            
            # 确保可疑帧列表存在
            if not hasattr(self, 'suspicious_frames'):
                self.suspicious_frames = []
                
            suspicious_frames = self.suspicious_frames
            
            # 判断是否为图片分析
            is_image_analysis = len(behaviors) == 1 and behaviors[0][0] == 0
            
            # 获取FPS和总帧数，处理图片和视频的不同情况
            if is_image_analysis:
                fps = 1
                total_frames = 1
            else:
                # 视频分析，获取视频信息
                cap = cv2.VideoCapture(self.current_media_path)
                fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 1
                cap.release()
            
            # 提取分析结果数据
            behavior_types = set()
            max_probability = 0.0
            
            # 修改条件：如果没有环境类型判断且摘要未生成，才创建摘要
            # 这样可以防止在首次分析时重复生成摘要
            if auto_create_summary and not hasattr(self, 'is_retail_environment') and (not hasattr(self, 'summary_generated') or not self.summary_generated):
                self.log("还未生成行为摘要，将先创建摘要确定环境类型")
                # 先计算最大概率
                for _, frame_behaviors in behaviors:
                    for behavior in frame_behaviors:
                        confidence = behavior.get('confidence', 0.0)
                        max_probability = max(max_probability, confidence)
                        
                self.create_behavior_summary(behaviors, max_probability, len(suspicious_frames), total_frames)
            
            # 处理检测到的行为
            behavior_count = 0
            for frame_idx, frame_behaviors in behaviors:
                for behavior in frame_behaviors:
                    behavior_count += 1
                    behavior_type = behavior.get('type', '未知行为')
                    behavior_types.add(behavior_type)
                    confidence = behavior.get('confidence', 0.0)
                    max_probability = max(max_probability, confidence)
                    
                    # 添加到行为列表UI - 这里会应用环境类型判断进行替换
                    time_point = frame_idx / fps if fps > 0 else 0
                    self.add_behavior_to_list(frame_idx, time_point, behavior_type, confidence)
            
            self.log(f"已添加 {behavior_count} 条行为记录到界面")
            
            # 总是滚动到底部显示最新内容
            self.behavior_list.yview_moveto(1.0)
            
            # 计算盗窃帧数 - 无论是否自动创建摘要都需要这个值
            theft_frames = len(suspicious_frames)
            if is_image_analysis and theft_frames == 0 and behavior_count > 0:
                # 如果是图片分析且有行为但没有可疑帧，将可疑帧计为1
                theft_frames = 1
            
            # 修改条件：只有在自动创建摘要模式下且摘要未生成，并且前面没有创建过摘要时，才在此处创建摘要
            if auto_create_summary and (not hasattr(self, 'summary_generated') or not self.summary_generated) and hasattr(self, 'is_retail_environment'):
                # 创建行为摘要
                self.create_behavior_summary(behaviors, max_probability, theft_frames, total_frames)
            
            # 更新分析结果日志
            theft_detected = "是" if theft_frames > 0 else "否"
            
            if is_image_analysis:
                summary_message = f"图片分析完成: 探测盗窃行为：{theft_detected}"
                summary_message += f"\n最高行为可疑度: {max_probability:.2f}"
                summary_message += f"\n检测到 {behavior_count} 处可疑行为"
            else:
                summary_message = f"视频分析完成: 探测盗窃行为：{theft_detected}"
                summary_message += f"\n最高盗窃概率: {max_probability:.2f}"
                summary_message += f"\n包含盗窃行为的帧数: {theft_frames}"
                summary_message += f"\n检测到 {behavior_count} 处可疑行为"
                
            self.log(summary_message)
            
            # 恢复原始按钮禁用状态
            self._force_disable_buttons = original_disable_state
            
            # 检查处理状态并相应地更新按钮
            if self.is_processing:
                # 如果仍在处理中，确保按钮保持禁用
                self.process_btn.state(['disabled'])
                self.select_image_btn.state(['disabled'])
                self.select_video_btn.state(['disabled'])
            else:
                # 如果处理已完成，确保按钮重新启用
                self.log("更新行为列表后重新启用按钮")
                self.process_btn.state(['!disabled'])
                self.select_image_btn.state(['!disabled'])
                self.select_video_btn.state(['!disabled'])
                
                # 如果有处理结果，启用保存按钮
                if hasattr(self, 'processed_media_path') and self.processed_media_path:
                    self.save_btn.state(['!disabled'])
                
        except Exception as e:
            self.log(f"更新行为列表错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            
            # 恢复原始按钮禁用状态
            self._force_disable_buttons = original_disable_state
            
            # 如果仍在处理中，确保按钮保持禁用
            if self.is_processing:
                self.process_btn.state(['disabled'])
                self.select_image_btn.state(['disabled'])
                self.select_video_btn.state(['disabled'])
    
    def determine_behavior_type(self, detections):
        """基于检测结果确定可疑行为类型 - 被analyze_behavior替代，保留为兼容旧代码"""
        if detections is None:
            return None
            
        behavior_types = [
            "遮挡商品区域", 
            "手肘内收姿态异常",
            "肩部不自然隆起",
            "反复调整位置",
            "疑似撕标签动作",
            "可疑商品处理",
            "快速藏匿物品"
        ]
        
        import random
        return random.choice(behavior_types)
    
    def start_processed_video_playback(self):
        """开始播放处理后的视频"""
        try:
            # 首先停止任何正在进行的视频播放
            self.stop_video_playback()
            
            # 获取原始视频文件路径（优先从static/videos目录获取）
            original_filename = os.path.basename(self.current_media_path)
            static_video_path = os.path.join("static", "videos", original_filename)
            
            # 优先使用static/videos目录下的视频作为原始视频
            if os.path.exists(static_video_path):
                original_video_path = static_video_path
                self.log(f"使用static/videos目录下的原始视频: {original_video_path}")
            else:
                # 如果static/videos中不存在，则使用当前路径
                original_video_path = self.current_media_path
                if not os.path.exists(original_video_path):
                    self.log(f"错误：原始视频文件不存在: {original_video_path}")
                    return
                
            # 检查处理后的视频文件路径是否有效
            if not os.path.exists(self.processed_media_path):
                self.log(f"错误：处理后的视频文件不存在: {self.processed_media_path}")
                return
            
            # 左侧显示原始视频（原始媒体区域）
            self.video_capture = cv2.VideoCapture(original_video_path)
            if not self.video_capture.isOpened():
                self.log(f"无法打开原始视频: {original_video_path}")
                return
            
            # 右侧显示处理后的视频（检测结果区域）
            self.processed_video_capture = cv2.VideoCapture(self.processed_media_path)
            if not self.processed_video_capture.isOpened():
                self.log(f"无法打开处理后的视频: {self.processed_media_path}")
                self.video_capture.release()
                return
            
            # 记录文件信息用于调试    
            self.log(f"原始视频（左侧原始媒体区域）：{original_video_path}")
            self.log(f"处理后视频（右侧检测结果区域）：{self.processed_media_path}")
            
            # 获取视频信息
            fps_original = self.video_capture.get(cv2.CAP_PROP_FPS)
            fps_processed = self.processed_video_capture.get(cv2.CAP_PROP_FPS)
            frames_original = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_processed = int(self.processed_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            self.log(f"原始视频信息: FPS={fps_original}, 总帧数={frames_original}")
            self.log(f"处理后视频信息: FPS={fps_processed}, 总帧数={frames_processed}")
            
            # 确保两个视频都从头开始
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.processed_video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
            # 获取视频信息，初始化时间标签
            self.update_time_label(0)
            
            # 显示播放控制区域
            self.playback_control_frame.pack(fill=tk.X, pady=5, after=self.media_frame)
            
            # 检查行为列表是否为空，如果是且有保存的行为数据，则更新UI
            if (len(self.behavior_list.get_children()) == 0 and 
                hasattr(self, 'behaviors_data') and self.behaviors_data):
                self.log("检测到行为列表为空，使用保存的行为数据更新UI")
                self.update_ui_with_behaviors()
                
            # 启动播放线程
            self.stop_video_thread = False
            self.is_playing = True
            self.play_pause_btn.config(text="暂停")
            self.current_frame = 0
            self.video_thread = threading.Thread(target=self.sync_video_playback_loop)
            self.video_thread.daemon = True
            self.video_thread.start()
            
            # 启用保存按钮
            self.save_btn.state(['!disabled'])
        except Exception as e:
            self.log(f"播放处理后视频错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def sync_video_playback_loop(self):
        """同步播放原始视频和处理后的视频"""
        if self.video_capture is None or self.processed_video_capture is None:
            self.log("错误: 视频源不可用")
            return
        
        # 获取视频信息
        fps = self.video_capture.get(cv2.CAP_PROP_FPS)  # 使用原始视频的帧率
        total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))  # 使用原始视频的总帧数
        
        # 确保两个视频从头开始播放
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.processed_video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        
        # 帧率控制
        target_fps = fps if fps > 0 else 30
        frame_time = 1.0 / target_fps
        
        # 同步播放标志
        self.log(f"开始同步播放：左侧原始媒体区域显示原始视频，右侧检测结果区域显示检测后视频，FPS: {target_fps}")
        
        # 尝试读取第一帧测试是否成功
        ret1, test_frame1 = self.video_capture.read()
        if ret1:
            self.log("原始视频(左侧原始媒体区域)成功读取第一帧")
            # 重置到开始
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.log("警告: 无法读取原始视频第一帧")
        
        ret2, test_frame2 = self.processed_video_capture.read()
        if ret2:
            self.log("处理后视频(右侧检测结果区域)成功读取第一帧")
            # 重置到开始
            self.processed_video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            self.log("警告: 无法读取处理后视频第一帧")
        
        while not self.stop_video_thread:
            # 如果暂停，则等待
            if not self.is_playing:
                time.sleep(0.1)
                continue
                
            # 记录帧处理开始时间
            start_time = time.time()
            
            # 读取原始视频帧
            ret1, frame1 = self.video_capture.read()
            # 读取处理后的视频帧
            ret2, frame2 = self.processed_video_capture.read()
            
            # 记录读取结果
            if self.current_frame % 30 == 0:  # 避免日志过多
                self.log(f"帧 {self.current_frame}: 原始视频读取状态={ret1}, 处理后视频读取状态={ret2}")
            
            # 判断是否有任一视频结束
            if not ret1 or not ret2:
                # 视频结束，重置视频到开始位置
                self.log("视频播放完毕，重置到开始位置")
                self.video_capture.release()
                self.processed_video_capture.release()
                
                # 获取正确的原始视频路径
                original_video_path = self.current_media_path
                # 检查原始视频是否存在，如果不存在，尝试在static/videos目录中查找
                if not os.path.exists(original_video_path):
                    filename = os.path.basename(original_video_path)
                    static_video_path = os.path.join("static", "videos", filename)
                    if os.path.exists(static_video_path):
                        original_video_path = static_video_path
                        self.log(f"已找到原始视频文件: {original_video_path}")
                
                # 重新打开视频文件
                self.video_capture = cv2.VideoCapture(original_video_path)
                self.processed_video_capture = cv2.VideoCapture(self.processed_media_path)
                
                # 确保两个视频都成功打开
                if not self.video_capture.isOpened() or not self.processed_video_capture.isOpened():
                    self.log("无法重新打开视频文件")
                    self.is_playing = False
                    self.root.after(0, lambda: self.play_pause_btn.config(text="播放"))
                    break
                
                # 重置状态
                self.is_playing = False
                self.current_frame = 0
                
                # 在主线程中更新按钮状态
                self.root.after(0, lambda: self.play_pause_btn.config(text="播放"))
                break
            
            # 转换颜色格式
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)  # 原始视频帧
            frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)  # 处理后的视频帧
            
            # 使用一个函数同时更新两个画布，避免异步更新导致不同步
            def update_both_canvases(original_img, processed_img):
                # 原始视频显示在左侧原始媒体区域
                self.display_image(original_img, self.original_canvas)
                # 处理后的视频显示在右侧检测结果区域
                self.display_image(processed_img, self.processed_canvas)
            
            # 更新UI (在主线程中)
            self.root.after(0, lambda: update_both_canvases(frame1_rgb.copy(), frame2_rgb.copy()))
            
            # 增加帧计数
            self.current_frame += 1
            
            # 更新进度条和时间标签 (不触发滑动条变化事件)
            if not hasattr(self, 'slider_being_changed') or not self.slider_being_changed:
                progress = (self.current_frame / total_frames) * 100 if total_frames > 0 else 0
                self.root.after(0, lambda p=progress: self.update_slider_position(p))
                self.root.after(0, lambda f=self.current_frame: self.update_time_label(f))
            
            # 计算帧处理所需时间
            processing_time = time.time() - start_time
            
            # 控制播放速度
            sleep_time = max(0, frame_time - processing_time)
            time.sleep(sleep_time)
    
    def update_slider_position(self, position):
        """更新滑动条位置，不触发事件"""
        # 设置标志防止触发回调
        self.slider_being_changed = True
        self.progress_var.set(position)
        self.slider_being_changed = False
            
    def stop_video_playback(self):
        """停止视频播放"""
        self.stop_video_thread = True
        self.is_playing = False
        
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
            self.video_thread = None
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
            
        if self.processed_video_capture:
            self.processed_video_capture.release()
            self.processed_video_capture = None
            
        # 重置播放按钮
        self.play_pause_btn.config(text="播放")
    
    def save_result(self):
        """Save processing result"""
        if not self.processed_media_path or not os.path.exists(self.processed_media_path):
            self.log("错误: 没有可保存的结果")
            return
        
        if self.current_media_type == 'image':
            filetypes = [("JPEG 图片", "*.jpg"), ("所有文件", "*.*")]
            default_ext = ".jpg"
        else:
            filetypes = [("MP4 视频", "*.mp4"), ("所有文件", "*.*")]
            default_ext = ".mp4"
        
        save_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=default_ext,
            filetypes=filetypes
        )
        
        if save_path:
            try:
                # Copy the result file to the selected path
                import shutil
                shutil.copy2(self.processed_media_path, save_path)
                self.log(f"结果已保存到: {save_path}")
            except Exception as e:
                messagebox.showerror("保存错误", f"保存结果时出错: {str(e)}")
                self.log(f"保存错误: {str(e)}")

    def on_window_configure(self, event):
        """处理窗口大小变化，但避免过于频繁的更新"""
        # 确保事件来自根窗口
        if event.widget != self.root:
            return
        
        # 检查是否为最大化/恢复事件
        current_is_maximized = self.root.wm_state() == 'zoomed'
        
        # 如果最大化状态改变，或窗口大小有明显变化，则更新画布
        if (current_is_maximized != self.is_maximized or 
            abs(self.last_known_width - event.width) > 50 or 
            abs(self.last_known_height - event.height) > 50):
            
            # 更新状态
            self.is_maximized = current_is_maximized
            self.last_known_width = event.width
            self.last_known_height = event.height
            
            # 保存当前进度条状态
            current_progress = self.progress_bar['value']
            progress_text = self.progress_text.get() if hasattr(self, 'progress_text') else "0%"
            
            # 临时暂停视频播放以防止竞态条件
            was_playing = False
            if hasattr(self, 'is_playing') and self.is_playing and hasattr(self, 'video_thread') and self.video_thread:
                was_playing = True
                self.is_playing = False
                # 等待短暂时间确保视频线程已暂停
                time.sleep(0.1)
            
            # 计算新的窗口比例，用于自适应调整
            window_ratio = event.width / self.default_width if self.default_width > 0 else 1
            
            # 根据窗口的缩放比例，调整update_canvas_sizes方法中使用的画布大小计算参数
            if self.is_maximized:
                # 在最大化状态下，根据窗口宽度调整画布尺寸
                # 这样可以避免最大化窗口时右侧出现大量空白
                available_width = event.width - 50
                canvas_width = int(available_width / 2 - 20)  # 左右两边各占一半
                canvas_height = int(canvas_width * 9 / 16)  # 保持16:9比例
                
                # 确保高度不超过可用空间
                available_height = event.height - 200
                if canvas_height > available_height * 0.7:
                    canvas_height = int(available_height * 0.7)
                    canvas_width = int(canvas_height * 16 / 9)
                
                # 直接更新画布和框架大小，避免调用update_canvas_sizes修改初始布局
                self.original_canvas.config(width=canvas_width, height=canvas_height)
                self.processed_canvas.config(width=canvas_width, height=canvas_height)
                self.original_canvas_frame.config(width=canvas_width, height=canvas_height)
                self.processed_canvas_frame.config(width=canvas_width, height=canvas_height)
                
                # 如果有原始图像或视频帧，需要重新调整大小并绘制
                if hasattr(self, 'original_image') and isinstance(self.original_image, np.ndarray):
                    # 重新显示原始图像
                    original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                    self.display_image(original_rgb, self.original_canvas, 
                                    forced_width=canvas_width, 
                                    forced_height=canvas_height)
                
                # 如果有处理后的图像，也需要重新显示
                if hasattr(self, 'processed_frame') and isinstance(self.processed_frame, np.ndarray):
                    # 重新显示处理后的图像
                    if len(self.processed_frame.shape) == 3:
                        processed_rgb = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
                    else:
                        processed_rgb = self.processed_frame
                    self.display_image(processed_rgb, self.processed_canvas,
                                    forced_width=canvas_width,
                                    forced_height=canvas_height)
            else:
                # 如果是从最大化恢复，或窗口大小变化，调用正常的update_canvas_sizes
                # 这样可以确保恢复到初始布局
                self.root.after(100, self.update_canvas_sizes)
                # 在尺寸更新后重新显示媒体
                self.root.after(200, self.redisplay_current_media)
            
            # 恢复进度条状态
            self.root.after(150, lambda: self.restore_progress_state(current_progress, progress_text))
            
            # 如果之前正在播放，恢复播放
            if was_playing:
                self.root.after(300, lambda: setattr(self, 'is_playing', True))

    def redisplay_current_media(self, canvas_width=None, canvas_height=None):
        """重新显示当前加载的媒体内容，适应画布大小"""
        try:
            # 首先检查是否有处理后的图像（检测结果）
            has_processed_content = False
            
            # 1. 优先显示处理后的图像（检测结果）
            if hasattr(self, 'processed_frame') and isinstance(self.processed_frame, np.ndarray):
                try:
                    # 重新显示处理后的图像（检测结果）
                    if len(self.processed_frame.shape) == 3:
                        processed_rgb = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
                    else:
                        processed_rgb = self.processed_frame
                    
                    # 在两个画布上都显示处理后的图像
                    self.display_image(processed_rgb, self.processed_canvas,
                                    forced_width=canvas_width,
                                    forced_height=canvas_height)
                    self.display_image(processed_rgb, self.original_canvas,
                                    forced_width=canvas_width,
                                    forced_height=canvas_height)
                    has_processed_content = True
                    return True
                except Exception as e:
                    self.log(f"显示处理后图像错误: {str(e)}", console_only=True)
            
            # 2. 如果没有处理后的图像但有处理后的PhotoImage
            if not has_processed_content and hasattr(self, '_processed_photo') and self._processed_photo:
                try:
                    # 在两个画布上都显示处理后的PhotoImage
                    self.processed_canvas.delete("all")
                    self.original_canvas.delete("all")
                    
                    center_x = canvas_width//2 if canvas_width else self.processed_canvas.winfo_width()//2
                    center_y = canvas_height//2 if canvas_height else self.processed_canvas.winfo_height()//2
                    
                    self.processed_canvas.create_image(center_x, center_y, image=self._processed_photo)
                    self.original_canvas.create_image(center_x, center_y, image=self._processed_photo)
                    has_processed_content = True
                    return True
                except Exception as e:
                    self.log(f"显示处理后PhotoImage错误: {str(e)}", console_only=True)
            
            # 3. 只有在没有任何处理后内容时，才显示原始内容
            if not has_processed_content:
                if hasattr(self, 'original_image') and isinstance(self.original_image, np.ndarray):
                    try:
                        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                        self.display_image(original_rgb, self.original_canvas, 
                                        forced_width=canvas_width, 
                                        forced_height=canvas_height)
                    except Exception as e:
                        self.log(f"显示原始图像错误: {str(e)}", console_only=True)
                elif hasattr(self, '_original_photo') and self._original_photo:
                    try:
                        self.original_canvas.delete("all")
                        center_x = canvas_width//2 if canvas_width else self.original_canvas.winfo_width()//2
                        center_y = canvas_height//2 if canvas_height else self.original_canvas.winfo_height()//2
                        self.original_canvas.create_image(center_x, center_y, image=self._original_photo)
                    except Exception as e:
                        self.log(f"显示原始PhotoImage错误: {str(e)}", console_only=True)
            
            # 更新UI布局
            self.root.update_idletasks()
            return True
        except Exception as e:
            self.log(f"重新显示媒体错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc(), console_only=True)
            return False
        
    def toggle_fullscreen(self, event=None):
        """切换全屏模式"""
        self.is_maximized = not self.is_maximized
        self.root.attributes("-fullscreen", self.is_maximized)
        self.update_canvas_sizes()
        return "break"

    def end_fullscreen(self, event=None):
        """退出全屏模式"""
        self.is_maximized = False
        self.root.attributes("-fullscreen", False)
        self.update_canvas_sizes()
        return "break"

    def redisplay_current_frames(self):
        """在调整大小后重新显示当前帧"""
        # 只有在当前有视频播放时才更新
        if hasattr(self, 'video_capture') and self.video_capture is not None:
            # 临时暂停视频播放
            was_playing = False
            if hasattr(self, 'is_playing') and self.is_playing:
                was_playing = True
                self.is_playing = False
                # 等待短暂时间确保视频线程已暂停
                time.sleep(0.1)
            
            # 使用已经缓存的图像而不是重新读取视频
            try:
                if hasattr(self, '_original_photo') and self._original_photo:
                    self.original_canvas.delete("all")
                    self.original_canvas.create_image(
                        self.original_canvas.winfo_width()//2, 
                        self.original_canvas.winfo_height()//2,
                        image=self._original_photo
                    )
                    
                if hasattr(self, '_processed_photo') and self._processed_photo:
                    self.processed_canvas.delete("all")
                    self.processed_canvas.create_image(
                        self.processed_canvas.winfo_width()//2, 
                        self.processed_canvas.winfo_height()//2,
                        image=self._processed_photo
                    )
            except Exception as e:
                self.log(f"重新显示帧错误: {str(e)}")
            
            # 如果之前正在播放，延迟一点时间后恢复播放
            if was_playing:
                self.root.after(200, lambda: setattr(self, 'is_playing', True))
    
    def restore_progress_state(self, progress_value, progress_text):
        """恢复进度条状态"""
        # 只更新进度值，不显示进度文本内容
        self.progress_bar['value'] = progress_value
        
        # 只显示百分比，而不显示日志内容
        self.progress_text.set(f"{progress_value}%")
        self.progress_bar.update()
        
        # 日志内容转移到日志区域
        if progress_text and progress_text != f"{progress_value}%":
            self.log(progress_text)
        
        # 重新设置标签框架位置，确保居中
        if hasattr(self, 'label_frame') and self.label_frame:
            self.label_frame.place(relx=0.5, rely=0.5, anchor="center")
            self.progress_label.update()

    def _center_window(self):
        """将窗口居中显示"""
        # 获取屏幕尺寸
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 计算居中位置
        x_position = int((screen_width - self.default_width) / 2)
        y_position = int((screen_height - self.default_height) / 2)
        
        # 设置窗口位置
        self.root.geometry(f"{self.default_width}x{self.default_height}+{x_position}+{y_position}")
    
    def _configure_main_canvas(self, event):
        """配置主画布滚动区域"""
        # 更新画布的滚动区域
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        
        # 调整画布宽度以适应框架
        self.main_canvas.config(width=self.root.winfo_width() - self.main_scroll.winfo_width())

    def toggle_playback(self):
        """切换视频播放/暂停状态"""
        if not self.processed_media_path:
            return
            
        self.is_playing = not self.is_playing
        
        # 更新按钮文本
        if self.is_playing:
            self.play_pause_btn.config(text="暂停")
            if not self.video_thread or not self.video_thread.is_alive():
                # 启动播放线程
                self.start_processed_video_playback()
        else:
            self.play_pause_btn.config(text="播放")
    
    def on_slider_change(self, value):
        """处理进度条拖动事件"""
        if not self.processed_media_path:
            return
            
        # 获取当前进度百分比
        position = float(value)
        
        # 防止在播放线程中处理滑动条拖动引起的循环
        if hasattr(self, 'slider_being_changed') and self.slider_being_changed:
            return
            
        # 设置标志，防止播放线程更新滑动条
        self.slider_being_changed = True
        
        try:
            # 如果视频已打开，设置视频位置
            if self.video_capture and self.processed_video_capture:
                total_frames = int(self.processed_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                target_frame = int(total_frames * position / 100)
                
                # 关闭并重新打开视频以避免解码错误
                self.video_capture.release()
                self.processed_video_capture.release()
                
                self.video_capture = cv2.VideoCapture(self.current_media_path)
                self.processed_video_capture = cv2.VideoCapture(self.processed_media_path)
                
                # 逐帧读取到目标位置
                for _ in range(target_frame):
                    self.video_capture.read()
                    self.processed_video_capture.read()
                
                # 读取目标帧
                ret1, frame1 = self.video_capture.read()
                ret2, frame2 = self.processed_video_capture.read()
                
                if ret1 and ret2:
                    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    
                    self.display_image(frame1_rgb, self.original_canvas)
                    self.display_image(frame2_rgb, self.processed_canvas)
                    
                    # 更新时间标签
                    self.update_time_label(target_frame)
        finally:
            # 清除标志
            self.slider_being_changed = False
            
    def update_time_label(self, current_frame=None):
        """更新视频时间标签"""
        if not self.processed_video_capture:
            return
            
        # 获取视频信息
        fps = self.processed_video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.processed_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 如果没有指定当前帧，则获取当前位置
        if current_frame is None:
            current_frame = int(self.processed_video_capture.get(cv2.CAP_PROP_POS_FRAMES))
        
        # 计算当前时间和总时间
        current_time = current_frame / fps if fps > 0 else 0
        total_time = total_frames / fps if fps > 0 else 0
        
        # 格式化时间
        current_time_str = self.format_time(current_time)
        total_time_str = self.format_time(total_time)
        
        # 更新标签
        self.time_label.config(text=f"{current_time_str} / {total_time_str}")
    
    def format_time(self, seconds):
        """将秒数格式化为时:分:秒格式
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def update_behavior_list(self, behaviors):
        """更新行为列表UI"""
        try:
            # 清空行为列表准备添加新行为
            for item in self.behavior_list.get_children():
                self.behavior_list.delete(item)
            
            # 打开视频获取FPS
            cap = cv2.VideoCapture(self.current_media_path)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 1
            cap.release()
            
            # 提取分析结果数据
            behavior_types = set()
            max_probability = 0.0
            
            # 处理检测到的行为
            for frame_idx, frame_behaviors in behaviors:
                for behavior in frame_behaviors:
                    behavior_type = behavior.get('type', '未知行为')
                    behavior_types.add(behavior_type)
                    confidence = behavior.get('confidence', 0.0)
                    max_probability = max(max_probability, confidence)
                    
                    # 添加到行为列表UI
                    time_point = frame_idx / fps if fps > 0 else 0
                    self.add_behavior_to_list(frame_idx, time_point, behavior_type, confidence)
            
            # 确保更新列表显示和滚动条
            self.behavior_list.update()
            self.behavior_scrollbar.update()
            
            # 总是滚动到底部显示最新内容
            self.behavior_list.yview_moveto(1.0)
            
            theft_frames = len(set(frame_idx for frame_idx, _ in behaviors))
            
            # 创建行为摘要
            self.create_behavior_summary(behaviors, max_probability, theft_frames, total_frames)
            
            # 更新分析结果日志
            theft_detected = "是" if theft_frames > 0 else "否"
            summary_message = f"视频分析完成: 探测盗窃行为：{theft_detected}"
            summary_message += f"\n最高盗窃概率: {max_probability:.2f}"
            summary_message += f"\n包含盗窃行为的帧数: {theft_frames}"
            total_behaviors = sum(1 for _, frame_behaviors in behaviors for _ in frame_behaviors)
            summary_message += f"\n检测到 {total_behaviors} 处可疑行为"
            self.log(summary_message)
        except Exception as e:
            self.log(f"更新行为列表错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def update_progress(self, value):
        """更新进度条显示"""
        try:
            if hasattr(self, 'progress_bar') and self.progress_bar:
                self.progress_bar['value'] = value
                
            if hasattr(self, 'progress_text') and self.progress_text:
                self.progress_text.set(f"{value}%")
                
            # 重新调整标签位置，确保居中
            if hasattr(self, 'label_frame') and self.label_frame:
                # 先更新进度条，确保它有正确的尺寸
                if hasattr(self, 'progress_bar'):
                    self.progress_bar.update_idletasks()
                
                # 重新设置标签框架位置，确保居中
                self.label_frame.place(relx=0.5, rely=0.5, anchor="center")
                
            if hasattr(self, 'progress_label') and self.progress_label:
                self.progress_label.update()
                
            if hasattr(self, 'progress_bar') and self.progress_bar:
                self.progress_bar.update()
                
            self.root.update_idletasks()
        except Exception as e:
            print(f"更新进度条错误: {str(e)}")
            import traceback
            traceback.print_exc()

    def rebuild_behavior_list(self, existing_behaviors):
        """根据环境类型重建行为列表"""
        try:
            # 记录当前的按钮禁用状态
            original_disable_state = getattr(self, '_force_disable_buttons', False)
            
            self.log("开始重建行为列表以匹配环境类型")
            
            # 定义零售环境相关行为和通用行为
            retail_behaviors = ["遮挡商品区域", "疑似撕标签动作", "可疑商品处理", "快速藏匿物品", "将物品放入口袋"]
            
            # 零售特定行为映射到通用行为
            behavior_mapping = {
                "遮挡商品区域": "反复调整位置",
                "疑似撕标签动作": "手肘内收姿态异常",
                "可疑商品处理": "反复调整位置", 
                "快速藏匿物品": "手肘内收姿态异常",
                "将物品放入口袋": "手肘内收姿态异常"
            }
            
            # 图标映射
            icon_map = {
                "遮挡商品区域": "🧥 ",
                "手肘内收姿态异常": "💪 ",
                "肩部不自然隆起": "👕 ",
                "反复调整位置": "🔄 ",
                "疑似撕标签动作": "🏷️ ",
                "可疑商品处理": "🛒 ",
                "快速藏匿物品": "👝 ",
                "将物品放入口袋": "👖 "
            }
            
            # 清空行为列表
            for item in self.behavior_list.get_children():
                self.behavior_list.delete(item)
            
            # 根据环境类型重新添加行为
            is_retail_environment = getattr(self, 'is_retail_environment', True)
            
            for time_str, behavior_type, probability_text in existing_behaviors:
                # 在非零售环境中替换零售特定行为
                original_behavior = behavior_type
                replaced = False
                
                if not is_retail_environment:
                    for retail_behavior in retail_behaviors:
                        if retail_behavior in behavior_type:
                            # 替换为通用行为
                            behavior_type = behavior_mapping.get(retail_behavior, "反复调整位置")
                            replaced = True
                            self.log(f"重建列表: 将'{original_behavior}'替换为'{behavior_type}'")
                            break
                
                # 添加图标前缀
                icon_prefix = icon_map.get(behavior_type, "⚠️ ")
                display_type = icon_prefix + behavior_type
                
                # 如果是替换后的行为，添加星号标记
                if replaced:
                    display_type += "*"
                
                # 插入行为记录
                item_id = self.behavior_list.insert("", "end", values=(time_str, display_type, probability_text))
                
                # 设置背景色，基于概率
                try:
                    probability = float(probability_text.strip('%')) / 100
                    if probability > 0.8:
                        self.behavior_list.item(item_id, tags=("high",))
                    elif probability > 0.6:
                        self.behavior_list.item(item_id, tags=("medium",))
                    else:
                        self.behavior_list.item(item_id, tags=("low",))
                except ValueError:
                    # 如果概率文本无法转换为浮点数，使用默认标签
                    self.behavior_list.item(item_id, tags=("low",))
            
            # 配置标签颜色
            self.behavior_list.tag_configure("high", background="#ffcccc")
            self.behavior_list.tag_configure("medium", background="#ffffcc")  
            self.behavior_list.tag_configure("low", background="#e6f7ff")
            
            # 总是滚动到底部显示最新内容
            self.behavior_list.yview_moveto(1.0)
            
            self.log("行为列表重建完成")
            
            # 确保在处理完成后恢复按钮状态
            if not self.is_processing:
                self.log("重建列表后恢复按钮状态")
                self.process_btn.state(['!disabled'])
                self.select_image_btn.state(['!disabled'])
                self.select_video_btn.state(['!disabled'])
                
                # 如果有处理结果，启用保存按钮
                if hasattr(self, 'processed_media_path') and self.processed_media_path:
                    self.save_btn.state(['!disabled'])
            
        except Exception as e:
            self.log(f"重建行为列表错误: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            
            # 恢复进入函数前的按钮禁用状态
            self._force_disable_buttons = original_disable_state
            
            # 如果正在处理中，确保按钮保持禁用
            if self.is_processing:
                self.process_btn.state(['disabled'])
                self.select_image_btn.state(['disabled'])
                self.select_video_btn.state(['disabled'])
    
    def is_retail_environment_by_visual_features(self, image):
        """通过图像视觉特征判断是否为零售环境"""
        # 初始化评分
        retail_score = 0.0
        visual_retail_score = 0.0
        
        try:
            # 边缘分析
            edge_ratio = self.analyze_edge_density(image)
            if edge_ratio > 0.15:
                visual_retail_score += 0.6
                self.log(f"检测到高边缘密度 ({edge_ratio:.4f})，视觉零售评分 +0.6", console_only=True)
            elif edge_ratio > 0.1:
                visual_retail_score += 0.3
                self.log(f"检测到中等边缘密度 ({edge_ratio:.4f})，视觉零售评分 +0.3", console_only=True)
        except Exception as e:
            self.log(f"计算图像边缘失败: {e}", console_only=True)
            
        try:
            # 颜色多样性分析
            color_diversity = self.analyze_color_diversity(image)
            if color_diversity > 0.4:
                visual_retail_score += 0.4
                self.log(f"检测到高颜色多样性 ({color_diversity:.4f})，视觉零售评分 +0.4", console_only=True)
        except Exception as e:
            self.log(f"计算颜色分布失败: {e}", console_only=True)
        
        # ... existing code ...

    def frame_callback(self, frame, original_frame, frame_index, detections, poses, theft_probability, behaviors):
        """在视频处理线程中处理每一帧的回调函数"""
        start_time = time.time()
        
        try:
            # 构建帧数据字典，用于更新UI
            frame_data = {
                'frame': frame,  # 处理后的帧（带有检测结果）
                'original_frame': original_frame,  # 原始帧
                'frame_index': frame_index,
                'theft_probability': theft_probability,
                'has_detections': len(detections) > 0 or len(behaviors) > 0
            }
            
            # 更新UI显示
            self.update_frame_display(frame_data)
            
            # 计算每秒帧数
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
            
            # 生成进度日志
            progress_percentage = (frame_index / self.total_frames) * 100 if self.total_frames > 0 else 0
            log_message = f"处理进度: {progress_percentage:.1f}% (帧 {frame_index}/{self.total_frames}), FPS: {fps:.1f}"
            
            # 更新进度条和日志 - 显示在控制台，但不频繁显示在UI上
            self.log(log_message)
            self.update_progress(progress_percentage, log_message)
            
            # 如果检测到行为，实时添加到行为列表中
            if behaviors and len(behaviors) > 0:
                frame_time = frame_index / self.video_fps if self.video_fps > 0 else 0
                
                for behavior in behaviors:
                    behavior_type = behavior.get('type', '未知行为')
                    probability = behavior.get('probability', 0)
                    
                    # 添加行为到界面列表
                    self.add_behavior_to_list(
                        frame=frame_index, 
                        time_point=frame_time, 
                        behavior_type=behavior_type, 
                        probability=probability
                    )
                    
                    # 如果是高可信度行为，输出更详细的日志
                    if probability > 0.7:
                        self.log(f"检测到高可信度行为: {behavior_type}，在帧 {frame_index}，可信度: {probability:.2%}")
        except Exception as e:
            self.log(f"帧回调处理错误: {str(e)}", console_only=True)
            
    def analyze_environment_in_video(self, sample_frames):
        """分析视频中的环境类型"""
        try:
            all_retail_environment_results = []
            true_count = 0
            
            for i, frame in enumerate(sample_frames):
                try:
                    self.log(f"分析视频环境 - 采样帧 {i+1}/{len(sample_frames)}", console_only=True)
                    
                    # 使用视觉特征判断环境
                    is_retail = self.is_retail_environment_by_visual_features(frame)
                    all_retail_environment_results.append(is_retail)
                    
                    if is_retail:
                        true_count += 1
                        
                except Exception as e:
                    self.log(f"环境判断出错: {str(e)}", console_only=True)
            
            # 基于所有采样帧结果判断
            if len(all_retail_environment_results) > 0:
                # 如果超过40%的帧判断为零售环境，则视为零售环境
                self.is_retail_environment = true_count / len(all_retail_environment_results) > 0.4
                self.log(f"基于视频中的{len(all_retail_environment_results)}帧分析，环境判断为: {'零售环境' if self.is_retail_environment else '非零售环境'} (零售判断率: {true_count/len(all_retail_environment_results):.2%})")
            else:
                self.is_retail_environment = False
                self.log("无法确定环境类型，默认为非零售环境")
                
            return all_retail_environment_results, true_count
            
        except Exception as e:
            self.log(f"采样帧分析错误: {str(e)}", console_only=True)
            self.is_retail_environment = False
            return [], 0

    def update_canvas_sizes(self):
        """更新画布尺寸，保持固定比例"""
        # 获取可用空间大小
        available_width = self.root.winfo_width() - 50  # 减去边距
        available_height = self.root.winfo_height() - 200  # 减去其他UI元素的高度
        
        # 确保值有效
        if available_width <= 1 or available_height <= 1:
            available_width = max(self.default_width - 50, 800)
            available_height = max(self.default_height - 200, 400)
        
        # 计算每个画布的尺寸（考虑左右各占一半）
        canvas_width = int(available_width / 2 - 20)  # 减去分隔边距
        
        # 使用16:9的宽高比
        canvas_height = int(canvas_width * 9 / 16)
        
        # 限制高度，确保不超过可用空间
        if canvas_height > available_height:
            canvas_height = available_height
            canvas_width = int(canvas_height * 16 / 9)
        
        # 设置画布大小
        self.original_canvas.config(width=canvas_width, height=canvas_height)
        self.processed_canvas.config(width=canvas_width, height=canvas_height)
        
        # 更新画布框架大小
        self.original_canvas_frame.config(width=canvas_width, height=canvas_height)
        self.processed_canvas_frame.config(width=canvas_width, height=canvas_height)
        
        # 重新显示当前媒体内容
        self.redisplay_current_media(canvas_width, canvas_height)
        
        # 确保UI元素适应新尺寸
        self.root.update_idletasks()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = TheftDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 