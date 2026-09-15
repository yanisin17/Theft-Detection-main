import os
import cv2
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ui')

class TheftDetectionUI:
    def __init__(self, detector):
        """
        Initialize the theft detection UI
        
        Args:
            detector: Object detector instance
        """
        self.detector = detector
        self.window = tk.Tk()
        self.window.title("AI盗窃行为监测系统")
        self.window.geometry("1280x720")
        self.window.resizable(True, True)
        
        # Set style
        self.style = ttk.Style()
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('TLabel', font=('Arial', 10))
        
        # Source variables
        self.source_path = None
        self.video_source = None
        self.is_video = False
        self.video_playing = False
        self.video_thread = None
        self.frame_delay = 30  # milliseconds between video frames
        
        # Detection variables
        self.detection_results = None
        self.detection_image = None
        self.behavior_detection_enabled = tk.BooleanVar(value=False)
        
        # Fixed size for video playback
        self.target_width = 960
        self.target_height = 540
        
        # Create UI components
        self.create_widgets()
        
        # Log
        logger.info("UI initialized")
        
    def create_widgets(self):
        """Create all UI widgets"""
        # Main frame
        self.main_frame = ttk.Frame(self.window, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top controls frame
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Source selection
        ttk.Label(self.controls_frame, text="选择来源:").pack(side=tk.LEFT, padx=5)
        self.source_btn = ttk.Button(self.controls_frame, text="图像/视频文件", command=self.select_source)
        self.source_btn.pack(side=tk.LEFT, padx=5)
        
        # Camera selection - for future implementation
        self.camera_btn = ttk.Button(self.controls_frame, text="摄像头", command=self.select_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5)
        
        # Detection controls
        ttk.Separator(self.controls_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.detect_btn = ttk.Button(self.controls_frame, text="开始检测", command=self.start_detection)
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        
        # Behavior detection checkbox
        self.behavior_check = ttk.Checkbutton(
            self.controls_frame, 
            text="启用可疑行为检测", 
            variable=self.behavior_detection_enabled
        )
        self.behavior_check.pack(side=tk.LEFT, padx=5)
        
        # Video controls
        ttk.Separator(self.controls_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.play_btn = ttk.Button(self.controls_frame, text="播放", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(self.controls_frame, text="停止", command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        # Save result button
        ttk.Separator(self.controls_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.save_btn = ttk.Button(self.controls_frame, text="保存结果", command=self.save_result, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Source info label
        self.info_label = ttk.Label(self.controls_frame, text="未选择来源")
        self.info_label.pack(side=tk.RIGHT, padx=5)
        
        # Create a PanedWindow for main content
        self.content_pane = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.content_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame for video/image display
        self.display_frame = ttk.Frame(self.content_pane, borderwidth=2, relief="groove")
        
        # Right frame for behavior information
        self.behavior_frame = ttk.Frame(self.content_pane, borderwidth=2, relief="groove", width=250)
        
        # Add frames to paned window
        self.content_pane.add(self.display_frame, weight=4)
        self.content_pane.add(self.behavior_frame, weight=1)
        
        # Canvas for displaying image/video
        self.canvas = tk.Canvas(self.display_frame, bg="black", width=self.target_width, height=self.target_height)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Behavior display (Treeview)
        self.behavior_label = ttk.Label(self.behavior_frame, text="可疑行为检测结果", font=('Arial', 12, 'bold'))
        self.behavior_label.pack(pady=10)
        
        # Create a frame for the treeview with scrollbar
        tree_frame = ttk.Frame(self.behavior_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar to the treeview
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create and configure treeview for behavior display
        self.behavior_tree = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set)
        self.behavior_tree["columns"] = ("置信度",)
        self.behavior_tree.column("#0", width=180, minwidth=180)
        self.behavior_tree.column("置信度", width=60, minwidth=60)
        self.behavior_tree.heading("#0", text="可疑行为类型")
        self.behavior_tree.heading("置信度", text="置信度")
        self.behavior_tree.pack(fill=tk.BOTH, expand=True)
        
        # Configure scrollbar
        tree_scroll.config(command=self.behavior_tree.yview)
        
        # Status bar
        self.status_var = tk.StringVar(value="就绪")
        self.status_bar = ttk.Label(self.window, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Set callback for window resize
        self.window.bind("<Configure>", self.on_resize)
        
    def select_source(self):
        """Open file dialog to select image or video source"""
        filetypes = [
            ("图像和视频文件", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"),
            ("图像文件", "*.jpg *.jpeg *.png"),
            ("视频文件", "*.mp4 *.avi *.mov"),
            ("所有文件", "*.*")
        ]
        self.source_path = filedialog.askopenfilename(
            title="选择图像或视频文件",
            filetypes=filetypes
        )
        
        if not self.source_path:
            return
            
        # Determine if it's an image or video
        _, ext = os.path.splitext(self.source_path)
        self.is_video = ext.lower() in ['.mp4', '.avi', '.mov']
        
        # Clear behavior tree
        self.behavior_tree.delete(*self.behavior_tree.get_children())
        
        if self.is_video:
            # Load video
            self.video_source = cv2.VideoCapture(self.source_path)
            if not self.video_source.isOpened():
                messagebox.showerror("错误", f"无法打开视频文件: {self.source_path}")
                return
                
            # Enable video controls
            self.play_btn["state"] = tk.NORMAL
            self.stop_btn["state"] = tk.NORMAL
            
            # Update info
            fps = self.video_source.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.video_source.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps else 0
            width = int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.info_label["text"] = f"视频: {os.path.basename(self.source_path)} - {width}x{height}, {fps:.1f}fps, {duration:.1f}秒"
            self.status_var.set(f"已加载视频: {os.path.basename(self.source_path)}")
            
            # Display first frame (statically, not playing)
            ret, frame = self.video_source.read()
            if ret:
                self.display_image(frame, maintain_fixed_size=True)
                # Rewind video
                self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        else:
            # Load image
            try:
                self.display_path = self.source_path
                image = cv2.imread(self.source_path)
                if image is None:
                    messagebox.showerror("错误", f"无法加载图像文件: {self.source_path}")
                    return
                
                # Display image
                self.display_image(image)
                
                # Update info
                height, width = image.shape[:2]
                self.info_label["text"] = f"图像: {os.path.basename(self.source_path)} - {width}x{height}"
                self.status_var.set(f"已加载图像: {os.path.basename(self.source_path)}")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载图像时出错: {str(e)}")
                logger.error(f"Error loading image: {e}")
    
    def select_camera(self):
        """Select camera as source"""
        # For future implementation
        messagebox.showinfo("提示", "摄像头功能尚未实现")
    
    def display_image(self, cv_image, maintain_fixed_size=False):
        """Display OpenCV image on canvas
        
        Args:
            cv_image: OpenCV image (BGR format)
            maintain_fixed_size: If True, maintain the fixed size for display
        """
        try:
            # Convert to RGB for PIL
            image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Get canvas dimensions
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Default to target size if canvas size not yet determined
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = self.target_width
                canvas_height = self.target_height
            
            # Get image dimensions
            img_height, img_width = image.shape[:2]
            
            # Determine target dimensions
            if maintain_fixed_size or (self.is_video and self.video_playing):
                # Use fixed size for videos and when specifically requested
                target_width = self.target_width
                target_height = self.target_height
            else:
                # For images, calculate based on canvas size while maintaining aspect ratio
                ratio = min(canvas_width / img_width, canvas_height / img_height)
                target_width = int(img_width * ratio)
                target_height = int(img_height * ratio)
            
            # Resize image if needed
            if target_width > 0 and target_height > 0 and (target_width != img_width or target_height != img_height):
                image = cv2.resize(image, (target_width, target_height))
            
            # Convert to PIL Image and then to PhotoImage
            self.pil_image = Image.fromarray(image)
            self.tk_image = ImageTk.PhotoImage(image=self.pil_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.config(width=target_width, height=target_height)  # Set canvas size to match image
            
            # Center the image
            x_pos = (canvas_width - target_width) // 2 if canvas_width > target_width else 0
            y_pos = (canvas_height - target_height) // 2 if canvas_height > target_height else 0
            
            self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_image)
            
        except Exception as e:
            logger.error(f"Error displaying image: {e}")
            messagebox.showerror("错误", f"显示图像时出错: {str(e)}")
    
    def start_detection(self):
        """Start object detection on current source"""
        if not self.source_path:
            messagebox.showinfo("提示", "请先选择图像或视频文件")
            return
            
        # Update status
        self.status_var.set("正在进行目标检测...")
        
        # Clear behavior tree
        self.behavior_tree.delete(*self.behavior_tree.get_children())
        
        # Get behavior detection setting
        enable_behavior = self.behavior_detection_enabled.get()
        
        if self.is_video:
            # Process video
            try:
                # Disable buttons during processing
                self.detect_btn["state"] = tk.DISABLED
                self.play_btn["state"] = tk.DISABLED
                self.stop_btn["state"] = tk.DISABLED
                self.source_btn["state"] = tk.DISABLED
                
                # Start processing in a separate thread
                threading.Thread(
                    target=self._process_video,
                    args=(enable_behavior,),
                    daemon=True
                ).start()
                
            except Exception as e:
                logger.error(f"Error processing video: {e}")
                messagebox.showerror("错误", f"处理视频时出错: {str(e)}")
                self.status_var.set("检测失败")
                
                # Re-enable buttons
                self.detect_btn["state"] = tk.NORMAL
                self.play_btn["state"] = tk.NORMAL
                self.stop_btn["state"] = tk.NORMAL
                self.source_btn["state"] = tk.NORMAL
                
        else:
            # Process image
            try:
                # Disable button during processing
                self.detect_btn["state"] = tk.DISABLED
                
                # Start processing in a separate thread
                threading.Thread(
                    target=self._process_image,
                    args=(enable_behavior,),
                    daemon=True
                ).start()
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                messagebox.showerror("错误", f"处理图像时出错: {str(e)}")
                self.status_var.set("检测失败")
                
                # Re-enable button
                self.detect_btn["state"] = tk.NORMAL
    
    def _process_image(self):
        """处理选择的图像文件"""
        if not self.source_path:
            logger.error("未选择图像文件")
            messagebox.showerror("错误", "请先选择一个图像文件")
            return
        
        self.status_var.set("正在处理图像...")
        self._update_status_bar()
        
        try:
            # 加载图像
            image = cv2.imread(self.source_path)
            if image is None:
                raise ValueError(f"无法加载图像: {self.source_path}")
            
            # 更改为使用正确的行为检测参数
            results = self.detector.detect_objects(image, behavior_detect=True)
            
            if results is None:
                raise ValueError("检测失败")
            
            # 绘制检测结果
            output_image = results[0].plot() if hasattr(results, '__getitem__') else results.plot()
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            
            # 保存处理后的图像
            output_filename = f"output_{os.path.basename(self.source_path)}"
            output_path = os.path.join("static", "output", output_filename)
            cv2.imwrite(output_path, output_image)
            
            # 获取可疑行为
            suspicious_behaviors = []
            if hasattr(results, 'suspicious_behaviors'):
                suspicious_behaviors = results.suspicious_behaviors
            
            # 更新界面显示
            self._update_after_image_detection(output_image, suspicious_behaviors)
            
            self.status_var.set(f"图像处理完成。输出保存至: {output_path}")
            logger.info(f"图像处理完成，输出保存至: {output_path}")
            
        except Exception as e:
            logger.error(f"处理图像时出错: {str(e)}")
            self.status_var.set(f"处理失败: {str(e)}")
            messagebox.showerror("处理失败", f"图像处理出错: {str(e)}")
        finally:
            self._update_status_bar()
    
    def _update_after_image_detection(self, image, suspicious_behaviors):
        """Update UI after image detection is complete"""
        # Display results
        self.display_image(image)
        
        # Update behavior treeview if behaviors detected
        if suspicious_behaviors:
            self._update_behavior_display(suspicious_behaviors)
            
            # Update status
            behavior_count = len(suspicious_behaviors)
            self.status_var.set(f"检测完成 - 发现 {behavior_count} 个可疑行为!")
        else:
            self.status_var.set("检测完成")
        
        # Enable save button
        self.save_btn["state"] = tk.NORMAL
        
        # Re-enable detect button
        self.detect_btn["state"] = tk.NORMAL
    
    def _update_behavior_display(self, behaviors):
        """Update behavior treeview with detected behaviors"""
        # Clear existing items
        self.behavior_tree.delete(*self.behavior_tree.get_children())
        
        # Add behaviors to treeview
        for i, behavior in enumerate(behaviors):
            behavior_desc = behavior.get('description', 'Unknown')
            confidence = behavior.get('confidence', 0.0)
            
            # Add to treeview
            self.behavior_tree.insert("", "end", text=behavior_desc, values=(f"{confidence:.2f}",))
    
    def _process_video(self, enable_behavior=False):
        """Process video for detection"""
        try:
            # Run detection on video
            start_time = time.time()
            
            # Create output directory if needed
            output_dir = os.path.join("static", "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Process video
            if enable_behavior:
                # For behavior analysis
                result = self.detector.analyze_video(
                    self.source_path, 
                    behavior_detect=True
                )
                
                # Unpack the result - should be (output_path, suspicious_frames, behaviors)
                if isinstance(result, tuple) and len(result) >= 3:
                    output_path, suspicious_frames, behaviors = result
                    # Count behaviors
                    behavior_count = sum(len(behaviors_at_frame) for _, behaviors_at_frame in behaviors) if behaviors else 0
                else:
                    # Handle unexpected return format gracefully
                    output_path = result if result else None
                    suspicious_frames = []
                    behaviors = []
                    behavior_count = 0
                    logger.warning("Unexpected return format from analyze_video method")
                
            else:
                # For standard object detection
                result = self.detector.analyze_video(
                    self.source_path,
                    behavior_detect=False
                )
                
                # Unpack the result - should be (output_path, [], [])
                if isinstance(result, tuple) and len(result) >= 3:
                    output_path, suspicious_frames, behaviors = result
                else:
                    # Handle unexpected return format
                    output_path = result if result else None
                    suspicious_frames = []
                    behaviors = []
                    
                behavior_count = 0
            
            elapsed = time.time() - start_time
            
            # Update processed video path and display first frame
            if output_path and os.path.exists(output_path):
                self.detection_video_path = output_path
                
                # Open the processed video
                cap = cv2.VideoCapture(output_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        # Display first frame statically (not playing)
                        self.window.after(0, lambda: self.display_image(frame, maintain_fixed_size=True))
                        
                        # Rewind the video to beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    cap.release()
                
                # Collect all behaviors for display
                all_behaviors = []
                if behaviors and isinstance(behaviors, list):
                    for frame_idx, frame_behaviors in behaviors:
                        if isinstance(frame_behaviors, list):
                            all_behaviors.extend(frame_behaviors)
                
                # Update UI with behaviors
                if all_behaviors:
                    self.window.after(0, lambda: self._update_behavior_display(all_behaviors))
                
                # Update UI status
                self.window.after(0, lambda: self._update_after_video_detection(elapsed, behavior_count, len(suspicious_frames)))
            else:
                raise ValueError("视频处理失败，未生成输出文件")
                
        except Exception as e:
            logger.error(f"Error in video detection: {e}")
            self.window.after(0, lambda: messagebox.showerror("错误", f"视频检测出错: {str(e)}"))
            self.window.after(0, lambda: self.status_var.set("检测失败"))
            self.window.after(0, lambda: setattr(self.detect_btn, "state", tk.NORMAL))
            self.window.after(0, lambda: setattr(self.play_btn, "state", tk.NORMAL))
            self.window.after(0, lambda: setattr(self.stop_btn, "state", tk.NORMAL))
            self.window.after(0, lambda: setattr(self.source_btn, "state", tk.NORMAL))
    
    def _update_after_video_detection(self, elapsed, behavior_count=0, suspicious_frames=0):
        """Update UI after video detection is complete"""
        # Update video source to the processed video
        self.video_source = cv2.VideoCapture(self.detection_video_path)
        
        # Update status
        if behavior_count > 0:
            self.status_var.set(f"视频检测完成 ({elapsed:.2f}秒) - 发现 {behavior_count} 个可疑行为! 在 {suspicious_frames} 帧中")
        else:
            self.status_var.set(f"视频检测完成 ({elapsed:.2f}秒)")
        
        # Enable buttons
        self.play_btn["state"] = tk.NORMAL
        self.stop_btn["state"] = tk.NORMAL
        self.save_btn["state"] = tk.NORMAL
        self.detect_btn["state"] = tk.NORMAL
        self.source_btn["state"] = tk.NORMAL
    
    def toggle_play(self):
        """Toggle video playback"""
        if not self.video_source:
            return
            
        if self.video_playing:
            # Pause video
            self.video_playing = False
            self.play_btn["text"] = "播放"
        else:
            # Play video
            self.video_playing = True
            self.play_btn["text"] = "暂停"
            
            # Start video playback thread
            if not self.video_thread or not self.video_thread.is_alive():
                self.video_thread = threading.Thread(target=self._play_video, daemon=True)
                self.video_thread.start()
    
    def _play_video(self):
        """Play video in a thread"""
        try:
            # Get video properties
            fps = self.video_source.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default if FPS cannot be determined
                
            # Calculate delay between frames in ms
            frame_delay = int(1000 / fps)
            
            # Play loop
            while self.video_playing:
                ret, frame = self.video_source.read()
                
                if not ret:
                    # End of video, rewind
                    self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # Display frame in UI thread
                self.window.after(0, lambda f=frame: self.display_image(f, maintain_fixed_size=True))
                
                # Wait appropriate time between frames
                time.sleep(frame_delay / 1000)
                
        except Exception as e:
            logger.error(f"Error in video playback: {e}")
            self.window.after(0, lambda: self.status_var.set("视频播放出错"))
            self.video_playing = False
            self.window.after(0, lambda: setattr(self.play_btn, "text", "播放"))
    
    def stop_video(self):
        """Stop video playback and rewind"""
        if not self.video_source:
            return
            
        # Stop playback
        self.video_playing = False
        self.play_btn["text"] = "播放"
        
        # Rewind to beginning
        self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Display first frame
        ret, frame = self.video_source.read()
        if ret:
            self.display_image(frame, maintain_fixed_size=True)
            # Rewind again after displaying first frame
            self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def save_result(self):
        """Save detection results"""
        if not self.source_path:
            return
            
        if self.is_video:
            # For video, just show where the processed file is
            if hasattr(self, 'detection_video_path') and os.path.exists(self.detection_video_path):
                messagebox.showinfo(
                    "保存成功", 
                    f"已处理的视频保存在:\n{os.path.abspath(self.detection_video_path)}"
                )
            else:
                messagebox.showerror("错误", "没有可保存的处理结果")
        else:
            # For image, let user choose where to save
            if hasattr(self, 'detection_image') and self.detection_image is not None:
                default_name = f"detected_{os.path.basename(self.source_path)}"
                save_path = filedialog.asksaveasfilename(
                    title="保存检测结果",
                    defaultextension=".jpg",
                    initialfile=default_name,
                    filetypes=[
                        ("JPEG图像", "*.jpg *.jpeg"),
                        ("PNG图像", "*.png"),
                        ("所有文件", "*.*")
                    ]
                )
                
                if save_path:
                    try:
                        cv2.imwrite(save_path, self.detection_image)
                        messagebox.showinfo("保存成功", f"检测结果已保存到:\n{save_path}")
                    except Exception as e:
                        logger.error(f"Error saving image: {e}")
                        messagebox.showerror("错误", f"保存图像时出错: {str(e)}")
            else:
                messagebox.showerror("错误", "没有可保存的处理结果")
    
    def on_resize(self, event):
        """Handle window resize event"""
        # Only respond to resizes of the main window
        if event.widget == self.window and not self.video_playing:
            # If we have an image displayed, update it
            if hasattr(self, 'detection_image') and self.detection_image is not None:
                self.display_image(self.detection_image)
            elif hasattr(self, 'display_path') and self.display_path:
                # If no detection result but we have source image
                try:
                    image = cv2.imread(self.display_path)
                    if image is not None:
                        self.display_image(image)
                except Exception:
                    pass
    
    def run(self):
        """Run the UI main loop"""
        self.window.mainloop()
        
    def cleanup(self):
        """Clean up resources"""
        if self.video_source and hasattr(self.video_source, 'release'):
            self.video_source.release()
            
    def __del__(self):
        """Destructor"""
        self.cleanup() 