import os
import sys
import logging
import time
import traceback
import mimetypes  # 添加mimetypes模块
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import json
import urllib.parse
import shutil
import subprocess  # 用于调用FFmpeg

# 添加MIME类型
mimetypes.add_type('video/mp4', '.mp4')
mimetypes.add_type('video/avi', '.avi')
mimetypes.add_type('video/x-msvideo', '.avi')  # AVI的另一种MIME类型
mimetypes.add_type('video/quicktime', '.mov')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入检测模块
from src.models.detection import TheftDetector
from src.models.behavior.image_behavior import ImageBehaviorDetector
from src.models.behavior.video_behavior import VideoBehaviorDetector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('app')

# 创建Flask应用
app = Flask(__name__, static_folder='../static', template_folder='../templates')
app.secret_key = 'theft-detection-secret-key'

# 配置上传文件
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB限制

# 确保必要的目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join('static', 'output'), exist_ok=True)
os.makedirs(os.path.join('logs'), exist_ok=True)

# 初始化检测器
detector = None
image_behavior_detector = None
video_behavior_detector = None

def allowed_file(filename):
    """检查文件类型是否允许上传"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_detectors():
    """初始化检测器"""
    global detector, image_behavior_detector, video_behavior_detector
    
    try:
        # 初始化物体检测器
        detector = TheftDetector(model_path='models/yolov11n.pt')
        
        # 初始化行为检测器
        image_behavior_detector = ImageBehaviorDetector()
        video_behavior_detector = VideoBehaviorDetector()
        
        logger.info("所有检测器初始化成功")
        return True
    except Exception as e:
        logger.error(f"初始化检测器失败: {e}")
        return False

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/test')
def test_video():
    """视频测试页面"""
    return render_template('test_video.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传"""
    # 检查是否有文件
    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    
    file = request.files['file']
    
    # 检查文件名
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400
    
    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({"error": "不支持的文件类型"}), 400
    
    # 保存文件
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # 判断文件类型
    file_ext = filename.rsplit('.', 1)[1].lower()
    is_video = file_ext in {'mp4', 'avi', 'mov'}
    
    try:
        # 根据文件类型进行处理
        if is_video:
            result = process_video_file(filepath)
        else:
            result = process_image_file(filepath)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"处理文件时出错: {e}")
        return jsonify({"error": str(e)}), 500

def process_image_file(filepath):
    """处理图像文件并返回处理结果"""
    # 确保检测器已初始化
    if detector is None or image_behavior_detector is None:
        if not init_detectors():
            raise Exception('检测器初始化失败，请检查日志')
    
    # 执行标准物体检测
    detection_output, detection_results = detector.process_image(filepath)
    
    # 执行行为分析
    behavior_output, behaviors = image_behavior_detector.analyze_image(filepath, detector)
    
    # 为前端准备序列化安全的行为数据
    serializable_behaviors = []
    for behavior in behaviors:
        serializable_behaviors.append({
            'type': behavior['type'],
            'description': behavior['description'],
            'confidence': float(behavior['confidence'])  # 确保浮点数可以被JSON序列化
        })
    
    # 准备结果数据
    filename = os.path.basename(filepath)
    result = {
        'original': f"/static/uploads/{filename}",
        'result': f"/{behavior_output}" if behavior_output else None,
        'behaviors': serializable_behaviors,
        'behaviors_count': len(behaviors),
        'type': 'image'
    }
    
    return result

def process_video_file(filepath):
    """处理视频文件并返回处理结果"""
    # 确保检测器已初始化
    if detector is None or video_behavior_detector is None:
        if not init_detectors():
            raise Exception('检测器初始化失败，请检查日志')
    
    # 确保输出目录存在
    output_dir = os.path.join('static', 'output')
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {output_dir}")
        except Exception as e:
            logger.error(f"创建输出目录失败: {e}")
            raise Exception(f"无法创建输出目录: {str(e)}")
    
    # 确保检测器具有所需的方法
    if not hasattr(detector, 'detect_theft') or not callable(getattr(detector, 'detect_theft')):
        logger.warning("检测器不具备 detect_theft 方法，将尝试通过 detect 方法进行适配")
    
    # 使用视频行为检测器进行分析
    output_path, suspicious_frames, behaviors = video_behavior_detector.analyze_video_behavior(filepath, detector)
    
    # 检查输出文件是否存在
    if output_path and os.path.exists(output_path):
        logger.info(f"输出视频文件成功创建: {output_path}")
    else:
        logger.error(f"输出视频文件不存在或创建失败: {output_path}")
        if output_path:
            # 尝试检查目录权限
            output_dir = os.path.dirname(output_path)
            if not os.access(output_dir, os.W_OK):
                logger.error(f"目录无写入权限: {output_dir}")
    
    # 准备行为数据
    behavior_data = []
    behavior_types = set()
    max_confidence = 0
    
    for frame_idx, frame_behaviors in behaviors:
        for behavior in frame_behaviors:
            behavior_types.add(behavior['type'])
            max_confidence = max(max_confidence, behavior['confidence'])
            
            time_point = frame_idx / 30.0  # 假设30fps
            behavior_data.append({
                'frame': frame_idx,
                'time': float(time_point),  # 确保浮点数可以被JSON序列化
                'type': behavior['type'],
                'description': behavior['description'],
                'confidence': float(behavior['confidence'])  # 确保浮点数可以被JSON序列化
            })
    
    # 准备结果数据
    filename = os.path.basename(filepath)
    
    # 统一处理路径格式
    formatted_output_path = None
    if output_path:
        # 确保路径格式正确 - 使用正斜杠并添加前导斜杠
        # 先转换所有反斜杠为正斜杠
        output_path = output_path.replace('\\', '/')
        
        # 为了更可靠的访问，确保输出路径格式为:/static/output/xxx.mp4
        if os.path.exists(output_path):
            # 获取真实的相对路径
            rel_path = os.path.relpath(output_path, '.')
            rel_path = rel_path.replace('\\', '/')
            
            # 确保以/static/开头
            if rel_path.startswith('static/'):
                formatted_output_path = '/' + rel_path
            else:
                formatted_output_path = '/static/' + rel_path.replace('static/', '')
            
            logger.info(f"格式化后的输出路径: {formatted_output_path}")
            
            # 将输出视频复制到uploads目录以确保原视频和结果视频在同一目录下
            try:
                # 构建目标路径 - 使用原始文件名加前缀"analyzed_"
                output_filename = os.path.basename(output_path)
                uploads_copy_path = os.path.join('static', 'uploads', 'analyzed_' + output_filename)
                
                # 复制文件
                shutil.copy2(output_path, uploads_copy_path)
                logger.info(f"已复制结果视频到uploads目录: {uploads_copy_path}")
                
                # 更新结果路径为uploads目录中的副本
                formatted_output_path = '/static/uploads/analyzed_' + output_filename
                logger.info(f"更新后的结果路径: {formatted_output_path}")
            except Exception as e:
                logger.error(f"复制结果视频到uploads目录失败: {str(e)}")
                # 继续使用原始路径，不影响主功能
        else:
            logger.error(f"输出文件不存在，无法格式化路径: {output_path}")
            formatted_output_path = None
    
    result = {
        'original': f"/static/uploads/{filename}",
        'result': formatted_output_path,
        'suspicious_frames': [int(frame) for frame in suspicious_frames],  # 确保整数可以被JSON序列化
        'behaviors': behavior_data,
        'behaviors_count': len(behavior_data),
        'behavior_types': list(behavior_types),
        'max_confidence': float(max_confidence),  # 确保浮点数可以被JSON序列化
        'type': 'video'
    }
    
    logger.info(f"视频处理结果: 原始视频={result['original']}, 结果视频={result['result']}")
    return result

@app.route('/api/process_example/<filename>', methods=['POST'])
def api_process_example(filename):
    """API处理示例文件"""
    # 构建文件路径
    filepath = os.path.join('static', 'examples', filename)
    
    # 检查文件是否存在
    if not os.path.exists(filepath):
        return jsonify({"error": "示例文件不存在"}), 404
    
    try:
        # 复制到上传目录
        dest_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'rb') as src_file, open(dest_path, 'wb') as dest_file:
            dest_file.write(src_file.read())
        
        # 判断文件类型并处理
        file_ext = filename.rsplit('.', 1)[1].lower()
        is_video = file_ext in {'mp4', 'avi', 'mov'}
        
        if is_video:
            result = process_video_file(dest_path)
        else:
            result = process_image_file(dest_path)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"处理示例文件时出错: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/examples', methods=['GET'])
def api_examples():
    """获取示例文件列表"""
    examples_dir = os.path.join('static', 'examples')
    os.makedirs(examples_dir, exist_ok=True)
    
    # 获取示例图像
    image_examples = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_examples.extend([f for f in os.listdir(examples_dir) if f.lower().endswith(f'.{ext}')])
    
    # 获取示例视频
    video_examples = []
    for ext in ['mp4', 'avi', 'mov']:
        video_examples.extend([f for f in os.listdir(examples_dir) if f.lower().endswith(f'.{ext}')])
    
    return jsonify({
        "images": image_examples,
        "videos": video_examples
    })

# 添加自定义静态文件发送函数，确保正确的MIME类型
@app.route('/static/<path:filename>')
def custom_static(filename):
    # 自定义MIME类型映射
    mime_mapping = {
        '.avi': 'video/x-msvideo',
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime'
    }
    
    # 获取文件扩展名
    ext = os.path.splitext(filename)[1].lower()
    
    # 设置自定义MIME类型（如果有）
    mimetype = None
    if ext in mime_mapping:
        mimetype = mime_mapping[ext]
    
    # 设置响应头，禁止缓存视频文件
    headers = {}
    if ext in ['.mp4', '.avi', '.mov']:
        headers = {
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    
    # 从正确的目录发送文件
    if filename.startswith('uploads/') or filename.startswith('output/'):
        directory = os.path.join('..', 'static', os.path.dirname(filename))
        basename = os.path.basename(filename)
        return send_from_directory(directory, basename, mimetype=mimetype, headers=headers)
    
    return send_from_directory(app.static_folder, filename, mimetype=mimetype, headers=headers)

# 专门用于直接访问输出视频的路由
@app.route('/video/<path:filename>')
def serve_video(filename):
    """直接访问视频文件的路由"""
    # 自定义MIME类型映射
    mime_mapping = {
        '.avi': 'video/x-msvideo',
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime'
    }
    
    # 获取文件扩展名
    ext = os.path.splitext(filename)[1].lower()
    
    # 设置自定义MIME类型（如果有）
    mimetype = None
    if ext in mime_mapping:
        mimetype = mime_mapping[ext]
    
    # 设置响应头，禁止缓存视频文件
    headers = {
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
        'Expires': '0'
    }
    
    # 从输出目录发送文件
    directory = os.path.join('static', 'output')
    logger.info(f"尝试从 {directory} 发送视频文件 {filename}")
    
    # 检查文件是否存在
    video_path = os.path.join(directory, filename)
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        
        # 尝试使用绝对路径检查
        absolute_path = os.path.abspath(video_path)
        logger.info(f"尝试使用绝对路径: {absolute_path}")
        if not os.path.exists(absolute_path):
            logger.error(f"绝对路径文件也不存在: {absolute_path}")
            
            # 尝试列出目录中的文件
            try:
                files = os.listdir(directory)
                logger.info(f"输出目录中的文件: {files}")
            except Exception as e:
                logger.error(f"列出目录文件时出错: {str(e)}")
                
            return jsonify({"error": "视频文件不存在"}), 404
    
    # 如果存在，返回文件
    logger.info(f"成功找到视频文件，准备发送: {video_path}")
    return send_from_directory(directory, filename, mimetype=mimetype, headers=headers)

# 添加一个API路由列出uploads目录下的文件
@app.route('/api/list-uploads', methods=['GET'])
def list_uploads():
    """列出uploads目录下的文件"""
    try:
        uploads_dir = os.path.join('static', 'uploads')
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir, exist_ok=True)
            
        files = os.listdir(uploads_dir)
        logger.info(f"找到uploads目录下的{len(files)}个文件")
        
        # 按最后修改时间排序，最新的在前
        files_with_time = [(f, os.path.getmtime(os.path.join(uploads_dir, f))) for f in files]
        files_with_time.sort(key=lambda x: x[1], reverse=True)
        sorted_files = [f[0] for f in files_with_time]
        
        return jsonify({
            "status": "success",
            "files": sorted_files,
            "count": len(sorted_files)
        })
    except Exception as e:
        logger.error(f"列出uploads目录文件时出错: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# 动态添加列出目录文件的API路由（供前端动态调用）
@app.route('/api/add-list-route', methods=['POST'])
def add_list_route():
    """确认已添加列出目录文件的API路由"""
    return jsonify({
        "status": "success",
        "message": "API路由已存在",
        "routes": ["/api/list-uploads"]
    })

# 添加一个路由用于浏览器友好的视频格式转换
@app.route('/web-video/<path:filename>')
def serve_web_video(filename):
    """提供浏览器友好格式的视频，将MP4V转换为H.264"""
    # 获取原始视频路径
    original_path = os.path.join('static', 'uploads', filename)
    
    # 检查原始文件是否存在
    if not os.path.exists(original_path):
        logger.error(f"原始视频文件不存在: {original_path}")
        return jsonify({"error": "视频文件不存在"}), 404
    
    # 为转换后的视频创建目录
    web_video_dir = os.path.join('static', 'web-videos')
    os.makedirs(web_video_dir, exist_ok=True)
    
    # 构建转换后的视频路径
    web_video_path = os.path.join(web_video_dir, f"web_{filename}")
    
    # 检查是否已经转换过
    if not os.path.exists(web_video_path) or os.path.getmtime(web_video_path) < os.path.getmtime(original_path):
        logger.info(f"开始转换视频格式为浏览器友好格式: {original_path} -> {web_video_path}")
        
        try:
            # 尝试使用FFmpeg（如果可用）
            try:
                # 构建FFmpeg命令
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-i', original_path,  # 输入文件
                    '-c:v', 'libx264',    # H.264编码
                    '-preset', 'fast',    # 编码速度/质量平衡
                    '-crf', '23',         # 质量参数（0-51，值越低质量越高）
                    '-c:a', 'aac',        # 音频编码
                    '-strict', 'experimental',
                    '-b:a', '128k',       # 音频比特率
                    '-y',                 # 覆盖输出文件
                    web_video_path        # 输出文件
                ]
                
                # 执行FFmpeg命令
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("FFmpeg转换成功")
                else:
                    logger.warning(f"FFmpeg转换失败: {result.stderr}")
                    raise Exception("FFmpeg转换失败")
                    
            except (subprocess.SubprocessError, Exception) as e:
                logger.warning(f"使用FFmpeg转换失败，尝试使用OpenCV: {str(e)}")
                
                # 使用OpenCV进行格式转换
                cap = cv2.VideoCapture(original_path)
                if not cap.isOpened():
                    raise Exception("无法打开视频文件")
                    
                # 获取视频属性
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # 创建VideoWriter - 尝试使用H.264编码
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264编码
                    out = cv2.VideoWriter(web_video_path, fourcc, fps, (width, height))
                    
                    if not out.isOpened():
                        raise Exception("AVC1编码器不可用")
                        
                except Exception as e:
                    logger.warning(f"使用AVC1编码器失败: {str(e)}，尝试替代编码器")
                    try:
                        # 尝试使用H264编码
                        fourcc = cv2.VideoWriter_fourcc(*'H264')
                        out = cv2.VideoWriter(web_video_path, fourcc, fps, (width, height))
                        
                        if not out.isOpened():
                            raise Exception("H264编码器不可用")
                    except Exception as e2:
                        logger.warning(f"使用H264编码器失败: {str(e2)}，回退到MP4V")
                        # 回退到MP4V
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(web_video_path, fourcc, fps, (width, height))
                
                # 逐帧处理
                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    out.write(frame)
                    frame_count += 1
                    
                    # 每100帧记录一次进度
                    if frame_count % 100 == 0:
                        progress = int(100 * frame_count / total_frames)
                        logger.info(f"视频转换进度: {progress}%")
                
                # 释放资源
                cap.release()
                out.release()
                logger.info(f"OpenCV转换完成，共处理{frame_count}帧")
            
            logger.info(f"视频格式转换完成: {web_video_path}")
            
        except Exception as e:
            logger.error(f"视频格式转换失败: {str(e)}")
            traceback.print_exc()
            # 如果转换失败，回退到原始文件
            return send_from_directory(os.path.dirname(original_path), os.path.basename(original_path))
    
    # 设置MIME类型和缓存控制
    headers = {
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
        'Expires': '0'
    }
    
    # 发送转换后的文件
    return send_from_directory(web_video_dir, f"web_{filename}", mimetype='video/mp4', headers=headers)

if __name__ == '__main__':
    # 初始化检测器
    init_detectors()
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000) 