import os
import cv2
import numpy as np
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('video_processor')

class VideoProcessor:
    """Class for video processing utilities"""
    
    def __init__(self):
        """Initialize the video processor"""
        # Ensure output directory exists
        self.output_dir = os.path.join("static", "output")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("VideoProcessor initialized")
    
    def get_video_info(self, video_path):
        """Get information about a video file"""
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Release video
            cap.release()
            
            return {
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "aspect_ratio": width / height if height > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            raise
    
    def extract_frame(self, video_path, frame_number=0):
        """Extract a specific frame from a video file"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")
            
            # Get frame count
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Validate frame number
            if frame_number < 0 or frame_number >= frame_count:
                logger.warning(f"Invalid frame number {frame_number}, using first frame")
                frame_number = 0
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise Exception(f"Could not read frame {frame_number} from video")
            
            return frame
        except Exception as e:
            logger.error(f"Error extracting frame: {str(e)}")
            raise
    
    def extract_frames(self, video_path, output_dir=None, interval=1, max_frames=None):
        """Extract frames from a video at regular intervals
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save extracted frames
            interval: Extract every Nth frame
            max_frames: Maximum number of frames to extract
        
        Returns:
            List of paths to extracted frames
        """
        try:
            if output_dir is None:
                output_dir = os.path.join(self.output_dir, f"frames_{int(time.time())}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Video has {frame_count} frames at {fps} FPS")
            
            # Limit frames if needed
            if max_frames is not None and max_frames > 0:
                frames_to_extract = min(frame_count, max_frames * interval)
            else:
                frames_to_extract = frame_count
            
            extracted_paths = []
            frame_number = 0
            extracted_count = 0
            
            while frame_number < frames_to_extract:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame
                if frame_number % interval == 0:
                    frame_path = os.path.join(output_dir, f"frame_{frame_number:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    extracted_paths.append(frame_path)
                    extracted_count += 1
                    
                    if max_frames is not None and extracted_count >= max_frames:
                        break
                
                frame_number += 1
            
            cap.release()
            
            logger.info(f"Extracted {len(extracted_paths)} frames to {output_dir}")
            return extracted_paths
        
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise
    
    def create_video_from_frames(self, frame_paths, output_path=None, fps=30, size=None):
        """Create a video from a list of image frames
        
        Args:
            frame_paths: List of paths to image frames
            output_path: Path to save the output video
            fps: Frames per second for output video
            size: (width, height) tuple for output video, or None to use first frame size
        
        Returns:
            Path to the created video
        """
        try:
            if not frame_paths:
                raise ValueError("No frames provided")
            
            if output_path is None:
                output_path = os.path.join(self.output_dir, f"video_{int(time.time())}.mp4")
            
            # Get the size from the first frame if not specified
            if size is None:
                first_frame = cv2.imread(frame_paths[0])
                if first_frame is None:
                    raise Exception(f"Could not read frame: {frame_paths[0]}")
                size = (first_frame.shape[1], first_frame.shape[0])
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, size)
            
            # Write frames to video
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is None:
                    logger.warning(f"Could not read frame: {frame_path}, skipping")
                    continue
                
                # Resize if needed
                if frame.shape[1] != size[0] or frame.shape[0] != size[1]:
                    frame = cv2.resize(frame, size)
                
                out.write(frame)
            
            out.release()
            
            logger.info(f"Created video with {len(frame_paths)} frames at {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            raise
    
    def process_video_with_detector(self, video_path, detector, output_path=None, process_interval=1, display_interval=30, progress_callback=None):
        """Process a video with a detector
        
        Args:
            video_path: Path to input video
            detector: Detector object with detect_theft and draw_detection methods
            output_path: Path to save output video
            process_interval: Process every Nth frame
            display_interval: Display progress every Nth frame
            progress_callback: Function to call with progress percentage
        
        Returns:
            Path to processed video, detection statistics
        """
        try:
            if output_path is None:
                output_path = os.path.join(self.output_dir, f"processed_{int(time.time())}.mp4")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process statistics
            processed_frames = 0
            theft_frames = 0
            max_probability = 0
            frame_number = 0
            
            # Process the video
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                if frame_number % process_interval == 0:
                    # Detect theft
                    detections, theft_proba = detector.detect_theft(frame)
                    
                    # Update statistics
                    processed_frames += 1
                    max_probability = max(max_probability, theft_proba)
                    if theft_proba > 0.5:
                        theft_frames += 1
                    
                    # Draw results
                    result_frame = detector.draw_detection(frame, detections, theft_proba)
                    
                    # Write frame
                    out.write(result_frame)
                    
                    # Report progress
                    if frame_number % display_interval == 0:
                        progress = int(100 * frame_number / frame_count)
                        logger.info(f"Processing frame {frame_number}/{frame_count} ({progress}%) - Theft: {theft_proba:.2%}")
                        
                        if progress_callback:
                            progress_callback(progress)
                else:
                    # Write original frame
                    out.write(frame)
                
                frame_number += 1
            
            # Release resources
            cap.release()
            out.release()
            
            # Calculate statistics
            theft_percentage = theft_frames / processed_frames if processed_frames > 0 else 0
            
            stats = {
                "total_frames": frame_count,
                "processed_frames": processed_frames,
                "theft_frames": theft_frames,
                "max_probability": max_probability,
                "theft_percentage": theft_percentage,
                "is_theft": max_probability > 0.5
            }
            
            logger.info(f"Video processing complete: {output_path}")
            logger.info(f"Statistics: {stats}")
            
            return output_path, stats
        
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
    
    def create_gif_from_video(self, video_path, output_path=None, max_frames=30, frame_interval=10, size=None):
        """Create an animated GIF from a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output GIF
            max_frames: Maximum number of frames in the GIF
            frame_interval: Extract every Nth frame
            size: (width, height) tuple for output GIF, or None to keep original size
            
        Returns:
            Path to the created GIF
        """
        try:
            if output_path is None:
                output_path = os.path.join(self.output_dir, f"animation_{int(time.time())}.gif")
            
            # Extract frames
            temp_dir = os.path.join(self.output_dir, f"temp_gif_{int(time.time())}")
            os.makedirs(temp_dir, exist_ok=True)
            
            frame_paths = self.extract_frames(
                video_path, 
                output_dir=temp_dir,
                interval=frame_interval,
                max_frames=max_frames
            )
            
            if not frame_paths:
                raise Exception("No frames extracted from video")
            
            # Use PIL to create animated GIF
            import imageio
            from PIL import Image
            
            frames = []
            for frame_path in frame_paths:
                img = Image.open(frame_path)
                
                # Resize if needed
                if size is not None:
                    img = img.resize(size, Image.LANCZOS)
                
                frames.append(np.array(img))
                
                # Clean up temporary frame
                os.remove(frame_path)
            
            # Remove temporary directory
            os.rmdir(temp_dir)
            
            # Save GIF
            imageio.mimsave(output_path, frames, fps=10)
            
            logger.info(f"Created GIF with {len(frames)} frames at {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error creating GIF: {str(e)}")
            raise

if __name__ == "__main__":
    # Test code
    processor = VideoProcessor()
    
    # Example: Extract a frame from a video
    test_video = "test.mp4"
    if os.path.exists(test_video):
        # Get video info
        info = processor.get_video_info(test_video)
        print(f"Video info: {info}")
        
        # Extract a frame
        frame = processor.extract_frame(test_video, frame_number=0)
        cv2.imwrite("test_frame.jpg", frame)
        print("Extracted test frame") 