import cv2, os
os.chdir('E:/Theft-Detection-main')
from ultralytics import YOLO
from detection_engine.models.behavior.video_behavior import VideoBehaviorDetector

y = YOLO('models/yolov11n.pt')
d = VideoBehaviorDetector()

f = cv2.imread('diag_frame_0.jpg')
r = y(f, verbose=False, conf=0.25)
print('YOLO detections:', [y.names[int(b.cls[0])] for b in r[0].boxes])

behaviors = d.detect_behaviors_in_image(f, r[0])
print(f'Behaviors returned: {len(behaviors)}')
for b in behaviors:
    btype = b.get('type', '?')
    bconf = b.get('confidence', 0)
    print(f'  {btype} conf={bconf:.3f}')
