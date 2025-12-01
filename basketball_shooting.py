import pyrealsense2 as rs
import numpy as np
import cv2
import math
import serial
import torch
import sys
import os
import argparse
from pathlib import Path

# --- Configuration ---
# Add YOLOv5 repository to path dynamically
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory of this project
YOLO_PATH = ROOT / 'yolov5' # Assumes yolov5 repo is cloned inside this folder

if str(YOLO_PATH) not in sys.path:
    sys.path.append(str(YOLO_PATH))

try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
except ImportError:
    print("Error: Could not import YOLOv5 modules. Make sure you have cloned 'ultralytics/yolov5' into this directory.")
    sys.exit(1)

def calculate_shooting_angle(D, V, delta_H=1.22, g=9.8, phi_min=45, phi_max=55):
    """
    Calculate the shooting angle (theta) for a projectile.
    """
    V_squared = V ** 2
    g_D_squared = g * D ** 2
    discriminant = (D ** 2) - (2 * g * D ** 2 * delta_H / V_squared) - (g ** 2 * D ** 4 / V_squared ** 2)

    if discriminant < 0:
        return None, f"Infeasible (V={V}m/s, D={D:.2f}m)"

    sqrt_discriminant = math.sqrt(discriminant)
    denominator = g_D_squared / V_squared
    k_larger = (D + sqrt_discriminant) / denominator
    theta_larger = math.atan(k_larger) * 180 / math.pi

    theta_rad = theta_larger * math.pi / 180
    t = D / (V * math.cos(theta_rad))
    V_y = V * math.sin(theta_rad) - g * t
    V_x = V * math.cos(theta_rad)
    phi_rad = math.atan(abs(V_y) / V_x)
    phi_deg = phi_rad * 180 / math.pi

    if phi_deg < phi_min:
        return None, f"Angle too shallow ({phi_deg:.1f}°)"
    elif phi_deg > phi_max:
        return None, f"Angle too steep ({phi_deg:.1f}°)"

    return round(theta_larger, 2), "Optimal"

def send_angle_to_esp32(angle, ser):
    """Send angle to ESP32 if serial connection exists."""
    if ser:
        try:
            ser.write(f"{angle}\n".encode())
            return True
        except serial.SerialException as e:
            print(f"Serial Error: {e}")
            return False
    return False

def load_yolov5_model(weights_path, device_str=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_str: device = torch.device(device_str)
    
    try:
        model = DetectMultiBackend(weights_path, device=device)
        model.eval()
        print(f"Model loaded on {device}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def detect_basketball_net(model, img, conf_thres=0.6, iou_thres=0.45):
    img_resized, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640))
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img_resized = np.ascontiguousarray(img_resized, dtype=np.float32) / 255.0
    img_resized = torch.from_numpy(img_resized).unsqueeze(0).to(model.device)

    pred = model(img_resized)
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    
    boxes = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_resized.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf), int(cls)])
    return boxes

def main(opt):
    # Initialize Serial
    ser = None
    try:
        ser = serial.Serial(opt.port, opt.baud, timeout=1)
        print(f"Connected to ESP32 on {opt.port}")
    except:
        print(f"Warning: Could not connect to ESP32 on {opt.port}. Running in simulation mode.")

    # Initialize RealSense
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("Camera initialized.")
    except Exception as e:
        print(f"Camera Error: {e}")
        return

    model = load_yolov5_model(opt.weights)
    
    delta_H = 1.22
    V = opt.velocity

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame: continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            boxes = detect_basketball_net(model, color_image)
            
            D, theta = None, None
            message = "Scanning..."
            
            if boxes:
                box = max(boxes, key=lambda x: x[4]) # Best confidence
                x1, y1, x2, y2, conf, _ = box
                pixel_x, pixel_y = (x1 + x2) // 2, y1
                
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
                
                # Get Depth (Average of a 3x3 kernel for stability)
                try:
                    depth_kernel = depth_image[pixel_y-1:pixel_y+2, pixel_x-1:pixel_x+2]
                    depth_value = np.mean(depth_kernel[depth_kernel > 0]) # Ignore zero depth
                except:
                    depth_value = depth_image[pixel_y, pixel_x]

                if depth_value > 0:
                    D = depth_value / 1000.0
                    theta, message = calculate_shooting_angle(D, V, delta_H)
                    
                    if theta:
                        angle_to_send = max(0, 90 - theta)
                        send_angle_to_esp32(angle_to_send, ser)
                        cv2.putText(color_image, f"Fire: {theta}°", (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # HUD
            cv2.putText(color_image, f"Dist: {D:.2f}m" if D else "No Target", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if D else (0,0,255), 2)
            cv2.putText(color_image, message, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow('Robocon Vision', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if ser: ser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='model.pt path')
    parser.add_argument('--port', type=str, default='/dev/ttyUSB0', help='ESP32 serial port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--velocity', type=float, default=8.5, help='Flywheel velocity (m/s)')
    opt = parser.parse_args()
    main(opt)
