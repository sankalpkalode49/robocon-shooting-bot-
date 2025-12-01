import pyrealsense2 as rs
import numpy as np
import cv2
import math
import serial
import torch
import sys
import os

# Add YOLOv5 repository to path (adjust path as needed)
sys.path.append('/home/ojas/linuxdataset/yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

def calculate_shooting_angle(D, V, delta_H=1.22, g=9.8, phi_min=45, phi_max=55):
    """
    Calculate the shooting angle (theta) for a projectile if landing angle is not too steep.

    Parameters:
    - D (float): Horizontal distance to the target (meters).
    - V (float): Initial velocity (m/s).
    - delta_H (float): Height difference (H_target - H_robot) (meters).
    - g (float): Gravity acceleration (m/s^2, default 9.8).
    - phi_min (float): Minimum acceptable landing angle (degrees, default 45).
    - phi_max (float): Maximum acceptable landing angle (degrees, default 55).

    Returns:
    - tuple: (theta_deg, message)
      - theta_deg (float or None): Larger shooting angle in degrees if shot is feasible
        and phi is in [phi_min, phi_max].
      - message (str): Explanation if theta is None (infeasible or suboptimal phi).
    """
    V_squared = V ** 2
    g_D_squared = g * D ** 2
    discriminant = (D ** 2) - (2 * g * D ** 2 * delta_H / V_squared) - (g ** 2 * D ** 4 / V_squared ** 2)

    if discriminant < 0:
        return None, f"Shot is infeasible with V={V} m/s at D={D} m, delta_H={delta_H} m. Move closer."

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
        return None, f"Landing angle ({phi_deg:.2f}°) too shallow. Move closer to increase phi."
    elif phi_deg > phi_max:
        return None, f"Landing angle ({phi_deg:.2f}°) too steep. Move farther to reduce phi."

    return round(theta_larger, 2), "Landing angle is optimal."

def send_angle_to_esp32(angle, port='/dev/ttyUSB0', baudrate=115200):
    """
    Send the calculated shooting angle to the ESP32 via serial communication.

    Parameters:
    - angle (float): The shooting angle to send.
    - port (str): The serial port to which the ESP32 is connected (default '/dev/ttyUSB0').
    - baudrate (int): The baud rate for serial communication (default 115200).
    """
    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            ser.write(f"{angle}\n".encode())
            print(f"Angle {angle}° sent to ESP32.")
    except serial.SerialException as e:
        print(f"Error: Could not send angle to ESP32. {e}")

def load_yolov5_model(weights_path='/home/ojas/linuxdataset/yolov5/runs/train/linuxdataset/weights/best.pt'):
    """
    Load the custom YOLOv5L model.

    Parameters:
    - weights_path (str): Path to the YOLOv5L model weights.

    Returns:
    - model: Loaded YOLOv5 model.
    """
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DetectMultiBackend(weights_path, device=device)
        model.eval()
        print(f"YOLOv5 model loaded on {device}")
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        exit()

def detect_basketball_net(model, img, conf_thres=0.6, iou_thres=0.45):
    """
    Detect basketball net in the image using YOLOv5.

    Parameters:
    - model: Loaded YOLOv5 model.
    - img: Input image (numpy array, BGR format).
    - conf_thres (float): Confidence threshold for detections.
    - iou_thres (float): IoU threshold for non-max suppression.

    Returns:
    - boxes: List of bounding boxes [x1, y1, x2, y2, conf, cls].
    """
    img_resized, ratio, (dw, dh) = letterbox(img, new_shape=(640, 640))
    img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img_resized = np.ascontiguousarray(img_resized, dtype=np.float32) / 255.0
    img_resized = torch.from_numpy(img_resized).unsqueeze(0).to(model.device)

    pred = model(img_resized)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False)
    
    boxes = []
    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img_resized.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                boxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf), int(cls)])
    return boxes

def main():
    # Initialize RealSense pipeline
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("RealSense camera initialized.")
    except Exception as e:
        print(f"Error: Could not initialize RealSense camera. {e}")
        exit()

    # Load YOLOv5 model
    model = load_yolov5_model()

    # Constants for shooting angle calculation
    delta_H = 1.22  # Height difference (H_target - H_robot) in meters
    V = 8.5        # Initial velocity in m/s

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert frames to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Detect basketball net
            boxes = detect_basketball_net(model, color_image)
            
            # Process the first detected basketball net
            D = None
            theta = None
            message = "No basketball net detected."
            target_pixel = None
            
            if boxes:
                # Get the bounding box with highest confidence
                box = max(boxes, key=lambda x: x[4])  # Sort by confidence
                x1, y1, x2, y2, conf, _ = box
                
                # Calculate middle x and top y pixel
                pixel_x = (x1 + x2) // 2
                pixel_y = y1
                
                # Draw bounding box and target pixel
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(color_image, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
                
                # Get depth value (in millimeters)
                depth_value = depth_image[pixel_y, pixel_x]
                if depth_value > 0:
                    D = depth_value / 1000.0  # Convert to meters
                    # Calculate shooting angle
                    theta, message = calculate_shooting_angle(D, V, delta_H)
                    if theta is not None:
                        # Adjust angle for ESP32 (assuming it expects angle relative to vertical)
                        angle_to_send = max(0, 90 - theta)  # Ensure non-negative angle
                        send_angle_to_esp32(angle_to_send)
                else:
                    message = "Invalid depth value at target pixel."
                target_pixel = (pixel_x, pixel_y)

            # Display results
            print(f"Distance: {D:.2f} m, {message}" if D else message)
            if theta is not None:
                print(f"Shooting angle: {theta}°, Sent to ESP32: {max(0, 90 - theta)}°")
                
            # Add text to display
            cv2.putText(color_image, f"Distance: {D:.2f} m" if D else message, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if theta is not None:
                cv2.putText(color_image, f"Angle: {theta}°", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(color_image, message, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display
            cv2.imshow('Basketball Net Detection', color_image)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program interrupted by user.")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
