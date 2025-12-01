🏀 Autonomous Basketball Shooting Bot (DD Robocon 2025)

Achievement: Secured All India Rank (AIR) 8 and the N-Motion Award at DD Robocon 2025.

📖 Overview

This repository contains the vision and control algorithm for an autonomous basketball-shooting robot. The system uses Computer Vision (YOLO) to detect the basketball hoop and an Intel RealSense Depth Camera to calculate the precise distance.

Using these inputs, a physics engine calculates the optimal projectile shooting angle in real-time and transmits it to an ESP32 microcontroller to adjust the shooting mechanism.

⚙️ How It Works

Detection: A custom-trained YOLO model identifies the basketball net in the video frame.

Depth Estimation: The center pixel of the bounding box is mapped to the RealSense depth map to get the distance ($D$) in millimeters.

Physics Solver: The system solves the inverse kinematics equation for projectile motion:


$$\theta = \arctan\left(\frac{v^2 \pm \sqrt{v^4 - g(gx^2 + 2yv^2)}}{gx}\right)$$


It selects the optimal angle $\theta$ (between 45°-55°) to ensure a high probability of scoring.

Actuation: The angle is sent via UART (Serial) to an ESP32, which controls the linear actuator/stepper motor.

🛠️ Hardware Requirements

Camera: Intel RealSense D435i (or compatible depth camera)

Compute: NVIDIA Jetson / Laptop with CUDA support (recommended)

Microcontroller: ESP32 (for motor control)

Robot: Differential drive base with adjustable flywheel shooter

📦 Installation

Clone the repository:

git clone [https://github.com/your-username/robocon-shooting-bot.git](https://github.com/your-username/Robocon-Shooting-Bot.git)
cd Robocon-Shooting-Bot


Install dependencies:

pip install -r requirements.txt


Setup YOLO:
This project requires the YOLOv5 repository structure.

git clone [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)


Add Weights:
Place your trained model weights (best.pt) inside a weights/ folder.

🚀 Usage

Run the main script:

python main.py --weights weights/best.pt --port /dev/ttyUSB0


Arguments:

--weights: Path to your custom trained .pt file (default: weights/best.pt)

--port: Serial port for ESP32 (default: /dev/ttyUSB0)

--velocity: Initial velocity of the ball in m/s (default: 8.5)

📂 File Structure

main.py: Core logic for detection, depth sensing, and angle calculation.

requirements.txt: Python dependencies.

🏆 Awards

N-Motion Award: Recognized for smooth and accurate autonomous navigation and shooting.

AIR 8: Top 10 finish among national competitors at DD Robocon 2025.
