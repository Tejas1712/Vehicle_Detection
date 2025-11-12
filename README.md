# Vehicle_Detection

📘 Overview

Traffic congestion is one of the most critical issues in urban areas, leading to time delays, fuel wastage, and increased pollution. This project leverages Artificial Intelligence to automate vehicle detection, traffic density estimation, and flow prediction using deep learning models.

The system combines Mask R-CNN for object detection and segmentation of vehicles from live camera feeds and Attention-based LSTM for temporal traffic flow prediction, enabling data-driven traffic signal control and congestion management.

🎯 Objectives

Detect vehicles in real-time using the Mask R-CNN model.

Classify and count vehicles (cars, trucks, buses, bikes, etc.).

Predict traffic density patterns using Attention LSTM.

Provide intelligent decision-making for adaptive signal control.

<img width="557" height="860" alt="image" src="https://github.com/user-attachments/assets/ba2b1ced-f91a-4ce9-9a46-a700c565a55a" />


🧠 Model Details
🔹 1. Mask R-CNN (Detection Model)

Base Model: mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8

Framework: TensorFlow 2.x Object Detection API

Task: Detect and segment vehicles in video frames

Training Dataset: Custom dataset (500+ annotated images using LabelImg)

Annotation Format: Pascal VOC → TFRecord

Evaluation Metrics: mAP, IoU, Precision-Recall Curve

🔹 2. Attention LSTM (Prediction Model)

Framework: TensorFlow / Keras

Input: Vehicle count sequences over time

Output: Predicted traffic density for next interval

Features: Incorporates attention weights for time-dependent features

Evaluation Metrics: RMSE, MAE, R² Score

🧾 Dataset
Type	Source	Description
📸 Images	Captured from CCTV traffic footage	Used for vehicle detection
🧾 Labels	Annotated using LabelImg	Bounding boxes and segmentation masks
📈 Time-Series	Extracted from vehicle counts	Used for LSTM-based prediction
⚙️ Installation
1️⃣ Clone Repository
git clone https://github.com/Tejas1712/Traffic-Management-AI.git
cd Traffic-Management-AI

download dataset :- https://drive.google.com/drive/folders/1GCgCvpnYiLcYN5vwNeoYycXfxt2Ni5tl?usp=drive_link

2️⃣ Create Conda Environment
conda create -n traffic_ai python=3.8.5
conda activate traffic_ai

3️⃣ Install Dependencies
pip install tensorflow==2.8
pip install opencv-python
pip install labelImg
pip install matplotlib
pip install pandas numpy

4️⃣ Set Up TensorFlow Object Detection API

Install Protobuf Compiler

Run:

protoc object_detection/protos/*.proto --python_out=.


Add models/research to Python PATH:

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

🚀 Training
🔹 Train Mask R-CNN
python model_main_tf2.py \
  --model_dir=models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8 \
  --pipeline_config_path=data/models/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8/pipeline.config

🔹 Train Attention LSTM
python train_lstm.py

📊 Evaluation Metrics
Model	Metric	Result
Mask R-CNN	mAP	0.86
Mask R-CNN	IoU	0.82
LSTM	RMSE	5.12
LSTM	R² Score	0.91
🖼️ Results

Real-time vehicle detection achieved with ~18 FPS on GPU.

Accurate prediction of traffic density 10 seconds ahead.

Potential for dynamic traffic signal control based on congestion level.

💡 Future Scope

Integrate IoT sensors for real-time weather and pollution data.

Deploy on edge devices (e.g., Jetson Nano).

Implement reinforcement learning for adaptive signal optimization.

Develop a web-based dashboard for live monitoring and analytics.

📚 References

TensorFlow Object Detection API: https://github.com/tensorflow/models

COCO Dataset: https://cocodataset.org

LabelImg: https://github.com/tzutalin/labelImg
