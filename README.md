# 🚧 EAGLEVISION – Equipment Activity Monitoring System

## 📌 Project Overview

EAGLEVISION is a computer vision system designed to monitor heavy equipment (e.g., excavators, trucks) and analyze their operational activity in real-time.

The system performs:

* Object detection & tracking
* Activity classification (ACTIVE / INACTIVE)
* Utilization analytics (active time, idle time, efficiency)
* Real-time visualization via a Streamlit dashboard

---

## ⚠️ Problem Faced

Initially, we used **YOLOv8 pre-trained weights** directly for detection.

However, we encountered a major issue:

* The model was detecting **excavators without their arms**
* This caused inaccurate motion analysis because:

  * The arm is the main moving part
  * Missing it → incorrect activity classification

---

## 🛠️ Solution

### 1. Custom Dataset Annotation

To fix this:

* We collected sample videos of excavators
* Used **Roboflow** to:

  * Annotate excavators including their **arms**
  * Improve bounding box quality
  * Ensure full object coverage

---

### 2. Model Retraining

* Trained a custom YOLOv8 model using the annotated dataset
* Exported the best-performing weights (`best.pt`)
* This significantly improved:

  * Detection accuracy
  * Tracking stability
  * ROI quality for motion analysis

---

### 3. Activity Classification using Optical Flow

Instead of training a separate activity model, we used:

**Dense Optical Flow (Farneback)** to:

* Measure motion between consecutive frames
* Compute average motion magnitude

#### Logic:

* Low motion → `INACTIVE`
* High motion → `ACTIVE`

This approach is:

* Lightweight
* Real-time friendly
* Effective for machinery monitoring

---

### 4. System Pipeline

```
Video Input
   ↓
YOLOv8 Detection + Tracking
   ↓
ROI Extraction per Machine
   ↓
Optical Flow Calculation
   ↓
Activity Classification (ACTIVE / INACTIVE)
   ↓
Kafka Streaming (JSON Telemetry)
   ↓
Streamlit Dashboard Visualization
```

---

## 📊 Output Metrics

For each machine:

* Activity Status (ACTIVE / INACTIVE)
* Total Active Time
* Total Idle Time
* Utilization %

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd <project-folder>
```

---

### 2. Install Requirements

Make sure you have Python 3.9+ installed, then:

```bash
pip install -r requirements.txt
```

---

### 3. Install Docker (Required for Kafka)

Download and install Docker:
👉 https://www.docker.com/products/docker-desktop

Verify installation:

```bash
docker --version
```

---

### 4. Run Kafka using Docker

If you have a `docker-compose.yml` file:

```bash
docker-compose up -d
```

Or manually run Kafka & Zookeeper:

```bash
docker run -d --name zookeeper -p 2181:2181 zookeeper
docker run -d --name kafka -p 9092:9092 \
-e KAFKA_ZOOKEEPER_CONNECT=host.docker.internal:2181 \
-e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
-e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
confluentinc/cp-kafka
```

---

### 5. Run the Computer Vision Service

```bash
python main.py
```

This will:

* Process the video
* Detect equipment
* Classify activity
* Send data to Kafka

---

### 6. Run Streamlit Dashboard

```bash
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 📁 Project Structure

```
├── main.py                # CV pipeline (YOLO + Optical Flow + Kafka)
├── app.py                 # Streamlit dashboard
├── best.pt                # Trained YOLO model
├── requirements.txt
├── docker-compose.yml     # Kafka setup
└── README.md
```

---

## 💡 Future Improvements

* Train a deep learning model for activity classification (instead of optical flow)
* Add pose/keypoint detection for excavator arm tracking
* Deploy as microservices (FastAPI + Kafka + UI)
* Add historical analytics & charts

---

## 👨‍💻 Authors

* Abdallah Hassan

---

## ✅ Summary

This project demonstrates how combining:

* Custom-trained detection models
* Motion analysis (optical flow)
* Streaming systems (Kafka)
* Visualization tools (Streamlit)

can build a **real-time intelligent monitoring system for heavy equipment**.
