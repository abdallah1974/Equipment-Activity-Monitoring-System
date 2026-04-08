import cv2
import numpy as np
import json
import time
from ultralytics import YOLO
from confluent_kafka import Producer

# ==============================
# Configuration
# ==============================
KAFKA_BROKER = 'localhost:9092'
TOPIC = 'equipment_telemetry'
VIDEO_PATH = r'C:\Users\3bdool\Documents\GitHub\Eagle_Vsion_Task\cv_service\test_video.mp4'

model = YOLO('best.pt')
producer = Producer({'bootstrap.servers': KAFKA_BROKER})

# ==============================
# Simplified Activity Detection
# ==============================
def classify_activity(roi_prev, roi_curr):
    """
    Returns only ACTIVE or INACTIVE based on motion magnitude.
    """
    flow = cv2.calcOpticalFlowFarneback(
        roi_prev, roi_curr, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    avg_motion = np.mean(mag)

    # Simple threshold
    if avg_motion > 0.3:
        return "ACTIVE"
    else:
        return "INACTIVE"


# ==============================
# Video Processing
# ==============================
def process_video():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_id = 0
    prev_gray_frame = None
    equipment_state = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model.track(frame, persist=True, classes=[0, 1])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                eq_id = f"EX-{track_id}"

                if eq_id not in equipment_state:
                    equipment_state[eq_id] = {
                        "tracked_frames": 0,
                        "active_frames": 0
                    }

                equipment_state[eq_id]["tracked_frames"] += 1

                curr_state = "INACTIVE"

                if prev_gray_frame is not None:
                    # Clamp ROI داخل الفريم
                    h, w = gray_frame.shape
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    if x2 > x1 and y2 > y1:
                        roi_curr = gray_frame[y1:y2, x1:x2]
                        roi_prev = prev_gray_frame[y1:y2, x1:x2]

                        if roi_curr.shape == roi_prev.shape and roi_curr.size > 0:
                            curr_state = classify_activity(roi_prev, roi_curr)

                            if curr_state == "ACTIVE":
                                equipment_state[eq_id]["active_frames"] += 1

                # Time analytics
                tracked_sec = equipment_state[eq_id]["tracked_frames"] / fps
                active_sec = equipment_state[eq_id]["active_frames"] / fps
                idle_sec = tracked_sec - active_sec
                util_pct = (active_sec / tracked_sec) * 100 if tracked_sec > 0 else 0

                payload = {
                    "frame_id": frame_id,
                    "equipment_id": eq_id,
                    "equipment_class": "excavator",
                    "timestamp": time.strftime('%H:%M:%S.000', time.gmtime(tracked_sec)),
                    "current_state": curr_state,
                    "current_activity": curr_state,  # same value
                    "motion_source": "optical_flow",
                    "time_analytics": {
                        "total_tracked_seconds": round(tracked_sec, 2),
                        "total_active_seconds": round(active_sec, 2),
                        "total_idle_seconds": round(idle_sec, 2),
                        "utilization_percent": round(util_pct, 2)
                    }
                }

                producer.produce(TOPIC, value=json.dumps(payload))

                # Visualization
                color = (0, 255, 0) if curr_state == "ACTIVE" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{eq_id}: {curr_state}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.imshow("Eaglevision Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray_frame = gray_frame

    cap.release()
    cv2.destroyAllWindows()
    producer.flush()


if __name__ == "__main__":
    process_video()