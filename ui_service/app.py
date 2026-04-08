import streamlit as st
import cv2
import tempfile
import numpy as np
import pandas as pd
import time
from ultralytics import YOLO

# ==============================
# Page Config
# ==============================
st.set_page_config(page_title="EAGLEVISION Demo", layout="wide")
st.title("Equipment Activity Monitoring")

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ==============================
# Activity Detection
# ==============================
def classify_activity(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_motion = np.mean(mag)

    return "ACTIVE" if avg_motion > 0.7 else "INACTIVE"

# ==============================
# Upload Video
# ==============================
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

video_placeholder = st.empty()
metrics_placeholder = st.empty()
table_placeholder = st.empty()

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    prev_gray = None

    # Store analytics
    machine_stats = {}

    st.success("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        results = model.track(frame, persist=True, classes=[0, 1])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = box
                eq_id = f"EX-{track_id}"

                # Init machine
                if eq_id not in machine_stats:
                    machine_stats[eq_id] = {
                        "tracked_frames": 0,
                        "active_frames": 0,
                        "state": "INACTIVE"
                    }

                machine_stats[eq_id]["tracked_frames"] += 1

                # Clamp ROI
                h, w = gray.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                state = "INACTIVE"

                if prev_gray is not None:
                    roi_curr = gray[y1:y2, x1:x2]
                    roi_prev = prev_gray[y1:y2, x1:x2]

                    if roi_curr.shape == roi_prev.shape and roi_curr.size > 0:
                        state = classify_activity(roi_prev, roi_curr)

                        if state == "ACTIVE":
                            machine_stats[eq_id]["active_frames"] += 1

                machine_stats[eq_id]["state"] = state

                # Draw box
                color = (0, 255, 0) if state == "ACTIVE" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{eq_id}: {state}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        prev_gray = gray

        # ==============================
        # UI Updates
        # ==============================

        video_placeholder.image(frame, channels="BGR")

        # Metrics
        with metrics_placeholder.container():
            cols = st.columns(len(machine_stats) if machine_stats else 1)

            for i, (eq_id, stats) in enumerate(machine_stats.items()):
                tracked_sec = stats["tracked_frames"] / fps
                active_sec = stats["active_frames"] / fps
                idle_sec = tracked_sec - active_sec
                util = (active_sec / tracked_sec) * 100 if tracked_sec > 0 else 0

                with cols[i]:
                    st.subheader(eq_id)
                    st.metric("Activity", stats["state"])
                    st.metric("Utilization %", f"{util:.2f}%")
                    st.text(f"Active: {active_sec:.2f}s")
                    st.text(f"Idle: {idle_sec:.2f}s")

        # Table
        with table_placeholder.container():
            df = []
            for eq_id, stats in machine_stats.items():
                tracked_sec = stats["tracked_frames"] / fps
                active_sec = stats["active_frames"] / fps
                idle_sec = tracked_sec - active_sec
                util = (active_sec / tracked_sec) * 100 if tracked_sec > 0 else 0

                df.append({
                    "Equipment ID": eq_id,
                    "Activity": stats["state"],
                    "Active Time (s)": round(active_sec, 2),
                    "Idle Time (s)": round(idle_sec, 2),
                    "Utilization (%)": round(util, 2)
                })

            st.dataframe(pd.DataFrame(df), use_container_width=True)

        # Control speed
        time.sleep(0.03)

    cap.release()
    st.success("Processing finished!")