CREATE TABLE IF NOT EXISTS equipment_telemetry (
    id SERIAL PRIMARY KEY,
    frame_id INT,
    equipment_id VARCHAR(50),
    equipment_class VARCHAR(50),
    timestamp VARCHAR(50),
    current_state VARCHAR(20),
    current_activity VARCHAR(50),
    motion_source VARCHAR(50),
    total_tracked_seconds FLOAT,
    total_active_seconds FLOAT,
    total_idle_seconds FLOAT,
    utilization_percent FLOAT
);