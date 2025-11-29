import numpy as np
import pandas as pd

def generate_sensor_event(accident=False, length=100):
    # Simulate accelerometer readings for one time window
    if not accident:
        # Normal driving: stable low variance data centered around 0 (x, y) and gravity (z)
        accel_x = np.random.normal(0, 0.5, length)
        accel_y = np.random.normal(0, 0.5, length)
        accel_z = np.random.normal(9.8, 0.2, length)  # gravity
    else:
        # Accident event: sudden spike in accel_x as signature of impact
        accel_x = np.random.normal(0, 0.5, length)
        spike_pos = np.random.randint(40, 60)
        accel_x[spike_pos:spike_pos + 5] += np.random.normal(20, 5, 5)
        accel_y = np.random.normal(0, 0.5, length)
        accel_z = np.random.normal(9.8, 0.2, length)
    return pd.DataFrame({
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'accident': accident
    })

# Generate half accident and half normal data windows for total 50,000 records
num_samples = 50_000
accident_count = num_samples // 2

frames = []
for _ in range(accident_count):
    frames.append(generate_sensor_event(accident=True))
for _ in range(accident_count):
    frames.append(generate_sensor_event(accident=False))

sensor_data = pd.concat(frames, ignore_index=True)

# Assign event IDs to group 100 samples per record/window
sensor_data['event_id'] = sensor_data.index // 100

sensor_data.head()
