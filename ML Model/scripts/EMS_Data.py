import numpy as np
import pandas as pd

num_records = 1_000_000

# Synthetic accident locations (latitude, longitude)
latitudes = np.random.uniform(12.8, 13.1, num_records)
longitudes = np.random.uniform(77.4, 77.7, num_records)

# Random timestamps over a one year period
timestamps = pd.date_range('2024-01-01', periods=num_records, freq='min').to_series().sample(n=num_records, replace=True).reset_index(drop=True)

# Severity classes with probabilities
severities = np.random.choice(['Low', 'Medium', 'High'], num_records, p=[0.6, 0.3, 0.1])

# Simulated ambulance locations
ambulance_lat = np.random.uniform(12.8, 13.1, 1000)
ambulance_lon = np.random.uniform(77.4, 77.7, 1000)

def min_distance(lat, lon):
    return np.min(np.sqrt((ambulance_lat - lat) ** 2 + (ambulance_lon - lon) ** 2))

distances = [min_distance(lat, lon) for lat, lon in zip(latitudes, longitudes)]
response_times = np.array(distances) * 50 + np.random.normal(0, 5, num_records)  # minutes

df_dispatch = pd.DataFrame({
    'latitude': latitudes,
    'longitude': longitudes,
    'timestamp': timestamps,
    'severity': severities,
    'nearest_ambulance_distance': distances,
    'estimated_response_time': response_times
})

df_dispatch.to_csv('synthetic_dispatch_data.csv', index=False)  # Save CSV
