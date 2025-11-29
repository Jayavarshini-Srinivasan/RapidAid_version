"""
Sample Data Generator
Creates a sample accident dataset for testing purposes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_samples=1000, output_file='sample_accident_data.csv'):
    """
    Generate sample accident data for testing
    
    Args:
        n_samples: Number of samples to generate
        output_file: Output CSV file path
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate dates
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 1000), 
                                     hours=random.randint(0, 23),
                                     minutes=random.randint(0, 59)) 
             for _ in range(n_samples)]
    
    # Generate severity (target variable)
    severity_levels = ['Low', 'Medium', 'Critical']
    severity = np.random.choice(severity_levels, n_samples, p=[0.5, 0.3, 0.2])
    
    # Generate weather conditions
    weather_conditions = ['Clear', 'Rainy', 'Foggy', 'Snowy', 'Windy']
    weather = np.random.choice(weather_conditions, n_samples, p=[0.4, 0.3, 0.1, 0.1, 0.1])
    
    # Generate road conditions
    road_conditions = ['Dry', 'Wet', 'Icy', 'Slippery']
    road_condition = np.random.choice(road_conditions, n_samples, p=[0.5, 0.3, 0.1, 0.1])
    
    # Generate accident causes
    accident_causes = ['Speeding', 'Distraction', 'Weather', 'Vehicle Failure', 'Other']
    accident_cause = np.random.choice(accident_causes, n_samples, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    
    # Generate location data (example: coordinates around a city center)
    center_lat, center_lon = 40.7128, -74.0060  # Example: NYC
    latitude = np.random.normal(center_lat, 0.1, n_samples)
    longitude = np.random.normal(center_lon, 0.1, n_samples)
    
    # Generate numerical features
    num_vehicles = np.random.randint(1, 5, n_samples)
    speed_limit = np.random.choice([25, 35, 45, 55, 65], n_samples)
    road_type = np.random.choice(['Highway', 'Street', 'Avenue', 'Road'], n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date_time': dates,
        'severity': severity,
        'weather': weather,
        'road_condition': road_condition,
        'accident_cause': accident_cause,
        'latitude': latitude,
        'longitude': longitude,
        'num_vehicles': num_vehicles,
        'speed_limit': speed_limit,
        'road_type': road_type
    })
    
    # Add some correlation between features and severity
    # Critical accidents more likely with bad weather
    critical_mask = df['severity'] == 'Critical'
    df.loc[critical_mask, 'weather'] = np.random.choice(
        ['Rainy', 'Foggy', 'Snowy'], 
        size=critical_mask.sum(), 
        p=[0.4, 0.3, 0.3]
    )
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Sample data generated: {output_file}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nSeverity distribution:")
    print(df['severity'].value_counts())
    
    return df

if __name__ == "__main__":
    generate_sample_data(n_samples=1000)

