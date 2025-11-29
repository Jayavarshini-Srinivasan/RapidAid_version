"""
Enhanced Feature Engineering Module
Implements comprehensive feature extraction including:
- Temporal features (day_of_week, hour_of_day, is_weekend, is_rush_hour)
- Spatial grid-based zones with accident aggregation
- Sensor time-window features with rolling statistics
- Frequency-domain features using FFT
- Lag features for time-series prediction
"""

import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineer:
    """Enhanced feature engineering with advanced transformations"""
    
    def __init__(self, df):
        """
        Initialize the EnhancedFeatureEngineer
        
        Args:
            df: pandas DataFrame with raw data
        """
        self.df = df.copy()
        self.grid_zones = None
        
    # ==================== TEMPORAL FEATURES ====================
    
    def extract_temporal_features(self, timestamp_column=None):
        """
        Extract comprehensive temporal features from timestamp columns
        
        Args:
            timestamp_column: Name of timestamp column. If None, auto-detect
        
        Returns:
            DataFrame with temporal features added
        """
        if timestamp_column is None:
            # Auto-detect timestamp columns
            timestamp_candidates = ['timestamp', 'datetime', 'date', 'time', 'Timestamp', 'DateTime']
            timestamp_column = None
            for col in timestamp_candidates:
                if col in self.df.columns:
                    timestamp_column = col
                    break
            
            # Try to find datetime columns
            if timestamp_column is None:
                datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
                if len(datetime_cols) > 0:
                    timestamp_column = datetime_cols[0]
        
        if timestamp_column is None or timestamp_column not in self.df.columns:
            print("Warning: No timestamp column found. Skipping temporal feature extraction.")
            return self.df
        
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_column]):
            self.df[timestamp_column] = pd.to_datetime(self.df[timestamp_column], errors='coerce')
        
        # Extract basic temporal features
        self.df['day_of_week'] = self.df[timestamp_column].dt.dayofweek  # 0=Monday, 6=Sunday
        self.df['hour_of_day'] = self.df[timestamp_column].dt.hour
        self.df['day_of_month'] = self.df[timestamp_column].dt.day
        self.df['month'] = self.df[timestamp_column].dt.month
        self.df['quarter'] = self.df[timestamp_column].dt.quarter
        self.df['year'] = self.df[timestamp_column].dt.year
        self.df['week_of_year'] = self.df[timestamp_column].dt.isocalendar().week
        
        # Binary temporal features
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)  # Saturday=5, Sunday=6
        self.df['is_weekday'] = (self.df['day_of_week'] < 5).astype(int)
        
        # Rush hour definition: 7-9 AM and 5-7 PM
        self.df['is_rush_hour'] = (
            ((self.df['hour_of_day'] >= 7) & (self.df['hour_of_day'] <= 9)) |
            ((self.df['hour_of_day'] >= 17) & (self.df['hour_of_day'] <= 19))
        ).astype(int)
        
        # Time of day categories
        self.df['is_morning'] = ((self.df['hour_of_day'] >= 6) & (self.df['hour_of_day'] < 12)).astype(int)
        self.df['is_afternoon'] = ((self.df['hour_of_day'] >= 12) & (self.df['hour_of_day'] < 18)).astype(int)
        self.df['is_evening'] = ((self.df['hour_of_day'] >= 18) & (self.df['hour_of_day'] < 22)).astype(int)
        self.df['is_night'] = ((self.df['hour_of_day'] >= 22) | (self.df['hour_of_day'] < 6)).astype(int)
        
        # Cyclical encoding for periodic features
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour_of_day'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour_of_day'] / 24)
        self.df['day_of_week_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_of_week_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        print(f"✓ Extracted temporal features from '{timestamp_column}'")
        print(f"  Added: day_of_week, hour_of_day, is_weekend, is_rush_hour, and cyclical encodings")
        
        return self.df
    
    # ==================== SPATIAL GRID FEATURES ====================
    
    def create_spatial_grid(self, lat_col=None, lon_col=None, grid_size=0.01, 
                           severity_col=None, create_aggregates=True):
        """
        Create grid-based zones from lat/lon coordinates and aggregate accident data
        
        Args:
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            grid_size: Size of grid cells in degrees (default 0.01 ≈ 1km)
            severity_col: Name of severity column for aggregation
            create_aggregates: Whether to create aggregated statistics per zone
        
        Returns:
            DataFrame with grid zone assignments and aggregated features
        """
        # Auto-detect lat/lon columns
        if lat_col is None:
            lat_candidates = ['latitude', 'lat', 'Latitude', 'LAT', 'y']
            for col in lat_candidates:
                if col in self.df.columns:
                    lat_col = col
                    break
        
        if lon_col is None:
            lon_candidates = ['longitude', 'lon', 'lng', 'Longitude', 'LONG', 'LON', 'x']
            for col in lon_candidates:
                if col in self.df.columns:
                    lon_col = col
                    break
        
        if lat_col is None or lon_col is None or lat_col not in self.df.columns or lon_col not in self.df.columns:
            print("Warning: Latitude/Longitude columns not found. Skipping spatial grid creation.")
            return self.df
        
        # Remove invalid coordinates
        valid_mask = (
            (self.df[lat_col] >= -90) & (self.df[lat_col] <= 90) &
            (self.df[lon_col] >= -180) & (self.df[lon_col] <= 180)
        )
        self.df = self.df[valid_mask].copy()
        
        # Create grid indices
        self.df['grid_lat'] = (self.df[lat_col] / grid_size).astype(int)
        self.df['grid_lon'] = (self.df[lon_col] / grid_size).astype(int)
        self.df['grid_zone'] = self.df['grid_lat'].astype(str) + '_' + self.df['grid_lon'].astype(str)
        
        if create_aggregates:
            # Calculate grid center coordinates
            grid_centers = self.df.groupby('grid_zone').agg({
                lat_col: 'mean',
                lon_col: 'mean'
            }).reset_index()
            grid_centers.columns = ['grid_zone', 'grid_center_lat', 'grid_center_lon']
            self.df = self.df.merge(grid_centers, on='grid_zone', how='left')
            
            # Aggregate accident counts per zone
            zone_counts = self.df.groupby('grid_zone').size().reset_index(name='accident_count_in_zone')
            self.df = self.df.merge(zone_counts, on='grid_zone', how='left')
            
            # Aggregate severity statistics if severity column exists
            if severity_col and severity_col in self.df.columns:
                # Encode severity if categorical
                if self.df[severity_col].dtype == 'object':
                    severity_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 3}
                    self.df['severity_encoded'] = self.df[severity_col].map(severity_map).fillna(1)
                else:
                    self.df['severity_encoded'] = self.df[severity_col]
                
                zone_severity = self.df.groupby('grid_zone').agg({
                    'severity_encoded': ['mean', 'std', 'min', 'max', 'count']
                }).reset_index()
                zone_severity.columns = ['grid_zone', 'zone_avg_severity', 'zone_std_severity', 
                                        'zone_min_severity', 'zone_max_severity', 'zone_total_accidents']
                self.df = self.df.merge(zone_severity, on='grid_zone', how='left')
                
                # Fill NaN values
                self.df['zone_avg_severity'] = self.df['zone_avg_severity'].fillna(1)
                self.df['zone_std_severity'] = self.df['zone_std_severity'].fillna(0)
            
            # Calculate distance from grid center
            self.df['distance_from_grid_center'] = np.sqrt(
                (self.df[lat_col] - self.df['grid_center_lat'])**2 +
                (self.df[lon_col] - self.df['grid_center_lon'])**2
            )
        
        print(f"✓ Created spatial grid with size {grid_size} degrees")
        print(f"  Total grid zones: {self.df['grid_zone'].nunique()}")
        
        return self.df
    
    # ==================== SENSOR TIME-WINDOW FEATURES ====================
    
    def create_sensor_rolling_features(self, sensor_columns, window_sizes=[5, 10], 
                                      timestamp_column=None, freq='1S'):
        """
        Create time-window features with rolling statistics for sensor data
        
        Args:
            sensor_columns: List of sensor column names (e.g., ['acceleration_x', 'acceleration_y'])
            window_sizes: List of window sizes in seconds (e.g., [5, 10])
            timestamp_column: Name of timestamp column for time-based rolling
            freq: Frequency string for resampling (default '1S' for 1 second)
        
        Returns:
            DataFrame with rolling statistics features
        """
        if timestamp_column is None:
            timestamp_candidates = ['timestamp', 'datetime', 'time']
            for col in timestamp_candidates:
                if col in self.df.columns:
                    timestamp_column = col
                    break
        
        if timestamp_column is None or timestamp_column not in self.df.columns:
            print("Warning: No timestamp column found. Using index-based rolling.")
            use_time_index = False
        else:
            use_time_index = True
            if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_column]):
                self.df[timestamp_column] = pd.to_datetime(self.df[timestamp_column], errors='coerce')
            self.df = self.df.sort_values(timestamp_column).reset_index(drop=True)
            self.df = self.df.set_index(timestamp_column)
        
        # Find available sensor columns
        available_sensors = [col for col in sensor_columns if col in self.df.columns]
        if len(available_sensors) == 0:
            print("Warning: No sensor columns found. Skipping rolling features.")
            if use_time_index:
                self.df = self.df.reset_index()
            return self.df
        
        print(f"Creating rolling features for sensors: {available_sensors}")
        
        for sensor_col in available_sensors:
            for window_size in window_sizes:
                if use_time_index:
                    # Time-based rolling window
                    window_str = f'{window_size}S'
                    rolling = self.df[sensor_col].rolling(window=window_str, min_periods=1)
                else:
                    # Index-based rolling window (assume 1 sample per second)
                    rolling = self.df[sensor_col].rolling(window=window_size, min_periods=1)
                
                # Rolling statistics
                self.df[f'{sensor_col}_rolling_mean_{window_size}s'] = rolling.mean()
                self.df[f'{sensor_col}_rolling_std_{window_size}s'] = rolling.std()
                self.df[f'{sensor_col}_rolling_min_{window_size}s'] = rolling.min()
                self.df[f'{sensor_col}_rolling_max_{window_size}s'] = rolling.max()
                self.df[f'{sensor_col}_rolling_median_{window_size}s'] = rolling.median()
                
                # Rate of change
                self.df[f'{sensor_col}_rolling_diff_{window_size}s'] = rolling.apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0)
        
        if use_time_index:
            self.df = self.df.reset_index()
        
        print(f"✓ Created rolling features with windows: {window_sizes} seconds")
        
        return self.df
    
    # ==================== FREQUENCY-DOMAIN FEATURES (FFT) ====================
    
    def extract_frequency_features(self, sensor_columns, sampling_rate=1.0, 
                                  n_fft=256, timestamp_column=None):
        """
        Generate frequency-domain features using FFT to detect sudden changes
        
        Args:
            sensor_columns: List of sensor column names
            sampling_rate: Sampling rate in Hz (default 1.0 for 1 sample per second)
            n_fft: Number of FFT points (default 256)
            timestamp_column: Name of timestamp column for time-series ordering
        
        Returns:
            DataFrame with frequency-domain features
        """
        if timestamp_column and timestamp_column in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_column]):
                self.df[timestamp_column] = pd.to_datetime(self.df[timestamp_column], errors='coerce')
            self.df = self.df.sort_values(timestamp_column).reset_index(drop=True)
        
        available_sensors = [col for col in sensor_columns if col in self.df.columns]
        if len(available_sensors) == 0:
            print("Warning: No sensor columns found. Skipping FFT features.")
            return self.df
        
        print(f"Extracting frequency features for sensors: {available_sensors}")
        
        for sensor_col in available_sensors:
            # Remove NaN values for FFT
            sensor_data = self.df[sensor_col].ffill().bfill().values
            
            # Apply FFT
            if len(sensor_data) >= n_fft:
                # Use sliding window FFT
                fft_features = []
                for i in range(len(sensor_data) - n_fft + 1):
                    window = sensor_data[i:i+n_fft]
                    fft_vals = np.abs(fft(window))
                    fft_freqs = fftfreq(n_fft, 1/sampling_rate)
                    
                    # Extract key frequency features
                    # Dominant frequency
                    dominant_freq_idx = np.argmax(fft_vals[1:n_fft//2]) + 1
                    dominant_freq = fft_freqs[dominant_freq_idx]
                    dominant_magnitude = fft_vals[dominant_freq_idx]
                    
                    # Spectral energy
                    spectral_energy = np.sum(fft_vals[1:n_fft//2]**2)
                    
                    # Spectral centroid
                    positive_freqs = fft_freqs[1:n_fft//2]
                    positive_mags = fft_vals[1:n_fft//2]
                    spectral_centroid = np.sum(positive_freqs * positive_mags) / (np.sum(positive_mags) + 1e-10)
                    
                    # Spectral spread
                    spectral_spread = np.sqrt(np.sum(((positive_freqs - spectral_centroid)**2) * positive_mags) / (np.sum(positive_mags) + 1e-10))
                    
                    fft_features.append({
                        'dominant_freq': dominant_freq,
                        'dominant_magnitude': dominant_magnitude,
                        'spectral_energy': spectral_energy,
                        'spectral_centroid': spectral_centroid,
                        'spectral_spread': spectral_spread
                    })
                
                # Pad with NaN for initial values
                padding = [np.nan] * (n_fft - 1)
                for key in fft_features[0].keys():
                    self.df[f'{sensor_col}_fft_{key}'] = padding + [f[key] for f in fft_features]
            else:
                # For short sequences, compute single FFT
                fft_vals = np.abs(fft(sensor_data))
                fft_freqs = fftfreq(len(sensor_data), 1/sampling_rate)
                
                dominant_freq_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
                dominant_freq = fft_freqs[dominant_freq_idx]
                dominant_magnitude = fft_vals[dominant_freq_idx]
                spectral_energy = np.sum(fft_vals[1:len(fft_vals)//2]**2)
                
                # Fill all rows with same values
                self.df[f'{sensor_col}_fft_dominant_freq'] = dominant_freq
                self.df[f'{sensor_col}_fft_dominant_magnitude'] = dominant_magnitude
                self.df[f'{sensor_col}_fft_spectral_energy'] = spectral_energy
        
        print(f"✓ Extracted frequency-domain features using FFT")
        
        return self.df
    
    # ==================== LAG FEATURES ====================
    
    def create_lag_features(self, columns, lags=[1, 2, 3, 5, 10], timestamp_column=None):
        """
        Create lag features for time-series prediction
        
        Args:
            columns: List of column names to create lags for
            lags: List of lag periods (e.g., [1, 2, 3] for 1, 2, 3 time steps back)
            timestamp_column: Name of timestamp column for proper ordering
        
        Returns:
            DataFrame with lag features
        """
        if timestamp_column and timestamp_column in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_column]):
                self.df[timestamp_column] = pd.to_datetime(self.df[timestamp_column], errors='coerce')
            self.df = self.df.sort_values(timestamp_column).reset_index(drop=True)
        
        available_cols = [col for col in columns if col in self.df.columns]
        if len(available_cols) == 0:
            print("Warning: No columns found for lag features.")
            return self.df
        
        print(f"Creating lag features for columns: {available_cols}")
        
        for col in available_cols:
            for lag in lags:
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
                
                # Also create difference features (change from lag period)
                if lag == 1:
                    self.df[f'{col}_diff_1'] = self.df[col] - self.df[col].shift(1)
        
        print(f"✓ Created lag features with lags: {lags}")
        
        return self.df
    
    # ==================== COMPREHENSIVE PIPELINE ====================
    
    def create_all_features(self, timestamp_column=None, lat_col=None, lon_col=None,
                           severity_col=None, sensor_columns=None, lag_columns=None,
                           grid_size=0.01):
        """
        Run complete feature engineering pipeline
        
        Args:
            timestamp_column: Name of timestamp column
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            severity_col: Name of severity column
            sensor_columns: List of sensor column names
            lag_columns: List of columns for lag features
            grid_size: Size of spatial grid cells
        
        Returns:
            DataFrame with all engineered features
        """
        print("\n" + "="*60)
        print("ENHANCED FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # 1. Temporal features
        self.df = self.extract_temporal_features(timestamp_column)
        
        # 2. Spatial grid features
        self.df = self.create_spatial_grid(lat_col, lon_col, grid_size, severity_col)
        
        # 3. Sensor rolling features
        if sensor_columns:
            self.df = self.create_sensor_rolling_features(sensor_columns, 
                                                         window_sizes=[5, 10],
                                                         timestamp_column=timestamp_column)
        
        # 4. Frequency-domain features
        if sensor_columns:
            self.df = self.extract_frequency_features(sensor_columns,
                                                     timestamp_column=timestamp_column)
        
        # 5. Lag features
        if lag_columns:
            self.df = self.create_lag_features(lag_columns, lags=[1, 2, 3, 5, 10],
                                             timestamp_column=timestamp_column)
        
        print("\n" + "="*60)
        print(f"Feature engineering complete! Final shape: {self.df.shape}")
        print("="*60 + "\n")
        
        return self.df


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    # Example usage
    print("Enhanced Feature Engineering Module")
    print("This module provides comprehensive feature extraction capabilities.")
    print("\nUsage:")
    print("  from enhanced_feature_engineering import EnhancedFeatureEngineer")
    print("  engineer = EnhancedFeatureEngineer(df)")
    print("  df_enhanced = engineer.create_all_features(...)")

