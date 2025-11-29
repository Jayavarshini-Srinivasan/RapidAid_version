"""
Feature Engineering Module
Handles encoding, temporal features, and geospatial processing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pandas.api.types import is_object_dtype, is_categorical_dtype
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Class for feature engineering and preprocessing"""
    
    def __init__(self, df):
        """
        Initialize the FeatureEngineer
        
        Args:
            df: Cleaned pandas DataFrame
        """
        self.df = df.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_columns = []
        self.numerical_columns = []
        
    def identify_column_types(self):
        """Identify categorical and numerical columns"""
        self.categorical_columns = []
        self.numerical_columns = []
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category':
                self.categorical_columns.append(col)
            elif self.df[col].dtype in ['int64', 'float64']:
                self.numerical_columns.append(col)
        
        print(f"Categorical columns: {self.categorical_columns}")
        print(f"Numerical columns: {self.numerical_columns}")
        
        return self.categorical_columns, self.numerical_columns
    
    def extract_temporal_features(self, date_column=None):
        """
        Extract temporal features from date column
        
        Args:
            date_column: Name of the date column. If None, auto-detect
        """
        if date_column is None:
            # Auto-detect datetime columns (prefer 'datetime' if exists, otherwise any datetime column)
            if 'datetime' in self.df.columns and self.df['datetime'].dtype == 'datetime64[ns]':
                date_column = 'datetime'
            else:
                date_columns = self.df.select_dtypes(include=['datetime64']).columns
                if len(date_columns) > 0:
                    date_column = date_columns[0]
                else:
                    print("No datetime column found. Skipping temporal feature extraction.")
                    return self.df
        
        if date_column not in self.df.columns:
            print(f"Date column '{date_column}' not found.")
            return self.df
        
        # Extract temporal features
        self.df['hour'] = self.df[date_column].dt.hour
        self.df['day_of_week'] = self.df[date_column].dt.dayofweek
        self.df['day_of_month'] = self.df[date_column].dt.day
        self.df['month'] = self.df[date_column].dt.month
        self.df['year'] = self.df[date_column].dt.year
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['is_night'] = ((self.df['hour'] >= 20) | (self.df['hour'] < 6)).astype(int)
        self.df['is_rush_hour'] = ((self.df['hour'] >= 7) & (self.df['hour'] <= 9) | 
                                   (self.df['hour'] >= 17) & (self.df['hour'] <= 19)).astype(int)
        
        print(f"Extracted temporal features from {date_column}")
        print(f"New features: hour, day_of_week, day_of_month, month, year, is_weekend, is_night, is_rush_hour")
        
        return self.df
    
    def encode_categorical_variables(self, columns=None, encoding_type='label'):
        """
        Encode categorical variables
        
        Args:
            columns: List of columns to encode. If None, encode all categorical columns
            encoding_type: 'label' (Label Encoding) or 'onehot' (One-Hot Encoding)
        """
        if columns is None:
            columns = self.categorical_columns.copy()
            # Exclude target variable if it exists
            if 'severity' in columns or 'Severity' in columns:
                target_col = 'severity' if 'severity' in columns else 'Severity'
                columns.remove(target_col)
        
        if encoding_type == 'label':
            for col in columns:
                if col in self.df.columns:
                    le = LabelEncoder()
                    self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
                    self.label_encoders[col] = le
                    print(f"Label encoded {col}")
        
        elif encoding_type == 'onehot':
            for col in columns:
                if col in self.df.columns:
                    dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                    self.df = pd.concat([self.df, dummies], axis=1)
                    print(f"One-hot encoded {col}")
        
        return self.df
    
    def process_geospatial_data(self, lat_col=None, lon_col=None):
        """
        Process geospatial data (latitude, longitude)
        
        Args:
            lat_col: Name of latitude column. If None, auto-detect
            lon_col: Name of longitude column. If None, auto-detect
        """
        # Auto-detect latitude/longitude columns
        if lat_col is None:
            possible_lat = ['latitude', 'lat', 'Latitude', 'LAT', 'y']
            for col in possible_lat:
                if col in self.df.columns:
                    lat_col = col
                    break
        
        if lon_col is None:
            possible_lon = ['longitude', 'lon', 'lng', 'Longitude', 'LONG', 'LON', 'x']
            for col in possible_lon:
                if col in self.df.columns:
                    lon_col = col
                    break
        
        if lat_col and lon_col and lat_col in self.df.columns and lon_col in self.df.columns:
            # Remove invalid coordinates
            self.df = self.df[
                (self.df[lat_col] >= -90) & (self.df[lat_col] <= 90) &
                (self.df[lon_col] >= -180) & (self.df[lon_col] <= 180)
            ]
            
            # Create distance from center (if needed)
            center_lat = self.df[lat_col].median()
            center_lon = self.df[lon_col].median()
            self.df['distance_from_center'] = np.sqrt(
                (self.df[lat_col] - center_lat)**2 + 
                (self.df[lon_col] - center_lon)**2
            )
            
            print(f"Processed geospatial data using {lat_col} and {lon_col}")
        else:
            print("Latitude/Longitude columns not found. Skipping geospatial processing.")
        
        return self.df
    
    def scale_numerical_features(self, columns=None):
        """
        Scale numerical features
        
        Args:
            columns: List of columns to scale. If None, scale all numerical columns
        """
        if columns is None:
            columns = [col for col in self.numerical_columns 
                      if col not in ['latitude', 'longitude', 'lat', 'lon', 'lng', 
                                    'Latitude', 'Longitude', 'LAT', 'LON']]
        
        if len(columns) > 0:
            self.df[columns] = self.scaler.fit_transform(self.df[columns])
            print(f"Scaled numerical features: {columns}")
        
        return self.df
    
    def prepare_features(self, target_column='severity', exclude_columns=None, drop_object=True):
        """
        Prepare final feature set for modeling
        
        Args:
            target_column: Name of the target variable
            exclude_columns: List of columns to exclude from features
        """
        if exclude_columns is None:
            exclude_columns = []
        
        # Find target column (case-insensitive)
        target_col = None
        for col in self.df.columns:
            if col.lower() == target_column.lower():
                target_col = col
                break
        
        if target_col is None:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Get feature columns
        feature_columns = [col for col in self.df.columns 
                          if col != target_col and col not in exclude_columns]

        if drop_object:
            feature_columns = [
                col for col in feature_columns
                if not (is_object_dtype(self.df[col]) or is_categorical_dtype(self.df[col]))
            ]
        
        X = self.df[feature_columns]
        y = self.df[target_col]
        
        # Encode target if it's categorical
        self.target_encoder = None
        if y.dtype == 'object' or y.dtype.name == 'category':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.target_encoder = le_target
            print(f"Encoded target variable. Classes: {le_target.classes_}")
        
        print(f"\nFeature preparation complete:")
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Feature columns: {list(X.columns)}")
        
        return X, y, feature_columns

