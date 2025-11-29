"""
Data Analysis and Preprocessing Module
Handles data loading, cleaning, and initial analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataAnalyzer:
    """Class for analyzing and preprocessing accident data"""
    
    def __init__(self, file_path):
        """
        Initialize the DataAnalyzer
        
        Args:
            file_path: Path to the accident data file (CSV or Excel)
        """
        self.file_path = file_path
        self.df = None
        self.missing_info = None
        
    def load_data(self):
        """Load data from file"""
        try:
            if self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path)
            elif self.file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(self.file_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        self.missing_info = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': [self.df[col].isnull().sum() for col in self.df.columns],
            'Missing_Percentage': [self.df[col].isnull().sum() / len(self.df) * 100 
                                  for col in self.df.columns],
            'Data_Type': [self.df[col].dtype for col in self.df.columns]
        })
        
        print("\n=== Missing Values Analysis ===")
        print(self.missing_info[self.missing_info['Missing_Count'] > 0])
        
        return self.missing_info
    
    def get_data_summary(self):
        """Get summary statistics of the dataset"""
        print("\n=== Dataset Summary ===")
        print(f"Total records: {len(self.df)}")
        print(f"Total features: {len(self.df.columns)}")
        print(f"\nColumn names: {list(self.df.columns)}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nBasic statistics:\n{self.df.describe()}")
        
        return self.df.describe()
    
    def convert_dates(self, date_columns=None, time_column=None):
        """
        Convert date columns to datetime format
        Handles DD-MM-YYYY format and combines with time if provided
        
        Args:
            date_columns: List of column names to convert. If None, auto-detect date columns
            time_column: Name of time column to combine with date (e.g., decimal hours)
        """
        if date_columns is None:
            # Auto-detect date columns
            date_columns = []
            for col in self.df.columns:
                col_lower = col.lower()
                if col_lower in ['date', 'datetime', 'timestamp']:
                    date_columns.append(col)
        
        # Handle date column
        for col in date_columns:
            if col in self.df.columns:
                try:
                    # Try DD-MM-YYYY format first
                    self.df[col] = pd.to_datetime(self.df[col], format='%d-%m-%Y', errors='coerce')
                    if self.df[col].isna().any():
                        # Try other common formats
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    print(f"Converted {col} to datetime format")
                except Exception as e:
                    print(f"Warning: Could not convert {col} to datetime: {str(e)}")
        
        # Combine date and time if both exist
        if time_column and time_column in self.df.columns:
            date_col = date_columns[0] if date_columns else None
            if date_col and date_col in self.df.columns:
                try:
                    # Convert decimal hours to time
                    def decimal_to_time(decimal_hour):
                        if pd.isna(decimal_hour):
                            return pd.NaT
                        hour = int(decimal_hour)
                        minutes = int((decimal_hour - hour) * 60)
                        return pd.Timedelta(hours=hour, minutes=minutes)
                    
                    time_delta = self.df[time_column].apply(decimal_to_time)
                    self.df['datetime'] = self.df[date_col] + time_delta
                    print(f"Combined {date_col} and {time_column} into datetime column")
                except Exception as e:
                    print(f"Warning: Could not combine date and time: {str(e)}")
        
        return self.df
    
    def handle_missing_values(self, strategy='auto'):
        """
        Handle missing values based on strategy
        
        Args:
            strategy: 'auto' (intelligent imputation), 'mean' (for numerical), 
                     'mode' (for categorical), 'drop' (drop rows with missing values)
        """
        if strategy == 'auto':
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    if self.df[col].dtype in ['int64', 'float64']:
                        # Numerical: use median (more robust than mean)
                        median_val = self.df[col].median()
                        if pd.notna(median_val):
                            self.df[col].fillna(median_val, inplace=True)
                            print(f"Filled {col} (numerical) with median: {median_val}")
                    elif self.df[col].dtype in ['datetime64[ns]']:
                        # Datetime: use forward fill
                        self.df[col] = self.df[col].ffill()
                        print(f"Filled {col} (datetime) with forward fill")
                    else:
                        # Categorical: use mode
                        mode_values = self.df[col].mode()
                        mode_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                        self.df[col].fillna(mode_value, inplace=True)
                        print(f"Filled {col} (categorical) with mode: {mode_value}")
        
        elif strategy == 'mean':
            self.df.fillna(self.df.select_dtypes(include=[np.number]).mean(), inplace=True)
        
        elif strategy == 'mode':
            for col in self.df.columns:
                if self.df[col].isnull().sum() > 0:
                    mode_values = self.df[col].mode()
                    mode_value = mode_values[0] if len(mode_values) > 0 else 'Unknown'
                    self.df[col].fillna(mode_value, inplace=True)
        
        elif strategy == 'drop':
            self.df.dropna(inplace=True)
            print(f"Dropped rows with missing values. New shape: {self.df.shape}")
        
        return self.df
    
    def normalize_severity(self, severity_column='severity'):
        """
        Normalize severity levels (map High to Critical, ensure consistent naming)
        
        Args:
            severity_column: Name of the severity column
        """
        if severity_column in self.df.columns:
            # Map High to Critical for consistency
            severity_mapping = {
                'High': 'Critical',
                'high': 'Critical',
                'HIGH': 'Critical'
            }
            self.df[severity_column] = self.df[severity_column].replace(severity_mapping)
            print(f"Normalized severity levels. Unique values: {self.df[severity_column].unique()}")
        
        return self.df
    
    def get_cleaned_data(self):
        """Return the cleaned dataframe"""
        return self.df.copy()

