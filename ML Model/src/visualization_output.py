"""
Visualization Output Module
Generates JSON output for frontend visualization
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


class VisualizationOutput:
    """Class for generating visualization-ready output"""
    
    def __init__(self, df, model, X_test, y_test, predictions, feature_columns, target_encoder=None):
        """
        Initialize the VisualizationOutput
        
        Args:
            df: Original dataframe with coordinates
            model: Trained model
            X_test: Test features
            y_test: Test targets
            predictions: Model predictions
            feature_columns: List of feature column names
            target_encoder: LabelEncoder used for target variable (optional)
        """
        self.df = df
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.predictions = predictions
        self.feature_columns = feature_columns
        self.target_encoder = target_encoder
        
        # Map severity levels - use target encoder if available
        self.severity_map = {}
        
        if target_encoder is not None and hasattr(target_encoder, 'classes_'):
            # Use the label encoder to map encoded values back to original labels
            for encoded_val, original_label in enumerate(target_encoder.classes_):
                self.severity_map[encoded_val] = str(original_label)
        elif hasattr(model, 'classes_'):
            # Fallback: use model classes
            for idx, class_val in enumerate(model.classes_):
                if isinstance(class_val, (int, np.integer)):
                    severity_names = ['Low', 'Medium', 'Critical']
                    if class_val < len(severity_names):
                        self.severity_map[class_val] = severity_names[class_val]
                    else:
                        self.severity_map[class_val] = f'Level_{class_val}'
                else:
                    self.severity_map[idx] = str(class_val)
        else:
            # Final fallback to default mapping
            self.severity_map = {0: 'Low', 1: 'Medium', 2: 'Critical'}
        
        print(f"Severity mapping: {self.severity_map}")
        
    def find_coordinate_columns(self):
        """Find latitude and longitude columns"""
        lat_col = None
        lon_col = None
        
        possible_lat = ['latitude', 'lat', 'Latitude', 'LAT', 'y']
        possible_lon = ['longitude', 'lon', 'lng', 'Longitude', 'LONG', 'LON', 'x']
        
        for col in self.df.columns:
            if col in possible_lat and lat_col is None:
                lat_col = col
            if col in possible_lon and lon_col is None:
                lon_col = col
        
        return lat_col, lon_col
    
    def generate_predictions_json(self, output_file='output/predictions.json', 
                                   include_test_indices=True):
        """
        Generate JSON file with predictions and coordinates for frontend
        
        Args:
            output_file: Path to output JSON file
            include_test_indices: Whether to include test set indices
        """
        # Get coordinate columns
        lat_col, lon_col = self.find_coordinate_columns()
        
        # Get predictions with probabilities
        predictions_proba = self.model.predict_proba(self.X_test)
        
        # Create output data
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_predictions': len(self.predictions),
                'model_type': 'Random Forest',
                'severity_levels': self.severity_map
            },
            'predictions': []
        }
        
        # Get original dataframe indices for test set
        if include_test_indices:
            test_indices = self.X_test.index if hasattr(self.X_test, 'index') else range(len(self.X_test))
        else:
            test_indices = range(len(self.X_test))
        
        for idx, (test_idx, pred, proba) in enumerate(zip(test_indices, self.predictions, predictions_proba)):
            prediction_entry = {
                'id': int(test_idx) if isinstance(test_idx, (int, np.integer)) else idx,
                'severity': self.severity_map.get(pred, f'Level_{pred}'),
                'severity_code': int(pred),
                'confidence': float(max(proba)),
                'probabilities': {
                    self.severity_map.get(i, f'Level_{i}'): float(prob) 
                    for i, prob in enumerate(proba)
                }
            }
            
            # Add coordinates if available
            if lat_col and lon_col and test_idx < len(self.df):
                try:
                    prediction_entry['latitude'] = float(self.df.iloc[test_idx][lat_col])
                    prediction_entry['longitude'] = float(self.df.iloc[test_idx][lon_col])
                except:
                    pass
            
            # Add actual severity if available
            if idx < len(self.y_test):
                actual = self.y_test.iloc[idx] if hasattr(self.y_test, 'iloc') else self.y_test[idx]
                prediction_entry['actual_severity'] = self.severity_map.get(actual, f'Level_{actual}')
                prediction_entry['actual_severity_code'] = int(actual)
                prediction_entry['prediction_correct'] = bool(pred == actual)
            
            output_data['predictions'].append(prediction_entry)
        
        # Save to file
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nPredictions JSON saved to {output_file}")
        print(f"Total predictions: {len(output_data['predictions'])}")
        
        return output_file
    
    def generate_summary_statistics(self, output_file='output/summary_stats.json'):
        """
        Generate summary statistics JSON
        
        Args:
            output_file: Path to output JSON file
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, self.predictions, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, self.predictions, average='weighted', zero_division=0)
        cm = confusion_matrix(self.y_test, self.predictions)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        summary = {
            'model_performance': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'confusion_matrix': cm.tolist(),
            'severity_distribution': {
                'predicted': {self.severity_map.get(i, f'Level_{i}'): int(count) 
                             for i, count in enumerate(np.bincount(self.predictions))},
                'actual': {self.severity_map.get(i, f'Level_{i}'): int(count) 
                          for i, count in enumerate(np.bincount(self.y_test))}
            },
            'top_features': [
                {
                    'feature': row['feature'],
                    'importance': float(row['importance'])
                }
                for _, row in feature_importance.head(10).iterrows()
            ]
        }
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary statistics saved to {output_file}")
        
        return output_file

