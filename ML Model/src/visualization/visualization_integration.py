"""
Visualization & Integration Module
Creates interactive dashboard system with:
- Map visualizations showing accident hotspots using heatmaps (folium/plotly)
- Ambulance locations and optimal routes
- Time-series charts for demand prediction
- Filtering capabilities (by date range, location, severity level)
- Backend API structure for real-time predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    print("Warning: folium not available. Install with: pip install folium")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly")

try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: flask not available. Install with: pip install flask")


class VisualizationDashboard:
    """Interactive dashboard for accident data visualization"""
    
    def __init__(self, df, lat_col=None, lon_col=None, timestamp_col=None, severity_col=None):
        """
        Initialize the VisualizationDashboard
        
        Args:
            df: DataFrame with accident data
            lat_col: Name of latitude column
            lon_col: Name of longitude column
            timestamp_col: Name of timestamp column
            severity_col: Name of severity column
        """
        self.df = df.copy()
        
        # Auto-detect columns
        self.lat_col = lat_col or self._detect_column(['latitude', 'lat', 'Latitude', 'LAT', 'y'])
        self.lon_col = lon_col or self._detect_column(['longitude', 'lon', 'lng', 'Longitude', 'LONG', 'LON', 'x'])
        self.timestamp_col = timestamp_col or self._detect_column(['timestamp', 'datetime', 'date', 'time'])
        self.severity_col = severity_col or self._detect_column(['severity', 'Severity', 'SEVERITY'])
        
        # Ensure timestamp is datetime
        if self.timestamp_col and self.timestamp_col in self.df.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.df[self.timestamp_col]):
                self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col], errors='coerce')
    
    def _detect_column(self, candidates):
        """Auto-detect column from candidates"""
        for col in candidates:
            if col in self.df.columns:
                return col
        return None
    
    # ==================== HEATMAP VISUALIZATIONS ====================
    
    def create_heatmap_folium(self, severity_filter=None, date_range=None, 
                              save_path='accident_heatmap.html'):
        """
        Create heatmap using folium
        
        Args:
            severity_filter: List of severity levels to include (e.g., ['High', 'Critical'])
            date_range: Tuple of (start_date, end_date) for filtering
            save_path: Path to save HTML file
        """
        if not FOLIUM_AVAILABLE:
            print("Error: folium is not installed. Install with: pip install folium")
            return None
        
        # Filter data
        df_filtered = self._filter_data(severity_filter, date_range)
        
        if len(df_filtered) == 0:
            print("No data to visualize after filtering.")
            return None
        
        # Calculate center of map
        center_lat = df_filtered[self.lat_col].mean()
        center_lon = df_filtered[self.lon_col].mean()
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Prepare heatmap data
        heat_data = [[row[self.lat_col], row[self.lon_col]] 
                     for idx, row in df_filtered.iterrows()]
        
        # Add heatmap
        HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
        
        # Add marker cluster for individual accidents
        marker_cluster = MarkerCluster().add_to(m)
        for idx, row in df_filtered.head(1000).iterrows():  # Limit to 1000 for performance
            severity = row.get(self.severity_col, 'Unknown') if self.severity_col else 'Unknown'
            popup_text = f"Severity: {severity}"
            if self.timestamp_col:
                popup_text += f"<br>Time: {row[self.timestamp_col]}"
            
            color = self._get_severity_color(severity)
            folium.CircleMarker(
                location=[row[self.lat_col], row[self.lon_col]],
                radius=5,
                popup=popup_text,
                color=color,
                fill=True,
                fillColor=color
            ).add_to(marker_cluster)
        
        # Save map
        m.save(save_path)
        print(f"✓ Saved heatmap to {save_path}")
        return m
    
    def create_heatmap_plotly(self, severity_filter=None, date_range=None,
                              save_path='accident_heatmap_plotly.html'):
        """
        Create heatmap using plotly
        
        Args:
            severity_filter: List of severity levels to include
            date_range: Tuple of (start_date, end_date) for filtering
            save_path: Path to save HTML file
        """
        if not PLOTLY_AVAILABLE:
            print("Error: plotly is not installed. Install with: pip install plotly")
            return None
        
        # Filter data
        df_filtered = self._filter_data(severity_filter, date_range)
        
        if len(df_filtered) == 0:
            print("No data to visualize after filtering.")
            return None
        
        # Create density heatmap
        fig = px.density_mapbox(
            df_filtered,
            lat=self.lat_col,
            lon=self.lon_col,
            z=self.severity_col if self.severity_col else None,
            radius=10,
            center=dict(lat=df_filtered[self.lat_col].mean(), 
                       lon=df_filtered[self.lon_col].mean()),
            zoom=11,
            mapbox_style="open-street-map",
            title="Accident Hotspots Heatmap"
        )
        
        fig.write_html(save_path)
        print(f"✓ Saved plotly heatmap to {save_path}")
        return fig
    
    # ==================== AMBULANCE LOCATIONS & ROUTES ====================
    
    def plot_ambulance_locations(self, ambulance_locations, accident_locations=None,
                                save_path='ambulance_map.html'):
        """
        Plot ambulance locations and nearby accidents
        
        Args:
            ambulance_locations: DataFrame with columns [lat, lon, ambulance_id] or list of tuples
            accident_locations: DataFrame with accident locations (optional)
            save_path: Path to save HTML file
        """
        if not FOLIUM_AVAILABLE:
            print("Error: folium is not installed.")
            return None
        
        # Convert ambulance_locations to DataFrame if needed
        if isinstance(ambulance_locations, list):
            ambulance_df = pd.DataFrame(ambulance_locations, columns=['lat', 'lon', 'ambulance_id'])
        else:
            ambulance_df = ambulance_locations.copy()
        
        # Calculate center
        center_lat = ambulance_df['lat'].mean()
        center_lon = ambulance_df['lon'].mean()
        
        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add ambulance locations
        for idx, row in ambulance_df.iterrows():
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"Ambulance {row.get('ambulance_id', idx)}",
                icon=folium.Icon(color='green', icon='ambulance', prefix='fa')
            ).add_to(m)
        
        # Add accident locations if provided
        if accident_locations is not None:
            for idx, row in accident_locations.iterrows():
                severity = row.get(self.severity_col, 'Unknown') if self.severity_col else 'Unknown'
                color = self._get_severity_color(severity)
                folium.CircleMarker(
                    location=[row[self.lat_col], row[self.lon_col]],
                    radius=5,
                    popup=f"Severity: {severity}",
                    color=color,
                    fill=True
                ).add_to(m)
        
        m.save(save_path)
        print(f"✓ Saved ambulance map to {save_path}")
        return m
    
    def plot_optimal_routes(self, start_location, end_locations, 
                           route_data=None, save_path='routes_map.html'):
        """
        Plot optimal routes from start to multiple destinations
        
        Args:
            start_location: Tuple (lat, lon) of start point
            end_locations: List of tuples [(lat, lon), ...] or DataFrame
            route_data: Optional route data with coordinates
            save_path: Path to save HTML file
        """
        if not FOLIUM_AVAILABLE:
            print("Error: folium is not installed.")
            return None
        
        # Create map
        center_lat = (start_location[0] + np.mean([loc[0] for loc in end_locations])) / 2
        center_lon = (start_location[1] + np.mean([loc[1] for loc in end_locations])) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add start location
        folium.Marker(
            location=start_location,
            popup="Start Location",
            icon=folium.Icon(color='green', icon='play', prefix='fa')
        ).add_to(m)
        
        # Add end locations and routes
        for i, end_loc in enumerate(end_locations):
            folium.Marker(
                location=end_loc,
                popup=f"Destination {i+1}",
                icon=folium.Icon(color='red', icon='flag', prefix='fa')
            ).add_to(m)
            
            # Add route if provided
            if route_data and i < len(route_data):
                route_coords = route_data[i]
                folium.PolyLine(route_coords, color='blue', weight=3, opacity=0.7).add_to(m)
            else:
                # Simple straight line
                folium.PolyLine([start_location, end_loc], color='blue', 
                              weight=2, opacity=0.5, dashArray='5, 5').add_to(m)
        
        m.save(save_path)
        print(f"✓ Saved routes map to {save_path}")
        return m
    
    # ==================== TIME-SERIES CHARTS ====================
    
    def plot_demand_time_series(self, frequency='H', severity_filter=None,
                               save_path='demand_time_series.png'):
        """
        Plot time-series chart for demand prediction
        
        Args:
            frequency: Time frequency ('H' for hourly, 'D' for daily, 'W' for weekly)
            severity_filter: List of severity levels to include
            save_path: Path to save figure
        """
        if not self.timestamp_col or self.timestamp_col not in self.df.columns:
            print("Error: Timestamp column not found.")
            return None
        
        # Filter data
        df_filtered = self._filter_data(severity_filter, date_range=None)
        
        if len(df_filtered) == 0:
            print("No data to visualize.")
            return None
        
        # Resample by frequency
        df_filtered = df_filtered.set_index(self.timestamp_col)
        demand_series = df_filtered.resample(frequency).size()
        
        # Create plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Time series plot
        axes[0].plot(demand_series.index, demand_series.values, linewidth=2, color='steelblue')
        axes[0].set_title(f'Accident Demand Over Time ({frequency})', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Number of Accidents', fontsize=12)
        axes[0].grid(alpha=0.3)
        
        # Rolling average
        window_size = min(24, len(demand_series) // 10)  # Adaptive window
        if window_size > 1:
            rolling_avg = demand_series.rolling(window=window_size, center=True).mean()
            axes[0].plot(rolling_avg.index, rolling_avg.values, 
                        linewidth=2, color='red', linestyle='--', label=f'{window_size}-period average')
            axes[0].legend()
        
        # Distribution by hour/day
        if frequency == 'H':
            df_filtered['hour'] = df_filtered.index.hour
            hourly_dist = df_filtered.groupby('hour').size()
            axes[1].bar(hourly_dist.index, hourly_dist.values, color='coral', alpha=0.7)
            axes[1].set_title('Accident Distribution by Hour of Day', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Hour of Day', fontsize=12)
            axes[1].set_ylabel('Number of Accidents', fontsize=12)
            axes[1].grid(axis='y', alpha=0.3)
        elif frequency == 'D':
            df_filtered['day_of_week'] = df_filtered.index.dayofweek
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            daily_dist = df_filtered.groupby('day_of_week').size()
            axes[1].bar(range(7), [daily_dist.get(i, 0) for i in range(7)], 
                       color='coral', alpha=0.7)
            axes[1].set_xticks(range(7))
            axes[1].set_xticklabels(day_names)
            axes[1].set_title('Accident Distribution by Day of Week', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Number of Accidents', fontsize=12)
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved time-series plot to {save_path}")
        plt.show()
    
    # ==================== FILTERING ====================
    
    def _filter_data(self, severity_filter=None, date_range=None):
        """Filter data by severity and date range"""
        df_filtered = self.df.copy()
        
        # Filter by severity
        if severity_filter and self.severity_col and self.severity_col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[self.severity_col].isin(severity_filter)]
        
        # Filter by date range
        if date_range and self.timestamp_col and self.timestamp_col in df_filtered.columns:
            start_date, end_date = date_range
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            df_filtered = df_filtered[
                (df_filtered[self.timestamp_col] >= start_date) &
                (df_filtered[self.timestamp_col] <= end_date)
            ]
        
        return df_filtered
    
    def _get_severity_color(self, severity):
        """Get color based on severity level"""
        severity_lower = str(severity).lower()
        if 'critical' in severity_lower or 'high' in severity_lower:
            return 'red'
        elif 'medium' in severity_lower:
            return 'orange'
        else:
            return 'yellow'
    
    # ==================== INTERACTIVE DASHBOARD ====================
    
    def create_interactive_dashboard(self, save_path='dashboard.html'):
        """
        Create an interactive dashboard with multiple visualizations
        
        Args:
            save_path: Path to save HTML dashboard
        """
        if not PLOTLY_AVAILABLE:
            print("Error: plotly is required for interactive dashboard.")
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accident Hotspots', 'Demand Over Time', 
                           'Severity Distribution', 'Hourly Pattern'),
            specs=[[{"type": "scattermapbox"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Map (if coordinates available)
        if self.lat_col and self.lon_col:
            sample_data = self.df.head(1000)  # Limit for performance
            fig.add_trace(
                go.Scattermapbox(
                    lat=sample_data[self.lat_col],
                    lon=sample_data[self.lon_col],
                    mode='markers',
                    marker=dict(size=5, color='red', opacity=0.6),
                    text=sample_data.get(self.severity_col, '') if self.severity_col else '',
                    name='Accidents'
                ),
                row=1, col=1
            )
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(lat=sample_data[self.lat_col].mean(),
                              lon=sample_data[self.lon_col].mean()),
                    zoom=11
                )
            )
        
        # 2. Time series
        if self.timestamp_col:
            df_ts = self.df.set_index(self.timestamp_col)
            demand = df_ts.resample('H').size()
            fig.add_trace(
                go.Scatter(x=demand.index, y=demand.values, mode='lines', name='Demand'),
                row=1, col=2
            )
        
        # 3. Severity distribution
        if self.severity_col:
            severity_counts = self.df[self.severity_col].value_counts()
            fig.add_trace(
                go.Bar(x=severity_counts.index, y=severity_counts.values, name='Severity'),
                row=2, col=1
            )
        
        # 4. Hourly pattern
        if self.timestamp_col:
            self.df['hour'] = pd.to_datetime(self.df[self.timestamp_col]).dt.hour
            hourly_counts = self.df['hour'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=hourly_counts.index, y=hourly_counts.values, name='Hourly'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Accident Analysis Dashboard", showlegend=False)
        fig.write_html(save_path)
        print(f"✓ Saved interactive dashboard to {save_path}")
        return fig


# ==================== API STRUCTURE ====================

class PredictionAPI:
    """Backend API structure for real-time predictions"""
    
    def __init__(self, model, feature_columns, scaler=None):
        """
        Initialize the PredictionAPI
        
        Args:
            model: Trained model (must have .predict() method)
            feature_columns: List of feature column names in order
            scaler: Optional scaler for feature normalization
        """
        self.model = model
        self.feature_columns = feature_columns
        self.scaler = scaler
        
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            self._setup_routes()
        else:
            self.app = None
            print("Warning: Flask not available. API routes not set up.")
    
    def _setup_routes(self):
        """Set up Flask API routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint"""
            return jsonify({'status': 'healthy', 'model_loaded': self.model is not None})
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """Prediction endpoint"""
            try:
                data = request.get_json()
                
                # Extract features
                features = []
                for col in self.feature_columns:
                    if col not in data:
                        return jsonify({'error': f'Missing feature: {col}'}), 400
                    features.append(data[col])
                
                # Convert to numpy array
                X = np.array([features])
                
                # Scale if scaler provided
                if self.scaler:
                    X = self.scaler.transform(X)
                
                # Make prediction
                prediction = self.model.predict(X)[0]
                
                # Get probabilities if available
                probabilities = None
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X)[0].tolist()
                
                return jsonify({
                    'prediction': float(prediction) if isinstance(prediction, (np.integer, np.floating)) else str(prediction),
                    'probabilities': probabilities
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/predict_batch', methods=['POST'])
        def predict_batch():
            """Batch prediction endpoint"""
            try:
                data = request.get_json()
                records = data.get('records', [])
                
                if not records:
                    return jsonify({'error': 'No records provided'}), 400
                
                # Extract features for all records
                X = []
                for record in records:
                    features = []
                    for col in self.feature_columns:
                        if col not in record:
                            return jsonify({'error': f'Missing feature: {col}'}), 400
                        features.append(record[col])
                    X.append(features)
                
                X = np.array(X)
                
                # Scale if scaler provided
                if self.scaler:
                    X = self.scaler.transform(X)
                
                # Make predictions
                predictions = self.model.predict(X).tolist()
                
                # Get probabilities if available
                probabilities = None
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X).tolist()
                
                return jsonify({
                    'predictions': predictions,
                    'probabilities': probabilities
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """
        Run the Flask API server
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        if self.app is None:
            print("Error: Flask is not installed. Cannot run API server.")
            return
        
        print(f"Starting API server on http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)
    
    def predict_single(self, features_dict):
        """
        Make a single prediction (without Flask)
        
        Args:
            features_dict: Dictionary with feature names as keys
        
        Returns:
            Prediction result
        """
        # Extract features in correct order
        features = [features_dict[col] for col in self.feature_columns]
        X = np.array([features])
        
        # Scale if scaler provided
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        result = {'prediction': prediction}
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            result['probabilities'] = self.model.predict_proba(X)[0].tolist()
        
        return result


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("Visualization & Integration Module")
    print("This module provides dashboard, maps, and API structure.")
    print("\nUsage:")
    print("  from visualization_integration import VisualizationDashboard, PredictionAPI")
    print("  dashboard = VisualizationDashboard(df)")
    print("  dashboard.create_heatmap_folium()")

