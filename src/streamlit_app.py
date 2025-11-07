"""
Streamlit web interface for time series analysis.

This module provides an interactive web interface for exploring time series
data, training models, and visualizing results.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data_handler import DataLoader
from models import ARIMAForecaster, ProphetForecaster, LSTMForecaster, AnomalyDetector
from visualization import TimeSeriesVisualizer

# Configure page
st.set_page_config(
    page_title="Time Series Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitApp:
    """Main Streamlit application class."""
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.config = self._load_config()
        self.data_loader = DataLoader(self.config.get('data', {}))
        self.visualizer = TimeSeriesVisualizer()
        
        # Initialize session state
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'models' not in st.session_state:
            st.session_state.models = {}
        if 'results' not in st.session_state:
            st.session_state.results = {}
    
    def _load_config(self) -> Dict:
        """Load configuration."""
        try:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return {}
    
    def render_header(self):
        """Render the main header."""
        st.markdown('<h1 class="main-header">📈 Time Series Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render the sidebar with controls."""
        st.sidebar.header("⚙️ Configuration")
        
        # Data source selection
        data_source = st.sidebar.selectbox(
            "Data Source",
            ["Synthetic Data", "Stock Data"],
            help="Choose between synthetic data or real stock data"
        )
        
        if data_source == "Stock Data":
            symbol = st.sidebar.text_input(
                "Stock Symbol",
                value="AAPL",
                help="Enter a valid stock symbol (e.g., AAPL, GOOGL, MSFT)"
            )
            start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
            end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
        else:
            symbol = "AAPL"
            start_date = pd.to_datetime("2020-01-01")
            end_date = pd.to_datetime("2023-12-31")
        
        # Model selection
        st.sidebar.header("🤖 Models")
        use_arima = st.sidebar.checkbox("ARIMA", value=True)
        use_prophet = st.sidebar.checkbox("Prophet", value=True)
        use_lstm = st.sidebar.checkbox("LSTM", value=False, 
                                      help="Note: LSTM training may take longer")
        
        # Forecasting parameters
        st.sidebar.header("🔮 Forecasting")
        forecast_horizon = st.sidebar.slider(
            "Forecast Horizon (days)",
            min_value=7,
            max_value=90,
            value=30,
            help="Number of days to forecast into the future"
        )
        
        # Anomaly detection
        st.sidebar.header("🚨 Anomaly Detection")
        detect_anomalies = st.sidebar.checkbox("Enable Anomaly Detection", value=True)
        contamination = st.sidebar.slider(
            "Expected Anomaly Rate",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
            help="Expected proportion of anomalies in the data"
        )
        
        return {
            'data_source': data_source,
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'use_arima': use_arima,
            'use_prophet': use_prophet,
            'use_lstm': use_lstm,
            'forecast_horizon': forecast_horizon,
            'detect_anomalies': detect_anomalies,
            'contamination': contamination
        }
    
    def load_data(self, config: Dict):
        """Load data based on configuration."""
        with st.spinner("Loading data..."):
            try:
                if config['data_source'] == "Synthetic Data":
                    st.session_state.data = self.data_loader.generate_synthetic_data(
                        n_samples=1000,
                        trend_strength=0.1,
                        seasonality_period=12,
                        noise_level=0.05
                    )
                    st.success("✅ Synthetic data loaded successfully!")
                else:
                    st.session_state.data = self.data_loader.load_stock_data(
                        symbol=config['symbol'],
                        start_date=config['start_date'].strftime('%Y-%m-%d'),
                        end_date=config['end_date'].strftime('%Y-%m-%d')
                    )
                    st.success(f"✅ Stock data loaded for {config['symbol']}!")
                
                # Preprocess data
                st.session_state.data, _ = self.data_loader.preprocess_data(
                    st.session_state.data, scale=False
                )
                
                # Split data
                train_data, test_data = self.data_loader.split_data(st.session_state.data, 0.2)
                st.session_state.train_data = train_data
                st.session_state.test_data = test_data
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                return False
        
        return True
    
    def train_models(self, config: Dict):
        """Train selected models."""
        if st.session_state.data is None:
            st.error("Please load data first!")
            return
        
        st.session_state.models = {}
        
        with st.spinner("Training models..."):
            progress_bar = st.progress(0)
            total_models = sum([config['use_arima'], config['use_prophet'], config['use_lstm']])
            current_model = 0
            
            # ARIMA
            if config['use_arima']:
                current_model += 1
                progress_bar.progress(current_model / total_models)
                st.text("Training ARIMA model...")
                
                try:
                    arima_model = ARIMAForecaster(auto_arima=True)
                    arima_model.fit(st.session_state.train_data)
                    st.session_state.models['ARIMA'] = arima_model
                    st.success("✅ ARIMA model trained!")
                except Exception as e:
                    st.error(f"❌ ARIMA training failed: {str(e)}")
            
            # Prophet
            if config['use_prophet']:
                current_model += 1
                progress_bar.progress(current_model / total_models)
                st.text("Training Prophet model...")
                
                try:
                    prophet_model = ProphetForecaster()
                    prophet_model.fit(st.session_state.train_data)
                    st.session_state.models['Prophet'] = prophet_model
                    st.success("✅ Prophet model trained!")
                except Exception as e:
                    st.error(f"❌ Prophet training failed: {str(e)}")
            
            # LSTM
            if config['use_lstm']:
                current_model += 1
                progress_bar.progress(current_model / total_models)
                st.text("Training LSTM model...")
                
                try:
                    lstm_model = LSTMForecaster(epochs=50)  # Reduced for demo
                    lstm_model.fit(st.session_state.train_data)
                    st.session_state.models['LSTM'] = lstm_model
                    st.success("✅ LSTM model trained!")
                except Exception as e:
                    st.error(f"❌ LSTM training failed: {str(e)}")
            
            progress_bar.progress(1.0)
            st.success("🎉 All models trained successfully!")
    
    def detect_anomalies(self, contamination: float):
        """Detect anomalies in the data."""
        if st.session_state.data is None:
            st.error("Please load data first!")
            return None
        
        with st.spinner("Detecting anomalies..."):
            try:
                detector = AnomalyDetector(contamination=contamination)
                detector.fit(st.session_state.data)
                anomaly_data = detector.detect_anomalies(st.session_state.data)
                
                n_anomalies = anomaly_data['is_anomaly'].sum()
                st.success(f"✅ Detected {n_anomalies} anomalies!")
                return anomaly_data
            except Exception as e:
                st.error(f"❌ Anomaly detection failed: {str(e)}")
                return None
    
    def generate_forecasts(self, horizon: int):
        """Generate forecasts using trained models."""
        if not st.session_state.models:
            st.error("Please train models first!")
            return {}
        
        forecasts = {}
        
        with st.spinner("Generating forecasts..."):
            for model_name, model in st.session_state.models.items():
                try:
                    forecast = model.predict(horizon)
                    forecasts[model_name] = forecast
                except Exception as e:
                    st.error(f"❌ {model_name} forecast failed: {str(e)}")
        
        if forecasts:
            st.success(f"✅ Generated forecasts for {len(forecasts)} models!")
        
        return forecasts
    
    def render_data_overview(self):
        """Render data overview section."""
        if st.session_state.data is None:
            return
        
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(st.session_state.data))
        
        with col2:
            st.metric("Training Samples", len(st.session_state.train_data))
        
        with col3:
            st.metric("Test Samples", len(st.session_state.test_data))
        
        with col4:
            mean_value = st.session_state.data.iloc[:, 0].mean()
            st.metric("Mean Value", f"{mean_value:.2f}")
        
        # Plot time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state.data.index,
            y=st.session_state.data.iloc[:, 0],
            mode='lines',
            name='Time Series',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title="Time Series Data",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_model_results(self, forecasts: Dict):
        """Render model results and forecasts."""
        if not st.session_state.models:
            return
        
        st.header("🤖 Model Results")
        
        # Model comparison on test set
        if st.session_state.test_data is not None:
            st.subheader("Model Performance on Test Set")
            
            actual = st.session_state.test_data.iloc[:, 0].values
            predictions = {}
            
            for model_name, model in st.session_state.models.items():
                try:
                    pred = model.predict(len(actual))
                    predictions[model_name] = pred
                except Exception as e:
                    st.error(f"Error generating {model_name} predictions: {str(e)}")
            
            if predictions:
                fig = go.Figure()
                
                # Add actual values
                fig.add_trace(go.Scatter(
                    y=actual,
                    mode='lines',
                    name='Actual',
                    line=dict(color='black', width=2)
                ))
                
                # Add predictions
                colors = ['red', 'blue', 'green', 'orange']
                for i, (model_name, pred) in enumerate(predictions.items()):
                    fig.add_trace(go.Scatter(
                        y=pred,
                        mode='lines',
                        name=model_name,
                        line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                    ))
                
                fig.update_layout(
                    title="Model Comparison on Test Set",
                    xaxis_title="Time Steps",
                    yaxis_title="Value",
                    hovermode='x unified',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Forecasts
        if forecasts:
            st.subheader("Future Forecasts")
            
            forecast_dates = pd.date_range(
                start=st.session_state.data.index[-1] + pd.Timedelta(days=1),
                periods=len(list(forecasts.values())[0]),
                freq='D'
            )
            
            fig = go.Figure()
            
            # Add historical data
            fig.add_trace(go.Scatter(
                x=st.session_state.data.index,
                y=st.session_state.data.iloc[:, 0],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
            
            # Add forecasts
            colors = ['red', 'green', 'orange', 'purple']
            for i, (model_name, forecast) in enumerate(forecasts.items()):
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast,
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
            
            fig.update_layout(
                title="Future Forecasts",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode='x unified',
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_anomaly_results(self, anomaly_data: pd.DataFrame):
        """Render anomaly detection results."""
        if anomaly_data is None:
            return
        
        st.header("🚨 Anomaly Detection Results")
        
        n_anomalies = anomaly_data['is_anomaly'].sum()
        anomaly_rate = n_anomalies / len(anomaly_data) * 100
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Anomalies Detected", n_anomalies)
        
        with col2:
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        
        # Plot anomalies
        fig = go.Figure()
        
        # Add normal data
        normal_mask = ~anomaly_data['is_anomaly']
        fig.add_trace(go.Scatter(
            x=anomaly_data.index[normal_mask],
            y=anomaly_data.iloc[normal_mask, 0],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Add anomalies
        anomaly_mask = anomaly_data['is_anomaly']
        fig.add_trace(go.Scatter(
            x=anomaly_data.index[anomaly_mask],
            y=anomaly_data.iloc[anomaly_mask, 0],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10)
        ))
        
        fig.update_layout(
            title="Anomaly Detection Results",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='closest',
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the Streamlit application."""
        self.render_header()
        
        # Render sidebar and get configuration
        config = self.render_sidebar()
        
        # Main content area
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("📥 Load Data", type="primary"):
                self.load_data(config)
        
        with col2:
            if st.button("🤖 Train Models"):
                self.train_models(config)
        
        with col3:
            if st.button("🔮 Generate Forecasts"):
                forecasts = self.generate_forecasts(config['forecast_horizon'])
                st.session_state.forecasts = forecasts
        
        # Render results
        if st.session_state.data is not None:
            self.render_data_overview()
            
            if st.session_state.models:
                self.render_model_results(st.session_state.get('forecasts', {}))
            
            if config['detect_anomalies']:
                anomaly_data = self.detect_anomalies(config['contamination'])
                if anomaly_data is not None:
                    self.render_anomaly_results(anomaly_data)


def main():
    """Main function to run the Streamlit app."""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()
