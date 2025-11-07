"""
Main application module for time series analysis.

This module provides the main interface for running time series analysis,
forecasting, and anomaly detection with multiple models.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data_handler import DataLoader, load_multiple_stocks
from models import (
    ARIMAForecaster, 
    ProphetForecaster, 
    LSTMForecaster, 
    AnomalyDetector,
    evaluate_forecast
)
from visualization import TimeSeriesVisualizer

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging(config: Dict) -> None:
    """Setup logging configuration."""
    log_level = config.get('logging', {}).get('level', 'INFO')
    log_format = config.get('logging', {}).get('format', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = config.get('logging', {}).get('file', 'logs/timeseries.log')
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


class TimeSeriesAnalyzer:
    """Main class for time series analysis and forecasting."""
    
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialize the analyzer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_loader = DataLoader(self.config['data'])
        self.visualizer = TimeSeriesVisualizer(
            figsize=tuple(self.config['visualization']['figure_size'])
        )
        
        # Initialize models
        self.models = {}
        self._initialize_models()
        
        # Data storage
        self.data = None
        self.train_data = None
        self.test_data = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'data': {
                'sources': {
                    'stock_symbols': ['AAPL'],
                    'start_date': '2020-01-01',
                    'end_date': '2023-12-31'
                },
                'synthetic': {
                    'enabled': True,
                    'n_samples': 1000
                }
            },
            'models': {
                'arima': {'order': [5, 1, 2]},
                'prophet': {'yearly_seasonality': True},
                'lstm': {'sequence_length': 60, 'epochs': 100}
            },
            'forecasting': {'horizon': 30},
            'visualization': {'figure_size': [12, 8]},
            'logging': {'level': 'INFO'}
        }
    
    def _initialize_models(self) -> None:
        """Initialize forecasting models."""
        model_config = self.config['models']
        
        # ARIMA
        arima_config = model_config.get('arima', {})
        self.models['ARIMA'] = ARIMAForecaster(
            order=tuple(arima_config.get('order', [5, 1, 2])),
            auto_arima=arima_config.get('auto_arima', True)
        )
        
        # Prophet
        prophet_config = model_config.get('prophet', {})
        self.models['Prophet'] = ProphetForecaster(
            yearly_seasonality=prophet_config.get('yearly_seasonality', True),
            weekly_seasonality=prophet_config.get('weekly_seasonality', True),
            daily_seasonality=prophet_config.get('daily_seasonality', False)
        )
        
        # LSTM
        lstm_config = model_config.get('lstm', {})
        self.models['LSTM'] = LSTMForecaster(
            sequence_length=lstm_config.get('sequence_length', 60),
            hidden_units=lstm_config.get('hidden_units', 50),
            epochs=lstm_config.get('epochs', 100)
        )
        
        # Anomaly Detector
        self.anomaly_detector = AnomalyDetector()
        
        self.logger.info("Models initialized successfully")
    
    def load_data(
        self, 
        use_synthetic: bool = False,
        symbol: str = "AAPL"
    ) -> None:
        """
        Load time series data.
        
        Args:
            use_synthetic: Whether to use synthetic data
            symbol: Stock symbol for real data
        """
        self.logger.info("Loading time series data")
        
        if use_synthetic or self.config['data']['synthetic']['enabled']:
            synthetic_config = self.config['data']['synthetic']
            self.data = self.data_loader.generate_synthetic_data(
                n_samples=synthetic_config.get('n_samples', 1000),
                trend_strength=synthetic_config.get('trend_strength', 0.1),
                seasonality_period=synthetic_config.get('seasonality_period', 12),
                noise_level=synthetic_config.get('noise_level', 0.05)
            )
            self.logger.info("Synthetic data loaded")
        else:
            sources_config = self.config['data']['sources']
            self.data = self.data_loader.load_stock_data(
                symbol=symbol,
                start_date=sources_config['start_date'],
                end_date=sources_config['end_date']
            )
            self.logger.info(f"Stock data loaded for {symbol}")
        
        # Preprocess data
        self.data, _ = self.data_loader.preprocess_data(self.data, scale=False)
        
        # Split data
        test_size = self.config['forecasting'].get('cross_validation', {}).get('test_size', 0.2)
        self.train_data, self.test_data = self.data_loader.split_data(self.data, test_size)
        
        self.logger.info(f"Data loaded: {len(self.train_data)} train, {len(self.test_data)} test")
    
    def train_models(self) -> Dict[str, Dict]:
        """
        Train all forecasting models.
        
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training forecasting models")
        results = {}
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Training {model_name} model")
                model.fit(self.train_data)
                
                # Make predictions on test set
                horizon = len(self.test_data)
                predictions = model.predict(horizon)
                
                # Evaluate performance
                actual = self.test_data.iloc[:, 0].values
                metrics = evaluate_forecast(actual, predictions)
                
                results[model_name] = {
                    'predictions': predictions,
                    'metrics': metrics,
                    'status': 'success'
                }
                
                self.logger.info(f"{model_name} trained successfully")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {e}")
                results[model_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        return results
    
    def detect_anomalies(self) -> pd.DataFrame:
        """
        Detect anomalies in the data.
        
        Returns:
            DataFrame with anomaly labels
        """
        self.logger.info("Detecting anomalies")
        
        self.anomaly_detector.fit(self.data)
        anomaly_data = self.anomaly_detector.detect_anomalies(self.data)
        
        n_anomalies = anomaly_data['is_anomaly'].sum()
        self.logger.info(f"Detected {n_anomalies} anomalies")
        
        return anomaly_data
    
    def generate_forecasts(self, horizon: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate forecasts using all trained models.
        
        Args:
            horizon: Number of steps to forecast
            
        Returns:
            Dictionary with forecasts from each model
        """
        if horizon is None:
            horizon = self.config['forecasting']['horizon']
        
        self.logger.info(f"Generating {horizon}-step forecasts")
        forecasts = {}
        
        for model_name, model in self.models.items():
            try:
                forecast = model.predict(horizon)
                forecasts[model_name] = forecast
                self.logger.info(f"{model_name} forecast generated")
            except Exception as e:
                self.logger.error(f"Error generating {model_name} forecast: {e}")
        
        return forecasts
    
    def visualize_results(
        self, 
        training_results: Dict,
        anomaly_data: Optional[pd.DataFrame] = None,
        forecasts: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Visualize analysis results.
        
        Args:
            training_results: Results from model training
            anomaly_data: Anomaly detection results
            forecasts: Forecast results
        """
        self.logger.info("Creating visualizations")
        
        # Plot original time series
        self.visualizer.plot_time_series(
            self.data, 
            title="Time Series Data",
            interactive=self.config['visualization']['interactive']
        )
        
        # Plot model comparison on test set
        if training_results:
            actual = self.test_data.iloc[:, 0].values
            predictions = {
                name: result['predictions'] 
                for name, result in training_results.items() 
                if result['status'] == 'success'
            }
            
            if predictions:
                self.visualizer.plot_model_comparison(
                    actual, predictions, "Model Comparison on Test Set",
                    interactive=self.config['visualization']['interactive']
                )
        
        # Plot anomalies
        if anomaly_data is not None:
            self.visualizer.plot_anomalies(
                self.data, anomaly_data, "Anomaly Detection",
                interactive=self.config['visualization']['interactive']
            )
        
        # Plot forecasts
        if forecasts:
            forecast_dates = pd.date_range(
                start=self.data.index[-1] + pd.Timedelta(days=1),
                periods=len(list(forecasts.values())[0]),
                freq='D'
            )
            
            for model_name, forecast in forecasts.items():
                self.visualizer.plot_forecast(
                    self.data, forecast, forecast_dates,
                    f"{model_name} Forecast",
                    interactive=self.config['visualization']['interactive']
                )
        
        # Plot decomposition
        self.visualizer.plot_decomposition(
            self.data, "Time Series Decomposition",
            interactive=self.config['visualization']['interactive']
        )
    
    def save_models(self, models_dir: str = "models/saved") -> None:
        """Save trained models."""
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            try:
                filepath = os.path.join(models_dir, f"{model_name.lower()}_model.pkl")
                model.save_model(filepath)
                self.logger.info(f"{model_name} model saved to {filepath}")
            except Exception as e:
                self.logger.error(f"Error saving {model_name} model: {e}")
    
    def run_full_analysis(
        self, 
        use_synthetic: bool = False,
        symbol: str = "AAPL"
    ) -> Dict:
        """
        Run complete time series analysis.
        
        Args:
            use_synthetic: Whether to use synthetic data
            symbol: Stock symbol for real data
            
        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("Starting full time series analysis")
        
        # Load data
        self.load_data(use_synthetic, symbol)
        
        # Train models
        training_results = self.train_models()
        
        # Detect anomalies
        anomaly_data = self.detect_anomalies()
        
        # Generate forecasts
        forecasts = self.generate_forecasts()
        
        # Visualize results
        self.visualize_results(training_results, anomaly_data, forecasts)
        
        # Save models
        self.save_models()
        
        # Compile results
        results = {
            'training_results': training_results,
            'anomaly_data': anomaly_data,
            'forecasts': forecasts,
            'data_info': {
                'train_size': len(self.train_data),
                'test_size': len(self.test_data),
                'total_size': len(self.data)
            }
        }
        
        self.logger.info("Full analysis completed successfully")
        return results


def main():
    """Main function to run the analysis."""
    analyzer = TimeSeriesAnalyzer()
    
    # Run analysis with synthetic data
    results = analyzer.run_full_analysis(use_synthetic=True)
    
    # Print summary
    print("\n" + "="*50)
    print("TIME SERIES ANALYSIS SUMMARY")
    print("="*50)
    
    for model_name, result in results['training_results'].items():
        if result['status'] == 'success':
            print(f"\n{model_name} Model:")
            for metric, value in result['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"\n{model_name} Model: Failed - {result['error']}")
    
    n_anomalies = results['anomaly_data']['is_anomaly'].sum()
    print(f"\nAnomalies Detected: {n_anomalies}")
    
    print(f"\nData Summary:")
    print(f"  Training samples: {results['data_info']['train_size']}")
    print(f"  Test samples: {results['data_info']['test_size']}")
    print(f"  Total samples: {results['data_info']['total_size']}")


if __name__ == "__main__":
    main()
