"""
Unit tests for time series analysis components.

This module contains comprehensive tests for data handling, models,
and visualization components.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_handler import DataLoader, load_multiple_stocks
from models import (
    ARIMAForecaster, 
    ProphetForecaster, 
    LSTMForecaster, 
    AnomalyDetector,
    evaluate_forecast
)


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'sources': {
                'stock_symbols': ['AAPL'],
                'start_date': '2020-01-01',
                'end_date': '2023-01-01'
            },
            'synthetic': {
                'enabled': True,
                'n_samples': 100
            }
        }
        self.loader = DataLoader(self.config)
    
    def test_init(self):
        """Test DataLoader initialization."""
        self.assertIsNotNone(self.loader)
        self.assertEqual(self.loader.config, self.config)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        data = self.loader.generate_synthetic_data(n_samples=100)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 100)
        self.assertIn('Close', data.columns)
        self.assertIsInstance(data.index, pd.DatetimeIndex)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        # Create test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100) * 10 + 100
        data = pd.DataFrame({'Close': values}, index=dates)
        
        # Test preprocessing
        processed_data, scaler = self.loader.preprocess_data(data, scale=True)
        
        self.assertIsInstance(processed_data, pd.DataFrame)
        self.assertIsNotNone(scaler)
        self.assertEqual(len(processed_data), len(data))
    
    def test_create_sequences(self):
        """Test sequence creation for time series."""
        # Create test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100)
        data = pd.DataFrame({'Close': values}, index=dates)
        
        # Test sequence creation
        X, y = self.loader.create_sequences(data, sequence_length=10)
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[1], 10)
    
    def test_split_data(self):
        """Test data splitting."""
        # Create test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100)
        data = pd.DataFrame({'Close': values}, index=dates)
        
        # Test splitting
        train_data, test_data = self.loader.split_data(data, test_size=0.2)
        
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)
        self.assertEqual(len(train_data) + len(test_data), len(data))
        self.assertAlmostEqual(len(test_data) / len(data), 0.2, places=1)


class TestARIMAForecaster(unittest.TestCase):
    """Test cases for ARIMAForecaster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = ARIMAForecaster(order=(1, 1, 1))
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.cumsum(np.random.randn(100)) + 100
        self.data = pd.DataFrame({'Close': values}, index=dates)
    
    def test_init(self):
        """Test ARIMAForecaster initialization."""
        self.assertEqual(self.forecaster.order, (1, 1, 1))
        self.assertFalse(self.forecaster.auto_arima)
        self.assertIsNone(self.forecaster.model_fit)
    
    def test_fit(self):
        """Test ARIMA model fitting."""
        self.forecaster.fit(self.data)
        self.assertIsNotNone(self.forecaster.model_fit)
    
    def test_predict(self):
        """Test ARIMA predictions."""
        self.forecaster.fit(self.data)
        predictions = self.forecaster.predict(steps=10)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), 10)
    
    def test_predict_without_fit(self):
        """Test predictions without fitting."""
        with self.assertRaises(ValueError):
            self.forecaster.predict(steps=10)


class TestProphetForecaster(unittest.TestCase):
    """Test cases for ProphetForecaster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.forecaster = ProphetForecaster()
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.cumsum(np.random.randn(100)) + 100
        self.data = pd.DataFrame({'Close': values}, index=dates)
    
    def test_init(self):
        """Test ProphetForecaster initialization."""
        self.assertTrue(self.forecaster.yearly_seasonality)
        self.assertTrue(self.forecaster.weekly_seasonality)
        self.assertFalse(self.forecaster.daily_seasonality)
        self.assertEqual(self.forecaster.seasonality_mode, "multiplicative")
        self.assertIsNone(self.forecaster.model)
    
    def test_fit(self):
        """Test Prophet model fitting."""
        self.forecaster.fit(self.data)
        self.assertIsNotNone(self.forecaster.model)
    
    def test_predict(self):
        """Test Prophet predictions."""
        self.forecaster.fit(self.data)
        predictions = self.forecaster.predict(steps=10)
        
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), 10)
    
    def test_predict_without_fit(self):
        """Test predictions without fitting."""
        with self.assertRaises(ValueError):
            self.forecaster.predict(steps=10)


class TestAnomalyDetector(unittest.TestCase):
    """Test cases for AnomalyDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = AnomalyDetector(contamination=0.1)
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.random.randn(100) * 10 + 100
        self.data = pd.DataFrame({'Close': values}, index=dates)
    
    def test_init(self):
        """Test AnomalyDetector initialization."""
        self.assertEqual(self.detector.contamination, 0.1)
        self.assertFalse(self.detector.fitted)
    
    def test_fit(self):
        """Test anomaly detector fitting."""
        self.detector.fit(self.data)
        self.assertTrue(self.detector.fitted)
    
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        self.detector.fit(self.data)
        result = self.detector.detect_anomalies(self.data)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('is_anomaly', result.columns)
        self.assertIn('anomaly_score', result.columns)
        self.assertEqual(len(result), len(self.data))
    
    def test_detect_anomalies_without_fit(self):
        """Test anomaly detection without fitting."""
        with self.assertRaises(ValueError):
            self.detector.detect_anomalies(self.data)


class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for evaluation metrics."""
    
    def test_evaluate_forecast(self):
        """Test forecast evaluation."""
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluate_forecast(actual, predicted)
        
        self.assertIn('MAE', metrics)
        self.assertIn('MSE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('MAPE', metrics)
        
        # Check that all metrics are positive
        for metric, value in metrics.items():
            self.assertGreaterEqual(value, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'sources': {
                'stock_symbols': ['AAPL'],
                'start_date': '2020-01-01',
                'end_date': '2023-01-01'
            },
            'synthetic': {
                'enabled': True,
                'n_samples': 200
            }
        }
        self.loader = DataLoader(self.config)
    
    def test_complete_pipeline(self):
        """Test complete analysis pipeline."""
        # Generate data
        data = self.loader.generate_synthetic_data(n_samples=200)
        
        # Preprocess data
        processed_data, scaler = self.loader.preprocess_data(data, scale=True)
        
        # Split data
        train_data, test_data = self.loader.split_data(processed_data, test_size=0.2)
        
        # Train ARIMA model
        arima_model = ARIMAForecaster(order=(1, 1, 1))
        arima_model.fit(train_data)
        
        # Make predictions
        predictions = arima_model.predict(steps=len(test_data))
        
        # Evaluate
        actual = test_data.iloc[:, 0].values
        metrics = evaluate_forecast(actual, predictions)
        
        # Assertions
        self.assertIsInstance(metrics, dict)
        self.assertIn('MAE', metrics)
        self.assertGreater(len(predictions), 0)
    
    @patch('yfinance.download')
    def test_load_multiple_stocks(self, mock_download):
        """Test loading multiple stocks."""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Close': np.random.randn(100) * 10 + 100
        }, index=pd.date_range('2020-01-01', periods=100, freq='D'))
        mock_download.return_value = mock_data
        
        # Test loading multiple stocks
        symbols = ['AAPL', 'GOOGL']
        stock_data = load_multiple_stocks(symbols, '2020-01-01', '2023-01-01')
        
        self.assertIsInstance(stock_data, dict)
        self.assertEqual(len(stock_data), len(symbols))
        
        for symbol in symbols:
            self.assertIn(symbol, stock_data)
            self.assertIsInstance(stock_data[symbol], pd.DataFrame)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestDataLoader,
        TestARIMAForecaster,
        TestProphetForecaster,
        TestAnomalyDetector,
        TestEvaluationMetrics,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"{'='*50}")
