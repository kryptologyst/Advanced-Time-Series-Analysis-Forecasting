"""
Advanced time series forecasting models.

This module implements various forecasting methods including ARIMA, Prophet,
LSTM, and anomaly detection techniques.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for time series forecasters."""
    
    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the model to the data."""
        pass
    
    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """Make predictions for the next steps."""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        pass


class ARIMAForecaster(BaseForecaster):
    """ARIMA model for time series forecasting."""
    
    def __init__(
        self, 
        order: Tuple[int, int, int] = (5, 1, 2),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        auto_arima: bool = False
    ) -> None:
        """
        Initialize ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal ARIMA order (P, D, Q, s)
            auto_arima: Whether to use auto ARIMA for parameter selection
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_arima = auto_arima
        self.model = None
        self.model_fit = None
        
    def _check_stationarity(self, data: pd.Series) -> bool:
        """Check if the time series is stationary."""
        result = adfuller(data.dropna())
        return result[1] <= 0.05
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit ARIMA model to the data.
        
        Args:
            data: Time series data with datetime index
        """
        logger.info("Fitting ARIMA model")
        
        if self.auto_arima:
            logger.info("Using auto ARIMA for parameter selection")
            self.model = pm.auto_arima(
                data.iloc[:, 0],
                seasonal=True,
                m=12,
                suppress_warnings=True,
                stepwise=True
            )
            self.model_fit = self.model
        else:
            if self.seasonal_order:
                self.model = ARIMA(
                    data.iloc[:, 0], 
                    order=self.order,
                    seasonal_order=self.seasonal_order
                )
            else:
                self.model = ARIMA(data.iloc[:, 0], order=self.order)
            
            self.model_fit = self.model.fit()
        
        logger.info("ARIMA model fitted successfully")
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Make predictions for the next steps.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        if self.model_fit is None:
            raise ValueError("Model must be fitted before making predictions")
        
        forecast = self.model_fit.forecast(steps=steps)
        return forecast.values if hasattr(forecast, 'values') else forecast
    
    def save_model(self, filepath: str) -> None:
        """Save the trained ARIMA model."""
        if self.model_fit is None:
            raise ValueError("No model to save")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model_fit, f)
        logger.info(f"ARIMA model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained ARIMA model."""
        with open(filepath, 'rb') as f:
            self.model_fit = pickle.load(f)
        logger.info(f"ARIMA model loaded from {filepath}")


class ProphetForecaster(BaseForecaster):
    """Prophet model for time series forecasting."""
    
    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = False,
        seasonality_mode: str = "multiplicative"
    ) -> None:
        """
        Initialize Prophet forecaster.
        
        Args:
            yearly_seasonality: Whether to include yearly seasonality
            weekly_seasonality: Whether to include weekly seasonality
            daily_seasonality: Whether to include daily seasonality
            seasonality_mode: Seasonality mode ('additive' or 'multiplicative')
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.model = None
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit Prophet model to the data.
        
        Args:
            data: Time series data with datetime index
        """
        logger.info("Fitting Prophet model")
        
        # Prepare data for Prophet
        df_prophet = pd.DataFrame({
            'ds': data.index,
            'y': data.iloc[:, 0]
        })
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode
        )
        
        self.model.fit(df_prophet)
        logger.info("Prophet model fitted successfully")
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Make predictions for the next steps.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        
        # Return only the new predictions
        return forecast['yhat'].iloc[-steps:].values
    
    def save_model(self, filepath: str) -> None:
        """Save the trained Prophet model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Prophet model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained Prophet model."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Prophet model loaded from {filepath}")


class LSTMForecaster(BaseForecaster):
    """LSTM neural network for time series forecasting."""
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_units: int = 50,
        dropout: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ) -> None:
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length: Length of input sequences
            hidden_units: Number of hidden units in LSTM
            dropout: Dropout rate
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        
    def _create_model(self, input_shape: Tuple[int, int]) -> nn.Module:
        """Create LSTM model architecture."""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, dropout_rate):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(dropout_rate)
                self.linear = nn.Linear(hidden_size, 1)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                lstm_out = self.dropout(lstm_out[:, -1, :])
                output = self.linear(lstm_out)
                return output
        
        return LSTMModel(input_shape[1], self.hidden_units, self.dropout)
    
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit LSTM model to the data.
        
        Args:
            data: Time series data with datetime index
        """
        logger.info("Fitting LSTM model")
        
        # Prepare data
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data.iloc[:, 0].values.reshape(-1, 1))
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i])
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create model
        self.model = self._create_model(X.shape)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Train model
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        logger.info("LSTM model fitted successfully")
    
    def predict(self, steps: int) -> np.ndarray:
        """
        Make predictions for the next steps.
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Array of predictions
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        predictions = []
        
        # Use the last sequence_length points as input
        last_sequence = self.scaler.transform(
            self.last_data.iloc[-self.sequence_length:].values.reshape(-1, 1)
        )
        
        with torch.no_grad():
            current_input = torch.FloatTensor(last_sequence).unsqueeze(0)
            
            for _ in range(steps):
                output = self.model(current_input)
                predictions.append(output.item())
                
                # Update input for next prediction
                current_input = torch.cat([
                    current_input[:, 1:, :],
                    output.unsqueeze(0).unsqueeze(0)
                ], dim=1)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        
        return predictions
    
    def save_model(self, filepath: str) -> None:
        """Save the trained LSTM model."""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'sequence_length': self.sequence_length,
            'hidden_units': self.hidden_units,
            'dropout': self.dropout
        }, filepath)
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained LSTM model."""
        checkpoint = torch.load(filepath)
        self.scaler = checkpoint['scaler']
        self.sequence_length = checkpoint['sequence_length']
        self.hidden_units = checkpoint['hidden_units']
        self.dropout = checkpoint['dropout']
        
        # Recreate model and load weights
        self.model = self._create_model((1, self.sequence_length))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"LSTM model loaded from {filepath}")


class AnomalyDetector:
    """Anomaly detection using Isolation Forest."""
    
    def __init__(self, contamination: float = 0.1) -> None:
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.fitted = False
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit anomaly detection model.
        
        Args:
            data: Time series data
        """
        logger.info("Fitting anomaly detection model")
        self.model.fit(data.iloc[:, 0].values.reshape(-1, 1))
        self.fitted = True
        logger.info("Anomaly detection model fitted successfully")
    
    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in the data.
        
        Args:
            data: Time series data
            
        Returns:
            DataFrame with anomaly labels
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before detecting anomalies")
        
        predictions = self.model.predict(data.iloc[:, 0].values.reshape(-1, 1))
        anomaly_scores = self.model.decision_function(data.iloc[:, 0].values.reshape(-1, 1))
        
        result = data.copy()
        result['is_anomaly'] = predictions == -1
        result['anomaly_score'] = anomaly_scores
        
        return result


def evaluate_forecast(
    actual: np.ndarray, 
    predicted: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate forecast performance.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        Dictionary with evaluation metrics
    """
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }
