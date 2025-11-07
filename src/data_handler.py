"""
Data handling module for time series analysis.

This module provides utilities for loading, preprocessing, and generating
synthetic time series data for forecasting and analysis.
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and preprocessing of time series data."""
    
    def __init__(self, config: Dict) -> None:
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary containing data parameters
        """
        self.config = config
        self.scaler = StandardScaler()
        
    def load_stock_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str
    ) -> pd.DataFrame:
        """
        Load stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with stock data
            
        Raises:
            ValueError: If data cannot be loaded
        """
        try:
            logger.info(f"Loading stock data for {symbol} from {start_date} to {end_date}")
            data = yf.download(symbol, start=start_date, end=end_date)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
                
            # Clean column names and ensure we have Close prices
            if 'Close' in data.columns:
                data = data[['Close']].copy()
            elif len(data.columns) == 1:
                data.columns = ['Close']
            else:
                raise ValueError("Unexpected data format")
                
            data.dropna(inplace=True)
            logger.info(f"Successfully loaded {len(data)} data points")
            return data
            
        except Exception as e:
            logger.error(f"Error loading stock data: {e}")
            raise ValueError(f"Failed to load data for {symbol}: {e}")
    
    def generate_synthetic_data(
        self, 
        n_samples: int = 1000,
        trend_strength: float = 0.1,
        seasonality_period: int = 12,
        noise_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Generate synthetic time series data for testing.
        
        Args:
            n_samples: Number of data points to generate
            trend_strength: Strength of the trend component
            seasonality_period: Period of seasonal component
            noise_level: Level of random noise
            
        Returns:
            DataFrame with synthetic time series data
        """
        logger.info(f"Generating synthetic data with {n_samples} samples")
        
        # Generate time index
        dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
        
        # Generate components
        trend = np.linspace(100, 100 + trend_strength * n_samples, n_samples)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_samples) / seasonality_period)
        noise = np.random.normal(0, noise_level * 100, n_samples)
        
        # Combine components
        values = trend + seasonal + noise
        
        # Create DataFrame
        data = pd.DataFrame({
            'Close': values
        }, index=dates)
        
        logger.info("Synthetic data generated successfully")
        return data
    
    def preprocess_data(
        self, 
        data: pd.DataFrame, 
        scale: bool = True
    ) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
        """
        Preprocess time series data.
        
        Args:
            data: Input time series data
            scale: Whether to apply standardization
            
        Returns:
            Tuple of (processed_data, scaler)
        """
        logger.info("Preprocessing time series data")
        
        processed_data = data.copy()
        
        # Handle missing values
        if processed_data.isnull().any().any():
            logger.warning("Found missing values, filling with forward fill")
            processed_data.fillna(method='ffill', inplace=True)
        
        # Apply scaling if requested
        scaler = None
        if scale:
            scaler = StandardScaler()
            processed_data.iloc[:, 0] = scaler.fit_transform(
                processed_data.iloc[:, 0].values.reshape(-1, 1)
            ).flatten()
            logger.info("Applied standardization to data")
        
        logger.info("Data preprocessing completed")
        return processed_data, scaler
    
    def create_sequences(
        self, 
        data: pd.DataFrame, 
        sequence_length: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series forecasting.
        
        Args:
            data: Time series data
            sequence_length: Length of input sequences
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        logger.info(f"Creating sequences with length {sequence_length}")
        
        values = data.iloc[:, 0].values
        X, y = [], []
        
        for i in range(sequence_length, len(values)):
            X.append(values[i-sequence_length:i])
            y.append(values[i])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences")
        return X, y
    
    def split_data(
        self, 
        data: pd.DataFrame, 
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            data: Time series data
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        split_idx = int(len(data) * (1 - test_size))
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        logger.info(f"Split data: {len(train_data)} train, {len(test_data)} test")
        return train_data, test_data


def load_multiple_stocks(
    symbols: List[str], 
    start_date: str, 
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple stock symbols.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        Dictionary mapping symbols to their data
    """
    loader = DataLoader({})
    stock_data = {}
    
    for symbol in symbols:
        try:
            stock_data[symbol] = loader.load_stock_data(symbol, start_date, end_date)
        except ValueError as e:
            logger.warning(f"Skipping {symbol}: {e}")
            continue
    
    return stock_data
