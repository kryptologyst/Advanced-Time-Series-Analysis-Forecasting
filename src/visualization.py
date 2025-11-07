"""
Advanced visualization module for time series analysis.

This module provides comprehensive plotting capabilities for time series data,
forecasts, anomalies, and model comparisons.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class TimeSeriesVisualizer:
    """Comprehensive visualization for time series analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        
    def plot_time_series(
        self, 
        data: pd.DataFrame, 
        title: str = "Time Series",
        interactive: bool = False
    ) -> None:
        """
        Plot basic time series data.
        
        Args:
            data: Time series data with datetime index
            title: Plot title
            interactive: Whether to create interactive plot
        """
        if interactive:
            self._plot_interactive_time_series(data, title)
        else:
            self._plot_static_time_series(data, title)
    
    def _plot_static_time_series(self, data: pd.DataFrame, title: str) -> None:
        """Create static matplotlib plot."""
        plt.figure(figsize=self.figsize)
        plt.plot(data.index, data.iloc[:, 0], linewidth=2)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_interactive_time_series(self, data: pd.DataFrame, title: str) -> None:
        """Create interactive plotly plot."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.iloc[:, 0],
            mode='lines',
            name='Time Series',
            line=dict(width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_forecast(
        self,
        historical_data: pd.DataFrame,
        forecast_values: np.ndarray,
        forecast_dates: pd.DatetimeIndex,
        title: str = "Time Series Forecast",
        confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        interactive: bool = True
    ) -> None:
        """
        Plot historical data with forecast.
        
        Args:
            historical_data: Historical time series data
            forecast_values: Forecasted values
            forecast_dates: Dates for forecast period
            title: Plot title
            confidence_intervals: Optional confidence intervals (lower, upper)
            interactive: Whether to create interactive plot
        """
        if interactive:
            self._plot_interactive_forecast(
                historical_data, forecast_values, forecast_dates, 
                title, confidence_intervals
            )
        else:
            self._plot_static_forecast(
                historical_data, forecast_values, forecast_dates,
                title, confidence_intervals
            )
    
    def _plot_static_forecast(
        self,
        historical_data: pd.DataFrame,
        forecast_values: np.ndarray,
        forecast_dates: pd.DatetimeIndex,
        title: str,
        confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Create static forecast plot."""
        plt.figure(figsize=self.figsize)
        
        # Plot historical data
        plt.plot(historical_data.index, historical_data.iloc[:, 0], 
                label='Historical', linewidth=2, color='blue')
        
        # Plot forecast
        plt.plot(forecast_dates, forecast_values, 
                label='Forecast', linewidth=2, color='red', linestyle='--')
        
        # Plot confidence intervals if provided
        if confidence_intervals:
            lower, upper = confidence_intervals
            plt.fill_between(forecast_dates, lower, upper, 
                           alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_interactive_forecast(
        self,
        historical_data: pd.DataFrame,
        forecast_values: np.ndarray,
        forecast_dates: pd.DatetimeIndex,
        title: str,
        confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]]
    ) -> None:
        """Create interactive forecast plot."""
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data.iloc[:, 0],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add confidence intervals if provided
        if confidence_intervals:
            lower, upper = confidence_intervals
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=lower,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.2)',
                name='Confidence Interval'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_anomalies(
        self,
        data: pd.DataFrame,
        anomaly_data: pd.DataFrame,
        title: str = "Anomaly Detection",
        interactive: bool = True
    ) -> None:
        """
        Plot time series with detected anomalies.
        
        Args:
            data: Original time series data
            anomaly_data: Data with anomaly labels
            title: Plot title
            interactive: Whether to create interactive plot
        """
        if interactive:
            self._plot_interactive_anomalies(data, anomaly_data, title)
        else:
            self._plot_static_anomalies(data, anomaly_data, title)
    
    def _plot_static_anomalies(
        self, 
        data: pd.DataFrame, 
        anomaly_data: pd.DataFrame, 
        title: str
    ) -> None:
        """Create static anomaly plot."""
        plt.figure(figsize=self.figsize)
        
        # Plot normal data
        normal_mask = ~anomaly_data['is_anomaly']
        plt.plot(data.index[normal_mask], data.iloc[normal_mask, 0], 
                'o', color='blue', alpha=0.6, label='Normal')
        
        # Plot anomalies
        anomaly_mask = anomaly_data['is_anomaly']
        plt.plot(data.index[anomaly_mask], data.iloc[anomaly_mask, 0], 
                'o', color='red', markersize=8, label='Anomaly')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_interactive_anomalies(
        self, 
        data: pd.DataFrame, 
        anomaly_data: pd.DataFrame, 
        title: str
    ) -> None:
        """Create interactive anomaly plot."""
        fig = go.Figure()
        
        # Add normal data
        normal_mask = ~anomaly_data['is_anomaly']
        fig.add_trace(go.Scatter(
            x=data.index[normal_mask],
            y=data.iloc[normal_mask, 0],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Add anomalies
        anomaly_mask = anomaly_data['is_anomaly']
        fig.add_trace(go.Scatter(
            x=data.index[anomaly_mask],
            y=data.iloc[anomaly_mask, 0],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            hovermode='closest',
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_model_comparison(
        self,
        actual: np.ndarray,
        predictions: Dict[str, np.ndarray],
        title: str = "Model Comparison",
        interactive: bool = True
    ) -> None:
        """
        Compare multiple model predictions.
        
        Args:
            actual: Actual values
            predictions: Dictionary of model predictions
            title: Plot title
            interactive: Whether to create interactive plot
        """
        if interactive:
            self._plot_interactive_comparison(actual, predictions, title)
        else:
            self._plot_static_comparison(actual, predictions, title)
    
    def _plot_static_comparison(
        self, 
        actual: np.ndarray, 
        predictions: Dict[str, np.ndarray], 
        title: str
    ) -> None:
        """Create static comparison plot."""
        plt.figure(figsize=self.figsize)
        
        # Plot actual values
        plt.plot(actual, label='Actual', linewidth=2, color='black')
        
        # Plot predictions
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for i, (model_name, pred) in enumerate(predictions.items()):
            plt.plot(pred, label=model_name, linewidth=2, 
                    color=colors[i], linestyle='--')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Time Steps', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def _plot_interactive_comparison(
        self, 
        actual: np.ndarray, 
        predictions: Dict[str, np.ndarray], 
        title: str
    ) -> None:
        """Create interactive comparison plot."""
        fig = go.Figure()
        
        # Add actual values
        fig.add_trace(go.Scatter(
            y=actual,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=2)
        ))
        
        # Add predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, pred) in enumerate(predictions.items()):
            fig.add_trace(go.Scatter(
                y=pred,
                mode='lines',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time Steps',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_decomposition(
        self,
        data: pd.DataFrame,
        title: str = "Time Series Decomposition",
        interactive: bool = True
    ) -> None:
        """
        Plot time series decomposition (trend, seasonal, residual).
        
        Args:
            data: Time series data
            title: Plot title
            interactive: Whether to create interactive plot
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Perform decomposition
        decomposition = seasonal_decompose(data.iloc[:, 0], model='additive', period=12)
        
        if interactive:
            self._plot_interactive_decomposition(decomposition, title)
        else:
            self._plot_static_decomposition(decomposition, title)
    
    def _plot_static_decomposition(self, decomposition, title: str) -> None:
        """Create static decomposition plot."""
        fig, axes = plt.subplots(4, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        decomposition.observed.plot(ax=axes[0], title='Original')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def _plot_interactive_decomposition(self, decomposition, title: str) -> None:
        """Create interactive decomposition plot."""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
            vertical_spacing=0.1
        )
        
        # Add traces
        fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed,
                               mode='lines', name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend,
                               mode='lines', name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal,
                               mode='lines', name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid,
                               mode='lines', name='Residual'), row=4, col=1)
        
        fig.update_layout(
            title=title,
            height=800,
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        title: str = "Correlation Matrix"
    ) -> None:
        """
        Plot correlation matrix for multivariate time series.
        
        Args:
            data: Multivariate time series data
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        correlation_matrix = data.corr()
        
        sns.heatmap(
            correlation_matrix,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.2f'
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
