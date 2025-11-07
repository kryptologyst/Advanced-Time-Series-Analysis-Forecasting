# Advanced Time Series Analysis & Forecasting

A comprehensive Python project for time series analysis, forecasting, and anomaly detection using state-of-the-art methods including ARIMA, Prophet, LSTM, and Isolation Forest.

## Features

### **Advanced Forecasting Models**
- **ARIMA**: AutoRegressive Integrated Moving Average with automatic parameter selection
- **Prophet**: Facebook's robust forecasting tool with trend and seasonality detection
- **LSTM**: Deep learning approach using Long Short-Term Memory networks
- **Auto-ARIMA**: Automatic parameter optimization using pmdarima

### **Anomaly Detection**
- **Isolation Forest**: Unsupervised anomaly detection for identifying outliers
- **Configurable contamination rates** for different anomaly detection sensitivity
- **Interactive visualization** of detected anomalies

### **Rich Visualizations**
- **Interactive Plotly charts** for exploration and analysis
- **Static matplotlib/seaborn plots** for publication-ready figures
- **Time series decomposition** (trend, seasonal, residual)
- **Model comparison dashboards**
- **Forecast confidence intervals**

### **Data Sources**
- **Real stock data** via Yahoo Finance API
- **Synthetic data generation** with configurable trends and seasonality
- **Multiple stock symbols** support
- **Flexible date ranges**

### **User Interfaces**
- **Streamlit web dashboard** for interactive analysis
- **Command-line interface** for batch processing
- **Jupyter notebook support** for research and development

### **Modern Architecture**
- **Type hints** throughout the codebase
- **Comprehensive logging** with configurable levels
- **YAML configuration** management
- **Model persistence** with save/load functionality
- **Unit tests** with pytest
- **PEP8 compliant** code style

## Quick Start

### Prerequisites
- Python 3.10+
- pip or conda package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/kryptologyst/Advanced-Time-Series-Analysis-Forecasting.git
cd Advanced-Time-Series-Analysis-Forecasting
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit dashboard**
```bash
streamlit run src/streamlit_app.py
```

4. **Or run the command-line interface**
```bash
python src/main.py
```

## Project Structure

```
time-series-analysis/
├── src/                          # Source code
│   ├── data_handler.py           # Data loading and preprocessing
│   ├── models.py                 # Forecasting models (ARIMA, Prophet, LSTM)
│   ├── visualization.py          # Plotting and visualization utilities
│   ├── main.py                   # Command-line interface
│   └── streamlit_app.py         # Web dashboard
├── config/                       # Configuration files
│   └── config.yaml              # Main configuration
├── data/                         # Data storage
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed data files
├── models/                       # Model storage
│   └── saved/                   # Saved model files
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
│   └── test_timeseries.py       # Test suite
├── logs/                         # Log files
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## Configuration

The project uses YAML configuration files for easy customization. Edit `config/config.yaml`:

```yaml
data:
  sources:
    stock_symbols: ["AAPL", "GOOGL", "MSFT"]
    start_date: "2020-01-01"
    end_date: "2023-12-31"
  
  synthetic:
    enabled: true
    n_samples: 1000
    trend_strength: 0.1
    seasonality_period: 12

models:
  arima:
    order: [5, 1, 2]
    auto_arima: true
  
  prophet:
    yearly_seasonality: true
    weekly_seasonality: true
  
  lstm:
    sequence_length: 60
    hidden_units: 50
    epochs: 100

forecasting:
  horizon: 30
  confidence_intervals: [0.8, 0.95]

visualization:
  figure_size: [12, 8]
  interactive: true
```

## Usage Examples

### Command Line Interface

```python
from src.main import TimeSeriesAnalyzer

# Initialize analyzer
analyzer = TimeSeriesAnalyzer("config/config.yaml")

# Run complete analysis with synthetic data
results = analyzer.run_full_analysis(use_synthetic=True)

# Run analysis with real stock data
results = analyzer.run_full_analysis(use_synthetic=False, symbol="AAPL")

# Access results
print("Training Results:", results['training_results'])
print("Anomalies:", results['anomaly_data']['is_anomaly'].sum())
print("Forecasts:", results['forecasts'])
```

### Programmatic Usage

```python
from src.data_handler import DataLoader
from src.models import ARIMAForecaster, ProphetForecaster
from src.visualization import TimeSeriesVisualizer

# Load data
loader = DataLoader({})
data = loader.load_stock_data("AAPL", "2020-01-01", "2023-01-01")

# Train models
arima_model = ARIMAForecaster(auto_arima=True)
arima_model.fit(data)

prophet_model = ProphetForecaster()
prophet_model.fit(data)

# Generate forecasts
arima_forecast = arima_model.predict(steps=30)
prophet_forecast = prophet_model.predict(steps=30)

# Visualize results
visualizer = TimeSeriesVisualizer()
visualizer.plot_forecast(data, arima_forecast, 
                        pd.date_range(data.index[-1], periods=30, freq='D'),
                        "ARIMA Forecast")
```

### Streamlit Dashboard

Launch the interactive web interface:

```bash
streamlit run src/streamlit_app.py
```

Features:
- **Data Source Selection**: Choose between synthetic or real stock data
- **Model Configuration**: Select which models to train
- **Interactive Visualizations**: Explore results with Plotly charts
- **Real-time Analysis**: Train models and generate forecasts on demand
- **Anomaly Detection**: Configure and visualize anomaly detection results

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python tests/test_timeseries.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

The test suite covers:
- Data loading and preprocessing
- Model training and prediction
- Anomaly detection
- Evaluation metrics
- Integration tests

## Model Performance

### Supported Metrics
- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

### Model Comparison
The framework automatically compares model performance on test data and provides:
- Side-by-side forecast comparisons
- Performance metrics for each model
- Confidence intervals for forecasts
- Visual comparison charts

## Anomaly Detection

The anomaly detection system uses Isolation Forest to identify outliers:

```python
from src.models import AnomalyDetector

# Initialize detector
detector = AnomalyDetector(contamination=0.1)

# Fit on data
detector.fit(data)

# Detect anomalies
anomaly_data = detector.detect_anomalies(data)

# Visualize results
visualizer.plot_anomalies(data, anomaly_data, "Anomaly Detection")
```

## Visualization Features

### Static Plots (Matplotlib/Seaborn)
- Time series plots with trends
- Model comparison charts
- Anomaly detection visualizations
- Decomposition plots

### Interactive Plots (Plotly)
- Zoomable time series charts
- Hover information
- Dynamic legend toggling
- Export capabilities

### Dashboard Features
- Real-time model training progress
- Interactive parameter adjustment
- Multiple chart types in one view
- Responsive design

## Advanced Configuration

### Logging Configuration
```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/timeseries.log"
```

### Model Persistence
```python
# Save trained models
analyzer.save_models("models/saved/")

# Load models
arima_model = ARIMAForecaster()
arima_model.load_model("models/saved/arima_model.pkl")
```

### Custom Data Sources
Extend the `DataLoader` class to support additional data sources:

```python
class CustomDataLoader(DataLoader):
    def load_custom_data(self, source):
        # Implement custom data loading
        pass
```

## Performance Optimization

### LSTM Training
- GPU support via PyTorch
- Configurable batch sizes
- Early stopping
- Learning rate scheduling

### Memory Management
- Efficient data preprocessing
- Chunked data loading for large datasets
- Model checkpointing

### Parallel Processing
- Multi-threaded data loading
- Parallel model training (where applicable)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run code formatting
black src/ tests/

# Run linting
flake8 src/ tests/

# Run type checking
mypy src/
```

## Dependencies

### Core Libraries
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scipy**: Scientific computing

### Time Series
- **statsmodels**: Statistical models
- **pmdarima**: Auto-ARIMA
- **prophet**: Facebook's forecasting tool
- **tslearn**: Time series machine learning
- **darts**: Time series forecasting
- **sktime**: Scikit-learn compatible time series

### Deep Learning
- **torch**: PyTorch for LSTM models
- **tensorflow**: Alternative deep learning backend

### Visualization
- **matplotlib**: Static plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive plotting

### Web Interface
- **streamlit**: Web dashboard
- **jupyter**: Notebook support

### Utilities
- **yfinance**: Stock data API
- **pyyaml**: Configuration management
- **python-dotenv**: Environment variables

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Facebook Prophet** team for the excellent forecasting library
- **Statsmodels** contributors for comprehensive time series tools
- **PyTorch** team for deep learning capabilities
- **Streamlit** for the intuitive web framework
- **Plotly** for interactive visualization capabilities

## Support

For questions, issues, or contributions:
- Open an issue on GitHub
- Check the documentation in the `notebooks/` directory
- Review the test cases for usage examples


# Advanced-Time-Series-Analysis-Forecasting
