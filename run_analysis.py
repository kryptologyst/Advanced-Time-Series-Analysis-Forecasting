#!/usr/bin/env python3
"""
Simple script to run time series analysis.

This script provides a quick way to run the complete time series analysis
with default settings.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import TimeSeriesAnalyzer


def main():
    """Run the complete time series analysis."""
    print("🚀 Starting Time Series Analysis...")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer()
    
    # Run analysis with synthetic data
    print("📊 Running analysis with synthetic data...")
    results = analyzer.run_full_analysis(use_synthetic=True)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📈 ANALYSIS SUMMARY")
    print("=" * 50)
    
    # Model performance
    print("\n🤖 Model Performance:")
    for model_name, result in results['training_results'].items():
        if result['status'] == 'success':
            print(f"\n{model_name} Model:")
            for metric, value in result['metrics'].items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"\n{model_name} Model: Failed - {result['error']}")
    
    # Anomaly detection
    n_anomalies = results['anomaly_data']['is_anomaly'].sum()
    print(f"\n🚨 Anomalies Detected: {n_anomalies}")
    
    # Data summary
    print(f"\n📊 Data Summary:")
    print(f"  Training samples: {results['data_info']['train_size']}")
    print(f"  Test samples: {results['data_info']['test_size']}")
    print(f"  Total samples: {results['data_info']['total_size']}")
    
    # Forecasts
    print(f"\n🔮 Forecasts Generated: {len(results['forecasts'])} models")
    for model_name, forecast in results['forecasts'].items():
        print(f"  {model_name}: {len(forecast)} steps")
    
    print("\n✅ Analysis completed successfully! 🎉")
    print("\n💡 Next steps:")
    print("  - Run 'streamlit run src/streamlit_app.py' for interactive dashboard")
    print("  - Check 'notebooks/timeseries_analysis_demo.ipynb' for detailed examples")
    print("  - Modify 'config/config.yaml' to customize settings")


if __name__ == "__main__":
    main()
