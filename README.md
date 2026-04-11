# Aegis - Adversarial Stress Testing & Tail Risk Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Latest-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

Aegis is an institutional-grade risk intelligence platform designed for adversarial scenario generation, tail risk quantification, and correlation breakdown simulation. Built with Streamlit, Plotly, and advanced financial modeling techniques.

**Live App**: [https://aegis-stress-testing-platform.streamlit.app/](https://aegis-stress-testing-platform.streamlit.app/)

---

## Overview

Aegis provides comprehensive tools for portfolio stress testing and risk analysis:

- **Monte Carlo Simulation** - Generate thousands of realistic market scenarios
- **Tail Risk Analytics** - VaR, CVaR, Maximum Drawdown, Kurtosis, Skewness
- **Market Regime Detection** - Classify market conditions (Bull, Normal, Stress, Crisis)
- **Advanced Charting** - 20+ candlestick variants, technical indicators, 3D visualizations
- **Stress Testing** - Volatility shock, drift shift, correlation crush scenarios
- **Candlestick Pattern Recognition** - AI-powered pattern detection and signal generation

---

## Features

### 📊 Comprehensive Analytics
- **Monte Carlo Simulation** - Run thousands of scenarios with customizable parameters
- **Tail Risk Analysis** - VaR, CVaR, Maximum Drawdown, Kurtosis, Skewness
- **Regime Detection Engine** - Classify market regimes (Bull, Normal, Stress, Crisis)

### 🕯️ Advanced Charting
- **20+ Candlestick Variants** - Standard, Heikin-Ashi, Renko, Kagi, Ichimoku, Zigzag, Multi-Timeframe
- **10 Candlestick Patterns** - Hammer, Doji, Engulfing, Morning/Evening Star, and more
- **Technical Analysis** - Bollinger Bands, RSI, MACD, Moving Averages
- **3D Risk Surface** - Multi-dimensional visualization of portfolio risk

### 🎨 Visualizations
- **Pie/Donut Charts** - Asset allocation visualization
- **Treemap** - Portfolio weight distribution
- **Radar Chart** - Multi-asset performance comparison
- **Correlation Matrix** - Asset correlation heatmap

### ⚡ Stress Testing
- **Volatility Shock** - Simulate VIX spikes
- **Drift Shift** - Model bear market scenarios
- **Correlation Crush** - Test portfolio resilience during market stress
- **Fat-Tail Scenarios** - VAE & Bootstrap tail amplification

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web UI Framework |
| **Plotly** | Interactive Visualizations |
| **NumPy** | Numerical Computing |
| **Pandas** | Data Manipulation |
| **SciPy** | Statistical Functions |
| **yfinance** | Market Data API |

---

## Getting Started

### Prerequisites
- Python 3.10+
- Windows/macOS/Linux

### Installation

1. Clone the repository:
```bash
git clone https://github.com/sourishdey2005/Aegis-Stress-Testing-Platform.git
cd Aegis-Stress-Testing-Platform
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the application:
```bash
streamlit run aegis_app.py
```

Or visit the live app: [https://aegis-stress-testing-platform.streamlit.app/](https://aegis-stress-testing-platform.streamlit.app/)

---

## Usage Guide

### Selecting Assets
- Use the **multiselect dropdown** to choose from 70+ trending tickers
- Add custom tickers manually via the text input
- Select asset class: Equities, Crypto, or Mixed

### Configuring Simulations
- **Monte Carlo**: Set number of simulations (500-25,000)
- **Forecast Horizon**: 30-252 trading days
- **Confidence Level**: 90-99%

### Scenario Models
- **VAE (Fat-Tail)** - Variational Autoencoder for extreme scenarios
- **Block Bootstrap** - Historical resampling with block size control
- **Ensemble** - Combine both models for robust results

### Stress Parameters
- **Volatility Shock**: Simulate VIX spikes
- **Drift Shift**: Model negative drift scenarios
- **Correlation Crush**: Test during correlation breakdown

### Candlestick Pattern Analysis
- Select a ticker for pattern analysis
- Toggle "Render pattern charts" to display
- Choose specific patterns or view all 10

---

## Sample Output

The platform provides:

- **KPI Cards** - VaR, CVaR, Max Drawdown, Tail Ratio
- **Path Simulation** - Fan chart of 5000+ scenarios
- **Tail Distribution** - Histogram with VaR/CVaR markers
- **Drawdown Analysis** - Waterfall and histogram views
- **3D Surface** - Time × Percentile × Portfolio Value
- **Candlestick Charts** - 20+ variants with pattern recognition

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Contact

For questions or feedback, reach out to:
- **Email**: sourish713321@gmail.com
- **GitHub**: [sourishdey2005](https://github.com/sourishdey2005)

---

<p align="center">
  Made with ❤️ by <strong>Sourish Dey</strong>
</p>