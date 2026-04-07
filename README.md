# Aegis - Adversarial Stress Testing & Tail Risk Intelligence Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Latest-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

Aegis is an institutional-grade risk intelligence platform designed for adversarial scenario generation, tail risk quantification, and correlation breakdown simulation. Built with Streamlit, Plotly, and financial modeling techniques.

![Aegis Platform](https://via.placeholder.com/800x400?text=Aegis+Stress+Testing+Platform)

---

## ✨ Features

### 📊 Comprehensive Analytics
- **Monte Carlo Simulation** - Run thousands of scenarios with customizable parameters
- **Tail Risk Analysis** - VaR, CVaR, Maximum Drawdown, Kurtosis, Skewness
- **Regime Detection Engine** - Classify market regimes (Bull, Normal, Stress, Crisis)

### 🕯️ Advanced Charting
- **20+ Candlestick Variants** - Standard, Heikin-Ashi, Renko, Ichimoku, and more
- **Technical Analysis** - Bollinger Bands, RSI, MACD, Moving Averages
- **3D Risk Surface** - Multi-dimensional visualization of portfolio risk

### 🎨 Visualizations
- **Pie/Donut Charts** - Asset allocation visualization
- **Treemap** - Portfolio weight distribution
- **Radar Chart** - Multi-asset performance comparison
- **Correlation Matrix** - Asset correlation heatmap

### ⚡ Stress Testing
- **Volatility Shock** - Simulate volatility spikes
- **Drift Shift** - Model bear market scenarios
- **Correlation Crush** - Test portfolio resilience during market stress
- **Tail Amplification** - Fat-tail scenario generation with VAE & Bootstrap

---

## 🚀 Getting Started

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

---

## 📖 Usage Guide

### Selecting Assets
- Use the **multiselect dropdown** to choose from 70+ trending tickers
- Add custom tickers manually via the text input
- Select asset class: Equities, Crypto, or Mixed

### Configuring Simulations
- **Monte Carlo**: Set number of simulations (500-50,000)
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

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web UI Framework |
| **Plotly** | Interactive Visualizations |
| **NumPy** | Numerical Computing |
| **Pandas** | Data Manipulation |
| **SciPy** | Statistical Functions |
| **yfinance** | Market Data API |

---

## 📊 Sample Output

The platform provides:

- **KPI Cards** - VaR, CVaR, Max Drawdown, Tail Ratio
- **Path Simulation** - Fan chart of 5000+ scenarios
- **Tail Distribution** - Histogram with VaR/CVaR markers
- **Drawdown Analysis** - Waterfall and histogram views
- **3D Surface** - Time × Percentile × Portfolio Value

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions or feedback, reach out to:
- **Email**: sourish713321@gmail.com
- **GitHub**: [sourishdey2005](https://github.com/sourishdey2005)

---

<p align="center">
  Made with ❤️ by <strong>Sourish Dey</strong>
</p>
