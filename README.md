# Algorithmic Trading ML Strategy

A comprehensive machine learning-driven algorithmic trading system that predicts market movements and executes trading strategies with proper backtesting, risk management, and performance evaluation.

## 🎯 Project Overview

This project implements a full-stack algorithmic trading system that:
- Collects market data from multiple sources (Alpha Vantage, NewsAPI, etc.)
- Engineers sophisticated features using technical analysis and sentiment data
- Trains multiple ML models (XGBoost, LSTM, ensemble methods)
- Backtests strategies with realistic trading conditions
- Manages risk through position sizing and portfolio constraints
- Tracks experiments and models using MLflow

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Ingestion│    │Feature Engine   │    │  ML Pipeline    │
│   Service       │───▶│   Service       │───▶│   Service       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │    MLflow       │
│   Raw Data      │    │   Cache Layer   │    │  Model Registry │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                 │
                                 ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │  Trading Engine │    │  Backtesting    │
                    │   Service       │───▶│   Service       │
                    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Poetry (for dependency management)
- Docker & Docker Compose
- PostgreSQL (or use Docker)
- Redis (or use Docker)

### Setup

1. **Clone and navigate to the project**:
   ```bash
   git clone <repository-url>
   cd algorithmic-trading-ml
   ```

2. **Set up the development environment**:
   ```bash
   make setup-dev
   ```
   This will:
   - Copy `.env.template` to `.env`
   - Install all dependencies
   - Set up pre-commit hooks
   - Start Docker services

3. **Edit your environment variables**:
   ```bash
   # Edit the .env file with your API keys
   vim .env
   
   # At minimum, you'll need:
   ALPHA_VANTAGE_API_KEY=your_key_here
   NEWSAPI_KEY=your_key_here  # optional
   ```

4. **Verify the setup**:
   ```bash
   # Check if all services are running
   docker-compose ps
   
   # Run tests to verify everything works
   make test
   ```

### API Keys Required

- **Alpha Vantage**: Primary market data source
  - Get free key at: https://www.alphavantage.co/support/#api-key
  - Free tier: 500 calls/day, 5 calls/minute
  - Paid tier: Unlimited calls, faster rates

- **NewsAPI** (Optional): For sentiment analysis
  - Get free key at: https://newsapi.org/register
  - Free tier: 1000 requests/month

## 📊 Usage

### Development Commands

```bash
# View all available commands
make help

# Start development environment
make docker-up

# Run tests
make test
make test-coverage

# Code quality checks
make lint
make format

# Data collection
make collect-data

# Model training
make train-model

# Backtesting
make run-backtest

# Start Jupyter for exploration
make jupyter

# Cache management
python -m src.cli.cache stats        # View cache statistics
python -m src.cli.cache clear-expired # Clear expired entries
python -m src.cli.cache clear-all     # Clear all cache
python -m src.cli.cache invalidate AAPL # Clear data for symbol
```

### Services URLs (when running with Docker)

- **MLflow UI**: http://localhost:5000
- **FastAPI Docs**: http://localhost:8000/docs
- **Streamlit Dashboard**: http://localhost:8501
- **Jupyter Lab**: http://localhost:8888
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## 📁 Project Structure

```
algorithmic-trading-ml/
├── src/                          # Source code
│   ├── data_collection/          # Market data ingestion
│   ├── feature_engineering/      # Feature creation and transformation
│   ├── models/                   # ML model implementations
│   ├── strategy/                 # Trading strategies and signal generation
│   ├── backtesting/              # Historical testing framework
│   ├── risk_management/          # Risk controls and portfolio management
│   └── api/                      # FastAPI web interface
├── data/                         # Data storage
│   ├── raw/                      # Raw market data
│   ├── processed/                # Engineered features
│   ├── cache/                    # API response cache (auto-generated)
│   └── external/                 # Economic indicators, news, etc.
├── notebooks/                    # Jupyter notebooks for analysis
├── tests/                        # Test suites
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── e2e/                      # End-to-end tests
├── config/                       # Configuration files
├── docker/                       # Docker configurations
└── results/                      # Backtest results and reports
```

## 🧪 Testing Strategy

The project implements comprehensive testing:

- **Unit Tests**: Individual function/class testing (70% coverage target)
- **Integration Tests**: Service-to-service communication
- **End-to-End Tests**: Complete workflow testing
- **Performance Tests**: Benchmarking critical paths

Run tests with:
```bash
make test           # All tests
make test-unit      # Unit tests only
make test-coverage  # With coverage report
```

## 📈 Features

### Data Collection
- Multiple data sources with fallback mechanisms
- **Intelligent Caching System**: Minimizes API calls with smart expiration policies
- Rate limit handling with exponential backoff
- Real-time and batch data processing
- Data validation and quality checks
- Cache management CLI for monitoring and cleanup

### Feature Engineering
- 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Sentiment analysis from news and social media
- Macro economic indicators
- Rolling window statistics and transformations

### Machine Learning Models
- **XGBoost/LightGBM**: Gradient boosting for structured data
- **LSTM/GRU**: Deep learning for sequential patterns
- **Ensemble Methods**: Combining multiple model predictions
- **AutoML**: Automated hyperparameter optimization with Optuna

### Trading Strategies
- **Mean Reversion**: Buy oversold, sell overbought
- **Momentum**: Follow trending assets
- **Pairs Trading**: Long/short correlated pairs
- **Multi-factor**: Combine technical, fundamental, macro signals

### Risk Management
- Position sizing based on volatility
- Portfolio diversification constraints
- Stop-loss and take-profit mechanisms
- Dynamic risk adjustment based on market conditions

### Backtesting
- Realistic trading costs and slippage
- Walk-forward validation
- Multiple performance metrics (Sharpe, Sortino, Calmar ratios)
- Statistical significance testing

## ⚠️ Risk Disclaimer

This is a research and educational project for **paper trading only**. The system:
- Does NOT execute real trades
- Is NOT financial advice
- Should NOT be used with real money without extensive additional testing
- Past performance does not guarantee future results

## 🔧 Configuration

Key configuration parameters in `.env`:

```bash
# Trading Parameters
INITIAL_CAPITAL=100000.0          # Starting capital for backtests
MAX_POSITION_SIZE=0.05            # Max 5% per position
MAX_SECTOR_EXPOSURE=0.20          # Max 20% per sector
MAX_DRAWDOWN_THRESHOLD=0.10       # Stop trading at 10% drawdown

# Risk Management
STOP_LOSS_PCT=0.05               # 5% stop loss
TAKE_PROFIT_PCT=0.10             # 10% take profit

# Model Training
MODEL_RETRAIN_INTERVAL_DAYS=7    # Retrain weekly
CROSS_VALIDATION_FOLDS=3         # 3-fold time series CV
```

## 📊 Performance Monitoring

The system tracks multiple performance metrics:

### Strategy Performance
- **Total Return**: Absolute and annualized returns
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

### Model Performance
- **Prediction Accuracy**: Direction prediction accuracy
- **Precision/Recall**: Signal quality metrics
- **Information Ratio**: Excess return per unit of tracking error

### System Performance
- **Data Latency**: Time from market event to signal
- **Model Inference Time**: Prediction generation speed
- **API Uptime**: Data source availability

## 🛠️ Development

### Code Quality Standards
- Type hints required for all functions
- Google-style docstrings
- Black code formatting (88 character line limit)
- Import sorting with isort
- Linting with flake8, mypy, bandit

### Git Workflow
- Feature branch development
- Pre-commit hooks for quality checks
- Conventional commit messages
- Pull request reviews required

## 📚 Documentation

- **API Documentation**: Available at `/docs` endpoint
- **Code Documentation**: Generated with Sphinx
- **Architecture Decision Records**: In `/docs/adr/`
- **User Guides**: In `/docs/guides/`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run quality checks: `make check-all`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Alpha Vantage](https://www.alphavantage.co/) for market data
- [TA-Lib](https://ta-lib.org/) for technical analysis indicators
- [MLflow](https://mlflow.org/) for experiment tracking
- [Backtrader](https://www.backtrader.com/) for backtesting framework

---

**⚠️ Important**: This is a research project for educational purposes only. Do not use for actual trading without proper due diligence and risk management.
