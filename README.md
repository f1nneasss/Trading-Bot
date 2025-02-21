
# Trading Bot

## Overview
This cryptocurrency trading bot is built with Python to automate trading strategies using technical analysis and reinforcement learning. It integrates with the Alpaca trading API for live trading and supports historical backtesting.

---

## Features
- **Trading Strategies**:
  - Buy and Hold Strategy
  - SMA Crossover Strategy
  - RSI Mean Reversion Strategy
  - Reinforcement Learning using PPO (Proximal Policy Optimization)
- **Backtesting**:
  - Historical backtesting on predefined market scenarios.
  - Performance metrics: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor.
- **Live Trading**:
  - Alpaca trading API integration.
  - Multi-source price fetching from Coinbase and Binance.
- **Notifications**:
  - Email notifications for trade alerts and errors.
- **Logging**:
  - Detailed logging for debugging and monitoring.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd trading-bot
   ```

2. **Install Dependencies**:
   Install the required Python packages using:
   ```bash
   pip install numpy pandas matplotlib ccxt torch gymnasium stable-baselines3 alpaca_trade_api psutil
   ```

3. **Set Up Environment Variables**:
   Configure the following environment variables:
   ```env
   ALPACA_KEY=your_alpaca_api_key
   ALPACA_SECRET=your_alpaca_api_secret
   EMAIL_SENDER=your_email@gmail.com
   EMAIL_RECEIVER=receiver_email@gmail.com
   EMAIL_PASSWORD=your_app_specific_password
   ```

---

## Usage

### Live Trading

To start live trading, ensure you have a **trained model** saved at:
```
./models/trained_model_live_trading.zip
```

Run the script with:
```bash
python 221618J_Trading_Bot_Code.py
```

Your main code:
- Loads the model from `./models/trained_model_live_trading.zip`.
- Starts live trading immediately upon execution.
- Trades for **6 hours** (`TRADING_DURATION = 360` minutes).
- Logs trading activity and any errors.

**Example Log Output:**
```
=== Starting Live Trading ===
Using model from: ./models/trained_model_live_trading.zip
Trading Duration: 360 minutes
[INFO] Buy Order Executed: 0.01 BTC @ $48,000
[INFO] Sell Order Executed: 0.01 BTC @ $50,000
Trading session completed successfully
```

---

### Requirements for Live Trading

1. **Trained Model**:
   - Ensure the model is saved as:
     ```
     ./models/trained_model_live_trading.zip
     ```

2. **Alpaca API Keys**:
   - Set the following environment variables:
     ```env
     ALPACA_KEY=your_alpaca_api_key
     ALPACA_SECRET=your_alpaca_api_secret
     ```

3. **Email Notifications**:
   - Configure the following for email alerts:
     ```env
     EMAIL_SENDER=your_email@gmail.com
     EMAIL_RECEIVER=receiver_email@gmail.com
     EMAIL_PASSWORD=your_app_specific_password
     ```

---

## Backtesting

### Overview
The trading bot supports historical backtesting to evaluate the performance of different strategies on past market data, including:
- **Buy and Hold Strategy**
- **SMA Crossover Strategy**
- **RSI Mean Reversion Strategy**
- **Reinforcement Learning Strategy (Optional)**

---

### Fetching Historical Data

To fetch historical market data from Binance:
```python
from 221618J_Trading_Bot_Code import fetch_market_data

# Fetch 1-hour candlestick data for Bitcoin (BTCUSDT) from Binance
data = fetch_market_data(
    symbol='BTCUSDT',
    start_date='2022-01-01',
    end_date='2022-12-31',
    timeframe='1h'
)

# Display the first few rows
print(data.head())
```

---

### Running a Backtest

To perform a backtest on a specific strategy:
```python
from 221618J_Trading_Bot_Code import Backtester, SMACrossoverStrategy

# Initialize the backtester
backtester = Backtester()

# Fetch historical data
data = fetch_market_data('BTCUSDT', '2022-01-01', '2022-12-31', '1h')

# Initialize the strategy
strategy = SMACrossoverStrategy(fast_period=20, slow_period=50)

# Run the backtest
results = backtester.backtest_strategy(strategy, data, period=None)

# Generate a performance report
report = backtester.generate_report({f'SMACrossover': results})
print(report)
```

---

### Performance Metrics

The following performance metrics are calculated:
- **Total Return**: Overall percentage return for the period
- **Max Drawdown**: Largest peak-to-trough decline in portfolio value
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / Gross losses (>1 indicates profitability)

---

### Plotting Equity Curves

To visualize the performance of each strategy:
```python
backtester.plot_equity_curves(results, period_name='2022-01-01_to_2022-12-31')
```

This will generate equity curves showing the portfolio value over time for each strategy.

---

## License
This project is licensed under the MIT License. 

---

## Disclaimer
This trading bot is for educational purposes only. Use at your own risk. Cryptocurrency trading is highly volatile and may result in significant financial loss.
