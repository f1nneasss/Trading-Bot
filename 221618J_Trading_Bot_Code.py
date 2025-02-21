# Python standard library
import os
import gc
import json
import time
import smtplib
import logging
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Any
import requests
import pandas as pd
import time
import ccxt
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from decimal import Decimal, ROUND_DOWN
import logging
import os
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Union, Optional
import os
import logging
import requests
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from datetime import datetime, timedelta
from typing import Dict, Optional, Union

# Third-party libraries
import numpy as np
import pandas as pd
import pandas_ta as ta
import gymnasium as gym
import matplotlib.pyplot as plt
import ccxt
import psutil
import torch.nn as nn
import alpaca_trade_api as tradeapi

# Stable-baselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_model_training.log'),
        logging.StreamHandler()
    ]
)


def get_memory_usage() -> Dict[str, Any]:
    """Get current memory usage statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss': memory_info.rss / 1024 / 1024,  # RSS in MB
        'vms': memory_info.vms / 1024 / 1024,  # VMS in MB
        'percent': process.memory_percent(),
        'num_threads': process.num_threads()
    }

def log_memory_state(location: str):
    """Log current memory usage at a specific location in code"""
    mem = get_memory_usage()
    gc_count = gc.get_count()
    
    logging.info(f"\nMemory State at {location}:")
    logging.info(f"RSS Memory: {mem['rss']:.2f} MB")
    logging.info(f"Virtual Memory: {mem['vms']:.2f} MB")
    logging.info(f"Memory Usage: {mem['percent']:.1f}%")
    logging.info(f"Active Threads: {mem['num_threads']}")
    logging.info(f"GC Counts: {gc_count}\n")
    
    return mem

def monitor_section(section_name: str):
    """Decorator to monitor memory before and after a section of code"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logging.info(f"\nEntering section: {section_name}")
            before_mem = log_memory_state("start")
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                after_mem = log_memory_state("end")
                mem_diff = after_mem['rss'] - before_mem['rss']
                logging.info(f"Memory change in {section_name}: {mem_diff:.2f} MB\n")
                
                if mem_diff > 100:  # Alert if memory increased by more than 100MB
                    logging.warning(f"Large memory increase detected in {section_name}!")
                    
        return wrapper
    return decorator

@dataclass
class TradingConfig:
    ALPACA_ENDPOINT: str = "https://paper-api.alpaca.markets"
    ALPACA_KEY: str = "ALPACA_KEY"
    ALPACA_SECRET: str = "ALPACA_SECRET"
    EMAIL_SENDER: str = "EMAIL_SENDER@gmail.com"
    EMAIL_RECEIVER: str = "EMAIL_RECEIVER@gmail.com"
    EMAIL_PASSWORD: str = "EMAIL_PASSWORD"
    MAX_SESSIONS: int = 3
    DEFAULT_BALANCE: float = 100000.0
    LOGS_DIRECTORY: str = 'logs'
    PRICE_STALENESS_THRESHOLD: int = 30  # seconds
    MAX_TRADE_PERCENTAGE: float = 0.02  # 2% of account per trade

@dataclass
class BacktestPeriod:
    """Represents a specific period for backtesting"""
    name: str
    test_start: datetime
    test_end: datetime
    train_start: datetime
    train_end: datetime
    description: str
    market_conditions: str

@dataclass
class BacktestResult:
    """Contains the results of a backtest run"""
    strategy_name: str
    period_name: str
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    num_trades: int
    avg_profit_per_trade: float
    avg_loss_per_trade: float
    profit_factor: float
    trades: List[Dict]

class BaseStrategy:
    """Base class for all trading strategies"""
    def __init__(self, name: str):
        self.name = name
        self.trades = []
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate trading signals for the given data
        
        Args:
            df: DataFrame with price and indicator data
            
        Returns:
            Series of signals where:
            0 = Hold
            1 = Buy
            2 = Sell
        """
        raise NotImplementedError("Subclasses must implement generate_signals")
    
class BuyHoldStrategy(BaseStrategy):
    """Simple buy and hold strategy"""
    def __init__(self):
        super().__init__("Buy and Hold")
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates the simplest possible strategy:
        - Buy at the start
        - Hold until the end
        - Sell at the end
        """
        signals = pd.Series(0, index=df.index)
        signals.iloc[0] = 1  # Buy at start
        signals.iloc[-1] = 2  # Sell at end
        return signals

class SMACrossoverStrategy(BaseStrategy):
    """Moving average crossover strategy"""
    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__("SMA Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates signals based on SMA crossover:
        - Buy when fast SMA crosses above slow SMA
        - Sell when fast SMA crosses below slow SMA
        """
        price_col = 'Close BTC-USD'
        
        # Calculate SMAs
        df['SMA_fast'] = df[price_col].rolling(window=self.fast_period).mean()
        df['SMA_slow'] = df[price_col].rolling(window=self.slow_period).mean()
        
        # Generate signals
        signals = pd.Series(0, index=df.index)
        
        # Buy when fast crosses above slow
        signals[df['SMA_fast'] > df['SMA_slow']] = 1
        
        # Sell when fast crosses below slow
        signals[df['SMA_fast'] < df['SMA_slow']] = 2
        
        return signals

class RSIStrategy(BaseStrategy):
    """RSI mean reversion strategy"""
    def __init__(self, period: int = 14, overbought: int = 70, oversold: int = 30):
        super().__init__("RSI")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates signals based on RSI levels:
        - Buy when RSI drops below oversold level
        - Sell when RSI rises above overbought level
        
        Uses the RSI_medium column that's already calculated in the dataframe
        from the technical indicators
        """
        signals = pd.Series(0, index=df.index)
        
        # Buy on oversold conditions
        signals[df['RSI_medium'] < self.oversold] = 1
        
        # Sell on overbought conditions
        signals[df['RSI_medium'] > self.overbought] = 2
        
        return signals
    

config = TradingConfig()
# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)


# Update this line
alpaca_api = tradeapi.REST(config.ALPACA_KEY, config.ALPACA_SECRET, config.ALPACA_ENDPOINT)

class Trade:
    def __init__(self, 
                 symbol: str = 'BTCUSD', 
                 entry_price: Optional[float] = None, 
                 quantity: Optional[float] = None):
        """
        Enhanced Trade class with precise tracking and logging
        """
        self.symbol = symbol
        self.entry_price: Optional[Decimal] = Decimal(str(entry_price)) if entry_price is not None else None
        self.entry_time: Optional[datetime] = datetime.now()
        self.quantity: Optional[Decimal] = Decimal(str(quantity)) if quantity is not None else None
        self.exit_price: Optional[Decimal] = None
        self.exit_time: Optional[datetime] = None
        self.pnl: Optional[Decimal] = None
        self.status: str = "OPEN"
        self.exit_reason: Optional[str] = None

    def close(self, 
              exit_price: float, 
              exit_time: Optional[datetime] = None, 
              reason: Optional[str] = None) -> None:
        """
        Close the trade with precise calculations
        """
        self.exit_price = Decimal(str(exit_price))
        self.exit_time = exit_time or datetime.now()
        self.exit_reason = reason
        
        # Precise PnL calculation
        if self.entry_price is not None and self.quantity is not None:
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        
        self.status = "CLOSED"
        logging.info(f"Trade closed: Entry {self.entry_price}, Exit {self.exit_price}, PnL {self.pnl}")

    @property
    def duration(self) -> Optional[float]:
        """Calculate trade duration in minutes"""
        if self.exit_time and self.entry_time:
            return (self.exit_time - self.entry_time).total_seconds() / 60
        return None

class ImprovedTradingBot:
    def __init__(self, 
                 alpaca_key: str, 
                 alpaca_secret: str, 
                 alpaca_base_url: str = "https://paper-api.alpaca.markets"):
        """
        Initialize the improved trading bot with enhanced error handling and price tracking
        
        Args:
            alpaca_key: Alpaca API key
            alpaca_secret: Alpaca API secret
            alpaca_base_url: Alpaca API base URL (defaults to paper trading)
        """
        # Configure logging
        self._setup_logging()
        
        # Alpaca API connection
        self.api = tradeapi.REST(
            key_id=alpaca_key,
            secret_key=alpaca_secret,
            base_url=alpaca_base_url
        )
        
        # Backup price sources
        self.exchanges = {
            'coinbase': 'https://api.coinbase.com/v2/prices/BTC-USD/spot',
            'binance': 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
        }
        
        # Price caching mechanism
        self.last_price_cache = {
            'price': None,
            'timestamp': None,
            'sources': {}
        }
        
        # Configuration
        self.PRICE_CACHE_DURATION = timedelta(seconds=30)
        self.TRADE_SYMBOL = 'BTCUSD'

        # Enhanced price tracking
        self.price_history = {
            'alpaca': [],
            'coinbase': [],
            'binance': []
        }
        self.price_discrepancy_log = []

    def _setup_logging(self):
        """Configure logging with file and console output"""
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/trading_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _fetch_price_from_source(self, source: str) -> Optional[Decimal]:
        """
        Fetch price from a specific source without logging.
        """
        try:
            price = None

            if source == 'alpaca':
                try:
                    quote = self.api.get_latest_crypto_quotes(["BTC/USD"])
                    price = Decimal(str(quote["BTC/USD"].ap))
                except Exception:
                    return None

            elif source == 'coinbase':
                response = requests.get(self.exchanges['coinbase'], timeout=5)
                price = Decimal(response.json()['data']['amount'])

            elif source == 'binance':
                response = requests.get(self.exchanges['binance'], timeout=5)
                price = Decimal(response.json()['price'])

            if price is not None:
                self.last_price_cache['sources'][source] = price

            return price

        except Exception:
            return None




    def get_current_price(self, force_refresh: bool = False) -> Decimal:
        """
        Fetch current market price without logging price details.
        """
        try:
            sources = ['alpaca', 'coinbase', 'binance']
            prices = {}

            for source in sources:
                price = self._fetch_price_from_source(source)
                if price is not None:
                    prices[source] = price

            if not prices:
                raise ValueError("Unable to fetch price from any source")

            selected_price = prices.get('alpaca', sorted(prices.values())[len(prices) // 2])

            self.last_price_cache = {'price': selected_price, 'timestamp': datetime.now()}
            return selected_price

        except Exception:
            if self.last_price_cache.get('price'):
                return self.last_price_cache['price']
            raise




    def get_position(self, symbol: str = None) -> Decimal:
        """
        Get precise current position with decimal precision
        
        Args:
            symbol: Trading symbol (defaults to class default)
        
        Returns:
            Current position quantity as Decimal
        """
        symbol = symbol or self.TRADE_SYMBOL
        
        try:
            # Fetch all positions
            positions = self.api.list_positions()
            
            # Find position for specific symbol
            for position in positions:
                if position.symbol == symbol:
                    return Decimal(position.qty)
            
            # No position found
            return Decimal('0')
        
        except Exception as e:
            self.logger.error(f"Error retrieving position for {symbol}: {e}")
            return Decimal('0')

    def get_account_balance(self) -> Dict[str, Decimal]:
        """
        Calculate precise account balance
        
        Returns:
            Dictionary with cash balance, position value, and total portfolio value
        """
        try:
            # Fetch account details
            account = self.api.get_account()
            
            # Cash balance
            cash_balance = Decimal(account.cash)
            
            # Current market price
            current_price = self.get_current_price()
            
            # Get current position
            position_qty = self.get_position()
            position_value = position_qty * current_price
            
            # Total portfolio value
            total_portfolio_value = cash_balance + position_value
            
            return {
                'cash_balance': cash_balance,
                'position_qty': position_qty,
                'position_value': position_value,
                'total_portfolio_value': total_portfolio_value
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating account balance: {e}")
            raise

    def execute_trade(self, trade_type: str, quantity: Optional[Decimal] = None, symbol: Optional[str] = None) -> Dict:
        """
        Execute a trade with precise quantity management.
        """
        symbol = symbol or self.TRADE_SYMBOL
        
        try:
            current_price = self.get_current_price()
            current_position = self.get_position(symbol)

            if quantity is None:
                if trade_type == 'buy':
                    account_balance = self.get_account_balance()
                    max_buy_amount = account_balance['cash_balance'] * Decimal('0.02')
                    quantity = (max_buy_amount / current_price).quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
                elif trade_type == 'sell':
                    quantity = current_position
                else:
                    raise ValueError(f"Invalid trade type: {trade_type}")

            quantity = quantity.quantize(Decimal('0.00001'), rounding=ROUND_DOWN)

            # Validate trade execution
            if trade_type == 'buy':
                account_balance = self.get_account_balance()
                total_cost = quantity * current_price
                if total_cost > account_balance['cash_balance']:
                    max_qty = (account_balance['cash_balance'] / current_price).quantize(Decimal('0.00001'), rounding=ROUND_DOWN)
                    quantity = max_qty

            elif trade_type == 'sell' and quantity > current_position:
                quantity = current_position

            # Execute order
            order = self.api.submit_order(
                symbol=symbol,
                qty=float(quantity),
                side=trade_type,
                type='market',
                time_in_force='gtc'
            )

            # Log minimal trade info
            logging.info(f"{trade_type.upper()} {float(quantity)} BTC @ ${float(current_price):,.2f}")

            # Only get account balance if needed for the update
            account_balance = self.get_account_balance()
            
            # Position updates only for significant changes
            if abs(float(account_balance['position_value'])) > 0.01:  # Only log if position value changes significantly
                logging.debug(  # Changed to debug level
                    f"Position Update - "
                    f"Cash: ${float(account_balance['cash_balance']):,.2f}, "
                    f"BTC: {float(account_balance['position_qty'])}, "
                    f"Value: ${float(account_balance['position_value']):,.2f}"
                )

            return {
                'symbol': symbol,
                'type': trade_type,
                'quantity': float(quantity),
                'price': float(current_price),
                'total_value': float(quantity * current_price),
                'timestamp': datetime.now(timezone(timedelta(hours=8)))  # Use SGT timezone
            }

        except Exception as e:
            logging.error(f"Trade execution error: {e}")
            raise



    def generate_trade_report(self, trades=None) -> None:
        """
        Generate and print a well-structured Trading Session Report at the end of the session.
        If no trades are provided, it fetches all trades (including open and closed) from Alpaca.
        """
        try:
            # Get account state
            account_balance = self.get_account_balance()

            # Use provided trades or fetch all trades (open and closed) from Alpaca
            trades = trades or self.api.list_orders(status='all')

            # Initialize counters
            total_trades = len(trades)
            buy_trades = 0
            sell_trades = 0

            # Count Buy and Sell trades
            for trade in trades:
                # Check if trade is from Alpaca or custom Trade class
                if hasattr(trade, 'side'):  # Alpaca trade object
                    if trade.side == 'buy':
                        buy_trades += 1
                    elif trade.side == 'sell':
                        sell_trades += 1
                else:  # Custom Trade object
                    if trade.status == 'OPEN':
                        buy_trades += 1
                    elif trade.status == 'CLOSED':
                        sell_trades += 1

            # Account summary
            cash_balance = f"${float(account_balance['cash_balance']):,.2f}"
            position_qty = f"{float(account_balance['position_qty']):,.6f} BTC"
            position_value = f"${float(account_balance['position_value']):,.2f}"
            portfolio_value = f"${float(account_balance['total_portfolio_value']):,.2f}"

            # **Formatted Trade Report Output**
            report = (
                "\n" + "=" * 50 +
                "\n📊 TRADING SESSION REPORT" +
                "\n" + "=" * 50 +
                f"\nTotal Trades      : {total_trades}" +
                f"\nBuy Trades        : {buy_trades}" +
                f"\nSell Trades       : {sell_trades}" +
                f"\nPortfolio Value   : {portfolio_value}" +
                "\n" + "-" * 50 +
                "\n📈 ACCOUNT OVERVIEW" +
                "\n" + "-" * 50 +
                f"\nCash Balance      : {cash_balance}" +
                f"\nPosition Quantity : {position_qty}" +
                f"\nPosition Value    : {position_value}" +
                "\n" + "-" * 50 +
                "\n📌 RECENT TRADE EXECUTIONS"
                "\n" + "-" * 50
            )

            # Show all trades if passed in, else limit to the last 5
            display_trades = trades if trades else trades[:5]

            for trade in display_trades:
                # Check if trade is from Alpaca or custom Trade class
                if hasattr(trade, 'side'):  # Alpaca trade object
                    # Convert UTC to SGT
                    filled_time = trade.filled_at.replace(tzinfo=timezone.utc)
                    sgt_time = filled_time.astimezone(timezone(timedelta(hours=8)))
                    
                    report += (
                        f"\n{trade.side.upper()} {trade.qty} BTC @ ${float(trade.filled_avg_price):,.2f} "
                        f"on {sgt_time.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                else:  # Custom Trade object
                    # Use status to determine side
                    side = "BUY" if trade.status == "OPEN" else "SELL"
                    entry_time = trade.entry_time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    
                    report += (
                    f"\n{side} {trade.quantity} BTC @ ${float(trade.entry_price):,.2f} "
                    f"Entry: {entry_time}"
                    )   


            report += "\n" + "=" * 50  # Closing separator

            self.logger.info(report)  # Print nicely formatted report

        except Exception as e:
            self.logger.error(f"Error generating trade report: {e}")
            raise





class TradeManager:
    def __init__(self, initial_balance: float = config.DEFAULT_BALANCE):
        self.initial_balance = initial_balance
        self.open_trades: List[Trade] = []
        self.completed_trades: List[Trade] = []
        self.portfolio_values: List[float] = []
        self.metrics: Dict = {}
        self.notification_queue = deque(maxlen=100)
        
        self.first_trade_time = None
        self.last_trade_time = None
        self.total_trades_count = 0  

        # Initialize email notifications
        self.email_sender = config.EMAIL_SENDER
        self.email_receiver = config.EMAIL_RECEIVER
        self.email_password = config.EMAIL_PASSWORD

        # Initialize trading bot
        self.trading_bot = ImprovedTradingBot(
            alpaca_key=config.ALPACA_KEY,
            alpaca_secret=config.ALPACA_SECRET,
            alpaca_base_url=config.ALPACA_ENDPOINT
        )
        # New attribute to track session-specific trades
        self.session_trades = []
        self.session_start_time = datetime.now()
        
        # Logging setup
        os.makedirs(config.LOGS_DIRECTORY, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config.LOGS_DIRECTORY, 'trade_manager.log')),
                logging.StreamHandler()
            ]
        )

    def add_trade(self, trade: Trade) -> None:
        """
        Add a new trade and track it in the session
        """
        if trade not in self.open_trades:
            self.open_trades.append(trade)
            self.total_trades_count += 1
            
            # Track trade in session trades
            self.session_trades.append(trade)
            
            logging.info(f"Trade added. Total trades count: {self.total_trades_count}")
            logging.info(f"Session trades count: {len(self.session_trades)}")
        else:
            logging.warning(f"Attempted to add duplicate trade: {trade.entry_price}")

        self.update_metrics()  # Changed from _update_metrics to update_metrics
        self._notify("New Trade", f"Trade opened at ${trade.entry_price:.2f}, Quantity: {trade.quantity:.4f}")

    def get_session_trade_report(self) -> Dict:
        """
        Generate a detailed report of trades in the current session
        """
        # Calculate trade statistics
        buy_trades = [t for t in self.session_trades if t.status == 'OPEN']
        sell_trades = [t for t in self.session_trades if t.status == 'CLOSED']
        
        return {
            'total_trades': len(self.session_trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'trades': [
                {
                    'symbol': t.symbol,
                    'entry_price': float(t.entry_price),
                    'exit_price': float(t.exit_price) if t.exit_price else None,
                    'quantity': float(t.quantity),
                    'status': t.status,
                    'pnl': float(t.pnl) if t.pnl is not None else None
                } for t in self.session_trades
            ]
        }
    
    def close_trade(self, 
                trade: Trade, 
                exit_price: Union[float, Decimal], 
                exit_time: Optional[datetime] = None, 
                reason: Optional[str] = None) -> None:
        """
        Comprehensive trade closing method with enhanced error handling and tracking.
        """
        try:
            # Convert exit price to Decimal for precise calculations
            exit_price_decimal = Decimal(str(exit_price))
            exit_time = exit_time or datetime.now()
            
            # Validate trade can be closed
            if trade.status != "OPEN":
                logging.warning(f"Attempted to close already closed trade: {trade.entry_price}")
                return
            
            # Ensure trade has all necessary information
            if trade.entry_price is None or trade.quantity is None:
                logging.error("Trade missing critical information for closing")
                return
            
            # Calculate PnL with precise decimal calculation
            trade.pnl = (exit_price_decimal - trade.entry_price) * trade.quantity
            
            # ✅ **Skip logging for zero PnL trades**
            if trade.pnl == 0:
                return  

            # Close the trade
            trade.close(
                exit_price=exit_price_decimal, 
                exit_time=exit_time, 
                reason=reason
            )
            
            # Manage trade lists
            if trade in self.open_trades:
                self.open_trades.remove(trade)
                self.completed_trades.append(trade)
            
            # ✅ **Fix incorrect function call (_update_metrics → update_metrics)**
            self.update_metrics()  

            # Logging (only for valid trades)
            logging.info(
                f"Trade closed. "
                f"Entry: {trade.entry_price}, "
                f"Exit: {exit_price_decimal}, "
                f"Quantity: {trade.quantity}, "
                f"PnL: {trade.pnl:.2f}, "
                f"Reason: {reason}"
            )
            
            # Notify
            self._notify(
                "Trade Closed", 
                f"Trade closed at ${exit_price_decimal:.2f}, "
                f"PnL: ${trade.pnl:.2f}, "
                f"Reason: {reason}"
            )
        
        except Exception as e:
            logging.error(f"Error closing trade: {e}")
            self._notify("Trade Closing Error", f"Failed to close trade: {e}")


    def _close_trade(self, trade: Trade, current_price: Union[float, Decimal]) -> None:
        """
        Internal method to execute trade closing with automated order submission.
        """
        try:
            # Use the improved trading bot for precise balance and order execution
            account_balance = self.trading_bot.get_account_balance()
            
            # Validate trade quantity against available position
            current_position = self.trading_bot.get_position()
            
            if trade.quantity > current_position:
                logging.warning(
                    f"Sell quantity exceeds current position. "
                    f"Requested: {trade.quantity}, Available: {current_position}"
                )
                # Adjust quantity to available position
                trade.quantity = current_position
            
            # Determine close reason based on price movement
            current_price_decimal = Decimal(str(current_price))
            reason = (
                "STOP_LOSS" if current_price_decimal < trade.entry_price 
                else "TAKE_PROFIT"
            )
            
            # Execute sell order
            self.trading_bot.execute_trade(
                trade_type='sell', 
                quantity=trade.quantity
            )
            
            # Close the trade
            self.close_trade(
                trade=trade,
                exit_price=current_price,
                exit_time=datetime.now(),
                reason=reason
            )
        
        except Exception as e:
            logging.error(f"Error executing trade close: {e}")
            self._notify("Trade Closing Error", f"Failed to close trade: {e}")



    def update_metrics(self):
        """
        Update internal trading metrics after each trade.
        """
        try:
            self.metrics["total_completed_trades"] = len(self.completed_trades)
            self.metrics["current_open_trades"] = len(self.open_trades)
            self.metrics["portfolio_value"] = self.get_portfolio_value()

        except Exception as e:
            logging.error(f"Error updating metrics: {e}")

    def get_portfolio_value(self) -> float:
        """Calculate current portfolio value"""
        try:
            account = self.trading_bot.get_account_balance()
            return float(account['total_portfolio_value'])
        except Exception as e:
            logging.error(f"Error getting portfolio value: {e}")
            return float(self.initial_balance)



    def _notify(self, subject: str, body: str) -> None:
        """Send email notification"""
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.email_sender, self.email_password)
            
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = self.email_sender
            msg['To'] = self.email_receiver
            
            server.send_message(msg)
            server.quit()
            
            self.notification_queue.append({
                'timestamp': datetime.now(),
                'subject': subject,
                'body': body
            })
            
            logging.info(f"Notification sent: {subject}")
            
        except Exception as e:
            logging.error(f"Failed to send notification: {e}")

    def has_trading_activity(self, duration_threshold_minutes: int = 30) -> bool:
        """
        Check if there was meaningful trading activity
        
        Args:
            duration_threshold_minutes: Minimum duration to consider as active trading
        
        Returns:
            bool: Whether trading activity occurred
        """
        # No trades at all
        if not self.completed_trades and not self.open_trades:
            return False
        
        # If no first trade time, but trades exist (defensive check)
        if not self.first_trade_time:
            return len(self.completed_trades) + len(self.open_trades) > 0
        
        # Check if trading duration meets threshold
        trading_duration = (self.last_trade_time - self.first_trade_time).total_seconds() / 60
        
        return (
            len(self.completed_trades) > 0 or  # At least one completed trade
            len(self.open_trades) > 0 or       # Or open trades exist
            trading_duration >= duration_threshold_minutes  # Or trading lasted long enough
        )

    def get_report(self) -> str:
        """Generate performance report calculating PnL from portfolio value change"""
        try:
            # Get final portfolio value from Alpaca API
            account = alpaca_api.get_account()
            final_portfolio_value = float(account.portfolio_value)
            
            # Calculate PnL as the absolute change in portfolio value
            total_pnl = final_portfolio_value - self.initial_balance
            
            # Calculate return as the percentage change
            total_return = (total_pnl / self.initial_balance) * 100
            
            # Log final state for debugging
            logging.info(f"Final metrics before report: Total PnL: {total_pnl:.2f}, Portfolio Value: {final_portfolio_value:.2f}")

            # Generate report
            report_lines = [
                "=== Trading Session Results ===",
                f"Initial Portfolio Value: ${self.initial_balance:.2f}",
                f"Final Portfolio Value:   ${final_portfolio_value:.2f}",
                f"Total PnL:               ${total_pnl:.2f}",
                f"Total Return:            {total_return:.2f}%"
            ]

            return "\n".join(report_lines)
            
        except Exception as e:
            logging.error(f"Failed to get Alpaca portfolio value: {e}")
            return "Failed to generate trading report due to error fetching portfolio value"


class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000):
        super(TradingEnv, self).__init__()
        
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
            
        # Store the original DataFrame without resetting index
        self.df = df.copy()
        
        # Define feature columns to use
        self.feature_columns = [
            'Close BTC-USD', 'EMA_fast', 'EMA_medium', 'EMA_slow', 
            'MACD', 'MACD_Hist', 'RSI_fast', 'RSI_medium', 'ATR', 
            'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'ROC'
        ]
        
        # Validate columns and data
        missing_columns = [col for col in self.feature_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if self.df.isnull().any().any():
            logging.warning("DataFrame contains NaN values. Dropping them.")
            self.df.dropna(inplace=True)
        
        # Calculate price changes for rewards
        self.df['returns'] = self.df['Close BTC-USD'].pct_change()
        self.df['volatility'] = self.df['returns'].rolling(window=20).std()
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        
        # Enhanced observation space with more features
        num_features = len(self.feature_columns) + 5  # +5 for additional state info
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(num_features,),
            dtype=np.float32
        )
        
        # Initialize state variables
        self.initial_balance = float(initial_balance)
        self.reset()
        
        # Trading constraints
        self.max_position_size = 0.1  # Maximum 10% of balance per trade
        self.min_holding_period = 12   # Minimum holding period (1 hour for 5-min data)
        
        # Performance tracking
        self.max_portfolio_value = float(initial_balance)
        self.portfolio_values = []
        self.realized_pnl = []
        self.episode_trades = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.balance = float(self.initial_balance)
        self.shares_held = 0.0
        self.total_asset_value = float(self.balance)
        self.current_step = 0
        self.position_holding_time = 0
        self.last_trade_price = None
        self.trades = []
        self.state_memory = []
        self.max_portfolio_value = float(self.initial_balance)
        self.portfolio_values = [self.total_asset_value]
        self.realized_pnl = []
        self.episode_trades = 0
        
        observation = self._next_observation()
        
        # Ensure observation is valid
        if observation is None:
            logging.error("Failed to generate initial observation")
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        return observation, {}
            
    def _next_observation(self):
        try:
            if self.current_step >= len(self.df):
                raise ValueError(f"Current step {self.current_step} exceeds dataset length {len(self.df)}")
            
            # Get current market data
            current_row = self.df.iloc[self.current_step]
            
            # Market features
            market_features = []
            for col in self.feature_columns:
                value = float(current_row[col])
                market_features.append(value)
            
            # Enhanced position features
            position_features = [
                float(self.balance) / float(self.initial_balance),  # Normalized balance
                float(self.shares_held * current_row['Close BTC-USD']) / float(self.initial_balance),  # Position size
                float(self.position_holding_time) / 24.0,  # Holding time
                float(len(self.realized_pnl)) / 100.0 if self.realized_pnl else 0.0,  # Trade count
                float(np.mean(self.realized_pnl)) if self.realized_pnl else 0.0  # Average PnL
            ]
            
            # Combine and normalize features
            obs = np.array(market_features + position_features, dtype=np.float32)
            
            if np.isnan(obs).any():
                raise ValueError("NaN values detected in observation")
            
            # Normalize using recent history with clipping
            if len(self.state_memory) > 0:
                state_array = np.array(self.state_memory)
                state_mean = np.mean(state_array, axis=0)
                state_std = np.std(state_array, axis=0) + 1e-8
                obs = np.clip((obs - state_mean) / state_std, -10, 10)
            
            # Update state memory
            self.state_memory.append(obs)
            if len(self.state_memory) > 1000:
                self.state_memory.pop(0)
            
            return obs
            
        except Exception as e:
            logging.error(f"Error in _next_observation: {str(e)}")
            raise

    def _calculate_reward(self, prev_value, current_price, action):
        """Enhanced reward function with multiple components"""
        try:
            # Component 1: Portfolio return
            pct_change = float((self.total_asset_value - prev_value) / prev_value)
            reward = pct_change
            
            # Component 2: Trade quality
            if action != 0:  # If trading action taken
                # Penalize frequent trading
                if self.episode_trades > 0:
                    avg_trade_interval = self.current_step / self.episode_trades
                    if avg_trade_interval < self.min_holding_period:
                        reward *= 0.7
                
                # Penalize high turnover
                position_size = (self.shares_held * current_price) / self.total_asset_value
                if position_size > self.max_position_size:
                    reward *= 0.8
            
            # Component 3: Position management
            if self.shares_held > 0:
                # Reward for riding profits
                if self.last_trade_price and current_price > self.last_trade_price:
                    profit_ratio = (current_price - self.last_trade_price) / self.last_trade_price
                    reward *= (1 + profit_ratio)
                
                # Penalize holding losses too long
                elif self.last_trade_price and current_price < self.last_trade_price:
                    if self.position_holding_time > 24:  # More than 2 hours
                        loss_ratio = (self.last_trade_price - current_price) / self.last_trade_price
                        reward *= (1 - loss_ratio)
            
            # Component 4: Risk management
            current_drawdown = (self.max_portfolio_value - self.total_asset_value) / self.max_portfolio_value
            if current_drawdown > 0.02:  # 2% drawdown threshold
                reward *= (1 - current_drawdown)
            
            # Component 5: Trend alignment
            if len(self.portfolio_values) > 20:
                short_trend = np.mean(self.portfolio_values[-10:]) > np.mean(self.portfolio_values[-20:-10])
                if (short_trend and self.shares_held > 0) or (not short_trend and self.shares_held == 0):
                    reward *= 1.2
            
            # Scale final reward for better learning
            reward = np.clip(reward * 10, -1, 1)  # Clip reward between -1 and 1
            
            return float(reward)
            
        except Exception as e:
            logging.error(f"Error in _calculate_reward: {str(e)}")
            return 0.0

    def step(self, action):
        try:
            action = int(action)
            if action not in [0, 1, 2]:
                raise ValueError(f"Invalid action: {action}")
            
            # Get current price and store previous state
            current_price = float(self.df.iloc[self.current_step]['Close BTC-USD'])
            prev_value = float(self.total_asset_value)
            
            # Update position holding time
            if self.shares_held > 0:
                self.position_holding_time += 1
            else:
                self.position_holding_time = 0
            
            # Execute trading action
            if action == 1:  # Buy
                position_size = self._calculate_position_size(self.df.iloc[self.current_step]['ATR'])
                self._execute_buy(position_size, current_price)
                self.episode_trades += 1
            elif action == 2:  # Sell
                self._execute_sell(current_price)
                self.episode_trades += 1
            
            # Update portfolio value and tracking metrics
            self.total_asset_value = float(self.balance + self.shares_held * current_price)
            self.max_portfolio_value = max(self.max_portfolio_value, self.total_asset_value)
            self.portfolio_values.append(self.total_asset_value)
            
            # Calculate reward
            reward = self._calculate_reward(prev_value, current_price, action)
            
            # Update state
            self.current_step += 1
            done = self.current_step >= len(self.df) - 1
            
            # Get next observation
            next_obs = self._next_observation() if not done else np.zeros(self.observation_space.shape)
            
            info = {
                'total_value': float(self.total_asset_value),
                'shares_held': float(self.shares_held),
                'balance': float(self.balance),
                'current_price': float(current_price),
                'trade_count': self.episode_trades,
                'portfolio_value': float(self.total_asset_value)
            }
            
            return next_obs, float(reward), done, False, info
            
        except Exception as e:
            logging.error(f"Error in step: {str(e)}")
            raise

    def _calculate_position_size(self, volatility):
        """Calculate position size with improved risk management"""
        try:
            # Base position size (2% of balance)
            base_size = float(self.balance * 0.02)
            
            # Adjust for volatility
            vol_adjusted_size = base_size / (float(volatility) + 1e-8)
            
            # Apply constraints
            max_allowed = self.balance * self.max_position_size
            position_size = float(min(vol_adjusted_size, max_allowed))
            
            return position_size
            
        except Exception as e:
            logging.error(f"Error in _calculate_position_size: {str(e)}")
            raise
            
    def _execute_buy(self, position_size, current_price):
        try:
            shares_to_buy = float(position_size / current_price)
            cost = shares_to_buy * current_price
            
            if shares_to_buy > 0 and self.balance >= cost:
                self.shares_held += shares_to_buy
                self.balance -= cost
                self.last_trade_price = current_price
                self.trades.append({
                    'action': 'buy',
                    'price': float(current_price),
                    'shares': float(shares_to_buy),
                    'step': self.current_step
                })
        except Exception as e:
            logging.error(f"Error in _execute_buy: {str(e)}")
            raise
            
    def _execute_sell(self, current_price):
        try:
            if self.shares_held > 0:
                sale_proceeds = float(self.shares_held * current_price)
                self.balance += sale_proceeds
                
                # Calculate and store realized PnL
                if self.last_trade_price:
                    pnl = (current_price - self.last_trade_price) / self.last_trade_price
                    self.realized_pnl.append(pnl)
                
                self.trades.append({
                    'action': 'sell',
                    'price': float(current_price),
                    'shares': float(self.shares_held),
                    'step': self.current_step
                })
                
                self.shares_held = 0.0
                self.last_trade_price = None
                self.position_holding_time = 0
                
        except Exception as e:
            logging.error(f"Error in _execute_sell: {str(e)}")
            raise
    
class Backtester:
    """Main class for executing backtests"""
    
    def __init__(self):
        self.periods = self._initialize_periods()
        self.strategies = self._initialize_strategies()
        
    def _initialize_periods(self) -> List[BacktestPeriod]:
        """Initialize the testing periods"""
        return [
            BacktestPeriod(
                name="LUNA_Collapse",
                test_start=datetime(2022, 5, 1),
                test_end=datetime(2022, 6, 30),
                train_start=datetime(2021, 11, 1),
                train_end=datetime(2022, 4, 30),
                description="Terra/LUNA collapse period",
                market_conditions="Extreme market crash"
            ),
            BacktestPeriod(
                name="Bull_Market_Peak",
                test_start=datetime(2021, 10, 1),
                test_end=datetime(2021, 11, 30),
                train_start=datetime(2021, 4, 1),
                train_end=datetime(2021, 9, 30),
                description="Bitcoin all-time high period",
                market_conditions="Strong bull market"
            ),
            BacktestPeriod(
                name="ETF_Approval",
                test_start=datetime(2023, 12, 1),
                test_end=datetime(2024, 1, 31),
                train_start=datetime(2023, 6, 1),
                train_end=datetime(2023, 11, 30),
                description="Bitcoin ETF approval period",
                market_conditions="Institutional adoption"
            )
        ]
        
    def _initialize_strategies(self) -> List[BaseStrategy]:
        """Initialize the baseline strategies"""
        return [
            BuyHoldStrategy(),
            SMACrossoverStrategy(),
            RSIStrategy()
        ]
        
    def backtest_strategy(self, 
                        strategy: BaseStrategy, 
                        df: pd.DataFrame, 
                        period: BacktestPeriod) -> BacktestResult:
        """Execute backtest for a single strategy over a specific period"""
        trades = []
        position = None
        
        # Get signals from strategy
        signals = strategy.generate_signals(df)
        
        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        
        # Simulate trading based on signals
        for i in range(len(df)):
            current_price = df.iloc[i]['Close BTC-USD']
            signal = signals.iloc[i]
            
            # Open position on buy signal
            if signal == 1 and position is None:
                position = {
                    'strategy': strategy.name,
                    'period': period.name,
                    'entry_price': current_price,
                    'entry_time': df.index[i].to_pydatetime(),  # Convert to python datetime
                    'quantity': 100000 / current_price  # Simplified position sizing
                }
            # Close position on sell signal
            elif signal == 2 and position is not None:
                position['exit_price'] = current_price
                position['exit_time'] = df.index[i].to_pydatetime()  # Convert to python datetime
                trades.append(position)
                position = None
                
        # Force close any open position at the end
        if position is not None:
            position['exit_price'] = df.iloc[-1]['Close BTC-USD']
            position['exit_time'] = df.index[-1].to_pydatetime()  # Convert to python datetime
            trades.append(position)
            
        return self.calculate_metrics(trades, df)
        
    def calculate_metrics(self, trades: List[Dict], df: pd.DataFrame) -> BacktestResult:
        """Calculate performance metrics for a set of trades"""
        try:
            if not trades:
                logging.warning("No trades to calculate metrics for")
                return BacktestResult(
                    strategy_name="",
                    period_name="",
                    total_return=0.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0,
                    win_rate=0.0,
                    num_trades=0,
                    avg_profit_per_trade=0.0,
                    avg_loss_per_trade=0.0,
                    profit_factor=0.0,
                    trades=[]
                )
                
            # Calculate returns and track portfolio value
            returns = []
            current_balance = float(100000)  # Initial balance
            peak_balance = float(current_balance)
            max_drawdown = 0.0
            
            for trade in trades:
                try:
                    # Ensure numerical values are properly converted to float
                    entry_price = float(trade['entry_price'])
                    exit_price = float(trade['exit_price'])
                    quantity = float(trade['quantity'])
                    
                    # Calculate trade return
                    trade_return = (exit_price - entry_price) / entry_price
                    returns.append(trade_return)
                    
                    # Update portfolio value and track drawdown
                    trade_pnl = trade_return * quantity * entry_price
                    current_balance += trade_pnl
                    peak_balance = max(peak_balance, current_balance)
                    drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0.0
                    max_drawdown = max(max_drawdown, drawdown)
                    
                except Exception as e:
                    logging.error(f"Error processing trade: {str(e)}")
                    logging.error(f"Trade data: {trade}")
                    continue
            
            # Calculate core metrics with proper type handling
            total_return = float((current_balance - 100000) / 100000)
            
            winning_trades = [t for t in trades if float(t['exit_price']) > float(t['entry_price'])]
            losing_trades = [t for t in trades if float(t['exit_price']) <= float(t['entry_price'])]
            
            win_rate = float(len(winning_trades) / len(trades)) if trades else 0.0
            
            # Calculate average profits and losses
            if winning_trades:
                avg_profit = float(np.mean([
                    float(t['exit_price']) - float(t['entry_price']) 
                    for t in winning_trades
                ]))
            else:
                avg_profit = 0.0
                
            if losing_trades:
                avg_loss = float(np.mean([
                    float(t['entry_price']) - float(t['exit_price']) 
                    for t in losing_trades
                ]))
            else:
                avg_loss = 0.0
            
            # Calculate Sharpe Ratio with error handling
            try:
                returns_series = pd.Series(returns)
                excess_returns = returns_series - 0.02/252  # Daily risk-free rate
                if len(returns) > 1 and excess_returns.std() != 0:
                    sharpe_ratio = float(np.sqrt(252) * excess_returns.mean() / excess_returns.std())
                else:
                    sharpe_ratio = 0.0
            except Exception as e:
                logging.error(f"Error calculating Sharpe ratio: {str(e)}")
                sharpe_ratio = 0.0
            
            # Calculate Profit Factor with error handling
            try:
                gross_profit = sum(
                    float(t['exit_price']) - float(t['entry_price']) 
                    for t in winning_trades
                ) if winning_trades else 0.0
                
                gross_loss = sum(
                    float(t['entry_price']) - float(t['exit_price']) 
                    for t in losing_trades
                ) if losing_trades else 0.0
                
                profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss != 0 else float('inf')
            except Exception as e:
                logging.error(f"Error calculating profit factor: {str(e)}")
                profit_factor = 0.0
            
            result = BacktestResult(
                strategy_name=str(trades[0]['strategy']),
                period_name=str(trades[0]['period']),
                total_return=total_return,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                num_trades=len(trades),
                avg_profit_per_trade=avg_profit,
                avg_loss_per_trade=avg_loss,
                profit_factor=profit_factor,
                trades=trades
            )
            
            logging.info(f"Calculated metrics for {len(trades)} trades:")
            logging.info(f"  Total Return: {result.total_return:.2%}")
            logging.info(f"  Win Rate: {result.win_rate:.2%}")
            logging.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error in calculate_metrics: {str(e)}")
            logging.error(f"Number of trades: {len(trades)}")
            logging.error(f"DataFrame shape: {df.shape}")
            raise
        
    @monitor_section("backtest_rl_strategy")
    def backtest_rl_strategy(self, model: PPO, env: TradingEnv, period: BacktestPeriod) -> BacktestResult:
        """Execute backtest for the RL model with memory monitoring"""
        trades = []
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle gymnasium env reset return type
            
        done = False
        position = None
        
        # Ensure the DataFrame index is datetime
        env.df.index = pd.to_datetime(env.df.index)
        
        while not done:
            # Get model prediction with deterministic action
            action, _ = model.predict(obs, deterministic=True)
            
            # Open position on buy signal
            if action == 1 and position is None:
                position = {
                    'strategy': 'RL Strategy',
                    'period': period.name,
                    'entry_price': env.df.iloc[env.current_step]['Close BTC-USD'],
                    'entry_time': env.df.index[env.current_step].to_pydatetime(),
                    'quantity': env.balance / env.df.iloc[env.current_step]['Close BTC-USD']
                }
            # Close position on sell signal
            elif action == 2 and position is not None:
                position['exit_price'] = env.df.iloc[env.current_step]['Close BTC-USD']
                position['exit_time'] = env.df.index[env.current_step].to_pydatetime()
                trades.append(position)
                position = None
                
            # Take step in environment
            obs, _, done, _, _ = env.step(action)
            
        # Force close any open position at the end
        if position is not None:
            position['exit_price'] = env.df.iloc[-1]['Close BTC-USD']
            position['exit_time'] = env.df.index[-1].to_pydatetime()
            trades.append(position)
            
        return self.calculate_metrics(trades, env.df)
    
    @monitor_section("train_model")
    def train_model(self, train_data, period_name):
        """Train RL model with optimized parameters"""
        try:
            # Create directories
            os.makedirs("./tensorboard_logs", exist_ok=True)
            os.makedirs("./best_models", exist_ok=True)
            os.makedirs("./eval_logs", exist_ok=True)
            os.makedirs("./model_checkpoints", exist_ok=True)
            
            # Split data into train and validation (70-30 split)
            train_end_idx = int(len(train_data) * 0.7)
            train_data_split = train_data[:train_end_idx]
            validation_data = train_data[train_end_idx:]
            
            logging.info(f"Training data shape: {train_data_split.shape}")
            logging.info(f"Validation data shape: {validation_data.shape}")
            
            try:
                # Create and wrap environments
                def make_env(data):
                    env = TradingEnv(data)
                    env = Monitor(env, f"./eval_logs/{period_name}", allow_early_resets=True)
                    return env
                
                train_env = DummyVecEnv([lambda: make_env(train_data_split)])
                val_env = DummyVecEnv([lambda: make_env(validation_data)])
                
                # Learning rate schedule
                def linear_schedule(initial_value: float):
                    def schedule(progress_remaining: float):
                        return progress_remaining * initial_value
                    return schedule
                
                # Initialize model with optimized parameters
                policy_kwargs = dict(
                    net_arch=dict(
                        pi=[512, 256, 128],  # Policy network
                        vf=[512, 256, 128]   # Value network
                    ),
                    activation_fn=nn.ReLU,
                    ortho_init=True,         # Better weight initialization
                )
                
                model = PPO(
                    "MlpPolicy",
                    train_env,
                    learning_rate=linear_schedule(1e-4),  # Scheduled learning rate
                    n_steps=4096,                # Larger batch size
                    batch_size=256,              # Larger mini-batch
                    n_epochs=20,                 # More training epochs
                    gamma=0.99,                  # Discount factor
                    gae_lambda=0.95,             # GAE parameter
                    clip_range=0.2,              # Clip range
                    clip_range_vf=0.2,           # Value function clipping
                    ent_coef=0.01,               # Higher entropy for more exploration
                    vf_coef=1.0,                 # Higher value function importance
                    max_grad_norm=0.5,           # Gradient clipping
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=f"./tensorboard_logs/{period_name}",
                    verbose=1
                )
                
                # Callbacks
                eval_callback = EvalCallback(
                    val_env,
                    best_model_save_path=f"./best_models/{period_name}",
                    log_path=f"./eval_logs/{period_name}",
                    eval_freq=10000,
                    deterministic=True,
                    render=False,
                    n_eval_episodes=5,
                    warn=False
                )
                
                checkpoint_callback = CheckpointCallback(
                    save_freq=50000,
                    save_path=f"./model_checkpoints/{period_name}/",
                    name_prefix="ppo_trading"
                )
                
                # Train the model
                logging.info(f"Starting model training for {period_name}")
                
                model.learn(
                    total_timesteps=750_000,   # 750_000 steps for thorough training
                    callback=[eval_callback, checkpoint_callback],
                    progress_bar=True
                )
                
                # Save final model
                final_model_path = f"trained_model_{period_name}.zip"
                model.save(final_model_path)
                logging.info(f"Model saved to {final_model_path}")
                
                return model, final_model_path
                
            finally:
                # Clean up resources
                if 'train_env' in locals():
                    train_env.close()
                if 'val_env' in locals():
                    val_env.close()
                
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise

    def run_backtests(self, data_fetcher) -> Dict[str, List[BacktestResult]]:
        results = {}
        
        for period in self.periods:
            try:
                logging.info(f"\nProcessing period: {period.name}")
                
                # Fetch 5-minute training data for RL model
                train_data_5m = None
                test_data_5m = None
                test_data_1h = None
                model = None
                
                try:
                    logging.info("Fetching 5-minute training data...")
                    train_data_5m = data_fetcher(
                        symbol='BTCUSDT',
                        start_date=period.train_start.strftime('%Y-%m-%d'),
                        end_date=period.train_end.strftime('%Y-%m-%d'),
                        timeframe='5m'
                    )
                    
                    if train_data_5m is None or train_data_5m.empty:
                        raise ValueError("Failed to fetch training data")
                    
                    # Train model
                    model, model_path = self.train_model(train_data_5m, period.name)
                    
                    # Clear training data
                    del train_data_5m
                    train_data_5m = None
                    gc.collect()
                    
                    # Fetch test data
                    logging.info("Fetching test data...")
                    test_data_5m = data_fetcher(
                        symbol='BTCUSDT',
                        start_date=period.test_start.strftime('%Y-%m-%d'),
                        end_date=period.test_end.strftime('%Y-%m-%d'),
                        timeframe='5m'
                    )
                    
                    test_data_1h = data_fetcher(
                        symbol='BTCUSDT',
                        start_date=period.test_start.strftime('%Y-%m-%d'),
                        end_date=period.test_end.strftime('%Y-%m-%d'),
                        timeframe='1h'
                    )
                    
                    if test_data_5m is None or test_data_5m.empty or test_data_1h is None or test_data_1h.empty:
                        raise ValueError("Failed to fetch test data")
                    
                    # Test RL strategy
                    logging.info("Testing RL strategy...")
                    test_env = TradingEnv(test_data_5m)
                    rl_results = self.backtest_rl_strategy(model, test_env, period)
                    results[f"{period.name}_RL"] = rl_results
                    
                    # Clean up RL resources
                    test_env.close()
                    del test_env
                    del model
                    del test_data_5m
                    gc.collect()
                    
                    # Test traditional strategies
                    for strategy in self.strategies:
                        logging.info(f"Testing {strategy.name} strategy...")
                        result = self.backtest_strategy(strategy, test_data_1h.copy(), period)
                        results[f"{period.name}_{strategy.name}"] = result
                    
                finally:
                    # Clean up period resources
                    del train_data_5m
                    del test_data_5m
                    del test_data_1h
                    del model
                    gc.collect()
                    plt.close('all')
                
            except Exception as e:
                logging.error(f"Error processing period {period.name}: {e}")
                continue
        
        return results

    def generate_report(self, results: Dict[str, BacktestResult]) -> str:
        """Generate comprehensive backtest report with clear formatting"""
        report_lines = [
            "\n" + "="*100,
            "CRYPTOCURRENCY TRADING STRATEGY BACKTEST RESULTS",
            "="*100 + "\n"
        ]
        
        for period in self.periods:
            report_lines.extend([
                f"\nTESTING PERIOD: {period.name}",
                f"Time Range: {period.test_start.strftime('%Y-%m-%d')} to {period.test_end.strftime('%Y-%m-%d')}",
                f"Market Context: {period.market_conditions}",
                f"Description: {period.description}\n",
                "\nStrategy Performance Summary:",
                f"\n{'Strategy':<15} {'Return':<12} {'Drawdown':<12} {'Sharpe':<10} {'Win Rate':<12} {'# Trades':<10} {'Profit Factor':<12}"
                f"\n{'-'*80}"
            ])
            
            # Filter and sort results for current period
            period_results = {k: v for k, v in results.items() if period.name in k}
            sorted_results = sorted(period_results.items(), key=lambda x: x[1].total_return, reverse=True)
            
            # Add performance metrics for each strategy
            for strategy_name, result in sorted_results:
                # Get the actual strategy name, not the period name
                if "RL" in strategy_name:
                    strategy = "RL Strategy"
                else:
                    strategy = result.strategy_name
                    
                report_lines.append(
                    f"\n{strategy:<15} {result.total_return:>10.2%}  {result.max_drawdown:>10.2%}  {result.sharpe_ratio:>8.2f}  {result.win_rate:>10.2%}  {result.num_trades:>8}  {result.profit_factor:>10.2f}"
                )
            
            # Add detailed analysis for each strategy
            report_lines.extend(["\n", "Detailed Analysis by Strategy:", "-"*30])
            
            for strategy_name, result in sorted_results:
                # Get the actual strategy name
                if "RL" in strategy_name:
                    strategy = "RL Strategy"
                else:
                    strategy = result.strategy_name
                
                # Calculate additional metrics
                profitable_trades = len([t for t in result.trades if t['exit_price'] > t['entry_price']])
                avg_hold_time = np.mean([
                    (t['exit_time'] - t['entry_time']).total_seconds() / 5  # Convert to hours
                    for t in result.trades
                ])
                
                report_lines.extend([
                    f"\n{strategy}:",
                    f"  • Trading Statistics:",
                    f"    - Total Return: {result.total_return:.2%}",
                    f"    - Maximum Drawdown: {result.max_drawdown:.2%}",
                    f"    - Sharpe Ratio: {result.sharpe_ratio:.2f}",
                    f"    - Win Rate: {result.win_rate:.2%}",
                    f"    - Number of Trades: {result.num_trades}",
                    f"    - Profit Factor: {result.profit_factor:.2f}",
                    f"\n  • Trade Performance:",
                    f"    - Average Profit (Winning Trades): ${result.avg_profit_per_trade:.2f}",
                    f"    - Average Loss (Losing Trades): ${result.avg_loss_per_trade:.2f}",
                    f"    - Profitable Trades Ratio: {profitable_trades}/{len(result.trades)}",
                    f"    - Average Position Hold Time: {avg_hold_time:.1f} hours"
                ])
                
                # Add monthly returns if available
                if result.trades:
                    monthly_returns = self._calculate_monthly_returns(result.trades)
                    if monthly_returns:
                        report_lines.extend(["\n  • Monthly Returns:"])
                        for month, return_value in monthly_returns.items():
                            report_lines.append(f"    - {month}: {return_value:.2%}")
            
            report_lines.append("\n" + "="*100)
        
        # Add metrics explanation
        report_lines.extend([
            "\nMETRICS EXPLANATION",
            "-"*20,
            "• Return: Overall percentage return for the period",
            "• Drawdown: Largest peak-to-trough decline in portfolio value",
            "• Sharpe Ratio: Risk-adjusted return (higher is better, >1 is good)",
            "• Win Rate: Percentage of trades that were profitable",
            "• Profit Factor: Gross profits / Gross losses (>1 indicates profitability)",
            "• Hold Time: Average duration of trading positions",
            "\n" + "="*100
        ])
        
        return "\n".join(report_lines)

    def _calculate_monthly_returns(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate monthly returns from trades"""
        monthly_pnl = defaultdict(float)
        monthly_capital = defaultdict(float)
        
        for trade in trades:
            month_key = trade['entry_time'].strftime('%Y-%m')
            pnl = (trade['exit_price'] - trade['entry_price']) * trade['quantity']
            capital = trade['entry_price'] * trade['quantity']
            
            monthly_pnl[month_key] += pnl
            monthly_capital[month_key] += capital
        
        return {
            month: (monthly_pnl[month] / monthly_capital[month])
            for month in monthly_pnl.keys()
            if monthly_capital[month] > 0
        }

    def save_results(self, results: Dict[str, BacktestResult], filename: str = "backtest_results.json"):
        """Save backtest results to file"""
        # Convert results to serializable format
        serializable_results = {}
        for key, result in results.items():
            serializable_results[key] = {
                'strategy_name': result.strategy_name,
                'period_name': result.period_name,
                'total_return': float(result.total_return),
                'max_drawdown': float(result.max_drawdown),
                'sharpe_ratio': float(result.sharpe_ratio),
                'win_rate': float(result.win_rate),
                'num_trades': result.num_trades,
                'avg_profit_per_trade': float(result.avg_profit_per_trade),
                'avg_loss_per_trade': float(result.avg_loss_per_trade),
                'profit_factor': float(result.profit_factor),
                'trades': [
                    {
                        'entry_time': t['entry_time'].isoformat(),
                        'exit_time': t['exit_time'].isoformat(),
                        'entry_price': float(t['entry_price']),
                        'exit_price': float(t['exit_price']),
                        'quantity': float(t['quantity'])
                    }
                    for t in result.trades
                ]
            }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=4)
            
    def plot_equity_curves(self, results: Dict[str, BacktestResult], period_name: str, show_plot=True):
        """Plot equity curves for all strategies in a given period"""
        try:
            # Clear any existing plots
            plt.close('all')
            
            # Create new figure
            fig = plt.figure(figsize=(15, 8))
            
            # Filter results for the specified period
            period_results = {k: v for k, v in results.items() if period_name in k}
            
            # Get the period object for date information
            period = next((p for p in self.periods if p.name == period_name), None)
            if not period:
                raise ValueError(f"Period {period_name} not found")
            
            # Plot each strategy's equity curve
            for strategy_name, result in period_results.items():
                if not result.trades:
                    logging.warning(f"No trades found for {strategy_name}")
                    continue
                    
                # Sort trades by entry time
                sorted_trades = sorted(result.trades, key=lambda x: x['entry_time'])
                
                # Create timestamp points for x-axis
                timestamps = []
                equity_curve = []
                balance = 100000  # Initial balance
                
                # Add starting point
                timestamps.append(sorted_trades[0]['entry_time'])
                equity_curve.append(balance)
                
                # Calculate equity curve points
                for trade in sorted_trades:
                    # Add point at trade entry
                    timestamps.append(trade['entry_time'])
                    equity_curve.append(balance)
                    
                    # Calculate PnL
                    pnl = (float(trade['exit_price']) - float(trade['entry_price'])) * float(trade['quantity'])
                    balance += pnl
                    
                    # Add point at trade exit
                    timestamps.append(trade['exit_time'])
                    equity_curve.append(balance)
                
                # Convert to numpy arrays for easier manipulation
                timestamps = np.array(timestamps)
                equity_curve = np.array(equity_curve)
                
                # Get proper strategy label
                if "RL" in strategy_name:
                    strategy_label = "RL Strategy"
                else:
                    strategy_label = result.strategy_name
                
                # Plot the equity curve
                plt.plot(timestamps, equity_curve, 
                        label=f"{strategy_label} (Return: {result.total_return:.1%})",
                        linewidth=2)
            
            # Customize the plot
            plt.title(f'Strategy Performance Comparison - {period_name}\n{period.test_start.date()} to {period.test_end.date()}', 
                    fontsize=12, pad=15)
            plt.xlabel('Date', fontsize=10)
            plt.ylabel('Portfolio Value ($)', fontsize=10)
            
            # Format x-axis
            plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
            
            # Add grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Customize legend
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Add market context annotation
            plt.figtext(0.02, 0.02, f"Market Context: {period.market_conditions}", 
                    fontsize=8, alpha=0.7)
            
            # Save the plot
            plot_path = f'equity_curves_{period_name}.png'
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            logging.info(f"Plot saved to {plot_path}")
            
            if show_plot:
                plt.show()
            
            plt.close(fig)  # Explicitly close the figure
            
            return fig
            
        except Exception as e:
            logging.error(f"Error plotting equity curves: {str(e)}")
            plt.close('all')  # Clean up any open figures
            raise

def fetch_market_data(symbol: str, start_date: str, end_date: str = None, timeframe: str = '5m') -> Optional[pd.DataFrame]:
    """
    Fetch market data from Binance without any fallbacks.

    Args:
        symbol: Crypto symbol (e.g., 'BTCUSDT')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD) (optional)
        timeframe: Timeframe (supported: '5m', '1h', '1d')

    Returns:
        pd.DataFrame: Market data with timestamps, Open, High, Low, Close, Volume
    """
    logging.info(f"Fetching {symbol} {timeframe} data from Binance...")

    exchange = ccxt.binance()

    try:
        # Ensure time is correctly converted from SGT (UTC+8) to UTC (Binance requires UTC)
        sgt = timezone(timedelta(hours=8))  # Singapore Timezone (SGT)
        
        # If start_date is a datetime object, use it directly
        if isinstance(start_date, datetime):
            start_datetime = start_date.astimezone(timezone.utc)
        else:
            start_datetime = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=sgt).astimezone(timezone.utc)

        if end_date:
            if isinstance(end_date, datetime):
                end_datetime = end_date.astimezone(timezone.utc)
            else:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=sgt).astimezone(timezone.utc)
        else:
            end_datetime = datetime.now(sgt).astimezone(timezone.utc)  # Convert SGT to UTC

        # Convert to timestamp format required by Binance
        start_timestamp = int(start_datetime.timestamp() * 1000)
        end_timestamp = int(end_datetime.timestamp() * 1000)

        all_data = []
        current_timestamp = start_timestamp
        max_retries = 5  # Retry up to 5 times if no data is received
        retries = 0

        while current_timestamp < end_timestamp and retries < max_retries:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol,
                    timeframe,
                    since=current_timestamp,
                    limit=500  # Reduce limit to avoid rate limiting
                )

                if not ohlcv:
                    retries += 1
                    logging.warning(f"No data received, retrying ({retries}/{max_retries})...")
                    time.sleep(5 * retries)  # Wait before retrying, increasing delay
                    continue

                all_data.extend(ohlcv)
                current_timestamp = ohlcv[-1][0] + (parse_timeframe_to_minutes(timeframe) * 60 * 1000)
                retries = 0  # Reset retries on successful fetch

                logging.info(f"Fetched data up to {datetime.fromtimestamp(current_timestamp/1000, tz=timezone(timedelta(hours=8)))}")
                time.sleep(exchange.rateLimit / 1000)

            except ccxt.RateLimitExceeded:
                retries += 1
                logging.warning(f"Binance rate limit exceeded, retrying in {5 * retries} seconds...")
                time.sleep(5 * retries)
                continue
            except ccxt.NetworkError as e:
                logging.warning(f"Network error while fetching data, retrying: {e}")
                time.sleep(10)
                retries += 1
                continue
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                break

        if not all_data:
            logging.critical("No data retrieved from Binance after multiple attempts. Exiting...")
            return None  # Exit function safely

        # Convert data into DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)

        # Filter data within specified date range
        df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] < end_datetime)]

        if df.empty:
            logging.critical(f"No data found between {start_datetime} and {end_datetime}. Exiting...")
            return None  # Exit function safely

        # Set timestamp as index
        df.set_index('timestamp', inplace=True)

        # Rename columns to match expected format
        base_currency = symbol[:3] + '-USD'
        df.rename(columns={
            'Open': f'Open {base_currency}',
            'High': f'High {base_currency}',
            'Low': f'Low {base_currency}',
            'Close': f'Close {base_currency}',
            'Volume': f'Volume {base_currency}'
        }, inplace=True)

        # Add technical indicators
        df = add_technical_indicators(df)
        df.dropna(inplace=True)

        logging.info(f"Final dataset: {len(df)} candles from {df.index[0].astimezone(timezone(timedelta(hours=8)))} to {df.index[-1].astimezone(timezone(timedelta(hours=8)))}")
        return df

    except Exception as e:
        logging.error(f"Failed to fetch market data: {e}")
        return None  # Exit function safely


def parse_timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes"""
    unit = timeframe[-1]
    number = int(timeframe[:-1])
    
    if unit == 'm':
        return number
    elif unit == 'h':
        return number * 60
    elif unit == 'd':
        return number * 1440
    else:
        raise ValueError(f"Unsupported timeframe unit: {unit}")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators optimized for 5-minute crypto trading
    All calculations are performed on the Close price
    """
    try:
        df = df.copy()
        
        # Identify the close price column dynamically
        close_column = [col for col in df.columns if 'Close' in col and '-USD' in col][0]
        
        # Trend indicators
        # Use only EMAs for faster reaction to price changes
        df['EMA_fast'] = ta.ema(df[close_column], length=12)    # 1 hour
        df['EMA_medium'] = ta.ema(df[close_column], length=24)  # 2 hours
        df['EMA_slow'] = ta.ema(df[close_column], length=50)    # ~4 hours
        
        # Momentum indicators
        # MACD with optimized settings for crypto
        macd = ta.macd(
            df[close_column], 
            fast=8,    # Faster signal
            slow=21,   # Slower signal
            signal=5   # Signal smoothing
        )
        df['MACD'] = macd['MACD_8_21_5']
        df['MACD_Signal'] = macd['MACDs_8_21_5']
        df['MACD_Hist'] = macd['MACDh_8_21_5']
        
        # RSIs for different timeframes
        df['RSI_fast'] = ta.rsi(df[close_column], length=9)
        df['RSI_medium'] = ta.rsi(df[close_column], length=14)
        
        # Volatility indicators
        # ATR for measuring volatility
        # Identify high/low columns dynamically
        high_column = [col for col in df.columns if 'High' in col and '-USD' in col][0]
        low_column = [col for col in df.columns if 'Low' in col and '-USD' in col][0]
        
        df['ATR'] = ta.atr(
            high=df[high_column],
            low=df[low_column],
            close=df[close_column],
            length=14
        )
        
        # Bollinger Bands for volatility and trend
        bbands = ta.bbands(df[close_column], length=20, std=2.5)
        df['BB_Upper'] = bbands['BBU_20_2.5']
        df['BB_Lower'] = bbands['BBL_20_2.5']
        df['BB_Middle'] = bbands['BBM_20_2.5']
        
        # Volume analysis
        volume_column = [col for col in df.columns if 'Volume' in col and '-USD' in col][0]
        df['Volume_EMA'] = ta.ema(df[volume_column], length=20)
        df['Volume_Ratio'] = df[volume_column] / df['Volume_EMA']
        
        # Price momentum
        df['ROC'] = ta.roc(df[close_column], length=9)
        
        # Clean up NaN values
        df.dropna(inplace=True)
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {e}")
        raise

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate DataFrame to ensure it meets requirements for trading environment
    """
    # Check if DataFrame is empty
    if df is None or df.empty:
        logging.error("DataFrame is empty")
        return False
    
    # Check index is datetime with multiple unique timestamps
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.error("DataFrame index is not DatetimeIndex")
        return False
    
    # Check for multiple unique timestamps
    if len(df.index.unique()) <= 1:
        logging.error("DataFrame has fewer than two unique timestamps")
        logging.error(f"Unique timestamps: {df.index.unique()}")
        return False
    
    # Check for required columns
    required_columns = [
        'Close BTC-USD', 'EMA_fast', 'EMA_medium', 'EMA_slow', 
        'MACD', 'MACD_Hist', 'RSI_fast', 'RSI_medium', 'ATR', 
        'BB_Upper', 'BB_Lower', 'Volume_Ratio', 'ROC'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return False
    
    # Check for NaN values
    if df.isnull().any().any():
        logging.error("DataFrame contains NaN values")
        return False
    
    return True

def _should_exit_trade(trade, current_price, stop_loss_percent, take_profit_percent):
    """Helper function to check if a trade should be exited based on SL/TP"""
    if current_price <= trade.entry_price * (1 - stop_loss_percent):
        return "STOP_LOSS"
    elif current_price >= trade.entry_price * (1 + take_profit_percent):
        return "TAKE_PROFIT"
    return None

def execute_trades(model, env, trading_bot, trade_manager=None, trading_duration_minutes: int = 360):
    """
    Execute trades using the improved trading bot and reinforcement learning model
    
    Args:
        model: Trained RL model
        env: Trading environment
        trading_bot: ImprovedTradingBot instance
        trade_manager: Optional TradeManager instance (will create one if not provided)
        trading_duration_minutes: Duration of trading session
    
    Returns:
        List of completed trades
    """
    # Initialize trade manager if not provided
    if trade_manager is None:
        trade_manager = TradeManager(initial_balance=100000)
    
    try:
        # Reset the environment
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Set trading session parameters
        start_time = time.time()
        end_time = start_time + (trading_duration_minutes * 60)
        next_trade_time = start_time
        
        while time.time() < end_time:
            try:
                current_time = time.time()
                
                if current_time >= next_trade_time:
                    current_price = trading_bot.get_current_price()
                    action, _ = model.predict(obs, deterministic=True)
                    action = int(action) if not isinstance(action, int) else action
                    
                    if action == 1:  # Buy Signal
                        try:
                            buy_details = trading_bot.execute_trade('buy')
                            if buy_details:
                                new_trade = Trade(
                                    symbol='BTCUSD',
                                    entry_price=current_price,
                                    quantity=buy_details.get('qty', buy_details.get('quantity', 0))
                                )
                                trade_manager.add_trade(new_trade)
                                
                            else:
                                logging.warning("Buy trade did not return details")
                        except Exception as e:
                            logging.error(f"Buy trade failed: {e}")
                    
                    elif action == 2:  # Sell Signal
                        try:
                            for trade in trade_manager.open_trades[:]:
                                try:
                                    if trade.quantity > 0:  # Only execute if quantity is positive
                                        trading_bot.execute_trade('sell', quantity=trade.quantity)
                                        trade_manager.close_trade(
                                            trade, 
                                            exit_price=current_price, 
                                            exit_time=datetime.now(),
                                            reason="SELL_SIGNAL"
                                        )
                                except Exception as trade_close_error:
                                    logging.error(f"Error closing individual trade: {trade_close_error}")
                        except Exception as e:
                            logging.error(f"Sell trade failed: {e}")
                    
                    obs, reward, done, truncated, info = env.step(action)
                    next_trade_time = current_time + 300
                
                else:
                    time.sleep(1)
                
                if done:
                    obs = env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
            
            except Exception as e:
                logging.error(f"Error in trading iteration: {e}")
                continue
        
        # Only generate the final report at the end
        logging.info("Trading session completed")
        
        # Generate trade report using the TradeManager instance method
        # Pass session trades to ensure consistency
        trade_manager.trading_bot.generate_trade_report(trade_manager.session_trades)



        return trade_manager.completed_trades
    
    except Exception as e:
        logging.critical(f"Critical error in trading session: {e}")
        raise


def train_live_model():
    try:
        # Calculate date ranges for last 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months of data
        
        # Split into train (70%) and validation (30%) periods
        total_days = (end_date - start_date).days
        train_days = int(total_days * 0.7)
        
        train_start = start_date
        train_end = start_date + timedelta(days=train_days)
        val_start = train_end
        val_end = end_date
        
        logging.info(f"Training period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        logging.info(f"Validation period: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
        
        # Fetch training data
        logging.info("Fetching training data...")
        train_data = fetch_market_data(
            symbol='BTCUSDT',
            start_date=train_start.strftime('%Y-%m-%d'),
            end_date=train_end.strftime('%Y-%m-%d'),
            timeframe='5m'
        )
        
        logging.info("Fetching validation data...")
        val_data = fetch_market_data(
            symbol='BTCUSDT',
            start_date=val_start.strftime('%Y-%m-%d'),
            end_date=val_end.strftime('%Y-%m-%d'),
            timeframe='5m'
        )
        
        if train_data is None or train_data.empty or val_data is None or val_data.empty:
            raise ValueError("Failed to fetch market data")
            
        logging.info(f"Training data shape: {train_data.shape}")
        logging.info(f"Validation data shape: {val_data.shape}")
        
        # Create directories
        os.makedirs("./models", exist_ok=True)
        os.makedirs("./tensorboard_logs", exist_ok=True)
        os.makedirs("./model_checkpoints", exist_ok=True)
        
        # Create and wrap environments
        def make_env(data):
            env = TradingEnv(data)
            env = Monitor(env, "./model_checkpoints")
            return env
            
        train_env = DummyVecEnv([lambda: make_env(train_data)])
        val_env = DummyVecEnv([lambda: make_env(val_data)])
        
        # Initialize model with same parameters as backtesting
        policy_kwargs = dict(
            net_arch=dict(
                pi=[512, 256, 128],  # Policy network
                vf=[512, 256, 128]   # Value network
            ),
            activation_fn=nn.ReLU,
            ortho_init=True
        )
        
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=1e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=20,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=0.2,
            ent_coef=0.01,
            vf_coef=1.0,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tensorboard_logs",
            verbose=1
        )
        
        # Callbacks
        eval_callback = EvalCallback(
            val_env,
            best_model_save_path="./models",
            log_path="./model_checkpoints",
            eval_freq=10000,
            deterministic=True,
            render=False,
            n_eval_episodes=5,
            warn=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path="./model_checkpoints",
            name_prefix="ppo_trading"
        )
        
        # Train model
        logging.info("Starting model training...")
        model.learn(
            total_timesteps=750_000,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
        
        # Save final model
        final_model_path = "./models/trained_model_live_trading.zip"
        model.save(final_model_path)
        logging.info(f"Final model saved to {final_model_path}")
        
        return final_model_path, model, train_data  # Return path, model, and last training data
        
    except Exception as e:
        logging.error(f"Error during model training: {str(e)}")
        raise
    finally:
        if 'train_env' in locals():
            train_env.close()
        if 'val_env' in locals():
            val_env.close()


def verify_model_loading(model, env):
    """Verify that the loaded model can make predictions"""
    try:
        # Reset the environment to get a valid observation
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle gymnasium env reset return type
        
        # Try to predict an action
        action, _ = model.predict(obs, deterministic=True)
        
        # Check if action is valid
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action predicted: {action}")
        
        logging.info("Model loaded and prediction test passed successfully")
        return True
    except Exception as e:
        logging.error(f"Model loading verification failed: {e}")
        return False

def run_live_trading(model_path: str, trading_duration_minutes: int = 360):
    """
    Run live trading and print the final session report only at the end.
    """
    try:
        trading_bot = ImprovedTradingBot(
            alpaca_key=config.ALPACA_KEY,
            alpaca_secret=config.ALPACA_SECRET,
            alpaca_base_url=config.ALPACA_ENDPOINT
        )
        trade_manager = TradeManager(initial_balance=100000)

        # Use SGT for datetime calculations
        sgt = timezone(timedelta(hours=8))
        current_time_utc = datetime.now(sgt).astimezone(timezone.utc)
        start_time_utc = current_time_utc - timedelta(hours=48)

        logging.info("Fetching market data...")
        logging.info(f"Getting data from {start_time_utc} to {current_time_utc}")
        time.sleep(2)
        df = fetch_market_data('BTCUSDT', start_time_utc, current_time_utc, '5m')

        if df is None or df.empty:
            logging.error("Market data fetch failed. Exiting trading session.")
            return None

        # Create environment
        env = TradingEnv(df)

        # Load the model
        try:
            model = PPO.load(model_path)
            
            # Verify model loading
            if not verify_model_loading(model, env):
                raise ValueError("Model failed to load or predict correctly")
            
            logging.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logging.critical(f"Error loading model: {e}")
            trade_manager._notify("Model Loading Error", f"Failed to load model: {e}")
            return None

        logging.info("Starting live trading session...")

        # Execute trades without generating reports
        trades = execute_trades(
            model=model,
            env=env,
            trading_bot=trading_bot,
            trade_manager=trade_manager,
            trading_duration_minutes=trading_duration_minutes
        )

        # Only generate the final report once at the end
        if trades:
            logging.info("\nTrading Session Complete")
            logging.info("="*50)
            logging.info("📊 FINAL TRADING SESSION REPORT")
            logging.info("="*50)
            account_balance = trading_bot.get_account_balance()
            logging.info(f"""
Cash Balance      : ${float(account_balance['cash_balance']):,.2f}
Position Quantity : {float(account_balance['position_qty'])} BTC
Position Value    : ${float(account_balance['position_value']):,.2f}
Total Trades      : {len(trades)}
            """)
            logging.info("="*50)

        trade_manager._notify("Trading Session Complete", 
            f"Session completed with {len(trade_manager.session_trades)} trades. "
            f"Portfolio Value: ${float(trading_bot.get_account_balance()['total_portfolio_value']):,.2f}"
        )


        return trades

    except Exception as e:
        logging.error(f"Critical error in live trading: {e}")
        trade_manager._notify("Trading Error", f"Critical error: {e}")
        return None

if __name__ == "__main__":
    try:
        # Specify the path to the previously trained model
        model_path = "./models/trained_model_live_trading.zip"
        
        # Immediately start live trading using the saved model
        logging.info("\n=== Starting Live Trading ===")
        logging.info(f"Using model from: {model_path}")
        
        # Configure trading parameters
        TRADING_DURATION = 360  # 6 hours
        
        logging.info(f"Trading Duration: {TRADING_DURATION} minutes")
        
        # Start live trading using the saved model
        trades = run_live_trading(
            model_path=model_path,
            trading_duration_minutes=TRADING_DURATION
        )
        
        logging.info("Trading session completed successfully")
        
    except Exception as e:
        logging.critical(f"Session failed: {str(e)}")
        raise
