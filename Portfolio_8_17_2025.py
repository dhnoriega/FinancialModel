import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging
import warnings
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime
import sys
from collections import defaultdict, deque
import optuna
import pandas as pd
import pickle
import os
from functools import lru_cache
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil
import time

start_date='2018-01-01'
end_date='2025-08-14'

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set up console logging
logger = logging.getLogger('PerformanceLogger')
logger.setLevel(logging.INFO)
logger.handlers = []
logger.propagate = False
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

# Set up file logging for trades
trade_logger = logging.getLogger('TradeLogger_Portfolio')
trade_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('portfolio_trades.log')
file_handler.setFormatter(console_formatter)
trade_logger.handlers = []
trade_logger.addHandler(file_handler)
trade_logger.propagate = False

# Performance monitoring decorator
def performance_monitor(func):
    """Decorator to monitor function performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        logger.info(f"{func.__name__}: {end_time - start_time:.2f}s, "
                   f"Memory: {end_memory - start_memory:+.1f}MB")
        
        return result
    return wrapper

class OptimizedPortfolioStrategy:
    def __init__(self, tickers, start_date, end_date, initial_capital=100000, params=None):
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # Calculate split date (2 years before end date)
        self.backtest_end_date = self.end_date - pd.DateOffset(years=3, days=0)
        self.forward_test_start_date = self.backtest_end_date + pd.DateOffset(days=1)
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        
        # Create cache directory
        self.cache_dir = "strategy_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Memory-optimized data structures using numpy
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(tickers)}
        n_tickers = len(tickers)
        
        # Use efficient data types
        self.positions = np.zeros(n_tickers, dtype=np.int8)  # -1, 0, 1
        self.shares = np.zeros(n_tickers, dtype=np.int32)
        self.entry_prices = np.full(n_tickers, np.nan, dtype=np.float32)
        self.entry_dates = [None] * n_tickers  # Use Python list for datetime objects
        self.last_trade_dates = [None] * n_tickers  # Use Python list for datetime objects
        self.cooldown_days = np.zeros(n_tickers, dtype=np.int16)
        
        # Use deques for better performance
        self.trades = deque(maxlen=50000)
        self.forward_test_trades = deque(maxlen=50000)
        
        # Pre-allocate equity tracking
        estimated_days = (self.end_date - self.start_date).days + 100
        self.equity_values = np.zeros(estimated_days, dtype=np.float32)
        self.equity_dates = np.zeros(estimated_days, dtype='datetime64[D]')
        self.equity_idx = 0
        
        self.forward_test_equity_values = np.zeros(estimated_days, dtype=np.float32)
        self.forward_test_equity_dates = np.zeros(estimated_days, dtype='datetime64[D]')
        self.forward_test_equity_idx = 0
        
        self.data = {}
        self.models = {}
        self.transaction_cost_per_share = 0.01
        self.max_loss_per_trade = 0.01 * initial_capital
        
        # Position sizing constraints
        self.max_portfolio_risk = 0.03
        self.max_single_position = 0.20
        self.min_single_position = 0.005
        self.max_portfolio_leverage = 1.0
        
        # Kelly/Risk-based parameters
        self.base_kelly_fraction = 0.25
        self.volatility_lookback = 20
        
        # Signal tracking and momentum - using numpy arrays
        self.signal_debug = []
        self.momentum_scores = np.zeros(n_tickers, dtype=np.float32)
        self.trend_strength = np.zeros(n_tickers, dtype=np.float32)
        
        # Forward testing state tracking - using numpy arrays
        self.forward_test_cash = None
        self.forward_test_positions = None
        self.forward_test_shares = None
        self.forward_test_entry_prices = None
        self.forward_test_entry_dates = None
        
        # Parallel processing configuration
        self.n_workers = min(mp.cpu_count() - 1, 4)
        
        # Default parameters
        self.params = params or self._get_default_params()
        
        # Pre-computed indicators cache
        self.indicators_computed = False
        self.all_dates_cache = None

    def _get_default_params(self):
        """Get default parameters with proper structure"""
        return {
            'rsi_oversold': {ticker: 45 for ticker in self.tickers if ticker != '^DJI'},
            'rsi_overbought': {ticker: 65 for ticker in self.tickers if ticker != '^DJI'},
            'stoch_oversold': 25,
            'stoch_overbought': 75,
            'bb_period': {ticker: 20 for ticker in self.tickers if ticker != '^DJI'},
            'bb_std': {ticker: 1.8 for ticker in self.tickers if ticker != '^DJI'},
            'ma_short': {ticker: 10 for ticker in self.tickers if ticker != '^DJI'},
            'ma_long': {ticker: 30 for ticker in self.tickers if ticker != '^DJI'},
            'ema_fast': {ticker: 5 for ticker in self.tickers if ticker != '^DJI'},
            'ema_slow': {ticker: 15 for ticker in self.tickers if ticker != '^DJI'},
            'n_estimators': {ticker: 30 for ticker in self.tickers if ticker != '^DJI'},
            'max_depth': {ticker: 5 for ticker in self.tickers if ticker != '^DJI'},
            'confidence_threshold': {ticker: 0.52 for ticker in self.tickers if ticker != '^DJI'},
            'volume_multiplier': 0.8,
            'atr_multiplier_stop': 1.8,
            'atr_multiplier_profit': 3.5,
            'short_atr_multiplier_stop': 1.3,
            'short_atr_multiplier_profit': 2.8,
            'momentum_threshold': 0.3,
            'trend_strength_threshold': 0.4,
            'win_rate_estimate': {ticker: 0.60 for ticker in self.tickers if ticker != '^DJI'},
            'avg_win_loss_ratio': {ticker: 1.8 for ticker in self.tickers if ticker != '^DJI'},
        }

    @performance_monitor
    def fetch_data(self):
        """Optimized data loading with intelligent caching"""
        cache_key = f"data_{hash(tuple(self.tickers))}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        # Try to load from cache first
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.data = cached_data['data']
                    self.tickers = cached_data['valid_tickers']
                    logger.info(f"Loaded cached data for {len(self.tickers)} tickers")
                    return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, loading fresh data")

        # Load fresh data
        valid_tickers = []
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        for ticker in self.tickers:
            try:
                filename = f'{ticker}_features_updated.csv'
                data = pd.read_csv(filename, parse_dates=['Date'], index_col='Date')
                
                if not all(col in data.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    logger.error(f"Missing required columns in {filename}")
                    continue
                    
                data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                data = data.loc[self.start_date:self.end_date]
                data = data[~data.index.duplicated(keep='first')]
                
                if data.empty:
                    logger.error(f"No data for {ticker}")
                    continue
                    
                # Convert to more memory-efficient types
                data = data.astype({
                    'Open': 'float32',
                    'High': 'float32', 
                    'Low': 'float32',
                    'Close': 'float32',
                    'Volume': 'int32'
                })
                
                data['Ticker'] = ticker
                self.data[ticker] = data
                valid_tickers.append(ticker)
                logger.info(f"Loaded {ticker}: {len(data)} rows")
                
            except FileNotFoundError:
                logger.error(f"File {filename} not found")
            except Exception as e:
                logger.error(f"Error loading {ticker}: {e}")

        self.tickers = valid_tickers
        if not self.tickers:
            logger.error("No valid tickers available")
            sys.exit(1)

        # Update numpy arrays for new ticker count
        self._resize_arrays()
        
        # Cache the loaded data
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'data': self.data,
                    'valid_tickers': self.tickers
                }, f)
            logger.info(f"Cached data for future use")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")

    def _resize_arrays(self):
        """Resize numpy arrays for actual ticker count"""
        n_tickers = len(self.tickers)
        self.ticker_to_idx = {ticker: i for i, ticker in enumerate(self.tickers)}
        
        self.positions = np.zeros(n_tickers, dtype=np.int8)
        self.shares = np.zeros(n_tickers, dtype=np.int32)
        self.entry_prices = np.full(n_tickers, np.nan, dtype=np.float32)
        self.entry_dates = [None] * n_tickers  # Use Python list for datetime objects
        self.last_trade_dates = [None] * n_tickers  # Use Python list for datetime objects
        self.cooldown_days = np.zeros(n_tickers, dtype=np.int16)
        self.momentum_scores = np.zeros(n_tickers, dtype=np.float32)
        self.trend_strength = np.zeros(n_tickers, dtype=np.float32)

    @performance_monitor
    def calculate_all_indicators_vectorized(self):
        """Pre-compute all indicators using vectorized operations"""
        cache_key = f"indicators_{hash(tuple(self.tickers))}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        # Try to load cached indicators
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.data = pickle.load(f)
                    logger.info("Loaded cached indicators")
                    self.indicators_computed = True
                    return
            except Exception as e:
                logger.warning(f"Indicator cache load failed: {e}")

        logger.info("Computing indicators vectorized...")
        
        for ticker, data in self.data.items():
            if data is None or data.empty:
                continue
                
            try:
                # Convert to numpy for speed
                close = data['Close'].values.astype(np.float64)
                high = data['High'].values.astype(np.float64)
                low = data['Low'].values.astype(np.float64)
                volume = data['Volume'].values.astype(np.float64)
                n = len(close)
                
                # Pre-allocate result DataFrame
                result_data = data.copy()
                
                # Returns calculation
                returns = np.zeros(n)
                returns[1:] = np.diff(close) / close[:-1]
                result_data['Returns'] = returns
                
                # SMA 200 - using convolution for speed
                if n >= 200:
                    sma_200 = np.convolve(close, np.ones(200)/200, mode='same')
                    # Fix edges
                    for i in range(min(199, n)):
                        sma_200[i] = np.mean(close[:i+1]) if i > 0 else close[i]
                else:
                    sma_200 = np.cumsum(close) / np.arange(1, n+1)
                result_data['SMA_200'] = sma_200
                
                # ATR - vectorized calculation
                if n >= 14:
                    h_l = high - low
                    h_c = np.abs(high - np.roll(close, 1))
                    l_c = np.abs(low - np.roll(close, 1))
                    h_c[0] = h_l[0]  # Fix first value
                    l_c[0] = h_l[0]
                    
                    true_range = np.maximum(h_l, np.maximum(h_c, l_c))
                    atr = np.zeros(n)
                    atr[13] = np.mean(true_range[:14])
                    
                    # Exponential moving average for ATR
                    alpha = 1.0 / 14
                    for i in range(14, n):
                        atr[i] = alpha * true_range[i] + (1 - alpha) * atr[i-1]
                    
                    # Fill initial values
                    atr[:14] = atr[13]
                else:
                    atr = h_l if n > 0 else np.zeros(n)
                result_data['ATR'] = atr
                
                # Volume SMA 20
                if n >= 20:
                    vol_sma = np.convolve(volume, np.ones(20)/20, mode='same')
                    for i in range(min(19, n)):
                        vol_sma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
                else:
                    vol_sma = np.cumsum(volume) / np.arange(1, n+1)
                result_data['Volume_SMA_20'] = vol_sma
                
                # RSI - optimized calculation
                if n >= 14:
                    delta = np.diff(close)
                    gain = np.where(delta > 0, delta, 0)
                    loss = np.where(delta < 0, -delta, 0)
                    
                    avg_gain = np.zeros(n)
                    avg_loss = np.zeros(n)
                    
                    # Initial average
                    avg_gain[13] = np.mean(gain[:13])
                    avg_loss[13] = np.mean(loss[:13])
                    
                    # Exponential moving average
                    alpha = 1.0 / 14
                    for i in range(14, n):
                        avg_gain[i] = alpha * gain[i-1] + (1 - alpha) * avg_gain[i-1]
                        avg_loss[i] = alpha * loss[i-1] + (1 - alpha) * avg_loss[i-1]
                    
                    # Calculate RSI
                    rs = avg_gain / (avg_loss + 1e-10)
                    rsi = 100 - (100 / (1 + rs))
                    rsi[:14] = 50  # Fill initial values
                else:
                    rsi = np.full(n, 50)
                result_data['RSI'] = rsi
                
                # Stochastic Oscillator
                if n >= 14:
                    stoch_k = np.zeros(n)
                    stoch_d = np.zeros(n)
                    
                    for i in range(13, n):
                        period_high = np.max(high[i-13:i+1])
                        period_low = np.min(low[i-13:i+1])
                        if period_high != period_low:
                            stoch_k[i] = 100 * (close[i] - period_low) / (period_high - period_low)
                        else:
                            stoch_k[i] = 50
                    
                    # Smooth %K to get %D
                    if n >= 16:
                        for i in range(15, n):
                            stoch_d[i] = np.mean(stoch_k[i-2:i+1])
                    
                    stoch_k[:14] = 50
                    stoch_d[:16] = 50
                else:
                    stoch_k = np.full(n, 50)
                    stoch_d = np.full(n, 50)
                    
                result_data['Stoch_K'] = stoch_k
                result_data['Stoch_D'] = stoch_d
                
                # Bollinger Bands
                bb_period = self.params['bb_period'].get(ticker, 20)
                bb_std_mult = self.params['bb_std'].get(ticker, 1.8)
                
                if n >= bb_period:
                    bb_mid = np.convolve(close, np.ones(bb_period)/bb_period, mode='same')
                    
                    # Calculate rolling std efficiently
                    bb_std = np.zeros(n)
                    for i in range(bb_period-1, n):
                        bb_std[i] = np.std(close[i-bb_period+1:i+1])
                    
                    # Fill initial values
                    for i in range(bb_period-1):
                        bb_mid[i] = np.mean(close[:i+1]) if i > 0 else close[i]
                        bb_std[i] = np.std(close[:i+1]) if i > 0 else 0
                        
                    bb_upper = bb_mid + bb_std_mult * bb_std
                    bb_lower = bb_mid - bb_std_mult * bb_std
                else:
                    bb_mid = np.cumsum(close) / np.arange(1, n+1)
                    bb_std = np.zeros(n)
                    bb_upper = bb_mid
                    bb_lower = bb_mid
                
                result_data['BB_Mid'] = bb_mid
                result_data['BB_Std'] = bb_std
                result_data['BB_Upper'] = bb_upper
                result_data['BB_Lower'] = bb_lower
                
                # EMA Fast and Slow
                ema_fast_period = self.params['ema_fast'].get(ticker, 5)
                ema_slow_period = self.params['ema_slow'].get(ticker, 15)
                
                # EMA Fast
                alpha_fast = 2.0 / (ema_fast_period + 1)
                ema_fast = np.zeros(n)
                ema_fast[0] = close[0]
                for i in range(1, n):
                    ema_fast[i] = alpha_fast * close[i] + (1 - alpha_fast) * ema_fast[i-1]
                
                # EMA Slow
                alpha_slow = 2.0 / (ema_slow_period + 1)
                ema_slow = np.zeros(n)
                ema_slow[0] = close[0]
                for i in range(1, n):
                    ema_slow[i] = alpha_slow * close[i] + (1 - alpha_slow) * ema_slow[i-1]
                
                result_data['EMA_Fast'] = ema_fast
                result_data['EMA_Slow'] = ema_slow
                result_data['EMA_Signal'] = (ema_fast > ema_slow).astype(int)
                
                # MA Short and Long
                ma_short_period = self.params['ma_short'].get(ticker, 10)
                ma_long_period = self.params['ma_long'].get(ticker, 30)
                
                if n >= ma_short_period:
                    ma_short = np.convolve(close, np.ones(ma_short_period)/ma_short_period, mode='same')
                    for i in range(ma_short_period-1):
                        ma_short[i] = np.mean(close[:i+1]) if i > 0 else close[i]
                else:
                    ma_short = np.cumsum(close) / np.arange(1, n+1)
                    
                if n >= ma_long_period:
                    ma_long = np.convolve(close, np.ones(ma_long_period)/ma_long_period, mode='same')
                    for i in range(ma_long_period-1):
                        ma_long[i] = np.mean(close[:i+1]) if i > 0 else close[i]
                else:
                    ma_long = np.cumsum(close) / np.arange(1, n+1)
                
                result_data['MA_Short'] = ma_short
                result_data['MA_Long'] = ma_long
                result_data['MA_Crossover'] = (ma_short > ma_long).astype(int)
                
                # Momentum indicators
                price_momentum = np.zeros(n)
                if n >= 6:
                    price_momentum[5:] = close[5:] / close[:-5] - 1
                result_data['Price_Momentum'] = price_momentum
                
                volume_momentum = np.zeros(n)
                if n >= 6:
                    volume_momentum[5:] = volume[5:] / (volume[:-5] + 1) - 1
                result_data['Volume_Momentum'] = volume_momentum
                
                # Trend strength
                if n >= 5:
                    trend_strength = np.zeros(n)
                    price_change_5d = np.zeros(n)
                    price_change_5d[5:] = (close[5:] - close[:-5]) / close[:-5]
                    
                    for i in range(9, n):  # Need at least 5 values for rolling mean
                        trend_strength[i] = np.mean(price_change_5d[i-4:i+1])
                else:
                    trend_strength = np.zeros(n)
                    
                result_data['Trend_Strength'] = trend_strength
                
                # Volatility-adjusted momentum
                vol_adj_momentum = np.zeros(n)
                if n >= 10:
                    for i in range(10, n):
                        returns_std = np.std(returns[i-9:i+1])
                        if returns_std > 1e-6:
                            vol_adj_momentum[i] = price_momentum[i] / returns_std
                            
                result_data['Vol_Adj_Momentum'] = vol_adj_momentum
                
                # Store the enhanced data
                self.data[ticker] = result_data
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {ticker}: {e}")
                continue
        
        # Cache the computed indicators
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.data, f)
            logger.info("Cached computed indicators")
        except Exception as e:
            logger.warning(f"Failed to cache indicators: {e}")
            
        self.indicators_computed = True

    def years_between_dates(self, start_date, end_date):
        """Calculate years between dates"""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        delta = end - start
        return delta.days / 365.25

    def calculate_indicators(self, data, current_date, lookback=200):
        """Optimized indicator calculation - returns pre-computed slice"""
        if not self.indicators_computed:
            self.calculate_all_indicators_vectorized()
            
        if data is None or data.empty:
            return None
            
        # Return slice up to current date
        return data[data.index <= current_date]

    @lru_cache(maxsize=256)
    def prepare_ml_data_cached(self, ticker, current_date_str):
        """Cached ML data preparation"""
        current_date = pd.to_datetime(current_date_str)
        data = self.data.get(ticker)
        
        if data is None or data.empty:
            return None, None
            
        # Get data up to current date
        data_slice = data[data.index <= current_date].copy()
        
        # Enhanced feature set
        features = [
            'RSI', 'Stoch_K', 'Stoch_D', 'BB_Lower', 'BB_Upper', 'MA_Short', 'MA_Long', 
            'MA_Crossover', 'EMA_Fast', 'EMA_Slow', 'EMA_Signal', 'ATR', 'Volume_SMA_20',
            'Price_Momentum', 'Volume_Momentum', 'Trend_Strength', 'Vol_Adj_Momentum'
        ]
        
        # Check if all features exist
        missing_features = [f for f in features if f not in data_slice.columns]
        if missing_features:
            return None, None
            
        data_slice = data_slice.dropna(subset=features)
        if data_slice.empty:
            return None, None
        
        # Create target variables
        data_slice['Next_Return'] = data_slice['Returns'].shift(-1)
        data_slice['Next_Return_2d'] = data_slice['Returns'].shift(-1) + data_slice['Returns'].shift(-2)
        data_slice['Next_Return_3d'] = data_slice['Next_Return_2d'] + data_slice['Returns'].shift(-3)
        
        # Target creation
        data_slice['Target'] = ((data_slice['Next_Return'] > 0) | 
                               (data_slice['Next_Return_2d'] > 0.005) |
                               (data_slice['Next_Return_3d'] > 0.01)).astype(int)
        
        data_slice = data_slice.dropna(subset=['Target'])
        if data_slice.empty:
            return None, None
            
        X = data_slice[features].values.astype(np.float32)  # Convert to numpy for speed
        y = data_slice['Target'].values.astype(np.int8)
        
        return X, y

    def prepare_ml_data(self, data, current_date):
        """Wrapper for cached ML data preparation"""
        ticker = data['Ticker'].iloc[0] if 'Ticker' in data.columns and len(data) > 0 else None
        if ticker is None:
            return None, None
        return self.prepare_ml_data_cached(ticker, current_date.strftime('%Y-%m-%d'))

    @performance_monitor
    def train_ml_models_parallel(self):
        """Train ML models for all tickers in parallel"""
        
        def train_single_model(args):
            ticker, data, current_date, n_estimators, max_depth = args
            try:
                X, y = self.prepare_ml_data(data, current_date)
                if X is None or y is None or len(X) < 20:
                    return ticker, None
                
                # Use more recent data for training
                train_size = min(0.85, max(0.75, 50/len(X)))
                split_idx = int(len(X) * train_size)
                X_train, y_train = X[:split_idx], y[:split_idx]
                
                if len(X_train) < 8:
                    return ticker, None
                    
                model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=42, 
                    n_jobs=1,  # Use 1 job per model since we're already parallelizing
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    class_weight='balanced'
                )
                model.fit(X_train, y_train)
                return ticker, model
                
            except Exception as e:
                logger.error(f"Error training model for {ticker}: {e}")
                return ticker, None
        
        # Prepare training arguments
        initial_training_date = self.backtest_end_date - pd.DateOffset(days=300)
        
        training_args = []
        for ticker in self.tickers:
            if ticker == '^DJI' or self.data.get(ticker) is None:
                continue
            training_args.append((
                ticker,
                self.data[ticker],
                initial_training_date,
                self.params['n_estimators'].get(ticker, 30),
                self.params['max_depth'].get(ticker, 5)
            ))
        
        # Train models in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(train_single_model, training_args))
        
        # Store models
        for ticker, model in results:
            self.models[ticker] = model
            if model is not None:
                logger.info(f"Trained model for {ticker}")

    def train_ml_model(self, ticker, data, current_date, n_estimators, max_depth):
        """Single model training (kept for compatibility)"""
        X, y = self.prepare_ml_data(data, current_date)
        if X is None or y is None or len(X) < 20:
            return None
        
        try:
            train_size = min(0.85, max(0.75, 50/len(X)))
            split_idx = int(len(X) * train_size)
            X_train, y_train = X[:split_idx], y[:split_idx]
            
            if len(X_train) < 8:
                return None
                
            model = RandomForestClassifier(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                random_state=42, 
                n_jobs=-1,
                min_samples_split=3,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            logger.error(f"Error training ML model for {ticker}: {e}")
            return None

    def is_bear_market(self, date):
        """Optimized bear market detection"""
        dji_data = self.data.get('^DJI')
        if dji_data is None:
            return False
            
        # Use vectorized boolean indexing
        valid_data = dji_data[dji_data.index <= date]
        if valid_data.empty:
            return False
            
        latest_data = valid_data.iloc[-1]
        current_price = latest_data['Close']
        sma_200 = latest_data.get('SMA_200', current_price)
        
        if pd.isna(sma_200):
            return False
            
        return current_price < (sma_200 * 0.95)

    def calculate_momentum_score(self, row, ticker_idx):
        """Optimized momentum score calculation"""
        try:
            momentum_score = 0
            
            # Price momentum component
            price_momentum = row.get('Price_Momentum', 0)
            if not pd.isna(price_momentum):
                momentum_score += price_momentum * 2
            
            # Volume momentum component
            volume_momentum = row.get('Volume_Momentum', 0)
            if not pd.isna(volume_momentum) and volume_momentum > 0:
                momentum_score += min(volume_momentum, 0.5)
            
            # Trend strength component
            trend_strength = row.get('Trend_Strength', 0)
            if not pd.isna(trend_strength) and trend_strength > 0:
                momentum_score += trend_strength * 3
            
            # RSI momentum
            rsi = row.get('RSI', 50)
            if not pd.isna(rsi):
                if 30 < rsi < 50:
                    momentum_score += 0.2
                elif rsi < 35:
                    momentum_score += 0.1
            
            return momentum_score
        except:
            return 0

    def generate_signal_batch(self, tickers_data_list, current_date):
        """Generate signals for multiple tickers in batch"""
        
        def process_single_ticker(args):
            ticker, data, current_date = args
            return ticker, self.generate_signal(ticker, data, current_date)
        
        # Use parallel processing for signal generation
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(process_single_ticker, tickers_data_list))
        
        return dict(results)

    # def generate_signal(self, ticker, data, current_date):
    #     """Optimized signal generation"""
    #     if data is None or data.empty or ticker == '^DJI':
    #         return 0, 0.5
            
    #     ticker_idx = self.ticker_to_idx.get(ticker, -1)
    #     if ticker_idx == -1:
    #         return 0, 0.5
            
    #     # Use pre-computed indicators
    #     current_data = data[data.index <= current_date]
    #     if current_data.empty:
    #         return 0, 0.5
            
    #     latest_data = current_data.iloc[-1]
    #     signal = 0
    #     confidence = 0.5
        
    #     # Check required features
    #     enhanced_features = [
    #         'RSI', 'Stoch_K', 'Stoch_D', 'BB_Lower', 'BB_Upper', 'MA_Short', 'MA_Long', 
    #         'MA_Crossover', 'EMA_Fast', 'EMA_Slow', 'EMA_Signal', 'ATR', 'Volume_SMA_20',
    #         'Price_Momentum', 'Volume_Momentum', 'Trend_Strength', 'Vol_Adj_Momentum'
    #     ]
        
    #     if any(pd.isna(latest_data.get(feat, np.nan)) for feat in enhanced_features):
    #         return signal, confidence
        
    #     # Volume condition
    #     volume_condition = (latest_data['Volume'] > 
    #                       latest_data['Volume_SMA_20'] * self.params['volume_multiplier'])
        
    #     bear_market = self.is_bear_market(current_date)
        
    #     # Calculate and store momentum score
    #     momentum_score = self.calculate_momentum_score(latest_data, ticker_idx)
    #     self.momentum_scores[ticker_idx] = momentum_score
        
    #     # Signal generation strategies
    #     signal_strength = 0
    #     signal_reasons = []
        
    #     # ML-based signal
    #     ml_confidence = 0.5
    #     model = self.models.get(ticker)
    #     if model is not None:
    #         try:
    #             # Prepare features as numpy array for speed
    #             features_array = np.array([latest_data[feat] for feat in enhanced_features]).reshape(1, -1)
    #             prob = model.predict_proba(features_array)[0]
    #             ml_confidence = prob[1] if len(prob) > 1 else 0.5
                
    #             if ml_confidence >= self.params['confidence_threshold'].get(ticker, 0.52):
    #                 signal_strength += ml_confidence
    #                 signal_reasons.append(f"ML({ml_confidence:.3f})")
    #         except Exception as e:
    #             logger.error(f"Error in ML prediction for {ticker}: {e}")
        
    #     # RSI + Stochastic oversold recovery
    #     rsi = latest_data['RSI']
    #     stoch_k = latest_data['Stoch_K']
    #     rsi_threshold = self.params['rsi_oversold'].get(ticker, 45)
    #     if (rsi < rsi_threshold - 2 or stoch_k < self.params['stoch_oversold'] - 3):
    #         signal_strength += 0.3
    #         signal_reasons.append("Oversold")
        
    #     # EMA crossover
    #     if latest_data['EMA_Signal'] == 1 and latest_data['MA_Crossover'] == 1:
    #         signal_strength += 0.4
    #         signal_reasons.append("EMA_Cross")
        
    #     # Bollinger Band bounce
    #     current_price = latest_data['Close']
    #     bb_lower = latest_data['BB_Lower']
    #     if current_price <= bb_lower * 1.01:
    #         signal_strength += 0.3
    #         signal_reasons.append("BB_Bounce")
        
    #     # Momentum-based signals
    #     if momentum_score > self.params['momentum_threshold'] + 0.05:
    #         signal_strength += min(momentum_score, 0.5)
    #         signal_reasons.append(f"Momentum({momentum_score:.2f})")
        
    #     # Volume breakout
    #     if latest_data['Volume'] > latest_data['Volume_SMA_20'] * 1.6:
    #         signal_strength += 0.2
    #         signal_reasons.append("Vol_Breakout")
        
    #     # Trend following
    #     if (latest_data['MA_Crossover'] == 1 and 
    #         not bear_market and 
    #         latest_data['Trend_Strength'] > 0):
    #         signal_strength += 0.3
    #         signal_reasons.append("Trend_Follow")
        
    #     # Mean reversion
    #     if (not bear_market and 
    #         rsi < 38 and 
    #         current_price > latest_data['MA_Long'] * 0.99):
    #         signal_strength += 0.25
    #         signal_reasons.append("Mean_Reversion")
        
    #     # Signal threshold and quality filters
    #     MIN_SIGNAL_STRENGTH = 0.28
    #     quality_bonus = 0
        
    #     strong_volume_condition = latest_data['Volume'] > latest_data['Volume_SMA_20'] * 0.85
        
    #     # Bonus for multiple signals
    #     if len(signal_reasons) >= 2:
    #         quality_bonus += 0.05
    #     if len(signal_reasons) >= 3:
    #         quality_bonus += 0.10
            
    #     # Bonus for strong ML confidence
    #     if ml_confidence > 0.6:
    #         quality_bonus += 0.08
    #     elif ml_confidence > 0.55:
    #         quality_bonus += 0.04
            
    #     adjusted_signal_strength = signal_strength + quality_bonus
        
    #     # Entry conditions
    #     entry_conditions = [
    #         adjusted_signal_strength >= MIN_SIGNAL_STRENGTH,
    #         strong_volume_condition or adjusted_signal_strength > 0.55,
    #         len(signal_reasons) >= 1
    #     ]
        
    #     if all(entry_conditions):
    #         signal = 1
    #         confidence = min(0.95, 0.5 + adjusted_signal_strength)
            
    #         if len(signal_reasons) >= 3:
    #             confidence += 0.12
    #         if ml_confidence > 0.65:
    #             confidence += 0.06
        
    #     # Store debug info (reduced frequency for performance)
    #     if signal == 1 or len(self.signal_debug) % 1000 == 0:
    #         debug_info = {
    #             'date': current_date,
    #             'ticker': ticker,
    #             'signal': signal,
    #             'confidence': confidence,
    #             'signal_strength': signal_strength,
    #             'adjusted_signal_strength': adjusted_signal_strength,
    #             'reasons': signal_reasons,
    #             'momentum_score': momentum_score,
    #             'rsi': rsi,
    #             'ema_signal': latest_data['EMA_Signal'],
    #             'volume_condition': strong_volume_condition,
    #             'bear_market': bear_market,
    #             'quality_bonus': quality_bonus
    #         }
    #         self.signal_debug.append(debug_info)
        
    #     return signal, confidence
    def generate_signal(self, ticker, data, current_date):
        """Enhanced signal generation for LONG-SHORT strategy"""
        if data is None or data.empty or ticker == '^DJI':
            return 0, 0.5
            
        ticker_idx = self.ticker_to_idx.get(ticker, -1)
        if ticker_idx == -1:
            return 0, 0.5
            
        # Use pre-computed indicators
        current_data = data[data.index <= current_date]
        if current_data.empty:
            return 0, 0.5
            
        latest_data = current_data.iloc[-1]
        signal = 0
        confidence = 0.5
        
        # Check required features
        enhanced_features = [
            'RSI', 'Stoch_K', 'Stoch_D', 'BB_Lower', 'BB_Upper', 'MA_Short', 'MA_Long', 
            'MA_Crossover', 'EMA_Fast', 'EMA_Slow', 'EMA_Signal', 'ATR', 'Volume_SMA_20',
            'Price_Momentum', 'Volume_Momentum', 'Trend_Strength', 'Vol_Adj_Momentum'
        ]
        
        if any(pd.isna(latest_data.get(feat, np.nan)) for feat in enhanced_features):
            return signal, confidence
        
        # Volume condition
        volume_condition = (latest_data['Volume'] > 
                          latest_data['Volume_SMA_20'] * self.params['volume_multiplier'])
        
        bear_market = self.is_bear_market(current_date)
        
        # Calculate and store momentum score
        momentum_score = self.calculate_momentum_score(latest_data, ticker_idx)
        self.momentum_scores[ticker_idx] = momentum_score
        
        # ========== LONG SIGNAL LOGIC ==========
        long_signal_strength = 0
        long_reasons = []
        
        # ML-based signal for longs
        ml_long_confidence = 0.5
        model = self.models.get(ticker)
        if model is not None:
            try:
                features_array = np.array([latest_data[feat] for feat in enhanced_features]).reshape(1, -1)
                prob = model.predict_proba(features_array)[0]
                ml_long_confidence = prob[1] if len(prob) > 1 else 0.5
                
                if ml_long_confidence >= self.params['confidence_threshold'].get(ticker, 0.52):
                    long_signal_strength += ml_long_confidence
                    long_reasons.append(f"ML_Long({ml_long_confidence:.3f})")
            except Exception as e:
                logger.error(f"Error in ML prediction for {ticker}: {e}")
        
        # RSI + Stochastic oversold recovery (LONG)
        rsi = latest_data['RSI']
        stoch_k = latest_data['Stoch_K']
        rsi_threshold = self.params['rsi_oversold'].get(ticker, 45)
        if (rsi < rsi_threshold - 2 or stoch_k < self.params['stoch_oversold'] - 3):
            long_signal_strength += 0.3
            long_reasons.append("Oversold_Long")
        
        # EMA crossover (LONG)
        if latest_data['EMA_Signal'] == 1 and latest_data['MA_Crossover'] == 1:
            long_signal_strength += 0.4
            long_reasons.append("EMA_Cross_Long")
        
        # Bollinger Band bounce (LONG)
        current_price = latest_data['Close']
        bb_lower = latest_data['BB_Lower']
        if current_price <= bb_lower * 1.01:
            long_signal_strength += 0.3
            long_reasons.append("BB_Bounce_Long")
        
        # Momentum-based signals (LONG)
        if momentum_score > self.params['momentum_threshold'] + 0.05:
            long_signal_strength += min(momentum_score, 0.5)
            long_reasons.append(f"Momentum_Long({momentum_score:.2f})")
        
        # Volume breakout (LONG)
        if latest_data['Volume'] > latest_data['Volume_SMA_20'] * 1.6:
            long_signal_strength += 0.2
            long_reasons.append("Vol_Breakout_Long")
        
        # Trend following (LONG)
        if (latest_data['MA_Crossover'] == 1 and 
            not bear_market and 
            latest_data['Trend_Strength'] > 0):
            long_signal_strength += 0.3
            long_reasons.append("Trend_Follow_Long")
        
        # Mean reversion (LONG)
        if (not bear_market and 
            rsi < 38 and 
            current_price > latest_data['MA_Long'] * 0.99):
            long_signal_strength += 0.25
            long_reasons.append("Mean_Reversion_Long")
        
        # ========== SHORT SIGNAL LOGIC ==========
        short_signal_strength = 0
        short_reasons = []
        
        # ML-based signal for shorts (inverse of long confidence)
        ml_short_confidence = 1 - ml_long_confidence
        if ml_short_confidence >= self.params['confidence_threshold'].get(ticker, 0.52):
            short_signal_strength += ml_short_confidence * 0.8  # Slightly less weight than longs
            short_reasons.append(f"ML_Short({ml_short_confidence:.3f})")
        
        # RSI + Stochastic overbought breakdown (SHORT)
        rsi_overbought_threshold = self.params['rsi_overbought'].get(ticker, 65)
        if (rsi > rsi_overbought_threshold + 2 or stoch_k > self.params['stoch_overbought'] + 3):
            short_signal_strength += 0.3
            short_reasons.append("Overbought_Short")
        
        # EMA bearish crossover (SHORT)
        if latest_data['EMA_Signal'] == 0 and latest_data['MA_Crossover'] == 0:
            short_signal_strength += 0.4
            short_reasons.append("EMA_Cross_Short")
        
        # Bollinger Band rejection (SHORT)
        bb_upper = latest_data['BB_Upper']
        if current_price >= bb_upper * 0.99:
            short_signal_strength += 0.3
            short_reasons.append("BB_Rejection_Short")
        
        # Negative momentum (SHORT)
        if momentum_score < -self.params['momentum_threshold'] - 0.05:
            short_signal_strength += min(abs(momentum_score), 0.5)
            short_reasons.append(f"Momentum_Short({momentum_score:.2f})")
        
        # Bear market trend following (SHORT)
        if (bear_market and 
            latest_data['MA_Crossover'] == 0 and 
            latest_data['Trend_Strength'] < -0.01):
            short_signal_strength += 0.4
            short_reasons.append("Bear_Trend_Short")
        
        # High volume distribution (SHORT)
        if (latest_data['Volume'] > latest_data['Volume_SMA_20'] * 1.8 and 
            rsi > 70 and 
            current_price < latest_data['Close']):  # Price declining on high volume
            short_signal_strength += 0.25
            short_reasons.append("Distribution_Short")
        
        # Gap down continuation (SHORT)
        if (current_price < latest_data['MA_Short'] * 0.95 and 
            latest_data['Trend_Strength'] < -0.02):
            short_signal_strength += 0.2
            short_reasons.append("Gap_Down_Short")
        
        # ========== SIGNAL DECISION LOGIC ==========
        MIN_SIGNAL_STRENGTH = 0.30
        
        # Quality bonuses
        long_quality_bonus = 0
        short_quality_bonus = 0
        
        strong_volume_condition = latest_data['Volume'] > latest_data['Volume_SMA_20'] * 0.85
        
        # Bonus for multiple signals
        if len(long_reasons) >= 2:
            long_quality_bonus += 0.05
        if len(long_reasons) >= 3:
            long_quality_bonus += 0.10
            
        if len(short_reasons) >= 2:
            short_quality_bonus += 0.05
        if len(short_reasons) >= 3:
            short_quality_bonus += 0.10
        
        # Bonus for strong ML confidence
        if ml_long_confidence > 0.6:
            long_quality_bonus += 0.08
        elif ml_long_confidence > 0.55:
            long_quality_bonus += 0.04
            
        if ml_short_confidence > 0.6:
            short_quality_bonus += 0.08
        elif ml_short_confidence > 0.55:
            short_quality_bonus += 0.04
        
        adjusted_long_strength = long_signal_strength + long_quality_bonus
        adjusted_short_strength = short_signal_strength + short_quality_bonus
        
        # Entry conditions for LONG
        long_entry_conditions = [
            adjusted_long_strength >= MIN_SIGNAL_STRENGTH,
            strong_volume_condition or adjusted_long_strength > 0.55,
            len(long_reasons) >= 1,
            adjusted_long_strength > adjusted_short_strength  # Long stronger than short
        ]
        
        # Entry conditions for SHORT
        short_entry_conditions = [
            adjusted_short_strength >= MIN_SIGNAL_STRENGTH,
            strong_volume_condition or adjusted_short_strength > 0.55,
            len(short_reasons) >= 1,
            adjusted_short_strength > adjusted_long_strength,  # Short stronger than long
            bear_market or rsi > 60  # Additional filter for shorts
        ]
        
        # Final signal determination
        if all(long_entry_conditions):
            signal = 1  # LONG
            confidence = min(0.95, 0.5 + adjusted_long_strength)
            signal_reasons = long_reasons
            
            if len(long_reasons) >= 3:
                confidence += 0.12
            if ml_long_confidence > 0.65:
                confidence += 0.06
                
        elif all(short_entry_conditions):
            signal = -1  # SHORT
            confidence = min(0.95, 0.5 + adjusted_short_strength)
            signal_reasons = short_reasons
            
            if len(short_reasons) >= 3:
                confidence += 0.12
            if ml_short_confidence > 0.65:
                confidence += 0.06
        else:
            signal_reasons = []
        
        # Store debug info (reduced frequency for performance)
        if signal != 0 or len(self.signal_debug) % 1000 == 0:
            debug_info = {
                'date': current_date,
                'ticker': ticker,
                'signal': signal,
                'confidence': confidence,
                'long_strength': long_signal_strength,
                'short_strength': short_signal_strength,
                'adjusted_long_strength': adjusted_long_strength,
                'adjusted_short_strength': adjusted_short_strength,
                'reasons': signal_reasons,
                'momentum_score': momentum_score,
                'rsi': rsi,
                'ema_signal': latest_data['EMA_Signal'],
                'volume_condition': strong_volume_condition,
                'bear_market': bear_market,
                'long_quality_bonus': long_quality_bonus,
                'short_quality_bonus': short_quality_bonus
            }
            self.signal_debug.append(debug_info)
        
        return signal, confidence

    def calculate_portfolio_value(self, current_date, is_forward_test=False):
        """Optimized portfolio value calculation"""
        if is_forward_test:
            portfolio_value = self.forward_test_cash
            positions = self.forward_test_positions
            shares = self.forward_test_shares
        else:
            portfolio_value = self.cash
            positions = self.positions
            shares = self.shares
        
        # Vectorized calculation where possible
        for i, ticker in enumerate(self.tickers):
            if ticker == '^DJI' or shares[i] == 0:
                continue
                
            data = self.data.get(ticker)
            if data is None or data.empty:
                continue
                
            # Use boolean indexing for speed
            valid_data = data[data.index <= current_date]
            if valid_data.empty:
                continue
                
            current_price = valid_data.iloc[-1]['Close']
            portfolio_value += shares[i] * current_price
            
        return max(portfolio_value, 0)

    def calculate_dynamic_position_size(self, ticker, price, confidence, portfolio_value):
        """Optimized dynamic position sizing"""
        ticker_idx = self.ticker_to_idx.get(ticker, -1)
        if ticker_idx == -1:
            return 0, 0
            
        base_position_pct = 0.65
        
        # Confidence multiplier
        confidence_multiplier = 0.5 + (confidence - 0.5) * 1.5
        
        # Momentum multiplier
        momentum_score = self.momentum_scores[ticker_idx]
        momentum_multiplier = 1.0 + min(momentum_score * 0.5, 0.5)
        
        # Diversification factor
        current_positions = np.count_nonzero(self.positions)
        diversification_factor = max(0.7, 1.0 - current_positions * 0.05)
        
        final_position_pct = (base_position_pct * 
                             confidence_multiplier * 
                             momentum_multiplier * 
                             diversification_factor)
        
        # Apply constraints
        final_position_pct = min(final_position_pct, self.max_single_position)
        final_position_pct = max(final_position_pct, self.min_single_position)
        
        target_position_value = portfolio_value * final_position_pct
        shares = int(target_position_value / price)
        
        return shares, final_position_pct

    def execute_trade(self, ticker, price, date, signal, pre_pv, confidence=0.5, close_full=False, is_forward_test=False):
        """Optimized trade execution"""
        ticker_idx = self.ticker_to_idx.get(ticker, -1)
        if ticker_idx == -1:
            return
            
        # Handle position closing
        if close_full:
            if is_forward_test:
                if self.forward_test_shares[ticker_idx] == 0:
                    return
                shares_to_close = abs(self.forward_test_shares[ticker_idx])
            else:
                if self.shares[ticker_idx] == 0:
                    return
                shares_to_close = abs(self.shares[ticker_idx])
                
            transaction_cost = shares_to_close * self.transaction_cost_per_share
            
            if is_forward_test:
                if self.forward_test_shares[ticker_idx] > 0:
                    proceeds = shares_to_close * price - transaction_cost
                    self.forward_test_cash += proceeds
                    action = 'sell'
                else:
                    cost = shares_to_close * price + transaction_cost
                    if self.forward_test_cash >= cost:
                        self.forward_test_cash -= cost
                        action = 'buy_to_cover'
                    else:
                        trade_logger.warning(f"[FORWARD] Insufficient cash to close short {ticker}")
                        return
                        
                self.forward_test_trades.append({
                    'date': date, 'ticker': ticker, 'action': action,
                    'shares': shares_to_close, 'price': price, 'transaction_cost': transaction_cost
                })
                
                trade_logger.info(f"[FORWARD] CLOSE {ticker}: {action} {shares_to_close} @ ${price:.2f}")
                
                self.forward_test_shares[ticker_idx] = 0
                self.forward_test_entry_prices[ticker_idx] = np.nan
                self.forward_test_entry_dates[ticker_idx] = None  # Reset to None
                self.update_position_status(ticker_idx, date, is_forward_test=True)
            else:
                if self.shares[ticker_idx] > 0:
                    proceeds = shares_to_close * price - transaction_cost
                    self.cash += proceeds
                    action = 'sell'
                else:
                    cost = shares_to_close * price + transaction_cost
                    if self.cash >= cost:
                        self.cash -= cost
                        action = 'buy_to_cover'
                    else:
                        trade_logger.warning(f"Insufficient cash to close short {ticker}")
                        return
                        
                self.trades.append({
                    'date': date, 'ticker': ticker, 'action': action,
                    'shares': shares_to_close, 'price': price, 'transaction_cost': transaction_cost
                })
                
                trade_logger.info(f"CLOSE {ticker}: {action} {shares_to_close} @ ${price:.2f}")
                
                self.shares[ticker_idx] = 0
                self.entry_prices[ticker_idx] = np.nan
                self.entry_dates[ticker_idx] = None  # Reset to None
                self.last_trade_dates[ticker_idx] = date  # Store as pandas Timestamp
                self.update_position_status(ticker_idx, date, is_forward_test=False)
            return
    
        # Open new positions
        if signal == 0:
            return
            
        shares_to_trade, position_pct = self.calculate_dynamic_position_size(ticker, price, confidence, pre_pv)
        
        if shares_to_trade <= 0:
            return
            
        transaction_cost = shares_to_trade * self.transaction_cost_per_share
        
        # Long trade
        if signal == 1:
            if is_forward_test:
                if self.forward_test_positions[ticker_idx] <= 0:
                    total_cost = shares_to_trade * price + transaction_cost
                    
                    if self.forward_test_cash >= total_cost:
                        if self.forward_test_positions[ticker_idx] == -1:
                            self.execute_trade(ticker, price, date, 0, pre_pv, close_full=True, is_forward_test=True)
                            total_cost = shares_to_trade * price + transaction_cost
                            if self.forward_test_cash < total_cost:
                                return
                        
                        self.forward_test_cash -= total_cost
                        self.forward_test_shares[ticker_idx] = shares_to_trade
                        self.forward_test_entry_prices[ticker_idx] = price
                        self.forward_test_entry_dates[ticker_idx] = date  # Store as pandas Timestamp
                        
                        trade_logger.info(f"[FORWARD] BUY {ticker}: {shares_to_trade} @ ${price:.2f}, "
                                        f"Position: {position_pct*100:.1f}%, Conf: {confidence:.3f}")
                        
                        self.forward_test_trades.append({
                            'date': date, 'ticker': ticker, 'action': 'buy',
                            'shares': shares_to_trade, 'price': price, 'transaction_cost': transaction_cost
                        })
            else:
                if self.positions[ticker_idx] <= 0:
                    total_cost = shares_to_trade * price + transaction_cost
                    
                    if self.cash >= total_cost:
                        if self.positions[ticker_idx] == -1:
                            self.execute_trade(ticker, price, date, 0, pre_pv, close_full=True)
                            total_cost = shares_to_trade * price + transaction_cost
                            if self.cash < total_cost:
                                return
                        
                        self.cash -= total_cost
                        self.shares[ticker_idx] = shares_to_trade
                        self.entry_prices[ticker_idx] = price
                        self.entry_dates[ticker_idx] = date  # Store as pandas Timestamp
                        self.last_trade_dates[ticker_idx] = date  # Store as pandas Timestamp
                        
                        trade_logger.info(f"BUY {ticker}: {shares_to_trade} @ ${price:.2f}, "
                                        f"Position: {position_pct*100:.1f}%, Conf: {confidence:.3f}")
                        
                        self.trades.append({
                            'date': date, 'ticker': ticker, 'action': 'buy',
                            'shares': shares_to_trade, 'price': price, 'transaction_cost': transaction_cost
                        })
        
        self.update_position_status(ticker_idx, date, is_forward_test=is_forward_test)

    def update_position_status(self, ticker_idx, date, is_forward_test=False):
        """Update position status using array indices"""
        if is_forward_test:
            if self.forward_test_shares[ticker_idx] > 0:
                self.forward_test_positions[ticker_idx] = 1
            elif self.forward_test_shares[ticker_idx] < 0:
                self.forward_test_positions[ticker_idx] = -1
            else:
                self.forward_test_positions[ticker_idx] = 0
        else:
            if self.shares[ticker_idx] > 0:
                self.positions[ticker_idx] = 1
            elif self.shares[ticker_idx] < 0:
                self.positions[ticker_idx] = -1
            else:
                self.positions[ticker_idx] = 0

    def record_equity(self, date, value, is_forward_test=False):
        """Efficiently record equity values"""
        if is_forward_test:
            if self.forward_test_equity_idx < len(self.forward_test_equity_values):
                self.forward_test_equity_values[self.forward_test_equity_idx] = value
                self.forward_test_equity_dates[self.forward_test_equity_idx] = np.datetime64(date)
                self.forward_test_equity_idx += 1
        else:
            if self.equity_idx < len(self.equity_values):
                self.equity_values[self.equity_idx] = value
                self.equity_dates[self.equity_idx] = np.datetime64(date)
                self.equity_idx += 1

    def get_equity_dataframe(self, is_forward_test=False):
        """Convert equity arrays to DataFrame"""
        if is_forward_test:
            valid_indices = self.forward_test_equity_idx
            return pd.DataFrame({
                'Date': pd.to_datetime(self.forward_test_equity_dates[:valid_indices]),
                'Equity': self.forward_test_equity_values[:valid_indices],
                'Phase': 'Forward Test'
            })
        else:
            valid_indices = self.equity_idx
            return pd.DataFrame({
                'Date': pd.to_datetime(self.equity_dates[:valid_indices]),
                'Equity': self.equity_values[:valid_indices],
                'Phase': 'Backtest'
            })

    def initialize_forward_test_state(self):
        """Initialize forward test state with CLEAN SLATE - no carried positions"""
        logger.info("Initializing forward test with clean slate (no carried positions)...")
        
        # Calculate final backtest portfolio value for logging
        backtest_portfolio_value = self.calculate_portfolio_value(self.backtest_end_date)
        logger.info(f"Backtest final portfolio value: ${backtest_portfolio_value:,.2f}")
        
        # CLEAN START: Initialize forward test with cash equal to final backtest value
        # This gives the forward test the same capital that the backtest achieved
        final_backtest_cash = self.cash
        
        # Calculate value of open positions to convert to cash
        position_values = 0
        for i, ticker in enumerate(self.tickers):
            if self.shares[i] != 0:
                data = self.data.get(ticker)
                if data is not None and not data.empty:
                    # Get price at backtest end date
                    valid_data = data[data.index <= self.backtest_end_date]
                    if not valid_data.empty:
                        price_at_split = valid_data.iloc[-1]['Close']
                        position_values += self.shares[i] * price_at_split
                        logger.info(f"Converting {ticker} position: {self.shares[i]} shares @ ${price_at_split:.2f} = ${self.shares[i] * price_at_split:,.2f}")
        
        # Forward test starts with ALL CASH, NO POSITIONS
        # self.forward_test_cash = final_backtest_cash + position_values
        self.forward_test_cash = 100000
        
        # Initialize empty arrays for forward test - CLEAN SLATE
        n_tickers = len(self.tickers)
        self.forward_test_positions = np.zeros(n_tickers, dtype=np.int8)  # All positions = 0
        self.forward_test_shares = np.zeros(n_tickers, dtype=np.int32)    # All shares = 0
        self.forward_test_entry_prices = np.full(n_tickers, np.nan, dtype=np.float32)  # No entry prices
        self.forward_test_entry_dates = [None] * n_tickers  # No entry dates
        
        logger.info(f"Forward test starting with clean slate:")
        logger.info(f"  Starting cash: ${self.forward_test_cash:,.2f}")
        logger.info(f"  Starting positions: 0 (all flat)")
        logger.info(f"  Forward test will build its own entry list from scratch")
        logger.info("="*50)

    @performance_monitor
    def backtest(self):
        """Main optimized backtesting method"""
        # Pre-compute all indicators first
        if not self.indicators_computed:
            self.calculate_all_indicators_vectorized()
        
        # Get all trading dates
        all_dates = sorted(set().union(*[data.index for data in self.data.values() 
                                        if data is not None and not data.empty]))
        
        if not all_dates:
            logger.error("No trading dates available")
            return
        
        # Split dates
        backtest_dates = [d for d in all_dates if d <= self.backtest_end_date]
        forward_test_dates = [d for d in all_dates if d > self.backtest_end_date]
        
        logger.info(f"Backtest: {len(backtest_dates)} days, Forward test: {len(forward_test_dates)} days")
        
        # PHASE 1: BACKTEST
        logger.info("="*50)
        logger.info("STARTING BACKTEST PHASE")
        logger.info("="*50)
        
        # Train initial models
        self.train_ml_models_parallel()
        
        signal_count = 0
        trade_attempts = 0
        
        # Process dates in batches for better performance
        BATCH_SIZE = 50
        
        for batch_start in range(0, len(backtest_dates), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(backtest_dates))
            batch_dates = backtest_dates[batch_start:batch_end]
            
            for i, date in enumerate(batch_dates):
                current_portfolio_value = self.calculate_portfolio_value(date)
                self.record_equity(date, current_portfolio_value, is_forward_test=False)
                
                # Get current prices for all tickers at once
                current_prices = {}
                atr_values = {}
                
                for ticker in self.tickers:
                    data = self.data.get(ticker)
                    if data is None or data.empty:
                        continue
                        
                    valid_data = data[data.index <= date]
                    if valid_data.empty:
                        continue
                        
                    latest_data = valid_data.iloc[-1]
                    price = latest_data['Close']
                    if price <= 0:
                        continue
                        
                    current_prices[ticker] = price
                    atr_values[ticker] = latest_data.get('ATR', price * 0.02)
                
                # Process stop losses and take profits for all positions
                for j, ticker in enumerate(self.tickers):
                    if ticker == '^DJI' or ticker not in current_prices:
                        continue
                        
                    price = current_prices[ticker]
                    atr = atr_values[ticker]
                    
                    if self.positions[j] == 1:  # Long position
                        entry_price = self.entry_prices[j]
                        if not np.isnan(entry_price):
                            stop_loss = entry_price - self.params['atr_multiplier_stop'] * atr
                            take_profit = entry_price + self.params['atr_multiplier_profit'] * atr
                            
                            # Trailing stop
                            if price > entry_price * 1.1:
                                trailing_stop = price - 2.0 * atr
                                stop_loss = max(stop_loss, trailing_stop)
                            
                            if price <= stop_loss or price >= take_profit:
                                self.execute_trade(ticker, price, date, -1, current_portfolio_value, close_full=True)
                                
                    elif self.positions[j] == -1:  # Short position
                        entry_price = self.entry_prices[j]
                        if not np.isnan(entry_price):
                            stop_loss = entry_price + self.params['short_atr_multiplier_stop'] * atr
                            take_profit = entry_price - self.params['short_atr_multiplier_profit'] * atr
                            
                            if price >= stop_loss or price <= take_profit:
                                self.execute_trade(ticker, price, date, 1, current_portfolio_value, close_full=True)
                
                # Retrain models less frequently for performance
                global_idx = batch_start + i
                if global_idx > 0 and global_idx % 100 == 0:  # Every 100 days instead of 20
                    logger.info(f"Retraining models at day {global_idx}")
                    self.train_ml_models_parallel()
                
                # Generate signals for positions that are flat
                tickers_to_process = []
                for j, ticker in enumerate(self.tickers):
                    if (ticker != '^DJI' and 
                        ticker in current_prices and 
                        self.positions[j] == 0):
                        tickers_to_process.append((ticker, self.data[ticker], date))
                
                # Process signals in parallel for better performance
                if tickers_to_process:
                    signals_batch = self.generate_signal_batch(tickers_to_process, date)
                    
                    for ticker, (signal, confidence) in signals_batch.items():
                        if signal != 0:
                            signal_count += 1
                            
                        if signal == 1 and ticker in current_prices:
                            trade_attempts += 1
                            self.execute_trade(ticker, current_prices[ticker], date, signal, 
                                             current_portfolio_value, confidence)
        
        logger.info(f"Backtest completed: {signal_count} signals, {trade_attempts} attempts, {len(self.trades)} trades")
        
        # PHASE 2: FORWARD TEST
        if forward_test_dates:
            logger.info("="*50)
            logger.info("STARTING FORWARD TEST PHASE")
            logger.info("="*50)
            
            self.initialize_forward_test_state()
            
            forward_signal_count = 0
            forward_trade_attempts = 0
            
            # Process forward test in batches
            for batch_start in range(0, len(forward_test_dates), BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, len(forward_test_dates))
                batch_dates = forward_test_dates[batch_start:batch_end]
                
                for date in batch_dates:
                    current_portfolio_value = self.calculate_portfolio_value(date, is_forward_test=True)
                    self.record_equity(date, current_portfolio_value, is_forward_test=True)
                    
                    # Get current prices
                    current_prices = {}
                    atr_values = {}
                    
                    for ticker in self.tickers:
                        data = self.data.get(ticker)
                        if data is None or data.empty:
                            continue
                            
                        valid_data = data[data.index <= date]
                        if valid_data.empty:
                            continue
                            
                        latest_data = valid_data.iloc[-1]
                        price = latest_data['Close']
                        if price <= 0:
                            continue
                            
                        current_prices[ticker] = price
                        atr_values[ticker] = latest_data.get('ATR', price * 0.02)
                    
                    # Process stop losses and take profits (forward test starts with no positions)
                    for j, ticker in enumerate(self.tickers):
                        if ticker == '^DJI' or ticker not in current_prices:
                            continue
                            
                        price = current_prices[ticker]
                        atr = atr_values[ticker]
                        
                        if self.forward_test_positions[j] == 1:  # Long position
                            entry_price = self.forward_test_entry_prices[j]
                            if not np.isnan(entry_price):
                                stop_loss = entry_price - self.params['atr_multiplier_stop'] * atr
                                take_profit = entry_price + self.params['atr_multiplier_profit'] * atr
                                
                                # Trailing stop
                                if price > entry_price * 1.1:
                                    trailing_stop = price - 2.0 * atr
                                    stop_loss = max(stop_loss, trailing_stop)
                                
                                if price <= stop_loss or price >= take_profit:
                                    self.execute_trade(ticker, price, date, -1, current_portfolio_value, 
                                                     close_full=True, is_forward_test=True)
                                    
                        elif self.forward_test_positions[j] == -1:  # Short position
                            entry_price = self.forward_test_entry_prices[j]
                            if not np.isnan(entry_price):
                                stop_loss = entry_price + self.params['short_atr_multiplier_stop'] * atr
                                take_profit = entry_price - self.params['short_atr_multiplier_profit'] * atr
                                
                                if price >= stop_loss or price <= take_profit:
                                    self.execute_trade(ticker, price, date, 1, current_portfolio_value, 
                                                     close_full=True, is_forward_test=True)
                    
                    # Generate signals for flat positions (NO MODEL RETRAINING - use backtest-trained models)
                    # Forward test starts with ALL positions flat, so all tickers will be evaluated initially
                    tickers_to_process = []
                    for j, ticker in enumerate(self.tickers):
                        if (ticker != '^DJI' and 
                            ticker in current_prices and 
                            self.forward_test_positions[j] == 0):
                            tickers_to_process.append((ticker, self.data[ticker], date))
                    
                    if tickers_to_process:
                        signals_batch = self.generate_signal_batch(tickers_to_process, date)
                        
                        for ticker, (signal, confidence) in signals_batch.items():
                            if signal != 0:
                                forward_signal_count += 1
                                
                            if signal == 1 and ticker in current_prices:
                                forward_trade_attempts += 1
                                self.execute_trade(ticker, current_prices[ticker], date, signal, 
                                                 current_portfolio_value, confidence, is_forward_test=True)
            
            logger.info(f"Forward test completed: {forward_signal_count} signals, {forward_trade_attempts} attempts, {len(self.forward_test_trades)} trades")
            logger.info("Forward test built its entry list completely independently from backtest")
        
        # Final summary
        logger.info("="*60)
        logger.info("COMPLETE STRATEGY RESULTS")
        logger.info("="*60)
        logger.info(f"Backtest: {len(self.trades)} trades over {len(backtest_dates)} days")
        if forward_test_dates:
            logger.info(f"Forward test: {len(self.forward_test_trades)} trades over {len(forward_test_dates)} days")
            logger.info("Forward test started with clean slate - no positions carried forward")

    @performance_monitor
    def performance_metrics(self):
        """Enhanced performance metrics with optimized calculations"""
        # Get equity data
        backtest_equity = self.get_equity_dataframe(is_forward_test=False)
        forward_test_equity = self.get_equity_dataframe(is_forward_test=True)
        
        # Calculate backtest performance
        backtest_metrics = self._calculate_performance_metrics(
            list(self.trades), backtest_equity, "BACKTEST", 
            self.cash, self.shares, self.entry_prices
        )
        
        # Calculate forward test performance if available
        forward_test_metrics = None
        if len(self.forward_test_trades) > 0 or len(forward_test_equity) > 0:
            forward_test_metrics = self._calculate_performance_metrics(
                list(self.forward_test_trades), forward_test_equity, "FORWARD TEST",
                self.forward_test_cash, self.forward_test_shares, self.forward_test_entry_prices
            )
        
        # Combined performance summary
        summary = self._generate_combined_summary(backtest_metrics, forward_test_metrics)
        
        return summary, backtest_metrics, forward_test_metrics
    
    def _calculate_performance_metrics(self, trades, equity_data, phase_name, cash, shares, entry_prices):
        """Optimized performance metrics calculation"""
        total_trades = len(trades)
        win_trades = 0
        loss_trades = 0
        total_profit = 0.0
        total_wins = 0.0
        total_losses = 0.0
        
        # Use numpy arrays for faster calculations
        profits = np.zeros(len(self.tickers))
        transaction_costs = np.zeros(len(self.tickers))
        trade_counts = np.zeros(len(self.tickers), dtype=int)
        win_counts = np.zeros(len(self.tickers), dtype=int)
        
        # Process trades more efficiently
        positions_dict = defaultdict(list)
        
        for trade in trades:
            ticker = trade['ticker']
            ticker_idx = self.ticker_to_idx.get(ticker, -1)
            if ticker_idx == -1:
                continue
                
            action = trade['action']
            shares_traded = trade['shares']
            price = trade['price']
            cost = trade['transaction_cost']
            
            transaction_costs[ticker_idx] += cost
            trade_counts[ticker_idx] += 1
            
            if action in ['buy', 'short']:
                positions_dict[ticker].append({
                    'shares': shares_traded, 
                    'price': price, 
                    'action': action, 
                    'date': trade['date']
                })
            elif action in ['sell', 'buy_to_cover']:
                shares_to_sell = shares_traded
                while shares_to_sell > 0 and positions_dict[ticker]:
                    pos = positions_dict[ticker][0]
                    pos_shares = pos['shares']
                    pos_price = pos['price']
                    pos_action = pos['action']
                    shares_matched = min(shares_to_sell, pos_shares)
                    
                    if pos_action == 'buy':
                        profit = (price - pos_price) * shares_matched
                    else:
                        profit = (pos_price - price) * shares_matched
                        
                    profits[ticker_idx] += profit
                    total_profit += profit
                    
                    if profit > 0:
                        win_trades += 1
                        win_counts[ticker_idx] += 1
                        total_wins += profit
                    else:
                        loss_trades += 1
                        total_losses += abs(profit)
                    
                    pos['shares'] -= shares_matched
                    shares_to_sell -= shares_matched
                    if pos['shares'] == 0:
                        positions_dict[ticker].pop(0)
        
        # Calculate ratios
        win_rate = win_trades / total_trades if total_trades > 0 else 0
        avg_win = total_wins / win_trades if win_trades > 0 else 0
        avg_loss = total_losses / loss_trades if loss_trades > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        total_transaction_costs = np.sum(transaction_costs)
        
        # Calculate final portfolio value efficiently
        final_portfolio_value = cash
        unrealized_gains = 0
        
        for i, ticker in enumerate(self.tickers):
            if ticker != '^DJI' and shares[i] != 0:
                data = self.data.get(ticker)
                if data is not None and not data.empty:
                    current_price = data.iloc[-1]['Close']
                    final_portfolio_value += shares[i] * current_price
                    
                    if not np.isnan(entry_prices[i]):
                        if shares[i] > 0:
                            unrealized_gains += (current_price - entry_prices[i]) * shares[i]
                        else:
                            unrealized_gains += (entry_prices[i] - current_price) * abs(shares[i])
        
        roi = ((final_portfolio_value - self.initial_capital) / self.initial_capital * 100) if self.initial_capital > 0 else 0
        
        # Enhanced risk metrics
        if equity_data.empty:
            logger.warning(f"No equity data recorded during {phase_name}")
            sharpe_ratio = 0
            max_drawdown = 0
            volatility = 0
            calmar_ratio = 0
        else:
            equity_series = equity_data.set_index('Date')['Equity']
            daily_returns = equity_series.pct_change().dropna()
            risk_free_rate = 0.02 / 252
            
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                excess_returns = daily_returns - risk_free_rate
                sharpe_ratio = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252)
                volatility = daily_returns.std() * np.sqrt(252) * 100
            else:
                sharpe_ratio = 0
                volatility = 0
            
            # Calculate maximum drawdown
            running_max = equity_series.expanding().max()
            drawdowns = (running_max - equity_series) / running_max * 100
            max_drawdown = drawdowns.max() if not drawdowns.empty else 0
            
            # Calmar ratio
            if phase_name == "BACKTEST":
                years = (self.backtest_end_date - self.start_date).days / 365.25
            else:
                years = (self.end_date - self.forward_test_start_date).days / 365.25
            annualized_return = roi / years if years > 0 else roi
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Convert numpy arrays back to dictionaries for compatibility
        profits_by_ticker = {self.tickers[i]: profits[i] for i in range(len(self.tickers))}
        trade_counts_by_ticker = {self.tickers[i]: trade_counts[i] for i in range(len(self.tickers))}
        win_counts_by_ticker = {self.tickers[i]: win_counts[i] for i in range(len(self.tickers))}
        transaction_costs_by_ticker = {self.tickers[i]: transaction_costs[i] for i in range(len(self.tickers))}
        
        return {
            'phase': phase_name,
            'total_trades': total_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'total_profit': total_profit,
            'unrealized_gains': unrealized_gains,
            'total_transaction_costs': total_transaction_costs,
            'final_portfolio_value': final_portfolio_value,
            'roi': roi,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'profits_by_ticker': profits_by_ticker,
            'trade_counts_by_ticker': trade_counts_by_ticker,
            'win_counts_by_ticker': win_counts_by_ticker,
            'transaction_costs_by_ticker': transaction_costs_by_ticker
        }
    
    def _generate_combined_summary(self, backtest_metrics, forward_test_metrics):
        """Generate comprehensive summary"""
        summary = "# OPTIMIZED Performance Metrics with Forward Testing\n\n"
        
        # Backtest results
        summary += "## BACKTEST RESULTS\n"
        summary += f"**Period**: {self.start_date.strftime('%Y-%m-%d')} to {self.backtest_end_date.strftime('%Y-%m-%d')}\n\n"
        summary += self._format_metrics_section(backtest_metrics)
        
        # Forward test results
        if forward_test_metrics:
            summary += "\n## FORWARD TEST RESULTS\n"
            summary += f"**Period**: {self.forward_test_start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n\n"
            summary += self._format_metrics_section(forward_test_metrics)
            
            # Comparison section
            summary += "\n## BACKTEST vs FORWARD TEST COMPARISON\n\n"
            summary += "| Metric | Backtest | Forward Test | Difference |\n"
            summary += "|--------|----------|--------------|------------|\n"
            summary += f"| Total Trades | {backtest_metrics['total_trades']} | {forward_test_metrics['total_trades']} | {forward_test_metrics['total_trades'] - backtest_metrics['total_trades']:+d} |\n"
            summary += f"| Win Rate | {backtest_metrics['win_rate']:.2%} | {forward_test_metrics['win_rate']:.2%} | {(forward_test_metrics['win_rate'] - backtest_metrics['win_rate'])*100:+.2f}% |\n"
            summary += f"| ROI | {backtest_metrics['roi']:.2f}% | {forward_test_metrics['roi']:.2f}% | {forward_test_metrics['roi'] - backtest_metrics['roi']:+.2f}% |\n"
            summary += f"| Sharpe Ratio | {backtest_metrics['sharpe_ratio']:.2f} | {forward_test_metrics['sharpe_ratio']:.2f} | {forward_test_metrics['sharpe_ratio'] - backtest_metrics['sharpe_ratio']:+.2f} |\n"
            summary += f"| Max Drawdown | {backtest_metrics['max_drawdown']:.2f}% | {forward_test_metrics['max_drawdown']:.2f}% | {forward_test_metrics['max_drawdown'] - backtest_metrics['max_drawdown']:+.2f}% |\n"
            summary += f"| Win/Loss Ratio | {backtest_metrics['win_loss_ratio']:.2f} | {forward_test_metrics['win_loss_ratio']:.2f} | {forward_test_metrics['win_loss_ratio'] - backtest_metrics['win_loss_ratio']:+.2f} |\n\n"
            
            # Strategy validation assessment
            summary += "## STRATEGY VALIDATION ASSESSMENT\n\n"
            
            roi_diff = (forward_test_metrics['roi'] - backtest_metrics['roi'])
            win_rate_diff = abs(forward_test_metrics['win_rate'] - backtest_metrics['win_rate'])
            sharpe_diff = (forward_test_metrics['sharpe_ratio'] - backtest_metrics['sharpe_ratio'])
            
            if (roi_diff > -5 or roi_diff < 10) and win_rate_diff < 0.1 and sharpe_diff > 0.5:
                validation_status = "✅ STRONG - Strategy shows consistent performance"
            elif (roi_diff < -10) and win_rate_diff < 0.15 and sharpe_diff < 1.0:
                validation_status = "⚠️ MODERATE - Some performance degradation observed"
            else:
                validation_status = "❌ WEAK - Significant performance degradation in forward test"
            
            summary += f"**Validation Status**: {validation_status}\n\n"
            summary += f"**Testing Method**: Forward test started with clean slate (no carried positions)\n\n"
            
            summary += "**Key Observations**:\n"
            if forward_test_metrics['roi'] > backtest_metrics['roi']:
                summary += f"- Forward test ROI ({forward_test_metrics['roi']:.1f}%) exceeded backtest ({backtest_metrics['roi']:.1f}%)\n"
            else:
                summary += f"- Forward test ROI ({forward_test_metrics['roi']:.1f}%) was lower than backtest ({backtest_metrics['roi']:.1f}%)\n"
            
            if forward_test_metrics['win_rate'] > backtest_metrics['win_rate']:
                summary += f"- Win rate improved from {backtest_metrics['win_rate']:.1%} to {forward_test_metrics['win_rate']:.1%}\n"
            else:
                summary += f"- Win rate declined from {backtest_metrics['win_rate']:.1%} to {forward_test_metrics['win_rate']:.1%}\n"
            
            if forward_test_metrics['max_drawdown'] < backtest_metrics['max_drawdown']:
                summary += f"- Maximum drawdown improved (reduced from {backtest_metrics['max_drawdown']:.1f}% to {forward_test_metrics['max_drawdown']:.1f}%)\n"
            else:
                summary += f"- Maximum drawdown worsened (increased from {backtest_metrics['max_drawdown']:.1f}% to {forward_test_metrics['max_drawdown']:.1f}%)\n"
            
            summary += f"- Forward test built entry list independently (no positions carried from backtest)\n"
                
        else:
            summary += "\n## FORWARD TEST RESULTS\n"
            summary += "**No forward test data available**\n\n"
        
        print(summary)
        return summary
    
    def _format_metrics_section(self, metrics):
        """Format metrics section"""
        section = ""
        section += f"**Total Trades**: {metrics['total_trades']}\n"
        section += f"**Winning Trades**: {metrics['win_trades']} ({metrics['win_rate']:.2%})\n"
        section += f"**Losing Trades**: {metrics['loss_trades']}\n"
        section += f"**Average Win**: ${metrics['avg_win']:,.2f}\n"
        section += f"**Average Loss**: ${metrics['avg_loss']:,.2f}\n"
        section += f"**Win/Loss Ratio**: {metrics['win_loss_ratio']:.2f}\n"
        section += f"**Total Profit/Loss**: ${metrics['total_profit']:,.2f}\n"
        section += f"**Unrealized Gains/Losses**: ${metrics['unrealized_gains']:,.2f}\n"
        section += f"**Total Transaction Costs**: ${metrics['total_transaction_costs']:,.2f}\n"
        section += f"**Final Portfolio Value**: ${metrics['final_portfolio_value']:,.2f}\n"
        section += f"**ROI**: {metrics['roi']:.2f}%\n"
        section += f"**Annualized Volatility**: {metrics['volatility']:.2f}%\n"
        section += f"**Sharpe Ratio**: {metrics['sharpe_ratio']:.2f}\n"
        section += f"**Max Drawdown**: {metrics['max_drawdown']:.2f}%\n"
        section += f"**Calmar Ratio**: {metrics['calmar_ratio']:.2f}\n"
        return section

    def export_features(self):
        """Export enhanced features"""
        for ticker in self.tickers:
            if self.data.get(ticker) is not None and not self.data[ticker].empty:
                filename = f"{ticker}_features_optimized.csv"
                self.data[ticker].to_csv(filename)
                logger.info(f"Exported optimized features to {filename}")

class OptimizedPlottingStrategy:
    """Optimized plotting with better performance"""
    
    def __init__(self, backtest_equity, forward_test_equity=None):
        if isinstance(backtest_equity, pd.DataFrame):
            self.backtest_equity = backtest_equity
        else:
            self.backtest_equity = pd.DataFrame(backtest_equity) if backtest_equity else pd.DataFrame()
            
        if isinstance(forward_test_equity, pd.DataFrame):
            self.forward_test_equity = forward_test_equity
        else:
            self.forward_test_equity = pd.DataFrame(forward_test_equity) if forward_test_equity else None
        
    @performance_monitor
    def plot_strategy(self):
        """Enhanced plotting with optimized performance"""
        if self.backtest_equity.empty:
            logger.warning("No backtest equity data to plot")
            return
            
        # Determine subplot layout
        n_plots = 3 if self.forward_test_equity is not None and not self.forward_test_equity.empty else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 5*n_plots))
        if n_plots == 2:
            ax1, ax2 = axes
        else:
            ax1, ax2, ax3 = axes
        
        # Plot 1: Combined equity curve
        backtest_eq = self.backtest_equity.set_index('Date')['Equity']
        ax1.plot(backtest_eq.index, backtest_eq.values, label='Backtest', color='blue', linewidth=2)
        
        if self.forward_test_equity is not None and not self.forward_test_equity.empty:
            forward_eq = self.forward_test_equity.set_index('Date')['Equity']
            ax1.plot(forward_eq.index, forward_eq.values, label='Forward Test', color='red', linewidth=2)
            
            # Add vertical line to separate periods
            split_date = forward_eq.index[0]
            ax1.axvline(x=split_date, color='gray', linestyle='--', alpha=0.7, label='Forward Test Start')
        
        # Add drawdowns
        running_max = backtest_eq.expanding().max()
        drawdowns = (running_max - backtest_eq) / running_max * 100
        ax1.fill_between(backtest_eq.index, 
                        backtest_eq - (drawdowns/100 * backtest_eq), 
                        backtest_eq, 
                        alpha=0.3, color='lightblue', label='Backtest Drawdowns')
        
        if self.forward_test_equity is not None and not self.forward_test_equity.empty:
            forward_running_max = forward_eq.expanding().max()
            forward_drawdowns = (forward_running_max - forward_eq) / forward_running_max * 100
            ax1.fill_between(forward_eq.index, 
                            forward_eq - (forward_drawdowns/100 * forward_eq), 
                            forward_eq, 
                            alpha=0.3, color='lightcoral', label='Forward Test Drawdowns')
        
        ax1.set_title('OPTIMIZED Portfolio Equity Curve: Backtest vs Forward Test')
        ax1.set_ylabel('Equity ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Returns distribution
        backtest_returns = backtest_eq.pct_change().dropna() * 100
        ax2.hist(backtest_returns, bins=30, alpha=0.7, color='blue', label='Backtest', density=True)
        ax2.axvline(backtest_returns.mean(), color='blue', linestyle='--', 
                   label=f'Backtest Mean: {backtest_returns.mean():.2f}%')
        
        if self.forward_test_equity is not None and not self.forward_test_equity.empty:
            forward_returns = forward_eq.pct_change().dropna() * 100
            ax2.hist(forward_returns, bins=30, alpha=0.7, color='red', label='Forward Test', density=True)
            ax2.axvline(forward_returns.mean(), color='red', linestyle='--', 
                       label=f'Forward Test Mean: {forward_returns.mean():.2f}%')
        
        ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Daily Returns Distribution: Backtest vs Forward Test')
        ax2.set_xlabel('Daily Return (%)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown comparison
        if n_plots == 3:
            ax3.plot(backtest_eq.index, -drawdowns, label='Backtest Drawdown', color='blue', linewidth=2)
            ax3.plot(forward_eq.index, -forward_drawdowns, label='Forward Test Drawdown', color='red', linewidth=2)
            ax3.fill_between(backtest_eq.index, 0, -drawdowns, alpha=0.3, color='lightblue')
            ax3.fill_between(forward_eq.index, 0, -forward_drawdowns, alpha=0.3, color='lightcoral')
            ax3.axvline(x=split_date, color='gray', linestyle='--', alpha=0.7)
            ax3.set_title('Drawdown Comparison: Backtest vs Forward Test')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Drawdown (%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = "optimized_backtest_vs_forward_test.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved optimized analysis plot to {filename}")

def objective_optimized(trial):
    """Optimized objective function"""
    tickers = ['AAPL', 'AMZN', 'GS', 'MA', 'NFLX', 'META', 
               'GOOGL', 'MSFT', 'GROY', 'EWO', 'SYM', 'AMIX', 
               'ESRT', 'INCY', 'BALL', 'MSS', 'FCAL', 'BME', 
               'LRGE', 'QSPT', 'BLV', 'SNAP', 'TSLA', 'V', 
               'USO', 'ALTG', 'QBTS', 'NLY', 'WTM', 'XSVM', 
               'AAT', 'XPAY', 'FEDM', 'CBUS', 'test_1', 'test_2', 
               'test_3', 'test_4', 'test_5', 'test_6', 'test_7', 
               'test_8', 'test_9', 'test_10']
    
    # Optimized parameter space (reduced for faster optimization)
    params = {
        'rsi_oversold': {},
        'rsi_overbought': {},
        'stoch_oversold': trial.suggest_int('stoch_oversold', 15, 35),
        'stoch_overbought': trial.suggest_int('stoch_overbought', 65, 85),
        'bb_period': {},
        'bb_std': {},
        'ma_short': {},
        'ma_long': {},
        'ema_fast': {},
        'ema_slow': {},
        'n_estimators': {},
        'max_depth': {},
        'confidence_threshold': {},
        'atr_multiplier_stop': trial.suggest_float('atr_multiplier_stop', 1.2, 2.5),
        'atr_multiplier_profit': trial.suggest_float('atr_multiplier_profit', 2.5, 5.0),
        'volume_multiplier': trial.suggest_float('volume_multiplier', 0.6, 1.2),
        'short_atr_multiplier_stop': trial.suggest_float('short_atr_multiplier_stop', 1.0, 2.0),
        'short_atr_multiplier_profit': trial.suggest_float('short_atr_multiplier_profit', 2.0, 4.0),
        'momentum_threshold': trial.suggest_float('momentum_threshold', 0.1, 0.6),
        'trend_strength_threshold': trial.suggest_float('trend_strength_threshold', 0.2, 0.8)
    }
    
    # Only optimize key parameters per ticker for speed
    for ticker in tickers[:-1]:  # Optimize only first 10 tickers for speed
        if ticker == '^DJI':
            continue
        params['rsi_oversold'][ticker] = trial.suggest_int(f'rsi_oversold_{ticker}', 35, 50)
        params['rsi_overbought'][ticker] = trial.suggest_int(f'rsi_overbought_{ticker}', 60, 75)
        params['confidence_threshold'][ticker] = trial.suggest_float(f'confidence_threshold_{ticker}', 0.45, 0.70)
    
    # Use defaults for remaining tickers
    for ticker in tickers[10:]:
        if ticker == '^DJI':
            continue
        params['rsi_oversold'][ticker] = 45
        params['rsi_overbought'][ticker] = 65
        params['bb_period'][ticker] = 20
        params['bb_std'][ticker] = 1.8
        params['ma_short'][ticker] = 10
        params['ma_long'][ticker] = 30
        params['ema_fast'][ticker] = 5
        params['ema_slow'][ticker] = 15
        params['n_estimators'][ticker] = 30
        params['max_depth'][ticker] = 5
        params['confidence_threshold'][ticker] = 0.52
    
    try:
        strategy = OptimizedPortfolioStrategy(tickers, start_date=start_date, end_date=end_date, 
                                            initial_capital=100000, params=params)
        strategy.fetch_data()
        strategy.backtest()
        summary, backtest_metrics, forward_test_metrics = strategy.performance_metrics()
        
        # Store metrics
        trial.set_user_attr('backtest_roi', backtest_metrics['roi'])
        trial.set_user_attr('backtest_max_drawdown', backtest_metrics['max_drawdown'])
        trial.set_user_attr('backtest_total_trades', backtest_metrics['total_trades'])
        trial.set_user_attr('backtest_win_rate', backtest_metrics['win_rate'])
        trial.set_user_attr('backtest_sharpe_ratio', backtest_metrics['sharpe_ratio'])
        
        if forward_test_metrics:
            trial.set_user_attr('forward_test_roi', forward_test_metrics['roi'])
            trial.set_user_attr('forward_test_max_drawdown', forward_test_metrics['max_drawdown'])
            trial.set_user_attr('forward_test_total_trades', forward_test_metrics['total_trades'])
            trial.set_user_attr('forward_test_win_rate', forward_test_metrics['win_rate'])
            trial.set_user_attr('forward_test_sharpe_ratio', forward_test_metrics['sharpe_ratio'])
        
        # Enhanced scoring
        if backtest_metrics['total_trades'] == 0:
            return -2000
        
        backtest_score = (
            backtest_metrics['roi'] * 0.4 +
            backtest_metrics['win_rate'] * 100 * 0.3 +
            backtest_metrics['sharpe_ratio'] * 20 * 0.2 +
            -backtest_metrics['max_drawdown'] * 0.2
        )
        
        # Forward test validation
        if forward_test_metrics and forward_test_metrics['total_trades'] > 0:
            roi_consistency = max(0, 100 - abs(forward_test_metrics['roi'] - backtest_metrics['roi']) * 2)
            win_rate_consistency = max(0, 100 - abs(forward_test_metrics['win_rate'] - backtest_metrics['win_rate']) * 200)
            
            forward_test_score = (
                forward_test_metrics['roi'] * 0.3 +
                forward_test_metrics['win_rate'] * 100 * 0.2 +
                roi_consistency * 0.3 +
                win_rate_consistency * 0.2
            )
            
            objective_score = backtest_score * 0.7 + forward_test_score * 0.3
        else:
            objective_score = backtest_score
        
        # Trade frequency bonus
        trade_count = backtest_metrics['total_trades']
        if 45 <= trade_count <= 180:
            objective_score += 50
        elif trade_count < 18:
            objective_score -= 200
        
        # Win rate bonus
        if backtest_metrics['win_rate'] > 0.6:
            objective_score += 200
        elif backtest_metrics['win_rate'] > 0.5:
            objective_score += 120
        elif backtest_metrics['win_rate'] > 0.4:
            objective_score += 60
        elif backtest_metrics['win_rate'] > 0.3:
            objective_score += 30
            
        #Drawdown bonus
        if abs(backtest_metrics['Max_drawdown']) > 30:
            objective_score -= 100
        if abs(backtest_metrics['Max_drawdown']) > 25:
            objective_score -= 500
        if abs(backtest_metrics['Max_drawdown']) > 20:
            objective_score -= 20
        if abs(backtest_metrics['Max_drawdown']) < 20:
            objective_score += 30
        if abs(backtest_metrics['Max_drawdown']) < 15:
            objective_score += 30
        if abs(backtest_metrics['Max_drawdown']) < 10:
            objective_score += 50
        if abs(backtest_metrics['Max_drawdown']) < 5:
            objective_score += 100
            
        logger.info(f'Objective Score: {objective_score}')
        
        return objective_score
        
    except Exception as e:
        logger.error(f"Error in optimized trial: {e}")
        return -2000

@performance_monitor
def main_optimized():
    """Optimized main function with performance improvements"""
    tickers = ['AAPL', 'AMZN', 'GS', 'MA', 'NFLX', 'META', 
               'GOOGL', 'MSFT', 'GROY', 'EWO', 'SYM', 'AMIX', 
               'ESRT', 'INCY', 'BALL', 'MSS', 'FCAL', 'BME', 
               'LRGE', 'QSPT', 'BLV', 'SNAP', 'TSLA', 'V', 
               'USO', 'ALTG', 'QBTS', 'NLY', 'WTM', 'XSVM', 
               'AAT', 'XPAY', 'FEDM', 'CBUS', 'test_1', 'test_2', 
               'test_3', 'test_4', 'test_5', 'test_6', 'test_7', 
               'test_8', 'test_9', 'test_10']
    
    logger.info("Starting OPTIMIZED strategy with enhanced performance...")
    logger.info(f"Available CPU cores: {mp.cpu_count()}")
    logger.info(f"Using {min(mp.cpu_count() - 1, 4)} workers for parallel processing")
    
    # Enhanced default parameters
    enhanced_params = {
        'rsi_oversold': {ticker: 42 for ticker in tickers if ticker != '^DJI'},
        'rsi_overbought': {ticker: 68 for ticker in tickers if ticker != '^DJI'},
        'stoch_oversold': 25,
        'stoch_overbought': 75,
        'bb_period': {ticker: 20 for ticker in tickers if ticker != '^DJI'},
        'bb_std': {ticker: 1.8 for ticker in tickers if ticker != '^DJI'},
        'ma_short': {ticker: 10 for ticker in tickers if ticker != '^DJI'},
        'ma_long': {ticker: 30 for ticker in tickers if ticker != '^DJI'},
        'ema_fast': {ticker: 5 for ticker in tickers if ticker != '^DJI'},
        'ema_slow': {ticker: 15 for ticker in tickers if ticker != '^DJI'},
        'n_estimators': {ticker: 30 for ticker in tickers if ticker != '^DJI'},
        'max_depth': {ticker: 5 for ticker in tickers if ticker != '^DJI'},
        'confidence_threshold': {ticker: 0.52 for ticker in tickers if ticker != '^DJI'},
        'atr_multiplier_stop': 1.8,
        'atr_multiplier_profit': 3.5,
        'volume_multiplier': 0.8,
        'short_atr_multiplier_stop': 1.3,
        'short_atr_multiplier_profit': 2.8,
        'momentum_threshold': 0.3,
        'trend_strength_threshold': 0.4,
        'win_rate_estimate': {ticker: 0.60 for ticker in tickers if ticker != '^DJI'},
        'avg_win_loss_ratio': {ticker: 1.8 for ticker in tickers if ticker != '^DJI'},
    }
    
    # Test run with optimized strategy
    logger.info("Running optimized test strategy...")
    test_strategy = OptimizedPortfolioStrategy(tickers, start_date=start_date, end_date=end_date, 
                                             initial_capital=100000, params=enhanced_params)
    test_strategy.fetch_data()
    test_strategy.backtest()
    summary, backtest_metrics, forward_test_metrics = test_strategy.performance_metrics()
    
    logger.info(f"Optimized test - Backtest: {backtest_metrics['total_trades']} trades, "
               f"{backtest_metrics['win_rate']:.1%} win rate, {backtest_metrics['roi']:.1f}% ROI")
    if forward_test_metrics:
        logger.info(f"Optimized test - Forward: {forward_test_metrics['total_trades']} trades, "
                   f"{forward_test_metrics['win_rate']:.1%} win rate, {forward_test_metrics['roi']:.1f}% ROI")
    
    if backtest_metrics['total_trades'] == 0:
        logger.error("No trades generated in optimized backtest")
        return summary
    
    # Run optimized optimization
    logger.info("Starting optimized parameter optimization...")
    
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective_optimized, n_trials=80)  # Fewer trials but faster execution
    
    logger.info("Optimization completed!")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best objective score: {study.best_trial.value:.2f}")
    logger.info(f"Best Backtest ROI: {study.best_trial.user_attrs.get('backtest_roi', 0):.2f}%")
    logger.info(f"Best Backtest Win Rate: {study.best_trial.user_attrs.get('backtest_win_rate', 0)*100:.2f}%")
    
    if 'forward_test_roi' in study.best_trial.user_attrs:
        logger.info(f"Best Forward Test ROI: {study.best_trial.user_attrs.get('forward_test_roi', 0):.2f}%")
        logger.info(f"Best Forward Test Win Rate: {study.best_trial.user_attrs.get('forward_test_win_rate', 0)*100:.2f}%")
    
    # Build final optimized parameters
    final_params = enhanced_params.copy()  # Start with defaults
    
    # Update with optimized parameters
    for key, value in study.best_trial.params.items():
        if key in ['stoch_oversold', 'stoch_overbought', 'atr_multiplier_stop', 
                  'atr_multiplier_profit', 'volume_multiplier', 'short_atr_multiplier_stop',
                  'short_atr_multiplier_profit', 'momentum_threshold', 'trend_strength_threshold']:
            final_params[key] = value
        elif key.startswith('rsi_oversold_'):
            ticker = key.replace('rsi_oversold_', '')
            final_params['rsi_oversold'][ticker] = value
        elif key.startswith('rsi_overbought_'):
            ticker = key.replace('rsi_overbought_', '')
            final_params['rsi_overbought'][ticker] = value
        elif key.startswith('confidence_threshold_'):
            ticker = key.replace('confidence_threshold_', '')
            final_params['confidence_threshold'][ticker] = value
    
    logger.info("Running final optimized strategy...")
    
    # Run final strategy with best parameters
    final_strategy = OptimizedPortfolioStrategy(tickers, start_date=start_date, end_date=end_date, 
                                               initial_capital=100000, params=final_params)
    final_strategy.fetch_data()
    final_strategy.backtest()
    final_strategy.export_features()
    summary, final_backtest_metrics, final_forward_test_metrics = final_strategy.performance_metrics()
    
    # Generate optimized plots
    backtest_equity = final_strategy.get_equity_dataframe(is_forward_test=False)
    forward_test_equity = final_strategy.get_equity_dataframe(is_forward_test=True)
    
    plotter = OptimizedPlottingStrategy(backtest_equity, forward_test_equity)
    plotter.plot_strategy()
    
    # Final performance summary
    logger.info("="*60)
    logger.info("FINAL OPTIMIZED STRATEGY RESULTS")
    logger.info("="*60)
    logger.info(f"BACKTEST - Trades: {final_backtest_metrics['total_trades']}, "
               f"Win Rate: {final_backtest_metrics['win_rate']:.2%}, "
               f"ROI: {final_backtest_metrics['roi']:.2f}%")
    if final_forward_test_metrics:
        logger.info(f"FORWARD TEST - Trades: {final_forward_test_metrics['total_trades']}, "
                   f"Win Rate: {final_forward_test_metrics['win_rate']:.2%}, "
                   f"ROI: {final_forward_test_metrics['roi']:.2f}%")
    
    # Performance summary
    logger.info("="*60)
    logger.info("PERFORMANCE OPTIMIZATIONS APPLIED")
    logger.info("="*60)
    logger.info("✅ Vectorized indicator calculations")
    logger.info("✅ Parallel signal generation")
    logger.info("✅ Intelligent data caching")
    logger.info("✅ Memory-optimized data structures")
    logger.info("✅ Batch processing optimization")
    logger.info("✅ Reduced ML model retraining frequency")
    logger.info("✅ Numpy array-based computations")
    logger.info("="*60)
    
    return summary

if __name__ == "__main__":
    main_optimized()