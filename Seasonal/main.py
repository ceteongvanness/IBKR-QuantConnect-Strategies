# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class SeasonalETFRotationStrategy(QCAlgorithm):
    """
    Seasonal ETF Rotation Strategy
    
    Strategy Logic:
    1. Seasonal Timing: 
       - Nov-Apr ("Winter"): Favor aggressive/cyclical sectors (XLV, XLI, XLY, XLB)
       - May-Oct ("Summer"): Favor defensive sectors (XLK, XLP, XLU, QQQ)
    
    2. Momentum Filter:
       - Only hold ETFs trading above their 200-day SMA
       - Rank by 3-month momentum within each seasonal group
    
    3. Top N Selection:
       - Select top 3 ETFs from the appropriate seasonal group
       - Equal weight allocation
    
    4. Safety Net:
       - If fewer than 2 ETFs pass the momentum filter, rotate to bonds (TLT/SHY)
       - Use market regime filter (SPY above 200-day SMA)
    
    5. Rebalancing:
       - Monthly rebalancing on first trading day
       - Seasonal switch at end of April and October
    """

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_end_date(2024, 12, 31)
        self.set_cash(100000)
        
        # Strategy parameters
        self.lookback_momentum = 63  # 3-month momentum (approx 63 trading days)
        self.sma_period = 200  # Trend filter period
        self.top_n = 3  # Number of ETFs to hold
        self.min_etfs = 2  # Minimum ETFs required, else go to safety
        
        # Aggressive/Cyclical sectors - historically perform better Nov-Apr
        # Healthcare, Industrials, Consumer Discretionary, Materials
        self.aggressive_etfs = ["XLV", "XLI", "XLY", "XLB"]
        
        # Defensive sectors - historically perform better May-Oct  
        # Technology, Consumer Staples, Utilities, Nasdaq-100
        self.defensive_etfs = ["XLK", "XLP", "XLU", "QQQ"]
        
        # Safety assets - when market conditions are poor
        self.safety_etfs = ["TLT", "SHY"]  # Long-term and short-term treasuries
        
        # Market regime filter
        self.market_etf = "SPY"
        
        # Combine all ETFs
        self.all_etfs = list(set(
            self.aggressive_etfs + 
            self.defensive_etfs + 
            self.safety_etfs + 
            [self.market_etf]
        ))
        
        # Data structures
        self.symbol_data = {}
        self.current_holdings = []  # Track what we currently hold
        
        # Add securities and create indicators
        for ticker in self.all_etfs:
            symbol = self.add_equity(ticker, Resolution.DAILY).symbol
            
            # Create indicators for each symbol
            self.symbol_data[ticker] = SymbolData(
                symbol=symbol,
                momentum=self.mom(symbol, self.lookback_momentum, Resolution.DAILY),
                sma=self.sma(symbol, self.sma_period, Resolution.DAILY),
                roc=self.rocp(symbol, self.lookback_momentum, Resolution.DAILY)
            )
        
        # Schedule monthly rebalancing on first trading day at market open + 30 min
        self.schedule.on(
            self.date_rules.month_start(self.market_etf),
            self.time_rules.after_market_open(self.market_etf, 30),
            self.rebalance
        )
        
        # Track last rebalance for logging
        self.last_rebalance = None
        
        # Warmup period for indicators
        self.set_warm_up(self.sma_period + 10, Resolution.DAILY)
        
        # Set benchmark
        self.set_benchmark(self.market_etf)

    def is_winter_season(self):
        """
        Returns True if we're in the 'aggressive' season (Nov-Apr)
        Returns False if we're in the 'defensive' season (May-Oct)
        """
        month = self.time.month
        return month >= 11 or month <= 4

    def is_market_bullish(self):
        """
        Market regime filter - check if SPY is above its 200-day SMA
        """
        spy_data = self.symbol_data[self.market_etf]
        if not spy_data.sma.is_ready:
            return True  # Default to bullish if indicator not ready
        
        current_price = self.securities[spy_data.symbol].price
        return current_price > spy_data.sma.current.value

    def get_etf_scores(self, etf_list):
        """
        Calculate scores for ETFs based on momentum and trend
        Returns list of (ticker, score) tuples, filtered by trend
        """
        scores = []
        
        for ticker in etf_list:
            data = self.symbol_data[ticker]
            
            # Skip if indicators not ready
            if not data.momentum.is_ready or not data.sma.is_ready:
                continue
            
            # Skip if not trading above SMA (trend filter)
            current_price = self.securities[data.symbol].price
            if current_price <= data.sma.current.value:
                continue
            
            # Score based on momentum (rate of change)
            momentum_score = data.roc.current.value if data.roc.is_ready else 0
            
            scores.append((ticker, momentum_score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def rebalance(self):
        """Monthly rebalancing logic"""
        
        if self.is_warming_up:
            return
        
        # Determine which seasonal group to use
        is_winter = self.is_winter_season()
        season_name = "Winter (Aggressive)" if is_winter else "Summer (Defensive)"
        primary_etfs = self.aggressive_etfs if is_winter else self.defensive_etfs
        
        # Check market regime
        market_bullish = self.is_market_bullish()
        
        self.log(f"\n{'='*50}")
        self.log(f"Rebalancing - {self.time.strftime('%Y-%m-%d')}")
        self.log(f"Season: {season_name}")
        self.log(f"Market Regime: {'Bullish' if market_bullish else 'Bearish'}")
        
        # Get scores for primary ETF group
        etf_scores = self.get_etf_scores(primary_etfs)
        
        self.log(f"ETF Scores: {etf_scores}")
        
        # Determine holdings
        target_holdings = []
        
        if market_bullish and len(etf_scores) >= self.min_etfs:
            # Normal operation - hold top N from seasonal group
            target_holdings = [ticker for ticker, _ in etf_scores[:self.top_n]]
            self.log(f"Selected ETFs: {target_holdings}")
        else:
            # Safety mode - rotate to bonds
            # Prefer TLT in falling rate environment, SHY in rising rate
            # For simplicity, use both equally
            target_holdings = self.safety_etfs
            self.log(f"Safety Mode - Holding: {target_holdings}")
        
        # Calculate target weights
        weight = 1.0 / len(target_holdings) if target_holdings else 0
        
        # Liquidate positions not in target
        for ticker in self.current_holdings:
            if ticker not in target_holdings:
                symbol = self.symbol_data[ticker].symbol
                if self.portfolio[symbol].invested:
                    self.liquidate(symbol)
                    self.log(f"Liquidated: {ticker}")
        
        # Set target positions
        for ticker in target_holdings:
            symbol = self.symbol_data[ticker].symbol
            self.set_holdings(symbol, weight)
            self.log(f"Set {ticker} to {weight*100:.1f}%")
        
        # Update current holdings tracker
        self.current_holdings = target_holdings.copy()
        
        self.last_rebalance = self.time
        self.log(f"{'='*50}\n")

    def on_data(self, data: Slice):
        """Called on each data slice - not used for trading but can add logging"""
        pass

    def on_end_of_algorithm(self):
        """Final summary"""
        self.log(f"\nAlgorithm completed.")
        self.log(f"Final portfolio value: ${self.portfolio.total_portfolio_value:,.2f}")


class SymbolData:
    """Container for symbol-specific data and indicators"""
    
    def __init__(self, symbol, momentum, sma, roc):
        self.symbol = symbol
        self.momentum = momentum
        self.sma = sma
        self.roc = roc
