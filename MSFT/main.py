# region imports
from AlgorithmImports import *
import numpy as np
# endregion

class MSFTBeatBuyHoldV4(QCAlgorithm):
    """
    Strategy V4: Optimized Vol-Scaled Momentum
    
    Improvements over V3 (Sharpe 0.863, CAGR 29.8%, DD 26.4%):
    1. Remove small position when below 200 SMA (was hurting Sharpe)
    2. Stricter trend requirements
    3. Slightly higher leverage in confirmed trends (1.6x vs 1.5x)
    4. Better vol floor to avoid overleveraging in quiet periods
    
    Benchmark: CAGR 26.5%, Sharpe 0.82, MaxDD 37.1%
    """

    def initialize(self):
        self.set_start_date(2015, 1, 1)
        self.set_end_date(2025, 1, 1)
        self.set_cash(100000)
        
        # Add MSFT
        self.msft = self.add_equity("MSFT", Resolution.DAILY)
        self.msft.set_data_normalization_mode(DataNormalizationMode.ADJUSTED)
        
        # Trend indicators
        self.sma_200 = self.sma("MSFT", 200, Resolution.DAILY)
        self.sma_50 = self.sma("MSFT", 50, Resolution.DAILY)
        
        # RSI for timing
        self.rsi_indicator = self.rsi("MSFT", 14, MovingAverageType.WILDERS, Resolution.DAILY)
        
        # Price and return history
        self.lookback_12m = 252
        self.price_history = RollingWindow[float](self.lookback_12m + 1)
        self.return_history = RollingWindow[float](25)
        
        # === OPTIMIZED PARAMETERS ===
        self.target_vol = 0.22           # Target 22% vol 
        self.max_leverage = 2.0          # Max leverage
        self.base_leverage = 1.25        # Base leverage
        self.strong_trend_leverage = 1.6 # Leverage for strong trends (up from 1.5)
        self.vol_floor = 0.12            # Min vol assumption
        
        # Warmup
        self.set_warm_up(self.lookback_12m + 5, Resolution.DAILY)
        self.set_benchmark("MSFT")
        
        # Weekly rebalancing
        self.schedule.on(
            self.date_rules.every(DayOfWeek.MONDAY),
            self.time_rules.after_market_open("MSFT", 30),
            self.rebalance
        )
        
        self.last_price = None

    def on_data(self, data):
        if not data.contains_key("MSFT") or not data["MSFT"]:
            return
        price = data["MSFT"].close
        if price <= 0:
            return
        self.price_history.add(price)
        if self.last_price and self.last_price > 0:
            self.return_history.add((price / self.last_price) - 1)
        self.last_price = price

    def get_12m_momentum(self):
        if self.price_history.count <= self.lookback_12m:
            return None
        return (self.price_history[0] / self.price_history[self.lookback_12m]) - 1

    def get_realized_vol(self):
        if self.return_history.count < 20:
            return None
        returns = [self.return_history[i] for i in range(20)]
        return np.std(returns) * np.sqrt(252)

    def rebalance(self):
        if self.is_warming_up:
            return
        if not self.sma_200.is_ready or not self.sma_50.is_ready:
            return
        if not self.rsi_indicator.is_ready:
            return
        if self.price_history.count <= self.lookback_12m:
            return
            
        momentum = self.get_12m_momentum()
        realized_vol = self.get_realized_vol()
        
        if momentum is None or realized_vol is None:
            return
            
        current_price = self.price_history[0]
        sma_200_val = self.sma_200.current.value
        sma_50_val = self.sma_50.current.value
        rsi_val = self.rsi_indicator.current.value
        
        above_sma_200 = current_price > sma_200_val
        above_sma_50 = current_price > sma_50_val
        golden_cross = sma_50_val > sma_200_val
        
        target_position = 0.0
        reason = ""
        
        # EXIT CONDITIONS: Must have positive momentum AND be above 200 SMA
        if momentum <= 0 or not above_sma_200:
            target_position = 0.0
            reason = f"EXIT: mom={momentum:.1%}, above_200={above_sma_200}"
        
        # STRONG UPTREND: All signals aligned (above both SMAs + golden cross)
        elif above_sma_50 and golden_cross:
            vol_scalar = self.target_vol / max(realized_vol, self.vol_floor)
            position = self.strong_trend_leverage * vol_scalar
            position = max(1.0, min(self.max_leverage, position))
            
            # RSI adjustments
            if rsi_val < 35:
                position = min(self.max_leverage, position * 1.15)
                reason = f"STRONG + DIP: RSI={rsi_val:.0f}"
            elif rsi_val > 72:
                position = max(1.0, position * 0.92)
                reason = f"STRONG + OVERBOUGHT: RSI={rsi_val:.0f}"
            else:
                reason = f"STRONG TREND: {position:.2f}x"
            
            target_position = position
        
        # MODERATE UPTREND: Above 200 SMA, positive momentum, but not golden cross
        else:
            vol_scalar = self.target_vol / max(realized_vol, self.vol_floor)
            position = self.base_leverage * vol_scalar
            target_position = max(0.9, min(1.5, position))
            reason = f"MODERATE: {target_position:.2f}x"
        
        # Execute if position changes meaningfully
        current = self.portfolio["MSFT"].holdings_value / self.portfolio.total_portfolio_value if self.portfolio.total_portfolio_value > 0 else 0
        
        if abs(current - target_position) > 0.10:
            self.set_holdings("MSFT", target_position)
            self.log(f"{reason} | Target: {target_position:.2f}x | Vol: {realized_vol:.1%}")
