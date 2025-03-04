import pandas as pd
from sqlmodel import Session, select
from stock_data_fetcher import StockPrice, get_db_engine

class Rule:
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def evaluate(self, stock_data):
        """Evaluate the rule on the stock data"""
        raise NotImplementedError("Subclasses must implement evaluate()")

class OpeningPriceHigherThanPreviousClosingRule(Rule):
    def __init__(self, days=10):
        desc = f"Buy if current day opening price is more than previous {days} days closing price average"
        super().__init__("Opening > Previous Closings", desc)
        self.days = days
    
    def evaluate(self, stock_data):
        """
        Returns True (buy signal) if current day opening price is higher than 
        the average closing price of the previous n days
        """
        if len(stock_data) < self.days + 1:
            return 0 # Not enough data
        
        # Get the most recent data point
        current_day = stock_data.iloc[-1]
        
        # Get the previous n days
        previous_days = stock_data.iloc[-(self.days+1):-1]
        
        # Calculate average closing price of previous days
        avg_closing = previous_days['close'].mean()
        
        # Check if current day opening price is higher than the average closing
        return 1 if current_day['open'] > avg_closing else 0

class MovingAverageCrossoverRule(Rule):
    def __init__(self, short_window=5, long_window=20):
        desc = f"Buy when {short_window}-day MA crosses above {long_window}-day MA"
        super().__init__(f"MA Crossover {short_window}/{long_window}", desc)
        self.short_window = short_window
        self.long_window = long_window
    
    def evaluate(self, stock_data):
        """
        Returns True (buy signal) if short-term moving average crosses above 
        the long-term moving average
        """
        if len(stock_data) < self.long_window + 1:
            return 0 # Not enough data
            
        # Calculate moving averages
        stock_data['short_ma'] = stock_data['close'].rolling(window=self.short_window).mean()
        stock_data['long_ma'] = stock_data['close'].rolling(window=self.long_window).mean()
        
        # Check if a crossover occurred in the last two days
        current = stock_data.iloc[-1]
        previous = stock_data.iloc[-2]
        
        # Current short MA above long MA and previous short MA below or equal to long MA
        if (current['short_ma'] > current['long_ma']) and (previous['short_ma'] <= previous['long_ma']):
            return 1
        # Short MA cross below Long MA => sell
        elif (current['short_ma'] < current['long_ma']) and (previous['short_ma'] >= previous['long_ma']):
            return -1
        else:
            return 0

class RSIOverboughtOversoldRule(Rule):
    def __init__(self, period=14, oversold=30, overbought=70):
        desc = f"Buy when RSI({period}) < {oversold}, Sell when RSI({period}) > {overbought}"
        super().__init__("RSI Overbought/Oversold", desc)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, data):
        # Calculate price changes
        delta = data['close'].diff()
        
        # Get gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculate relative strength (RS)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def evaluate(self, stock_data):
        """
        Returns:
        1 for buy signal (oversold)
        -1 for sell signal (overbought)
        0 for no signal
        """
        if len(stock_data) < self.period + 1:
            return 0
            
        # Calculate RSI
        stock_data['rsi'] = self.calculate_rsi(stock_data)
        
        current_rsi = stock_data.iloc[-1]['rsi']
        
        if pd.isna(current_rsi):
            return 0
            
        if current_rsi < self.oversold:
            return 1  # Buy signal
        elif current_rsi > self.overbought:
            return -1  # Sell signal
        else:
            return 0  # No signal

class MACDRule(Rule):
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        desc = f"Buy when MACD crosses above signal line, sell when crosses below"
        super().__init__(f"MACD ({fast_period}/{slow_period}/{signal_period})", desc)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def evaluate(self, stock_data):
        """
        Returns:
        1 for buy signal (MACD crosses above signal)
        -1 for sell signal (MACD crosses below signal)
        0 for no signal
        """
        if len(stock_data) < self.slow_period + self.signal_period:
            return 0
        
        # Calculate MACD
        exp1 = stock_data['close'].ewm(span=self.fast_period, adjust=False).mean()
        exp2 = stock_data['close'].ewm(span=self.slow_period, adjust=False).mean()
        macd = exp1 - exp2
        
        # Calculate Signal Line
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()
        
        # Get current and previous values
        curr_macd = macd.iloc[0]
        prev_macd = macd.iloc[1]
        curr_signal = signal.iloc[0]
        prev_signal = signal.iloc[1]
        
        # Check for crossover
        if curr_macd > curr_signal and prev_macd <= prev_signal:
            return 1  # Buy signal
        elif curr_macd < curr_signal and prev_macd >= prev_signal:
            return -1  # Sell signal
        else:
            return 0  # No signal

class BollingerBandsRule(Rule):
    def __init__(self, period=20, std_dev=2):
        desc = f"Buy when price crosses below lower band, sell when crosses above upper band"
        super().__init__(f"Bollinger Bands ({period}, {std_dev}σ)", desc)
        self.period = period
        self.std_dev = std_dev
    
    def evaluate(self, stock_data):
        """
        Returns:
        1 for buy signal (price crosses below lower band)
        -1 for sell signal (price crosses above upper band)
        0 for no signal
        """
        if len(stock_data) < self.period + 1:
            return 0
        
        # Calculate Bollinger Bands
        df = stock_data.copy()
        df['sma'] = df['close'].rolling(window=self.period).mean()
        df['std'] = df['close'].rolling(window=self.period).std()
        df['upper'] = df['sma'] + (df['std'] * self.std_dev)
        df['lower'] = df['sma'] - (df['std'] * self.std_dev)
        
        # Current and previous day
        curr = df.iloc[0]
        prev = df.iloc[1]
        
        # Check for crossing lower band (buy signal)
        if prev['close'] <= prev['lower'] and curr['close'] > curr['lower']:
            return 1  # Buy signal
        
        # Check for crossing upper band (sell signal)
        elif prev['close'] >= prev['upper'] and curr['close'] < curr['upper']:
            return -1  # Sell signal
        
        # No crossing
        else:
            return 0  # No signal


class VolumeBreakoutRule(Rule):
    def __init__(self, volume_factor=2, price_change_pct=1.0):
        desc = f"Buy when volume is {volume_factor}x average with {price_change_pct}% price increase"
        super().__init__(f"Volume Breakout", desc)
        self.volume_factor = volume_factor
        self.price_change_pct = price_change_pct
    
    def evaluate(self, stock_data):
        """
        Returns:
        1 for buy signal (volume spike with price increase)
        0 for no signal
        """
        if len(stock_data) < 11:
            return 0
        
        # Current day
        current_day = stock_data.iloc[0]
        
        # Calculate average volume (excluding current day)
        avg_volume = stock_data.iloc[1:11]['volume'].mean()
        
        # Calculate price change percentage
        prev_close = stock_data.iloc[1]['close']
        current_close = current_day['close']
        price_change_pct = ((current_close - prev_close) / prev_close) * 100
        
        # Check for volume breakout with price increase
        if (current_day['volume'] > self.volume_factor * avg_volume and 
            price_change_pct >= self.price_change_pct):
            return 1  # Buy signal
        else:
            return 0  # No signal


class SupplyDemandZoneRule(Rule):
    def __init__(self, lookback=20, zone_strength=3, zone_depth_pct=1.0):
        desc = f"Identify supply/demand zones and signal when price approaches them"
        super().__init__(f"Supply Demand Zones", desc)
        self.lookback = lookback
        self.zone_strength = zone_strength
        self.zone_depth_pct = zone_depth_pct
    
    def find_swing_highs_lows(self, data, strength=3):
        """Find swing highs and lows in the data"""
        highs = []
        lows = []
        
        for i in range(strength, len(data) - strength):
            # Check for swing high
            if all(data.iloc[i]['high'] > data.iloc[i-j]['high'] for j in range(1, strength+1)) and \
               all(data.iloc[i]['high'] > data.iloc[i+j]['high'] for j in range(1, strength+1)):
                highs.append((i, data.iloc[i]['high']))
            
            # Check for swing low
            if all(data.iloc[i]['low'] < data.iloc[i-j]['low'] for j in range(1, strength+1)) and \
               all(data.iloc[i]['low'] < data.iloc[i+j]['low'] for j in range(1, strength+1)):
                lows.append((i, data.iloc[i]['low']))
        
        return highs, lows
    
    def evaluate(self, stock_data):
        """
        Returns:
        1 for buy signal (price approaching demand zone)
        -1 for sell signal (price approaching supply zone)
        0 for no signal
        """
        if len(stock_data) < self.lookback + self.zone_strength:
            return 0
        
        # Get recent data for analysis
        analysis_data = stock_data.iloc[:self.lookback]
        
        # Find swing highs and lows
        swing_highs, swing_lows = self.find_swing_highs_lows(analysis_data, self.zone_strength)
        
        if not swing_highs or not swing_lows:
            return 0
        
        current_price = stock_data.iloc[0]['close']
        
        # Find nearest supply zone (swing high)
        nearest_supply = None
        min_distance_supply = float('inf')
        
        for _, price in swing_highs:
            if current_price < price:  # Price below supply zone
                distance = price - current_price
                if distance < min_distance_supply:
                    min_distance_supply = distance
                    nearest_supply = price
        
        # Find nearest demand zone (swing low)
        nearest_demand = None
        min_distance_demand = float('inf')
        
        for _, price in swing_lows:
            if current_price > price:  # Price above demand zone
                distance = current_price - price
                if distance < min_distance_demand:
                    min_distance_demand = distance
                    nearest_demand = price
        
        # Check if price is approaching a zone
        if nearest_supply:
            distance_pct = (nearest_supply - current_price) / current_price * 100
            if distance_pct <= self.zone_depth_pct:
                return -1  # Approaching supply zone (sell)
        
        if nearest_demand:
            distance_pct = (current_price - nearest_demand) / current_price * 100
            if distance_pct <= self.zone_depth_pct:
                return 1  # Approaching demand zone (buy)
        
        return 0  # No signal


class FibonacciRetracementRule(Rule):
    def __init__(self, trend_length=20, retracement_level=0.618):
        desc = f"Buy when price retraces to {retracement_level} Fibonacci level in uptrend"
        super().__init__(f"Fibonacci Retracement", desc)
        self.trend_length = trend_length
        self.retracement_level = retracement_level
    
    def evaluate(self, stock_data):
        """
        Returns:
        1 for buy signal (price at fibonacci retracement level in uptrend)
        0 for no signal
        """
        if len(stock_data) < self.trend_length + 1:
            return 0
        
        # Determine if we're in an uptrend by comparing current close to past closes
        close_prices = stock_data['close'].iloc[:self.trend_length]
        current_close = close_prices.iloc[0]
        
        # Get high and low of the trend
        trend_high = close_prices.max()
        trend_low = close_prices.min()
        
        # Calculate Fibonacci levels
        fib_range = trend_high - trend_low
        fib_level = trend_high - (fib_range * self.retracement_level)
        
        # Check if we're in an uptrend (current close > 50% of closes in our window)
        uptrend = (close_prices > close_prices.median()).sum() >= self.trend_length / 2
        
        # Check if price is near our Fibonacci level
        near_fib = abs(current_close - fib_level) / fib_level <= 0.01  # Within 1% of fib level
        
        if uptrend and near_fib:
            return 1  # Buy signal
        else:
            return 0  # No signal
        
class StockRulesEngine:
    def __init__(self, db_session):
        self.db_session = db_session
        self.rules = []
        
    def add_rule(self, rule):
        """Add a rule to the engine"""
        self.rules.append(rule)
        
    def get_stock_data(self, symbol, days=30):
        """Get stock data for a specific symbol"""
        query = (
            select(StockPrice)
            .where(StockPrice.symbol == symbol)
            .order_by(StockPrice.stock_date.desc())
            .limit(days)
        )
        stock_prices = self.db_session.exec(query).all()

        if not stock_prices:
            return pd.DataFrame()
        
        data = []
        for row in stock_prices:
            data.append({
                'symbol': row.symbol,
                'date': row.stock_date,
                'open': row.open,
                'high': row.high,
                'low': row.low,
                'close': row.close,
                'volume': row.volume
            })

        df = pd.DataFrame(data)
        stock_data = df.iloc[::-1].reset_index(drop=True) # Reverse the dataframe to ascending order
        
        return stock_data # return reversed dataframe
        
    def evaluate_rules(self, symbol, days_of_data=30):
        """Evaluate all rules for a given stock"""
        stock_data = self.get_stock_data(symbol, days_of_data)
        
        if stock_data.empty:
            return {
            "symbol": symbol,
            "date": None,
            "current_price": None,
            "error": "No data available",
            "signals": []
        }
            
        results = []
        for rule in self.rules:
            try:
                signal = rule.evaluate(stock_data)
                results.append({
                    "rule_name": rule.name,
                    "description": rule.description,
                    "signal": signal
                })
            except Exception as e:
                results.append({
                    "rule_name": rule.name,
                    "description": rule.description,
                    "error": str(e)
                })
        
        return {
            "symbol": symbol,
            "date": stock_data.iloc[-1]['date'].isoformat() if not stock_data.empty else None,
            "current_price": stock_data.iloc[-1]['close'] if not stock_data.empty else None,
            "signals": results
        }
    
    def get_recommendations(self, symbols):
        """Get recommendations for a list of stocks"""
        recommendations = []
        
        for symbol in symbols:
            result = self.evaluate_rules(symbol)
            buy_signals = sum(1 for signal in result["signals"] if signal.get("signal") == 1)
            sell_signals = sum(1 for signal in result["signals"] if signal.get("signal") == -1)

            strength = buy_signals - sell_signals
            recommendation = "STRONG BUY" if strength >= 2 else \
                            "BUY" if strength == 1 else \
                            "STRONG SELL" if strength <= -2 else \
                            "SELL" if strength == -1 else "HOLD"
            
            recommendations.append({
                "symbol": symbol,
                "price": result["current_price"],
                "date": result["date"],
                "recommendation": recommendation,
                "signals": result["signals"],
                "strength": strength
            })
            
        return recommendations

# Example usage
if __name__ == "__main__":
    engine = get_db_engine()
    with Session(engine) as session:
        rules_engine = StockRulesEngine(db_session=session)
        
        # Add some rules
        rules_engine.add_rule(OpeningPriceHigherThanPreviousClosingRule(days=10))
        rules_engine.add_rule(MovingAverageCrossoverRule(short_window=5, long_window=20))
        rules_engine.add_rule(RSIOverboughtOversoldRule())
        rules_engine.add_rule(BollingerBandsRule()) 
        rules_engine.add_rule(VolumeBreakoutRule())
        rules_engine.add_rule(SupplyDemandZoneRule())
        rules_engine.add_rule(FibonacciRetracementRule())

        # Get recommendations for stocks
        stocks = ["AAPL", "MSFT", "GOOG", "AMZN"]
        recommendations = rules_engine.get_recommendations(stocks)
        
        for rec in recommendations:
            print(
                f"{rec['symbol']}: {rec['recommendation']} "
                f"(Price: {rec['price']}) "
                f"(Signals: {rec['signals']}) "
                f"(Strength: {rec['strength']})"
            )