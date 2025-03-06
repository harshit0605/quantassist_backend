import json
import logging
from datetime import date
import os
import pandas as pd
import requests
import datetime
from contextlib import contextmanager
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict

from sqlmodel import (
    Field,
    Session,
    SQLModel,
    create_engine,
    select,
    Relationship
)  

from get_api_key import get_api_key
load_dotenv()
# Configure logging at the top of your module
logging.basicConfig(
    level=logging.DEBUG,  # Set DEBUG to capture all levels of logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

_db_engine = None

class StockPrice(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, max_length=10)
    stock_date: date = Field(index=True)
    open: float
    high: float
    low: float
    close: float
    volume: int

class Rule(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100, unique=True)
    description: str = Field(max_length=500)
    rule_type: str = Field(max_length=50)  # e.g., "RSI", "MACD", "MovingAverage"
    parameters: str = Field(max_length=1000)  # JSON string of parameters
    is_active: bool = Field(default=True)
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    updated_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    signals: List["Signal"] = Relationship(back_populates="rule")

class Signal(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    rule_id: int = Field(foreign_key="rule.id")
    symbol: str = Field(max_length=10, index=True)
    date: datetime.date = Field(index=True)
    signal_value: int  # 1 for buy, -1 for sell, 0 for hold
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)

    rule: Optional[Rule] = Relationship(back_populates="signals")

def get_db_connection():
    """Get database connection string from environment or use provided URL"""
    db_user = os.getenv('PGUSER')
    db_password = os.getenv('PGPASSWORD')
    db_host = os.getenv('PGHOST')
    db_name = os.getenv('PGDATABASE')
    db_url = f"postgresql://{db_user}:{db_password}@{db_host}/{db_name}?sslmode=require"
    
    return db_url

def get_db_engine():
    """
    Returns the global engine instance. If none exists, it creates one using SQLModel's create_engine.
    """
    connection_string = get_db_connection()
    global _db_engine
    if _db_engine is None:
        logger.debug(f"Creating new DB engine with URL: {connection_string}")
        _db_engine = create_engine(connection_string)
    return _db_engine

def init_db(engine):
    """
    Creates the database schema using SQLModel.
    Instead of Base.metadata.create_all, we call SQLModel.metadata.create_all.
    """
    logger.debug("Initializing database schema (SQLModel).")
    SQLModel.metadata.create_all(engine)
    return Session(engine)

@contextmanager
def db_engine_context():
    """
    Context manager for the DB engine. Disposes of the engine after usage.
    """
    global _db_engine
    try:
        logger.debug("Entering DB engine context.")
        engine = get_db_engine()
        yield engine
    finally:
        if _db_engine is not None:
            logger.debug("Disposing DB engine and resetting global reference.")
            _db_engine.dispose()
            _db_engine = None

class StockDataFetcher:
    def __init__(self, db_session: Session, cache_dir: str = "stock_cache"):
        logger.debug("Initializing StockDataFetcher.")
        self.api_key = get_api_key("ALPHA_VANTAGE_API_KEY")
        self.base_url = "https://www.alphavantage.co/query"
        self.db_session = db_session
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_path(self, symbol: str, full: bool = True):
        """Generate the cache file path for a given symbol and data size."""
        size = "full" if full else "compact"
        cache_path = os.path.join(self.cache_dir, f"{symbol}_{size}.json")
        logger.debug(f"Cache path for {symbol} ({size}): {cache_path}")
        return cache_path

    def _read_from_cache(self, symbol: str, full: bool = True):
        """Read stock data from cache if available."""
        cache_path = self._get_cache_path(symbol, full)
        logger.debug(f"Attempting to read cache from: {cache_path}")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache_data = json.load(f)
                
                # Check if cache is from today
                cache_date = cache_data.get('cache_date')
                if cache_date and cache_date == datetime.date.today().isoformat():
                    logger.info(f"Using cached data for {symbol}")
                    return cache_data.get('data')
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Error reading cache for {symbol}: {e}")
        
        logger.debug(f"No valid cache found for {symbol}.")
        return None

    def _write_to_cache(self, symbol: str, data: dict, full: bool = True):
        """Write stock data to cache."""
        cache_path = self._get_cache_path(symbol, full)
        logger.debug(f"Writing data to cache: {cache_path}")
        
        try:
            cache_data = {
                'cache_date': datetime.date.today().isoformat(),
                'data': data
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)

            logger.info(f"Cached data for {symbol} successfully.")
            return True
        except IOError as e:
            logger.warning(f"Error writing cache for {symbol}: {e}")
            return False

    def fetch_daily_data(self, symbol: str, full: bool = True):
        """
        Fetch daily stock data for a given symbol using Alpha Vantage API, returning a DataFrame.
        """
        logger.debug(f"Fetching daily data for symbol: {symbol}, full={full}")
        
        latest_entry = self.db_session.exec(
            select(StockPrice)
            .where(StockPrice.symbol == symbol)
            .order_by(StockPrice.stock_date.desc())
        ).first()

        if latest_entry and latest_entry.stock_date == datetime.date.today():
            logger.info(f"Data for {symbol} is already up-to-date.")
            return "up_to_date"
        
        # Try to get data from cache first
        cached_data = self._read_from_cache(symbol, full)
        
        if cached_data and "Time Series (Daily)" in cached_data:
            logger.debug(f"Using cached API response for {symbol}")
            data = cached_data
        else:
            output_size = "full" if full else "compact"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": output_size,
                "apikey": self.api_key
            }
            logger.info(f"Making API request for {symbol} with params: {params}")
            response = requests.get(self.base_url, params=params)
            data = response.json()
            if "Time Series (Daily)" in data:
                self._write_to_cache(symbol, data, full)

        if "Time Series (Daily)" not in data:
            logger.error(f"Error fetching data for {symbol}: {data.get('Information', 'Unknown error')}")
            return None

        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df = df.rename(
            columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            }
        )

        # Convert data types
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        df['volume'] = df['volume'].astype(int)

        df['symbol'] = symbol
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={'index': 'date'})
        logger.debug(f"Fetched {len(df)} rows for {symbol}")
        return df

    def save_to_db(self, df: pd.DataFrame):
        """
        Save the incoming DataFrame to the database via SQLModel.
        """
        if df is None or df.empty:
            logger.debug("No data to save; DataFrame is empty or None.")
            return False

        logger.debug(f"Saving {len(df)} entries to the database.")
        for _, row in df.iterrows():
            existing = self.db_session.exec(
                select(StockPrice).where(
                    StockPrice.symbol == row["symbol"],
                    StockPrice.stock_date == row["date"]
                )
            ).first()

            if not existing:
                stock_price = StockPrice(
                    symbol=row['symbol'],
                    stock_date=row['date'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                self.db_session.add(stock_price)

        self.db_session.commit()
        logger.info("Data saved to database successfully.")
        return True

    def update_stocks(self, symbols):
        """
        Update stock data for a list of symbols. By default, fetches recent data (compact).
        """
        logger.debug(f"Updating stocks for symbols: {symbols}")
        results = {}
        for symbol in symbols:
            logger.debug(f"Fetching data for {symbol}...")
            response = self.fetch_daily_data(symbol, full=False)

            if isinstance(response, str) and response == "up_to_date":
                logger.info(f"Data for {symbol} was already up to date.")
                results[symbol] = True
            elif response is None:
                logger.warning(f"Failed to fetch data for {symbol}.")
                results[symbol] = False
            else:
                success = self.save_to_db(response)
                results[symbol] = success
        return results

    def get_stock_data(self, symbol: str, days: int = 30):
        """
        Get up to `days` days of stock data for `symbol` from the database,
        returning a DataFrame (or empty DataFrame if none found).
        """
        logger.debug(f"Retrieving stock data for {symbol} over the past {days} days.")
        cutoff_date = datetime.datetime.now().date() - datetime.timedelta(days=days)
        logger.debug(f"Cutoff date: {cutoff_date}")
        query = (
            select(StockPrice)
            .where(StockPrice.symbol == symbol)
            .where(StockPrice.stock_date >= cutoff_date)
            .order_by(StockPrice.stock_date.asc())
        )
        stock_prices = self.db_session.exec(query).all()

        if not stock_prices:
            logger.debug(f"Retrieved 0 rows from DB for {symbol}.")
            return pd.DataFrame()

        logger.debug(f"Retrieved {len(stock_prices)} rows from DB for {symbol}.")

        data = []
        for row in stock_prices:
            data.append({
                'symbol': row.symbol,
                'date': row.stock_date.isoformat(),
                'open': row.open,
                'high': row.high,
                'low': row.low,
                'close': row.close,
                'volume': row.volume
            })

        df = pd.DataFrame(data)
        df.sort_values(by='date', inplace=True, ascending=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def get_all_stock_data(self, symbols, days=30):
        """
        Retrieves stock data for all requested symbols over the last N days.
        Returns a dictionary keyed by symbol, with values as DataFrames.
        """

        cutoff_date = datetime.date.today() - datetime.timedelta(days=days)

        # Fetch everything for the symbols in the date range
        query = (
            select(StockPrice)
            .where(StockPrice.symbol.in_(symbols))
            .where(StockPrice.stock_date >= cutoff_date)
            .order_by(StockPrice.symbol, StockPrice.stock_date.asc())
        )
        results = self.db_session.exec(query).all()

        # Group results by symbol
        symbol_data_map = defaultdict(list)
        for row in results:
            symbol_data_map[row.symbol].append({
                'symbol': row.symbol,
                'date': row.stock_date,
                'open': row.open,
                'high': row.high,
                'low': row.low,
                'close': row.close,
                'volume': row.volume
            })

        # Convert each list to a DataFrame, sorted ascending by date
        for symbol in symbol_data_map:
            df = pd.DataFrame(symbol_data_map[symbol]).sort_values(by='date')
            df.reset_index(drop=True, inplace=True)
            symbol_data_map[symbol] = df

        return symbol_data_map
    
if __name__ == "__main__":
    API_KEY = get_api_key("ALPHA_VANTAGE_API_KEY")  # Replace with your API key

    with db_engine_context() as engine:
        session = init_db(engine)
        fetcher = StockDataFetcher(session)

        # Example usage: uncomment to update data for selected stocks
        stocks_list = ["AAPL", "MSFT", "GOOG", "AMZN"]
        fetcher.update_stocks(stocks_list)

        # Or fetch from DB
        results = fetcher.get_stock_data("AAPL")
        logger.info(f"Data from DB: {results}")