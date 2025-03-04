# db_config.py
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, DateTime, Boolean, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.schema import UniqueConstraint
from datetime import datetime

# Define base class for SQLAlchemy models
Base = declarative_base()

# Models definition
class StockPrice(Base):
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), index=True)
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Add unique constraint to prevent duplicate data
    __table_args__ = (UniqueConstraint('symbol', 'date', name='uix_symbol_date'),)
    
    def __repr__(self):
        return f"<StockPrice(symbol='{self.symbol}', date='{self.date}', close='{self.close}')>"


class Rule(Base):
    __tablename__ = 'rules'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True)
    description = Column(String(500))
    rule_type = Column(String(50))  # e.g., "RSI", "MACD", "MovingAverage"
    parameters = Column(String(1000))  # JSON string of parameters
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    signals = relationship("Signal", back_populates="rule")
    
    def __repr__(self):
        return f"<Rule(name='{self.name}', type='{self.rule_type}')>"


class Signal(Base):
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    rule_id = Column(Integer, ForeignKey('rules.id'))
    symbol = Column(String(10), index=True)
    date = Column(Date, index=True)
    signal_value = Column(Integer)  # 1 for buy, -1 for sell, 0 for hold
    created_at = Column(DateTime, default=datetime.utcnow)
    
    rule = relationship("Rule", back_populates="signals")
    
    __table_args__ = (UniqueConstraint('rule_id', 'symbol', 'date', name='uix_rule_symbol_date'),)
    
    def __repr__(self):
        return f"<Signal(symbol='{self.symbol}', rule='{self.rule.name if self.rule else None}', value={self.signal_value})>"


class WatchList(Base):
    __tablename__ = 'watchlists'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    symbols = Column(String(1000))  # Comma-separated list of symbols
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<WatchList(name='{self.name}')>"


# Database connection functions
def get_db_connection(db_url=None):
    """Get database connection string from environment or use provided URL"""
    if db_url is None:
        # Get connection parameters from environment
        db_user = os.environ.get("DB_USER", "postgres")
        db_password = os.environ.get("DB_PASSWORD", "postgres")
        db_host = os.environ.get("DB_HOST", "localhost")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "stockdb")
        
        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    return db_url


def get_db_engine(db_url=None):
    """Create SQLAlchemy engine with the database connection"""
    connection_string = get_db_connection(db_url)
    return create_engine(connection_string, pool_size=10, max_overflow=20)


def init_db(engine):
    """Initialize database tables"""
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


# Migration functions
def migrate_from_sqlite(sqlite_path, postgres_url):
    """Migrate data from SQLite to PostgreSQL"""
    from sqlalchemy import create_engine as ce
    from sqlalchemy.orm import sessionmaker as sm
    import pandas as pd
    
    # Create connections
    sqlite_engine = ce(f"sqlite:///{sqlite_path}")
    sqlite_session = sm(bind=sqlite_engine)()
    
    postgres_engine = ce(postgres_url)
    init_db(postgres_engine)  # Create tables if they don't exist
    postgres_session = sm(bind=postgres_engine)()
    
    try:
        # Create tables in PostgreSQL
        Base.metadata.create_all(postgres_engine)
        
        # Migrate stock price data
        print("Migrating stock price data...")
        
        # We'll use chunking to handle large datasets
        chunk_size = 5000
        offset = 0
        
        while True:
            # Get a chunk of records from SQLite
            query = sqlite_session.execute(f"""
                SELECT symbol, date, open, high, low, close, volume
                FROM stock_prices
                LIMIT {chunk_size} OFFSET {offset}
            """)
            
            records = query.fetchall()
            if not records:
                break
                
            # Insert the records into PostgreSQL
            for record in records:
                symbol, date, open_price, high, low, close, volume = record
                
                # Check if record already exists
                existing = postgres_session.query(StockPrice).filter_by(
                    symbol=symbol,
                    date=date
                ).first()
                
                if not existing:
                    stock_price = StockPrice(
                        symbol=symbol,
                        date=date,
                        open=open_price,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume
                    )
                    postgres_session.add(stock_price)
            
            postgres_session.commit()
            print(f"Migrated {len(records)} records from offset {offset}")
            
            offset += chunk_size
        
        print("Migration completed successfully!")
    except Exception as e:
        postgres_session.rollback()
        print(f"Error during migration: {str(e)}")
    finally:
        sqlite_session.close()
        postgres_session.close()


if __name__ == "__main__":
    # Example for running migration
    sqlite_path = "stocks.db"
    postgres_url = get_db_connection()
    
    print(f"Migrating from SQLite ({sqlite_path}) to PostgreSQL ({postgres_url})")
    migrate_from_sqlite(sqlite_path, postgres_url)