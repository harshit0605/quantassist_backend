# scheduled_update.py
import os
import time
import schedule
import logging
from datetime import datetime
import pytz
from sqlmodel import Session
from stock_data_fetcher import StockDataFetcher, get_db_engine, init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_updates.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stock_updater")

# Get API key from environment
API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")
if not API_KEY:
    logger.error("ALPHA_VANTAGE_API_KEY environment variable not set!")
    raise ValueError("API key not found. Please set ALPHA_VANTAGE_API_KEY environment variable.")

# Database setup
engine = get_db_engine("postgresql://username:password@localhost:5432/stockdb")
Session = Session(engine)

# List of stock symbols to track
STOCK_SYMBOLS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "TSLA", "NVDA", "JPM", 
    "JNJ", "V", "PG", "UNH", "HD", "BAC", "XOM", "DIS", "ADBE",
    "NFLX", "CSCO", "INTC", "VZ", "CMCSA", "PEP", "KO", "AVGO"
]

def is_market_open():
    """Check if US stock market is currently open"""
    # Set timezone to US/Eastern
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    # Market is open on weekdays 9:30 AM - 4:00 PM ET
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    
    # Check market hours (9:30 AM - 4:00 PM)
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    
    return market_open <= now <= market_close

def update_stocks():
    """Update stock data for all tracked symbols"""
    logger.info(f"Starting scheduled stock data update for {len(STOCK_SYMBOLS)} symbols")
    
    # Skip updates when market is closed
    if not is_market_open():
        logger.info("Market is closed. Skipping update.")
        return
    
    session = Session()
    try:
        fetcher = StockDataFetcher(API_KEY, session)
        
        # Process in smaller batches to avoid API limits
        batch_size = 5
        for i in range(0, len(STOCK_SYMBOLS), batch_size):
            batch = STOCK_SYMBOLS[i:i+batch_size]
            logger.info(f"Processing batch: {batch}")
            
            results = fetcher.update_stocks(batch)
            
            # Log results
            success_count = sum(1 for result in results.values() if result)
            logger.info(f"Batch update completed: {success_count}/{len(batch)} successful")
            
            # Sleep between batches to respect API rate limits
            if i + batch_size < len(STOCK_SYMBOLS):
                logger.info("Sleeping between batches...")
                time.sleep(15)  # Wait 15 seconds between batches
                
        logger.info("Stock data update completed successfully")
    except Exception as e:
        logger.error(f"Error updating stock data: {str(e)}")
    finally:
        session.close()

def schedule_jobs():
    """Set up the scheduling for data updates"""
    # Update during market hours on weekdays
    schedule.every(30).minutes.do(update_stocks)
    
    # Also run once at market open and market close
    schedule.every().monday.at("09:35").do(update_stocks)
    schedule.every().monday.at("15:55").do(update_stocks)
    schedule.every().tuesday.at("09:35").do(update_stocks)
    schedule.every().tuesday.at("15:55").do(update_stocks)
    schedule.every().wednesday.at("09:35").do(update_stocks)
    schedule.every().wednesday.at("15:55").do(update_stocks)
    schedule.every().thursday.at("09:35").do(update_stocks)
    schedule.every().thursday.at("15:55").do(update_stocks)
    schedule.every().friday.at("09:35").do(update_stocks)
    schedule.every().friday.at("15:55").do(update_stocks)
    
    logger.info("Scheduled jobs have been set up")
    
    # Run the scheduler
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    # Run once at startup
    update_stocks()
    
    # Then schedule regular updates
    schedule_jobs()