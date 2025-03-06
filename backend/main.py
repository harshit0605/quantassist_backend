# main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session  # <-- Use Session from SQLModel
from typing import List, Optional
import os
from pydantic import BaseModel
import uuid

# Update: Use from stock_data_fetcher which uses SQLModel
from stock_data_fetcher import StockDataFetcher, get_db_engine, init_db
from rule_engine import (
    StockRulesEngine,
    OpeningPriceHigherThanPreviousClosingRule,
    MovingAverageCrossoverRule,
    RSIOverboughtOversoldRule,
    BollingerBandsRule,
    FibonacciRetracementRule,
    MACDRule,
    VolumeBreakoutRule,
    SupplyDemandZoneRule

)

app = FastAPI(title="Stock Recommender API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

jobs_registry = {}

def create_job_id() -> str:
    return str(uuid.uuid4())

# Database setup
engine = get_db_engine()
init_db(engine)  # Create tables if they don't exist

def get_db():
    """
    Dependency that yields a new SQLModel Session each time.
    """
    with Session(engine) as session:
        yield session

# Pydantic models for request/response
class StockSymbol(BaseModel):
    symbol: str

class StockList(BaseModel):
    symbols: List[str]

class RuleConfig(BaseModel):
    rule_type: str
    parameters: dict

class StockData(BaseModel):
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class SignalResponse(BaseModel):
    rule_name: str
    description: str
    signal: Optional[int] = None
    error: Optional[str] = None

class Recommendation(BaseModel):
    symbol: str
    price: Optional[float] = None
    date: Optional[str] = None
    recommendation: str
    signals: List[SignalResponse]
    strength: int

# Initialize the rules engine dependency
def get_rules_engine(db: Session):
    engine = StockRulesEngine(db)
    # Add default rules
    engine.add_rule(OpeningPriceHigherThanPreviousClosingRule(days=10))
    engine.add_rule(MovingAverageCrossoverRule(short_window=5, long_window=20))
    engine.add_rule(RSIOverboughtOversoldRule())
    engine.add_rule(BollingerBandsRule()) 
    engine.add_rule(VolumeBreakoutRule())
    engine.add_rule(SupplyDemandZoneRule())
    engine.add_rule(FibonacciRetracementRule())
    engine.add_rule(MACDRule())
    return engine

@app.get("/")
def read_root():
    return {"message": "Welcome to Stock Recommender API"}

@app.post("/update-stocks")
def update_stocks(request: StockList, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """
    Update stock data for the given symbols in the background
    """
    job_id = create_job_id()
    jobs_registry[job_id] = {"status": "pending", "symbols": request.symbols, "results": {}}

    def fetch_data(symbols, job_id):
        fetcher = StockDataFetcher(db)
        jobs_registry[job_id]["status"] = "in_progress"
        results = fetcher.update_stocks(symbols)
        jobs_registry[job_id]["results"] = results
        jobs_registry[job_id]["status"] = "completed"

    background_tasks.add_task(fetch_data, request.symbols, job_id)
    return {
        "message": f"Updating {len(request.symbols)} stocks in the background",
        "job_id": job_id
    }

@app.get("/stock/{symbol}", response_model=List[StockData])
def get_stock_data(symbol: str, days: int = 30, db: Session = Depends(get_db)):
    """
    Get historical stock data for a symbol
    """
    fetcher = StockDataFetcher(db)
    df = fetcher.get_stock_data(symbol, days)

    if df.empty:
        # No data found in DB -> attempt to update
        update_result = fetcher.update_stocks([symbol])
        # Re-fetch after update attempt
        df = fetcher.get_stock_data(symbol, days)

        # If still no data, raise 404
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
    data_list = df.to_dict(orient="records")
    return data_list

@app.get("/recommendations", response_model=List[Recommendation])
def get_recommendations(symbols: str, days: int = 30, db: Session = Depends(get_db)):
    """
    Get recommendations for a comma-separated list of stock symbols
    """
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    if not symbol_list:
        raise HTTPException(status_code=400, detail="No symbols provided")

    engine = get_rules_engine(db)
    recommendations = engine.get_recommendations(symbol_list, days)
    return recommendations

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """
    Return status of the background job.
    - status: pending, in_progress, completed, etc.
    - symbols: list of symbols being updated
    - results: dictionary of symbol -> update result (True, False, or error)
    """
    job = jobs_registry.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    
    return {
        "job_id": job_id,
        "status": job["status"],
        "symbols": job["symbols"],
        "results": job["results"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)