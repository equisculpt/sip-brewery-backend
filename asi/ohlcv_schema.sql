-- OHLCV table for all stocks (PostgreSQL/TimescaleDB compatible)
CREATE TABLE IF NOT EXISTS stock_ohlcv (
    symbol VARCHAR(16),
    date DATE,
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_stock_ohlcv_date ON stock_ohlcv(date);
