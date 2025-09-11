-- Create schemas and tables for the project
CREATE SCHEMA IF NOT EXISTS market;
CREATE SCHEMA IF NOT EXISTS backtests;
CREATE SCHEMA IF NOT EXISTS models;

-- Market OHLCV data
CREATE TABLE IF NOT EXISTS market.ohlcv (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  open NUMERIC(18,6),
  high NUMERIC(18,6),
  low NUMERIC(18,6),
  close NUMERIC(18,6),
  volume NUMERIC(20,2),
  timeframe TEXT NOT NULL DEFAULT 'daily',
  source TEXT NOT NULL,
  UNIQUE(symbol, ts, timeframe)
);

-- Feature store
CREATE TABLE IF NOT EXISTS market.features (
  id BIGSERIAL PRIMARY KEY,
  symbol TEXT NOT NULL,
  ts TIMESTAMPTZ NOT NULL,
  features JSONB NOT NULL,
  label NUMERIC(18,6),
  timeframe TEXT NOT NULL DEFAULT 'daily',
  source TEXT NOT NULL,
  UNIQUE(symbol, ts, timeframe)
);

-- Backtest runs
CREATE TABLE IF NOT EXISTS backtests.runs (
  id UUID PRIMARY KEY,
  strategy TEXT NOT NULL,
  params JSONB NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  notes TEXT
);

-- Backtest results by day
CREATE TABLE IF NOT EXISTS backtests.daily_metrics (
  id BIGSERIAL PRIMARY KEY,
  run_id UUID REFERENCES backtests.runs(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  equity NUMERIC(20,4) NOT NULL,
  pnl NUMERIC(20,4) NOT NULL,
  drawdown NUMERIC(10,4) NOT NULL
);

-- Trade logs
CREATE TABLE IF NOT EXISTS backtests.trades (
  id BIGSERIAL PRIMARY KEY,
  run_id UUID REFERENCES backtests.runs(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  side TEXT CHECK (side IN ('BUY','SELL')),
  qty NUMERIC(18,6) NOT NULL,
  entry_ts TIMESTAMPTZ NOT NULL,
  entry_px NUMERIC(18,6) NOT NULL,
  exit_ts TIMESTAMPTZ,
  exit_px NUMERIC(18,6),
  pnl NUMERIC(18,6)
);

-- Model registry (supplement to MLflow)
CREATE TABLE IF NOT EXISTS models.registry (
  id UUID PRIMARY KEY,
  name TEXT NOT NULL,
  version TEXT NOT NULL,
  params JSONB NOT NULL,
  metrics JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

