# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PandaFactor is a high-performance quantitative factor calculation and analysis system for financial data analysis, technical indicator computation, and factor construction in Chinese stock markets. The system supports multiple data sources (Tushare, RiceQuant, XTQuant, TQSDK) and provides a complete solution for factor generation, backtesting, and visualization.

## Architecture

This is a multi-module Python monorepo with the following structure:

- **panda_common**: Shared utilities, configuration management, database handlers, and common models
- **panda_data**: Data access layer for retrieving market data and factors from MongoDB
- **panda_data_hub**: Automated data update service with scheduled tasks for syncing market data from various providers
- **panda_factor**: Core factor calculation engine with analysis workflows and backtesting capabilities
- **panda_factor_server**: FastAPI REST API server exposing factor calculation and analysis endpoints
- **panda_llm**: LLM integration service (supports OpenAI-compatible APIs like Deepseek)
- **panda_web**: Frontend web interface

Each module has its own `setup.py` and is designed to be installed as a separate package with `pip install -e .`

## Development Setup

### Initial Setup (VSCode/Cursor)

Install each module in editable mode in the correct dependency order:

```bash
cd panda_common && pip install -e .
cd ../panda_data && pip install -e .
cd ../panda_data_hub && pip install -e .
cd ../panda_factor && pip install -e .
cd ../panda_llm && pip install -e .
cd ../panda_factor_server && pip install -e .
```

### PyCharm Setup

Mark each module directory as "Sources Root":
- Right-click on `panda_common`, `panda_data`, `panda_data_hub`, `panda_factor`, `panda_llm`, `panda_factor_server`
- Select "Mark Directory as" → "Sources Root"

### Configuration

The main configuration file is `panda_common/panda_common/config.yaml`. Key settings:

- **Database**: MongoDB connection (supports single, replica_set, or sharded modes)
- **Data Sources**: Configure tokens/credentials for Tushare, RiceQuant, XTQuant, TQSDK
- **Scheduling**: Data update times (default: 20:00 for stocks, 20:30 for factors)
- **LLM**: API key and endpoint for DeepSeek or OpenAI-compatible services

## Common Development Tasks

### Running Services

**Start Factor Server (Main API)**:
```bash
python -m panda_factor_server.server
# Runs on http://localhost:8001
```

**Start Data Hub (Auto-update Service)**:
```bash
python -m panda_data_hub
```

**Start LLM Service**:
```bash
python -m panda_llm
```

### Testing

Run tests with pytest:
```bash
pytest
```

### Using the Data API

```python
import panda_data

# Initialize with config
panda_data.init()

# Get market data
market_data = panda_data.get_market_data(
    start_date='20240101',
    end_date='20241231',
    symbols=['000001.SZ', '600000.SH']  # Optional
)

# Get factor data by name
factor_data = panda_data.get_factor_by_name(
    factor_name="VH03cc651",
    start_date='20240320',
    end_date='20250325'
)
```

## Factor Development

### Factor Structure

Factors must inherit from `Factor` base class and implement the `calculate()` method. The method receives a dictionary of base factors (close, open, high, low, volume, etc.) and must return a pandas Series with MultiIndex `['symbol', 'date']`.

**Python Mode (Recommended)**:

```python
from panda_factor.generate.factor_base import Factor

class CustomFactor(Factor):
    def calculate(self, factors):
        close = factors['close']
        volume = factors['volume']

        # Calculate 20-day return
        returns = (close / self.DELAY(close, 20)) - 1

        # Calculate volatility
        volatility = self.STDDEV(returns, 20)

        # Combine into final factor
        result = self.RANK(returns) * volatility

        return result  # Must be Series with MultiIndex ['symbol', 'date']
```

**Formula Mode**:

For users without programming background, factors can be expressed as string formulas:

```python
"RANK((CLOSE / DELAY(CLOSE, 20)) - 1) * STDDEV(VOLUME, 20)"
```

### Available Factor Methods

The `Factor` base class provides these built-in operators (accessible via `self.` in Python mode):

- `RANK(series)`: Cross-sectional ranking normalized to [-0.5, 0.5]
- `DELAY(series, period)`: Lag values by N periods
- `SUM(series, window)`: Rolling sum
- `STDDEV(series, window)`: Rolling standard deviation
- `CORRELATION(series1, series2, window)`: Rolling correlation
- `IF(condition, true_value, false_value)`: Conditional selection
- `RETURNS(close)`: Calculate returns from close prices

Additional methods are available via `FactorUtils` class, which is automatically imported.

### Factor Analysis Workflow

The `factor` class in `panda_factor/analysis/factor.py` provides comprehensive backtesting:

```python
from panda_factor.analysis.factor import factor

# Create factor analysis instance
f = factor(name="my_factor", group_number=10)

# Set backtest parameters
f.set_backtest_parameters(
    period=1,              # Holding period in days
    predict_direction=0,   # 0: lower is better, 1: higher is better
    commission=0.002,      # Commission + slippage (0.2%)
    mode=1                 # 1: full backtest, 0: simple
)

# Run backtest (requires prepared DataFrame with factor values)
f.start_backtest(df, df_benchmark_pct)

# Generate analysis outputs
f.draw_pct()          # Plot group returns
f.draw_ic()           # Plot IC time series
f.draw_ic_dacay()     # Plot IC decay
f.show_df_info(0)     # Display metrics
```

## Database Schema Conventions

- **Database**: `panda` (default)
- **Collections**:
  - `stock_market`: Daily stock OHLCV data
  - `user_factors`: User-defined factor metadata
  - `factor_analysis_results`: Backtest results and charts
  - `tasks`: Background task status
- **Indexes**: All collections have compound indexes on `['symbol', 'date']`
- **Date Format**: Stored as integers (YYYYMMDD format, e.g., 20240101)

## Code Style and Patterns

### Error Handling

Use Go-style explicit error handling where possible. Return tuples with error information rather than raising exceptions in data processing pipelines:

```python
# Prefer explicit error returns
def process_data(data):
    if data is None:
        return None, "Data is None"
    # ... processing
    return result, None
```

### Data Processing

- Use pandas MultiIndex with levels `['symbol', 'date']` for all factor/market data
- Always sort by date within symbol groups before applying time-series operations
- Handle NaN values explicitly (don't silently drop unless documented)

### Logging

Use `loguru` for logging throughout the codebase:

```python
from loguru import logger

logger.debug("Debug message")
logger.info("Info message")
logger.error("Error message")
```

## Key Architectural Patterns

### Module Dependencies

The dependency flow is: `panda_common` → `panda_data` → `panda_factor` → `panda_factor_server`

- `panda_common` has no internal dependencies
- `panda_data` depends only on `panda_common`
- `panda_factor` depends on `panda_common` (note: circular dependency with `panda_data` is removed)
- `panda_factor_server` depends on all modules

### Data Update Pipeline

The `panda_data_hub` module orchestrates scheduled data updates:

1. **Stock Instruments Update** (20:00 daily): Download stock list from data source
2. **Market Data Update** (20:00 daily): Fetch OHLCV data for all stocks
3. **Factor Update** (20:30 daily): Recalculate and persist factor values

Each data source (Tushare, RiceQuant, XTQuant, TQSDK) has dedicated cleaner classes implementing the update logic.

### Factor Calculation Flow

1. User submits factor code (Python class or formula string) via API
2. `panda_factor_server` validates and stores factor metadata in database
3. Factor code is dynamically loaded and instantiated
4. Base market data is fetched via `panda_data`
5. Factor's `calculate()` method is executed
6. Results are analyzed using `factor` class for backtesting
7. Analysis results (IC, returns, charts) are serialized to MongoDB

## Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -f Dockerfile -t panda-factor .

# Run container
docker run -p 8001:8001 panda-factor

# Or use the server-specific Dockerfile
docker build -f Dockerfile.server -t panda-factor-server .
```

## Important Notes

- MongoDB is required; the system expects either a local instance or replica set
- Initial database setup requires downloading the provided database dump (see README.md)
- Data sources require valid API credentials configured in `config.yaml`
- The system uses Chinese stock codes (e.g., 000001.SZ, 600000.SH)
- All date parameters should be in YYYYMMDD integer format (e.g., 20240101)
