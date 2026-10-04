# fall-check

Daily NIFTY 250 stock scanner that produces a research watchlist. It does not
place orders or make investment decisions.

## Requirements

- Python 3.10 or newer
- AWS credentials or an IAM role for the default Kite/S3/SSM integrations

Install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For the Yahoo Finance market-data provider, install `yfinance` as well:

```powershell
python -m pip install yfinance
```

## Run

The scanner reads the 250-stock universe from `nifty250.csv` by default:

```powershell
python .\fall-check-LuxAlgo-daily-final.py
```

The default market provider tries Zerodha Kite and falls back to Yahoo Finance.
To select a provider explicitly and use local watchlist state:

```powershell
python .\fall-check-LuxAlgo-daily-final.py --market-provider yahoo --state-backend local
```

Use `--help` to see all options. The command-line options and corresponding
environment variables allow overriding the universe, financial data, cache,
state backend, and notification settings.

## Configuration and generated files

Kite API credentials and the Telegram bot token are loaded from encrypted AWS
SSM parameters unless environment variables are supplied. AWS region is taken
from `AWS_REGION` or `AWS_DEFAULT_REGION`. The default state backend is
automatic: it tries S3 and falls back to a local watchlist state file.
Use `--state-backend local` for local state.

`nifty250.csv` must have `symbol,name` columns and contain exactly 250 stocks.
Optional local financial and third-party score inputs are JSON files. Financial
cache, daily market data, state, and report CSV files are generated at runtime
and are excluded from Git.

## Attribution

LuxAlgo Smart Money Concepts terminology and the attributed swing-structure
concepts are credited in the source file to LuxAlgo under CC BY-NC-SA 4.0.
This repository does not include the original LuxAlgo Pine source.
