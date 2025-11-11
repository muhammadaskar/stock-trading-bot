# stock-trading-bot

This is a simple stock trading bot implemented in Python. The bot uses historical stock data to make buy and sell decisions based on a moving average strategy.
## Features
- Fetches historical stock data using the yfinance library.
- Implements a moving average crossover strategy for trading decisions.
- Simulates buy and sell actions based on the strategy.
- Logs trades and calculates profit/loss.
## Requirements
- Python 3.6+
- yfinance
- pandas
- numpy
- requests
- python-dotenv
## Installation
1. Clone the repository:
   ```bash
   git clone git@github.com:muhammadaskar/stock-trading-bot.git
   cd stock-trading-bot
    ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Usage
1. Copy .env.example to .env and set your configuration:
   ```bash
   cp .env.example .env
   ```
2. Run the bot:
   ```bash
   python main.py
   ```
3. Monitor the console output for trade actions and profit/loss summary.
4. Customize the strategy parameters in `config.py` as needed.
5. Feel free to modify and enhance the bot to suit your trading strategies!
6. Happy trading!
7. ## Disclaimer
This bot is for educational purposes only. Trading stocks involves risk, and you should not trade with real money without proper knowledge and experience. Always do your own research before making any investment decisions.