import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
import warnings
import time

warnings.filterwarnings('ignore')


class IntradayTradingBot:
    """
    Intraday Trading Bot for Indonesian Stocks
    Strategy: Buy in the morning, Sell in the afternoon
    Focus: Quick profits from daily volatility
    """

    # High liquidity Indonesian stocks (good for day trading)
    STOCK_LIST = [
        'BBCA.JK',  # Bank BCA
        'BBRI.JK',  # Bank BRI
        'BMRI.JK',  # Bank Mandiri
        'TLKM.JK',  # Telkom
        'ASII.JK',  # Astra
        'GOTO.JK',  # GoTo (high volatility)
        'BBNI.JK',  # Bank BNI
        'AMMN.JK',  # Amman Mineral (volatile)
        'UNVR.JK',  # Unilever
        'BRIS.JK',  # Bank BRI Syariah
    ]

    # Trading hours IDX: 09:00 - 16:15 WIB
    MARKET_OPEN = dt_time(9, 0)
    MORNING_END = dt_time(10, 30)  # Ideal buying window
    AFTERNOON_START = dt_time(14, 30)  # Ideal selling window
    MARKET_CLOSE = dt_time(16, 15)

    # Minimum capital requirement (5 million IDR)
    MIN_CAPITAL = 5_000_000

    # Minimum trading volume (5 million shares for liquidity)
    MIN_VOLUME = 5_000_000

    def __init__(self, stocks=None, capital=10000000):  # Default 10 juta
        """Initialize intraday trading bot"""
        self.stocks = stocks if stocks else self.STOCK_LIST
        self.capital = capital
        self.results = []
        self.current_time = datetime.now().time()

        # Validate minimum capital
        if capital < self.MIN_CAPITAL:
            print(f"⚠️  Warning: Capital below recommended minimum (Rp {self.MIN_CAPITAL:,})")
            print(f"   Liquidity and position sizing may be limited.")

    def get_trading_phase(self):
        """Determine current trading phase"""
        now = self.current_time

        if now < self.MARKET_OPEN:
            return "PRE_MARKET"
        elif now <= self.MORNING_END:
            return "MORNING_SESSION"  # Best time to BUY
        elif now < self.AFTERNOON_START:
            return "MIDDAY_SESSION"
        elif now < self.MARKET_CLOSE:
            return "AFTERNOON_SESSION"  # Best time to SELL
        else:
            return "AFTER_MARKET"

    def fetch_intraday_data(self, ticker):
        """Fetch intraday data with 1-minute intervals"""
        try:
            stock = yf.Ticker(ticker)

            # Get today's 1-minute data
            intraday = stock.history(period='1d', interval='1m')

            # Get recent 5 days for reference
            daily = stock.history(period='5d', interval='1d')

            # Get current price
            if len(intraday) > 0:
                current_price = intraday['Close'].iloc[-1]
                open_price = intraday['Open'].iloc[0]
                today_volume = intraday['Volume'].sum()  # Total volume today
            else:
                current_price = daily['Close'].iloc[-1]
                open_price = daily['Open'].iloc[-1]
                today_volume = daily['Volume'].iloc[-1]

            # Calculate average daily volume (last 5 days)
            avg_volume = daily['Volume'].mean()

            info = stock.info

            return {
                'success': True,
                'ticker': ticker,
                'intraday': intraday,
                'daily': daily,
                'current_price': current_price,
                'open_price': open_price,
                'today_volume': today_volume,
                'avg_volume': avg_volume,
                'info': info
            }
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            return {'success': False, 'ticker': ticker}

    def calculate_intraday_indicators(self, intraday_data):
        """Calculate intraday technical indicators"""
        df = intraday_data.copy()

        if len(df) < 20:
            return None

        # Short-term EMAs for intraday
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_15'] = df['Close'].ewm(span=15, adjust=False).mean()
        df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()

        # Intraday RSI (faster period)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # VWAP (Volume Weighted Average Price) - important for intraday
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()

        # Intraday Momentum
        df['Momentum'] = df['Close'] - df['Close'].shift(10)

        # Volume analysis
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

        return df

    def intraday_analysis(self, data):
        """Analyze intraday trading opportunities"""
        intraday = data['intraday']
        current_price = data['current_price']
        open_price = data['open_price']

        df = self.calculate_intraday_indicators(intraday)

        if df is None or len(df) < 30:
            return None

        signals = []
        score = 0

        # Price movement from open
        price_change_pct = ((current_price - open_price) / open_price) * 100

        if price_change_pct > 0:
            signals.append(f"✓ Up {price_change_pct:.2f}% from open")
            score += 1
        else:
            signals.append(f"✗ Down {abs(price_change_pct):.2f}% from open")
            score -= 1

        # EMA Crossover (fast signal for intraday)
        ema_5 = df['EMA_5'].iloc[-1]
        ema_15 = df['EMA_15'].iloc[-1]
        ema_30 = df['EMA_30'].iloc[-1]

        if ema_5 > ema_15 > ema_30:
            signals.append("✓ Strong uptrend (EMA alignment)")
            score += 2
        elif ema_5 < ema_15 < ema_30:
            signals.append("✗ Strong downtrend (EMA alignment)")
            score -= 2
        elif ema_5 > ema_15:
            signals.append("✓ Short-term bullish")
            score += 1
        else:
            signals.append("✗ Short-term bearish")
            score -= 1

        # RSI (faster for intraday)
        rsi = df['RSI'].iloc[-1]
        if rsi < 35:
            signals.append(f"✓ RSI {rsi:.1f} (Oversold - Buy opportunity)")
            score += 2
        elif rsi > 65:
            signals.append(f"✗ RSI {rsi:.1f} (Overbought - Sell signal)")
            score -= 2
        else:
            signals.append(f"○ RSI {rsi:.1f} (Neutral)")

        # VWAP (critical for intraday)
        vwap = df['VWAP'].iloc[-1]
        if current_price > vwap:
            signals.append(f"✓ Price above VWAP (Institutional buying)")
            score += 2
        else:
            signals.append(f"✗ Price below VWAP (Weak momentum)")
            score -= 1

        # Volume analysis
        volume_ratio = df['Volume_Ratio'].iloc[-1]
        if volume_ratio > 1.5:
            signals.append(f"✓ High volume ({volume_ratio:.1f}x avg)")
            score += 1
        elif volume_ratio < 0.5:
            signals.append(f"⚠ Low volume ({volume_ratio:.1f}x avg)")
            score -= 1

        # Momentum
        momentum = df['Momentum'].iloc[-1]
        if momentum > 0:
            signals.append("✓ Positive momentum")
            score += 1
        else:
            signals.append("✗ Negative momentum")
            score -= 1

        return {
            'signals': signals,
            'score': score,
            'rsi': rsi,
            'vwap': vwap,
            'price_change_pct': price_change_pct,
            'volume_ratio': volume_ratio,
            'ema_5': ema_5,
            'ema_15': ema_15
        }

    def calculate_intraday_targets(self, current_price, open_price, analysis):
        """Calculate intraday target and stop loss"""
        # Average daily volatility for Indonesian stocks: 1-3%
        # Target: 1.5-2.5% profit
        # Stop loss: 0.5-1% loss

        # Dynamic target based on morning performance
        price_change = analysis['price_change_pct']

        if price_change > 0:
            # If already up, target smaller profit
            target_pct = 1.5
            stop_pct = 0.8
        else:
            # If down, look for bounce with higher target
            target_pct = 2.5
            stop_pct = 1.2

        target_price = current_price * (1 + target_pct / 100)
        stop_loss = current_price * (1 - stop_pct / 100)

        # Calculate potential profit
        shares = int(self.capital / current_price)
        potential_profit = (target_price - current_price) * shares
        potential_loss = (current_price - stop_loss) * shares

        return {
            'entry_price': current_price,
            'target_price': target_price,
            'target_pct': target_pct,
            'stop_loss': stop_loss,
            'stop_pct': -stop_pct,
            'shares': shares,
            'investment': shares * current_price,
            'potential_profit': potential_profit,
            'potential_loss': potential_loss,
            'risk_reward': potential_profit / potential_loss if potential_loss > 0 else 0
        }

    def generate_intraday_action(self, score, phase):
        """Generate trading action based on time and score"""
        if phase == "MORNING_SESSION":
            # Morning: Focus on buying
            if score >= 5:
                return "STRONG BUY", "🟢🟢", "Buy NOW - Strong momentum"
            elif score >= 2:
                return "BUY", "🟢", "Good entry point"
            elif score >= 0:
                return "WAIT", "🟡", "Wait for better signal"
            else:
                return "SKIP", "⚪", "Avoid - Weak setup"

        elif phase == "AFTERNOON_SESSION":
            # Afternoon: Focus on selling/taking profit
            if score >= 3:
                return "HOLD", "🟡", "Hold for higher target"
            elif score >= 0:
                return "TAKE PROFIT", "🟢", "Sell at profit"
            else:
                return "SELL NOW", "🔴", "Exit position - Weakness"

        else:
            # Midday or after hours
            if score >= 4:
                return "MONITOR", "👁️", "Watch for entry"
            else:
                return "NEUTRAL", "⚪", "Wait for trading hours"

    def calculate_confidence(self, score, volume_ratio, phase):
        """Calculate confidence for intraday trade"""
        base_confidence = 50

        # Score contribution
        base_confidence += score * 5

        # Volume boost
        if volume_ratio > 1.5:
            base_confidence += 15
        elif volume_ratio < 0.7:
            base_confidence -= 10

        # Time of day factor
        if phase == "MORNING_SESSION":
            base_confidence += 10  # Best time to enter
        elif phase == "AFTERNOON_SESSION":
            base_confidence += 5  # Good time to exit

        confidence = max(0, min(100, base_confidence))

        if confidence >= 75:
            level = "Very High"
        elif confidence >= 60:
            level = "High"
        elif confidence >= 45:
            level = "Medium"
        else:
            level = "Low"

        return int(confidence), level

    def analyze_stock(self, ticker, phase):
        """Analyze single stock for intraday trading"""
        data = self.fetch_intraday_data(ticker)

        if not data['success']:
            return None

        # Check volume requirement FIRST (most important for liquidity)
        if data['avg_volume'] < self.MIN_VOLUME:
            # Skip low volume stocks - hard to enter/exit
            return None

        analysis = self.intraday_analysis(data)

        if analysis is None:
            return None

        targets = self.calculate_intraday_targets(
            data['current_price'],
            data['open_price'],
            analysis
        )

        # Check if stock meets minimum capital requirement
        if targets['investment'] < self.MIN_CAPITAL:
            return None  # Skip stocks below 5M investment

        action, emoji, description = self.generate_intraday_action(
            analysis['score'],
            phase
        )

        confidence, conf_level = self.calculate_confidence(
            analysis['score'],
            analysis['volume_ratio'],
            phase
        )

        return {
            'ticker': ticker,
            'name': data['info'].get('longName', ticker),
            'current_price': data['current_price'],
            'open_price': data['open_price'],
            'today_volume': data['today_volume'],
            'avg_volume': data['avg_volume'],
            'action': action,
            'emoji': emoji,
            'description': description,
            'confidence': confidence,
            'conf_level': conf_level,
            'analysis': analysis,
            'targets': targets,
            'phase': phase,
            'timestamp': datetime.now()
        }

    def scan_all_stocks(self):
        """Scan all stocks for intraday opportunities"""
        phase = self.get_trading_phase()

        print("=" * 80)
        print("INTRADAY TRADING BOT - BUY MORNING, SELL AFTERNOON")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} WIB")
        print(f"Trading Phase: {phase}")
        print(f"Capital: Rp {self.capital:,.0f}")
        print(f"Scanning {len(self.stocks)} stocks...\n")

        self.results = []

        for ticker in self.stocks:
            print(f"⏳ Analyzing {ticker}...", end=" ")
            result = self.analyze_stock(ticker, phase)

            if result:
                self.results.append(result)
                print(f"✓ {result['action']}")
            else:
                print(f"✗ Failed")

            time.sleep(0.5)

        print(f"\n✅ Scan complete! {len(self.results)} stocks analyzed.\n")
        return self.results

    def display_results(self):
        """Display intraday trading opportunities"""
        if not self.results:
            print("No results to display.")
            return

        phase = self.get_trading_phase()
        sorted_results = sorted(self.results, key=lambda x: x['confidence'], reverse=True)

        print("=" * 80)
        if phase == "MORNING_SESSION":
            print("🌅 MORNING SESSION - TOP STOCKS TO BUY")
        elif phase == "AFTERNOON_SESSION":
            print("🌆 AFTERNOON SESSION - POSITION MANAGEMENT")
        else:
            print("INTRADAY OPPORTUNITIES")
        print("=" * 80)

        for i, r in enumerate(sorted_results, 1):
            # Only show actionable stocks
            if r['action'] in ['SKIP', 'NEUTRAL']:
                continue

            print(f"\n{i}. {r['ticker']} - {r['name'][:40]}")
            print("-" * 80)
            print(f"Action: {r['emoji']} {r['action']} - {r['description']}")
            print(f"Confidence: {r['confidence']}% ({r['conf_level']})")

            # Display volume info (important for liquidity)
            avg_vol_m = r['avg_volume'] / 1_000_000
            today_vol_m = r['today_volume'] / 1_000_000
            print(f"\n📊 LIQUIDITY:")
            print(f"   Avg Volume: {avg_vol_m:.2f}M shares/day")
            print(f"   Today Volume: {today_vol_m:.2f}M shares ✓ HIGH LIQUIDITY")

            print(f"\n💰 PRICES:")
            print(f"   Open: Rp {r['open_price']:,.0f}")
            print(f"   Current: Rp {r['current_price']:,.0f} ({r['analysis']['price_change_pct']:+.2f}%)")

            t = r['targets']
            print(f"\n🎯 INTRADAY TARGETS:")
            print(f"   Entry: Rp {t['entry_price']:,.0f}")
            print(f"   Target: Rp {t['target_price']:,.0f} (+{t['target_pct']:.1f}%)")
            print(f"   Stop Loss: Rp {t['stop_loss']:,.0f} ({t['stop_pct']:.1f}%)")
            print(f"   Risk/Reward: 1:{t['risk_reward']:.2f}")

            print(f"\n📊 POSITION SIZING:")
            print(f"   Shares: {t['shares']:,} lots")
            print(f"   Investment: Rp {t['investment']:,.0f}")
            print(f"   Potential Profit: Rp {t['potential_profit']:,.0f}")
            print(f"   Potential Loss: Rp {t['potential_loss']:,.0f}")

            print(f"\n📈 KEY SIGNALS:")
            for signal in r['analysis']['signals'][:4]:
                print(f"   {signal}")

        print("\n" + "=" * 80)

    def get_top_intraday_picks(self, limit=3):
        """Get top stocks for today's trading"""
        if not self.results:
            return []

        phase = self.get_trading_phase()

        if phase == "MORNING_SESSION":
            # Morning: prioritize BUY signals
            filtered = [r for r in self.results if 'BUY' in r['action']]
        elif phase == "AFTERNOON_SESSION":
            # Afternoon: show all positions
            filtered = [r for r in self.results if r['action'] != 'SKIP']
        else:
            filtered = self.results

        sorted_picks = sorted(filtered, key=lambda x: x['confidence'], reverse=True)
        return sorted_picks[:limit]

    def export_watchlist(self, filename='intraday_watchlist.csv'):
        """Export today's watchlist"""
        if not self.results:
            print("No results to export.")
            return

        data = []
        for r in self.results:
            if r['action'] not in ['SKIP', 'NEUTRAL']:
                data.append({
                    'Time': r['timestamp'].strftime('%H:%M:%S'),
                    'Ticker': r['ticker'],
                    'Action': r['action'],
                    'Current Price': r['current_price'],
                    'Target': r['targets']['target_price'],
                    'Stop Loss': r['targets']['stop_loss'],
                    'Shares': r['targets']['shares'],
                    'Investment': r['targets']['investment'],
                    'Potential Profit': r['targets']['potential_profit'],
                    'Confidence': r['confidence'],
                    'Avg Volume (M)': round(r['avg_volume'] / 1_000_000, 2),
                    'Today Volume (M)': round(r['today_volume'] / 1_000_000, 2),
                    'RSI': r['analysis']['rsi'],
                    'Change %': r['analysis']['price_change_pct']
                })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✅ Watchlist exported to {filename}")


# Main execution
if __name__ == "__main__":
    print("🚀 Starting Intraday Trading Bot...")

    # Set your capital (default 10 million IDR)
    TRADING_CAPITAL = 10_000_000

    # Initialize bot
    bot = IntradayTradingBot(capital=TRADING_CAPITAL)

    # Check trading phase
    phase = bot.get_trading_phase()
    print(f"\n⏰ Current phase: {phase}")

    if phase == "PRE_MARKET":
        print("📅 Market hasn't opened yet. Preparing watchlist...")
    elif phase == "AFTER_MARKET":
        print("🌙 Market closed. Review today's performance.")

    # Scan stocks
    results = bot.scan_all_stocks()

    # Display opportunities
    bot.display_results()

    # Show top picks
    print("\n" + "=" * 80)
    print("🏆 TOP INTRADAY PICKS")
    print("=" * 80)

    top_picks = bot.get_top_intraday_picks(3)

    for i, pick in enumerate(top_picks, 1):
        t = pick['targets']
        profit_pct = t['target_pct']

        print(f"\n{i}. {pick['ticker']} {pick['emoji']} {pick['action']}")
        print(f"   Entry: Rp {t['entry_price']:,.0f} → Target: Rp {t['target_price']:,.0f} (+{profit_pct:.1f}%)")
        print(f"   Investment: Rp {t['investment']:,.0f} → Profit: Rp {t['potential_profit']:,.0f}")
        print(f"   Confidence: {pick['confidence']}%")

    # Export watchlist
    print()
    bot.export_watchlist()

    print("\n" + "=" * 80)
    print("📋 INTRADAY TRADING RULES")
    print("=" * 80)
    print("✓ Best entry: 09:00 - 10:30 WIB")
    print("✓ Best exit: 14:30 - 16:00 WIB")
    print("✓ Always use stop loss (max 1% loss)")
    print("✓ Target: 1.5-2.5% profit per trade")
    print("✓ Don't hold overnight!")
    print("✓ Cut loss fast, let profit run")
    print("✓ MINIMUM VOLUME: 5M shares/day (for easy entry/exit)")
    print("✓ Focus on high liquidity stocks ONLY")
    print("✓ Check bid-ask spread before entering")
    print("=" * 80)

    print("\n⚠️  DISCLAIMER")
    print("This is for educational purposes only.")
    print("Day trading is risky. Only trade with money you can afford to lose.")
    print("=" * 80)