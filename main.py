import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
import warnings
import time
import requests
from typing import List, Dict
import sys
from dotenv import load_dotenv
import json

warnings.filterwarnings('ignore')

load_dotenv()


def get_required_env(name: str) -> str:
    """Return environment variable or exit with error (no fallback)."""
    val = os.getenv(name)
    if val is None or not val.strip():
        sys.exit(f"Environment variable {name} is required but not set")
    return val


class AIAnalyzer:
    """Multi-provider AI Analyzer for trading insights"""

    def __init__(self, provider: str = "deepseek", api_key: str = None):
        """
        Initialize AI Analyzer with multiple provider support

        Args:
            provider: 'deepseek', 'groq', 'ollama', or 'gemini'
            api_key: API key for the provider (not needed for ollama)
        """
        self.provider = provider.lower()
        self.api_key = api_key

        # Provider configurations
        self.configs = {
            'deepseek': {
                'url': 'https://api.deepseek.com/v1/chat/completions',
                'model': 'deepseek-chat',
                'requires_key': True
            },
            'groq': {
                'url': 'https://api.groq.com/openai/v1/chat/completions',
                'model': 'llama-3.3-70b-versatile',  # Free tier available
                'requires_key': True
            },
            'gemini': {
                'url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
                'model': 'gemini-pro',
                'requires_key': True
            },
            'ollama': {
                'url': 'http://localhost:11434/api/chat',
                'model': 'llama3.2',  # Local, completely free
                'requires_key': False
            }
        }

        if self.provider not in self.configs:
            raise ValueError(f"Unsupported provider: {provider}")

        config = self.configs[self.provider]

        if config['requires_key'] and not api_key:
            raise ValueError(f"{provider} requires an API key")

        self.base_url = config['url']
        self.model = config['model']

        if config['requires_key']:
            if self.provider == 'gemini':
                self.headers = {"Content-Type": "application/json"}
                self.base_url = f"{self.base_url}?key={api_key}"
            else:
                self.headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
        else:
            self.headers = {"Content-Type": "application/json"}

        print(f"✅ AI Analyzer initialized: {self.provider.upper()}")
        if api_key and config['requires_key']:
            print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")

        # Test connection
        if self.provider != 'ollama':  # Skip test for local ollama
            self._test_connection()

    def _test_connection(self):
        """Test API connection"""
        try:
            print("   Testing API connection...", end=" ")

            if self.provider == 'gemini':
                payload = {
                    "contents": [{"parts": [{"text": "Hi"}]}]
                }
            else:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10
                }

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                print("✅ Connected!")
            elif response.status_code == 401:
                print("❌ Authentication Failed!")
                raise ValueError("Invalid API key")
            elif response.status_code == 402:
                print("❌ Insufficient Balance!")
                print(f"   {self.provider.upper()} account needs credits/balance")
                raise ValueError(f"Please top up your {self.provider} account")
            else:
                print(f"⚠️  Status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")

    def analyze_stock(self, stock_data: Dict) -> Dict:
        """
        Analyze stock using AI

        Args:
            stock_data: Dictionary containing stock information

        Returns:
            Dictionary with AI analysis results
        """
        try:
            prompt = self._create_analysis_prompt(stock_data)

            # Prepare payload based on provider
            if self.provider == 'gemini':
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": f"You are an expert intraday stock trader. {prompt}"
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 800
                    }
                }
            elif self.provider == 'ollama':
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert intraday stock trader specializing in Indonesian stocks (IDX). Provide concise, actionable trading insights."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False
                }
            else:  # deepseek, groq, openai-compatible
                payload = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an expert intraday stock trader specializing in Indonesian stocks (IDX). Provide concise, actionable trading insights based on technical analysis and market data."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800
                }

            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()

                # Extract response based on provider
                if self.provider == 'gemini':
                    ai_response = result['candidates'][0]['content']['parts'][0]['text']
                elif self.provider == 'ollama':
                    ai_response = result['message']['content']
                else:
                    ai_response = result['choices'][0]['message']['content']

                analysis = self._parse_ai_response(ai_response, stock_data)
                return analysis
            else:
                error_detail = response.text
                print(f"⚠️  {self.provider.upper()} API error: {response.status_code}")
                print(f"   Detail: {error_detail[:200]}")
                return self._fallback_analysis(stock_data)

        except Exception as e:
            print(f"❌ {self.provider.upper()} analysis error: {e}")
            return self._fallback_analysis(stock_data)

    def _create_analysis_prompt(self, data: Dict) -> str:
        """Create detailed prompt for AI analysis"""

        prompt = f"""Analyze this Indonesian stock for INTRADAY trading (buy morning, sell afternoon):

STOCK: {data['ticker']} - {data['name']}
TRADING PHASE: {data['phase']}

PRICE DATA:
- Open: Rp {data['open_price']:,.0f}
- Current: Rp {data['current_price']:,.0f} ({data['analysis']['price_change_pct']:+.2f}%)
- Bid/Ask: Rp {data['bid']:,.0f} / Rp {data['ask']:,.0f}
- Spread: {data['spread_pct']:.2f}%

TECHNICAL INDICATORS:
- RSI: {data['analysis']['rsi']:.1f}
- EMA 5/15: {data['analysis']['ema_5']:.0f} / {data['analysis']['ema_15']:.0f}
- VWAP: {data['analysis']['vwap']:.0f}
- Volume Ratio: {data['analysis']['volume_ratio']:.2f}x

SIGNALS:
{chr(10).join(f"- {s}" for s in data['analysis']['signals'][:5])}

TARGETS:
- Entry: Rp {data['targets']['entry_price']:,.0f}
- Target: Rp {data['targets']['target_price']:,.0f} (+{data['targets']['target_pct']:.1f}%)
- Stop Loss: Rp {data['targets']['stop_loss']:,.0f} ({data['targets']['stop_pct']:.1f}%)

Provide:
1. RECOMMENDATION: (BUY/HOLD/SELL/WAIT) - one word
2. CONFIDENCE: (0-100) - number only
3. KEY_INSIGHT: 1-2 sentences on why
4. RISK_FACTORS: 1-2 main risks
5. TRADE_PLAN: Specific entry/exit strategy

Format as JSON."""

        return prompt

    def _parse_ai_response(self, ai_response: str, stock_data: Dict) -> Dict:
        """Parse AI response and extract insights"""
        try:
            # Try to parse as JSON first
            if '{' in ai_response and '}' in ai_response:
                json_start = ai_response.find('{')
                json_end = ai_response.rfind('}') + 1
                json_str = ai_response[json_start:json_end]
                ai_data = json.loads(json_str)
            else:
                # Fallback: parse text response
                ai_data = self._parse_text_response(ai_response)

            return {
                'ai_recommendation': ai_data.get('RECOMMENDATION', stock_data['action']),
                'ai_confidence': int(ai_data.get('CONFIDENCE', stock_data['confidence'])),
                'ai_insight': ai_data.get('KEY_INSIGHT', 'Analysis based on technical indicators'),
                'ai_risks': ai_data.get('RISK_FACTORS', 'Monitor stop loss levels'),
                'ai_trade_plan': ai_data.get('TRADE_PLAN', 'Follow intraday strategy'),
                'ai_raw_response': ai_response,
                'ai_analysis_success': True
            }

        except Exception as e:
            print(f"⚠️  Error parsing AI response: {e}")
            return {
                'ai_recommendation': stock_data['action'],
                'ai_confidence': stock_data['confidence'],
                'ai_insight': 'Technical analysis based on indicators',
                'ai_risks': 'Standard intraday risks apply',
                'ai_trade_plan': ai_response[:200] if ai_response else 'Follow standard strategy',
                'ai_raw_response': ai_response,
                'ai_analysis_success': False
            }

    def _parse_text_response(self, text: str) -> Dict:
        """Parse non-JSON text response"""
        data = {}

        # Extract recommendation
        for rec in ['BUY', 'SELL', 'HOLD', 'WAIT']:
            if rec in text.upper():
                data['RECOMMENDATION'] = rec
                break

        # Extract confidence (look for percentages or scores)
        import re
        confidence_match = re.search(r'(\d{1,3})%|\bconfidence[:\s]+(\d{1,3})', text, re.IGNORECASE)
        if confidence_match:
            data['CONFIDENCE'] = confidence_match.group(1) or confidence_match.group(2)

        # Extract insights (first few sentences)
        sentences = text.split('.')[:3]
        data['KEY_INSIGHT'] = '.'.join(sentences).strip()

        return data

    def _fallback_analysis(self, stock_data: Dict) -> Dict:
        """Fallback analysis if AI fails"""
        return {
            'ai_recommendation': stock_data['action'],
            'ai_confidence': stock_data['confidence'],
            'ai_insight': 'Analysis based on technical indicators and market data',
            'ai_risks': 'Monitor volume and stop loss levels closely',
            'ai_trade_plan': f"Entry at {stock_data['current_price']:,.0f}, target {stock_data['targets']['target_price']:,.0f}",
            'ai_raw_response': 'Fallback analysis',
            'ai_analysis_success': False
        }

    def batch_analyze(self, stocks_data: List[Dict], max_stocks: int = 5) -> List[Dict]:
        """Analyze multiple stocks (limited to avoid API rate limits)"""
        results = []

        sorted_stocks = sorted(stocks_data, key=lambda x: x['confidence'], reverse=True)
        top_stocks = sorted_stocks[:max_stocks]

        print(f"\n🤖 Running {self.provider.upper()} AI analysis on top {len(top_stocks)} stocks...")

        for i, stock in enumerate(top_stocks, 1):
            print(f"   [{i}/{len(top_stocks)}] Analyzing {stock['ticker']}...", end=" ")

            ai_analysis = self.analyze_stock(stock)
            stock.update(ai_analysis)
            results.append(stock)

            if ai_analysis['ai_analysis_success']:
                print("✓")
            else:
                print("⚠️")

            # Rate limiting
            time.sleep(1 if self.provider == 'ollama' else 2)

        print("✅ AI analysis complete!\n")
        return results


class TelegramNotifier:
    """Handle Telegram notifications for multiple chats"""

    def __init__(self, bot_token: str, chat_ids: list):
        self.bot_token = bot_token
        self.chat_ids = [chat_id for chat_id in chat_ids if chat_id and str(chat_id).strip()]
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        print(f"✅ Telegram configured for {len(self.chat_ids)} chats")

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """Send message to all registered chats"""
        if not self.chat_ids:
            print("⚠️  No valid chat IDs configured")
            return False

        success_count = 0
        for chat_id in self.chat_ids:
            try:
                if not chat_id or not str(chat_id).strip():
                    continue

                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": parse_mode
                }
                response = requests.post(url, json=payload, timeout=10)
                if response.status_code == 200:
                    success_count += 1
                    print(f"✅ Message sent to {chat_id}")
                else:
                    error_msg = response.json().get('description', 'Unknown error')
                    print(f"❌ Failed to send to {chat_id}: {error_msg}")
            except Exception as e:
                print(f"❌ Telegram error for {chat_id}: {e}")

        return success_count > 0

    def format_stock_message(self, stock: Dict) -> str:
        """Format single stock info with AI insights"""
        emoji_map = {
            'STRONG BUY': '🟢🟢',
            'BUY': '🟢',
            'HOLD': '🟡',
            'TAKE PROFIT': '💰',
            'SELL NOW': '🔴',
            'WAIT': '⏸',
            'SKIP': '⚪',
            'MONITOR': '👁️',
            'NEUTRAL': '⚪'
        }

        emoji = emoji_map.get(stock['action'], '📊')
        t = stock['targets']
        a = stock['analysis']

        avg_vol_m = stock['avg_volume'] / 1_000_000
        today_vol_m = stock['today_volume'] / 1_000_000

        message = f"""
{emoji} <b>{stock['ticker']}</b> - {stock['name'][:30]}
{'=' * 40}

<b>🎯 ACTION: {stock['action']}</b>
{stock['description']}

<b>💰 PRICES</b>
• Open: Rp {stock['open_price']:,.0f}
• Current: Rp {stock['current_price']:,.0f} (<b>{a['price_change_pct']:+.2f}%</b>)

<b>💹 BID / ASK</b>
• Bid: Rp {stock['bid']:,.0f} ({stock['bid_size']:,} lot)
• Ask: Rp {stock['ask']:,.0f} ({stock['ask_size']:,} lot)
• Spread: {stock['spread_pct']:.2f}%

<b>🎯 INTRADAY TARGETS</b>
• Entry: Rp {t['entry_price']:,.0f}
• Target: Rp {t['target_price']:,.0f} (<b>+{t['target_pct']:.1f}%</b>)
• Stop Loss: Rp {t['stop_loss']:,.0f} ({t['stop_pct']:.1f}%)
• R/R Ratio: 1:{t['risk_reward']:.2f}

<b>📊 POSITION</b>
• Shares: {t['shares']:,} shares
• Capital: Rp {t['investment']:,.0f}
• Potential Profit: Rp {t['potential_profit']:,.0f}
• Potential Loss: Rp {t['potential_loss']:,.0f}

<b>📈 INDICATORS</b>
• RSI: {a['rsi']:.1f}
• Volume: {today_vol_m:.1f}M (Avg: {avg_vol_m:.1f}M)
• Confidence: <b>{stock['confidence']}%</b> ({stock['conf_level']})
"""

        # Add AI insights if available
        if stock.get('ai_analysis_success'):
            message += f"""
<b>🤖 AI INSIGHTS</b>
• Recommendation: <b>{stock['ai_recommendation']}</b>
• AI Confidence: {stock['ai_confidence']}%
• Analysis: {stock['ai_insight'][:100]}
• Risks: {stock['ai_risks'][:100]}
"""

        message += f"""
<b>🔍 TOP SIGNALS</b>
"""
        for signal in a['signals'][:3]:
            clean_signal = signal.replace('✓', '').replace('✗', '').replace('○', '').replace('⚠', '').strip()
            message += f"• {clean_signal}\n"

        message += f"\n⏰ {datetime.now().strftime('%H:%M:%S')} WIB"

        return message

    def format_summary_message(self, results: List[Dict], phase: str, capital: float) -> str:
        """Format summary message"""
        buy_stocks = [r for r in results if 'BUY' in r['action']]
        sell_stocks = [r for r in results if 'SELL' in r['action'] or 'PROFIT' in r['action']]

        phase_emoji = {
            'MORNING_SESSION': '🌅',
            'AFTERNOON_SESSION': '🌆',
            'MIDDAY_SESSION': '☀️',
            'PRE_MARKET': '🌄',
            'AFTER_MARKET': '🌙'
        }

        message = f"""
{phase_emoji.get(phase, '📊')} <b>INTRADAY TRADING SUMMARY</b>
{'=' * 40}

⏰ <b>Time:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} WIB
📍 <b>Phase:</b> {phase.replace('_', ' ')}
💰 <b>Capital:</b> Rp {capital:,.0f}
🤖 <b>AI-Powered Analysis</b>

<b>📊 SCAN RESULTS</b>
• Total stocks: {len(results)}
• Buy signals: {len(buy_stocks)}
• Sell signals: {len(sell_stocks)}

<b>🏆 TOP 3 AI PICKS</b>
"""

        sorted_results = sorted(results, key=lambda x: x.get('ai_confidence', x['confidence']), reverse=True)
        for i, stock in enumerate(sorted_results[:3], 1):
            if stock['action'] not in ['SKIP', 'NEUTRAL']:
                t = stock['targets']
                ai_conf = stock.get('ai_confidence', stock['confidence'])
                message += f"{i}. {stock['ticker']} {stock['emoji']} +{t['target_pct']:.1f}% (AI: {ai_conf}%)\n"

        message += f"""
{'=' * 40}
📱 Individual stock details sent separately
"""
        return message


class IntradayTradingBot:
    """Intraday Trading Bot with DeepSeek AI Integration"""

    STOCK_LIST = [
        'BBCA.JK', 'BBRI.JK', 'BMRI.JK', 'TLKM.JK', 'ASII.JK',
        'GOTO.JK', 'BBNI.JK', 'AMMN.JK', 'UNVR.JK', 'BRIS.JK', 'BUMI.JK',
    ]

    MARKET_OPEN = dt_time(9, 0)
    MORNING_END = dt_time(10, 30)
    AFTERNOON_START = dt_time(14, 30)
    MARKET_CLOSE = dt_time(16, 0)
    HOLD_AFTER_TIME = dt_time(15, 40)

    MIN_CAPITAL = 5_000_000
    MIN_VOLUME = 5_000_000

    def __init__(self, stocks=None, capital=10000000, telegram_bot_token=None,
                 telegram_chat_id=None, ai_provider="deepseek", ai_api_key=None):
        """Initialize bot with AI integration"""
        self.stocks = stocks if stocks else self.STOCK_LIST
        self.capital = capital
        self.results = []
        self.current_time = datetime.now().time()

        # Setup AI Analyzer (supports multiple providers)
        self.ai_analyzer = None
        if ai_api_key or ai_provider == 'ollama':
            try:
                self.ai_analyzer = AIAnalyzer(provider=ai_provider, api_key=ai_api_key)
                print(f"✅ {ai_provider.upper()} AI enabled")
            except Exception as e:
                print(f"⚠️  AI disabled: {e}")
        else:
            print("⚠️  AI disabled (no API key)")

        # Setup Telegram
        self.telegram = None
        if telegram_bot_token:
            TELEGRAM_CHAT_IDS = [
                "1731964139",
                "-5042882073",
            ]
            self.telegram = TelegramNotifier(telegram_bot_token, TELEGRAM_CHAT_IDS)
        else:
            print("⚠️  Telegram notifications disabled")

        if capital < self.MIN_CAPITAL:
            print(f"⚠️  Warning: Capital below recommended minimum (Rp {self.MIN_CAPITAL:,})")

    def get_trading_phase(self):
        """Determine current trading phase"""
        now = self.current_time

        if now < self.MARKET_OPEN:
            return "PRE_MARKET"
        elif now <= self.MORNING_END:
            return "MORNING_SESSION"
        elif now < self.AFTERNOON_START:
            return "MIDDAY_SESSION"
        elif now < self.MARKET_CLOSE:
            return "AFTERNOON_SESSION"
        else:
            return "AFTER_MARKET"

    def fetch_intraday_data(self, ticker):
        """Fetch intraday data with 1-minute intervals + bid/ask info"""
        try:
            stock = yf.Ticker(ticker)
            intraday = stock.history(period='1d', interval='1m')
            daily = stock.history(period='5d', interval='1d')

            if len(intraday) > 0:
                current_price = intraday['Close'].iloc[-1]
                open_price = intraday['Open'].iloc[0]
                today_volume = intraday['Volume'].sum()
            else:
                current_price = daily['Close'].iloc[-1]
                open_price = daily['Open'].iloc[-1]
                today_volume = daily['Volume'].iloc[-1]

            avg_volume = daily['Volume'].mean()
            info = stock.info

            bid = info.get("bid", np.nan)
            ask = info.get("ask", np.nan)
            bid_size = info.get("bidSize", np.nan)
            ask_size = info.get("askSize", np.nan)

            if not np.isnan(bid) and not np.isnan(ask) and ask > 0:
                spread_pct = ((ask - bid) / ask) * 100
            else:
                spread_pct = np.nan

            return {
                'success': True,
                'ticker': ticker,
                'intraday': intraday,
                'daily': daily,
                'current_price': current_price,
                'open_price': open_price,
                'today_volume': today_volume,
                'avg_volume': avg_volume,
                'info': info,
                'bid': bid,
                'ask': ask,
                'bid_size': bid_size,
                'ask_size': ask_size,
                'spread_pct': spread_pct
            }

        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            return {'success': False, 'ticker': ticker}

    def calculate_intraday_indicators(self, intraday_data):
        """Calculate intraday technical indicators"""
        df = intraday_data.copy()

        if len(df) < 20:
            return None

        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_15'] = df['Close'].ewm(span=15, adjust=False).mean()
        df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()

        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=9).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=9).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['Momentum'] = df['Close'] - df['Close'].shift(10)
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

        price_change_pct = ((current_price - open_price) / open_price) * 100

        if price_change_pct > 0:
            signals.append(f"✓ Up {price_change_pct:.2f}% from open")
            score += 1
        else:
            signals.append(f"✗ Down {abs(price_change_pct):.2f}% from open")
            score -= 1

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

        rsi = df['RSI'].iloc[-1]
        if rsi < 35:
            signals.append(f"✓ RSI {rsi:.1f} (Oversold - Buy opportunity)")
            score += 2
        elif rsi > 65:
            signals.append(f"✗ RSI {rsi:.1f} (Overbought - Sell signal)")
            score -= 2
        else:
            signals.append(f"○ RSI {rsi:.1f} (Neutral)")

        vwap = df['VWAP'].iloc[-1]
        if current_price > vwap:
            signals.append(f"✓ Price above VWAP (Institutional buying)")
            score += 2
        else:
            signals.append(f"✗ Price below VWAP (Weak momentum)")
            score -= 1

        volume_ratio = df['Volume_Ratio'].iloc[-1]
        if volume_ratio > 1.5:
            signals.append(f"✓ High volume ({volume_ratio:.1f}x avg)")
            score += 1
        elif volume_ratio < 0.5:
            signals.append(f"⚠ Low volume ({volume_ratio:.1f}x avg)")
            score -= 1

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
        price_change = analysis['price_change_pct']

        if price_change > 0:
            target_pct = 1.5
            stop_pct = 0.8
        else:
            target_pct = 2.5
            stop_pct = 1.2

        target_price = current_price * (1 + target_pct / 100)
        stop_loss = current_price * (1 - stop_pct / 100)

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
            if score >= 5:
                return "STRONG BUY", "🟢🟢", "Buy NOW - Strong momentum"
            elif score >= 2:
                return "BUY", "🟢", "Good entry point"
            elif score >= 0:
                return "WAIT", "🟡", "Wait for better signal"
            else:
                return "SKIP", "⚪", "Avoid - Weak setup"

        elif phase == "AFTERNOON_SESSION":
            if score >= 3:
                return "HOLD", "🟡", "Hold for higher target"
            elif score >= 0:
                return "TAKE PROFIT", "🟢", "Sell at profit"
            else:
                return "SELL NOW", "🔴", "Exit position - Weakness"

        else:
            if score >= 4:
                return "MONITOR", "👁️", "Watch for entry"
            else:
                return "NEUTRAL", "⚪", "Wait for trading hours"

    def calculate_confidence(self, score, volume_ratio, phase):
        """Calculate confidence for intraday trade"""
        base_confidence = 50
        base_confidence += score * 5

        if volume_ratio > 1.5:
            base_confidence += 15
        elif volume_ratio < 0.7:
            base_confidence -= 10

        if phase == "MORNING_SESSION":
            base_confidence += 10
        elif phase == "AFTERNOON_SESSION":
            base_confidence += 5

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

        if data['avg_volume'] < self.MIN_VOLUME:
            return None

        analysis = self.intraday_analysis(data)

        if analysis is None:
            return None

        targets = self.calculate_intraday_targets(
            data['current_price'],
            data['open_price'],
            analysis
        )

        if targets['investment'] < self.MIN_CAPITAL:
            return None

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
            'bid': data['bid'],
            'ask': data['ask'],
            'bid_size': data['bid_size'],
            'ask_size': data['ask_size'],
            'spread_pct': data['spread_pct'],
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
        print("🤖 INTRADAY TRADING BOT - AI-POWERED ANALYSIS")
        print("=" * 80)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} WIB")
        print(f"Trading Phase: {phase}")
        print(f"Capital: Rp {self.capital:,.0f}")
        print(f"AI Status: {'✅ Enabled' if self.ai_analyzer else '⚠️ Disabled'}")
        print(f"Scanning {len(self.stocks)} stocks...\n")

        self.results = []

        for ticker in self.stocks:
            print(f"⏳ Analyzing {ticker}...", end=" ")
            result = self.analyze_stock(ticker, phase)

            if result:
                self.results.append(result)
                print(f"✓ {result['action']}")
            else:
                print(f"✗ Skipped")

            time.sleep(0.5)

        print(f"\n✅ Technical scan complete! {len(self.results)} stocks analyzed.")

        # Run AI analysis on top stocks
        if self.ai_analyzer and self.results:
            ai_enhanced = self.ai_analyzer.batch_analyze(self.results, max_stocks=5)

            # Update results with AI analysis
            for i, result in enumerate(self.results):
                for ai_result in ai_enhanced:
                    if result['ticker'] == ai_result['ticker']:
                        self.results[i].update(ai_result)
                        break

        return self.results

    def send_telegram_notifications(self, top_n=5):
        """Send notifications to Telegram (summary + individual stocks)"""
        if not self.telegram:
            print("⚠️  Telegram not configured. Skipping notifications.")
            return

        if not self.results:
            print("⚠️  No results to send.")
            return

        phase = self.get_trading_phase()

        print("\n📱 Sending Telegram notifications...")

        # 1. Send summary
        summary = self.telegram.format_summary_message(self.results, phase, self.capital)
        if self.telegram.send_message(summary):
            print("✅ Summary sent")
        else:
            print("❌ Failed to send summary")

        time.sleep(1)

        # 2. Send individual stock messages (prioritize AI-analyzed stocks)
        actionable_stocks = [r for r in self.results if r['action'] not in ['SKIP', 'NEUTRAL']]

        # Sort by AI confidence if available, otherwise use regular confidence
        sorted_stocks = sorted(
            actionable_stocks,
            key=lambda x: x.get('ai_confidence', x['confidence']),
            reverse=True
        )

        sent_count = 0
        for stock in sorted_stocks[:top_n]:
            stock_msg = self.telegram.format_stock_message(stock)
            if self.telegram.send_message(stock_msg):
                print(f"✅ {stock['ticker']} sent")
                sent_count += 1
            else:
                print(f"❌ {stock['ticker']} failed")

            time.sleep(1.5)

        print(f"\n✅ Sent {sent_count}/{top_n} stock notifications to Telegram")

    def display_results(self):
        """Display intraday trading opportunities with AI insights"""
        if not self.results:
            print("No results to display.")
            return

        phase = self.get_trading_phase()
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get('ai_confidence', x['confidence']),
            reverse=True
        )

        print("=" * 80)
        if phase == "MORNING_SESSION":
            print("🌅 MORNING SESSION - AI-POWERED BUY RECOMMENDATIONS")
        elif phase == "AFTERNOON_SESSION":
            print("🌆 AFTERNOON SESSION - AI-POWERED POSITION MANAGEMENT")
        else:
            print("🤖 AI-POWERED INTRADAY OPPORTUNITIES")
        print("=" * 80)

        for i, r in enumerate(sorted_results, 1):
            if r['action'] in ['SKIP', 'NEUTRAL']:
                continue

            print(f"\n{i}. {r['ticker']} - {r['name'][:40]}")
            print("-" * 80)
            print(f"Action: {r['emoji']} {r['action']} - {r['description']}")
            print(f"Confidence: {r['confidence']}% ({r['conf_level']})")

            # Display AI insights if available
            if r.get('ai_analysis_success'):
                print(f"\n🤖 AI ANALYSIS:")
                print(f"   Recommendation: {r['ai_recommendation']}")
                print(f"   AI Confidence: {r['ai_confidence']}%")
                print(f"   Key Insight: {r['ai_insight'][:150]}")
                print(f"   Risk Factors: {r['ai_risks'][:150]}")
                print(f"   Trade Plan: {r['ai_trade_plan'][:150]}")

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
            print(f"   Shares: {t['shares']:,} shares")
            print(f"   Investment: Rp {t['investment']:,.0f}")
            print(f"   Potential Profit: Rp {t['potential_profit']:,.0f}")
            print(f"   Potential Loss: Rp {t['potential_loss']:,.0f}")

            print(f"\n📈 KEY SIGNALS:")
            for signal in r['analysis']['signals'][:4]:
                print(f"   {signal}")

        print("\n" + "=" * 80)

    def get_top_intraday_picks(self, limit=3):
        """Get top AI-powered stocks for today's trading"""
        if not self.results:
            return []

        phase = self.get_trading_phase()

        if phase == "MORNING_SESSION":
            filtered = [r for r in self.results if 'BUY' in r['action']]
        elif phase == "AFTERNOON_SESSION":
            filtered = [r for r in self.results if r['action'] != 'SKIP']
        else:
            filtered = self.results

        sorted_picks = sorted(
            filtered,
            key=lambda x: x.get('ai_confidence', x['confidence']),
            reverse=True
        )
        return sorted_picks[:limit]

    def export_watchlist(self, filename='intraday_watchlist_ai.csv'):
        """Export AI-powered watchlist"""
        if not self.results:
            print("No results to export.")
            return

        data = []
        for r in self.results:
            if r['action'] not in ['SKIP', 'NEUTRAL']:
                row = {
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
                }

                # Add AI fields if available
                if r.get('ai_analysis_success'):
                    row.update({
                        'AI Recommendation': r['ai_recommendation'],
                        'AI Confidence': r['ai_confidence'],
                        'AI Insight': r['ai_insight'][:100]
                    })

                data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"✅ AI-powered watchlist exported to {filename}")


# Main execution
if __name__ == "__main__":
    print("🚀 Starting AI-Powered Intraday Trading Bot...")
    print("🤖 Powered by DeepSeek AI\n")

    # ============================================
    # CONFIGURATION - EDIT THIS SECTION
    # ============================================

    # 1. Trading Capital (default 10 million IDR)
    TRADING_CAPITAL = 10_000_000

    # 2. AI Configuration - Choose your provider
    # Options: 'deepseek', 'groq', 'gemini', 'ollama'

    AI_PROVIDER = os.getenv("AI_PROVIDER", "groq")  # Default to Groq (has free tier)

    # Get API key based on provider
    if AI_PROVIDER == "deepseek":
        AI_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    elif AI_PROVIDER == "groq":
        AI_API_KEY = os.getenv("GROQ_API_KEY")  # Get from https://console.groq.com
    elif AI_PROVIDER == "gemini":
        AI_API_KEY = os.getenv("GEMINI_API_KEY")  # Get from https://makersuite.google.com
    elif AI_PROVIDER == "ollama":
        AI_API_KEY = None  # Ollama runs locally, no API key needed
    else:
        AI_API_KEY = None

    # Debug: Check if API key is loaded
    if AI_API_KEY:
        print(f"✓ {AI_PROVIDER.upper()} API Key loaded: {AI_API_KEY[:15]}...")
    elif AI_PROVIDER == "ollama":
        print("✓ Using Ollama (local, no API key needed)")
    else:
        print(f"⚠️  {AI_PROVIDER.upper()} API Key not found")
        print(f"   Add to .env: {AI_PROVIDER.upper()}_API_KEY=your_key")

    # Set to None to disable AI analysis
    # AI_API_KEY = None

    # 3. Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_IDS = [c.strip() for c in os.getenv("TELEGRAM_CHAT_IDS", "").split(",") if c.strip()]

    # 4. Auto-run Configuration
    AUTO_RUN = True  # Set False to run once only
    RUN_INTERVAL_MINUTES = 30  # Scan every 30 minutes

    # ============================================

    # Initialize AI-powered bot
    bot = IntradayTradingBot(
        capital=TRADING_CAPITAL,
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_IDS,
        ai_provider=AI_PROVIDER,
        ai_api_key=AI_API_KEY
    )

    # Check trading phase
    phase = bot.get_trading_phase()
    print(f"\n⏰ Current phase: {phase}")

    if phase == "PRE_MARKET":
        print("📅 Market hasn't opened yet. Preparing AI-powered watchlist...")
    elif phase == "AFTER_MARKET":
        print("🌙 Market closed. Review today's AI-analyzed performance.")

    # Scan stocks with AI analysis
    results = bot.scan_all_stocks()

    # Display AI-powered opportunities
    bot.display_results()

    # Show top AI picks
    print("\n" + "=" * 80)
    print("🏆 TOP AI-POWERED INTRADAY PICKS")
    print("=" * 80)

    top_picks = bot.get_top_intraday_picks(3)

    for i, pick in enumerate(top_picks, 1):
        t = pick['targets']
        profit_pct = t['target_pct']

        ai_tag = " 🤖" if pick.get('ai_analysis_success') else ""

        print(f"\n{i}. {pick['ticker']} {pick['emoji']} {pick['action']}{ai_tag}")
        print(f"   Entry: Rp {t['entry_price']:,.0f} → Target: Rp {t['target_price']:,.0f} (+{profit_pct:.1f}%)")
        print(f"   Investment: Rp {t['investment']:,.0f} → Profit: Rp {t['potential_profit']:,.0f}")
        print(f"   Confidence: {pick['confidence']}%", end="")

        if pick.get('ai_analysis_success'):
            print(f" | AI: {pick['ai_confidence']}%")
            print(f"   AI Insight: {pick['ai_insight'][:120]}")
        else:
            print()

    # Export AI-powered watchlist
    print()
    bot.export_watchlist()

    # Send Telegram notifications with AI insights
    bot.send_telegram_notifications(top_n=5)

    print("\n" + "=" * 80)
    print("📋 AI-POWERED INTRADAY TRADING RULES")
    print("=" * 80)
    print("✓ AI analyzes top 5 stocks for deeper insights")
    print("✓ Combines technical indicators + AI recommendations")
    print("✓ Best entry: 09:00 - 10:30 WIB")
    print("✓ Best exit: 14:30 - 16:00 WIB")
    print("✓ Always use stop loss (max 1% loss)")
    print("✓ Target: 1.5-2.5% profit per trade")
    print("✓ Don't hold overnight!")
    print("✓ Follow AI risk warnings")
    print("✓ MINIMUM VOLUME: 5M shares/day")
    print("=" * 80)

    print("\n⚠️  DISCLAIMER")
    print("This is for educational purposes only.")
    print("AI provides insights but is not financial advice.")
    print("Day trading is risky. Trade responsibly.")
    print("=" * 80)

    print("\n📱 SETUP GUIDE")
    print("=" * 80)
    print("1. DeepSeek AI API:")
    print("   - Visit: https://platform.deepseek.com")
    print("   - Create account and get API key")
    print("   - Add to .env: DEEPSEEK_API_KEY=your_key")
    print()
    print("2. Telegram Bot:")
    print("   - Chat with @BotFather, create bot")
    print("   - Get Bot Token")
    print("   - Chat with @userinfobot for Chat ID")
    print("   - Add to .env: TELEGRAM_BOT_TOKEN=your_token")
    print("   - Add to .env: TELEGRAM_CHAT_IDS=chat_id1,chat_id2")
    print()
    print("3. Environment Variables (.env file):")
    print("   DEEPSEEK_API_KEY=sk-...")
    print("   TELEGRAM_BOT_TOKEN=123456:ABC...")
    print("   TELEGRAM_CHAT_IDS=1234567890,-1001234567890")
    print("=" * 80)

    print("\n🎯 FEATURES:")
    print("✅ Real-time stock scanning")
    print("✅ Technical indicator analysis")
    print("✅ DeepSeek AI insights & recommendations")
    print("✅ Risk factor analysis")
    print("✅ Automated Telegram notifications")
    print("✅ Multi-chat support")
    print("✅ AI-powered trade plans")
    print("=" * 80)