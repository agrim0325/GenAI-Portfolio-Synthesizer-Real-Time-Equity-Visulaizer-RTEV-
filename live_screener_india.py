import os
import sys
import time
import json
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Fix Windows console unicode printing
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Load Environment Variables
load_dotenv()
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')

def get_indian_universe():
    """Loads the 128-stock universe from your existing market-tools configuration."""
    sys.path.append(os.path.join(os.path.dirname(__file__), 'market-tools'))
    try:
        from india_screener import UNIVERSE, WATCHLIST
        return list(set(UNIVERSE + WATCHLIST))
    except Exception as e:
        print(f"Could not load universe dynamically: {e}")
        return ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']

def compute_intraday_indicators(df):
    """Calculates VWAP, RSI, and Volume averages on a 15-minute timeframe."""
    if df.empty or len(df) < 20:
        return None
    
    # Intraday VWAP (resets daily)
    df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical'] * df['Volume']).groupby(df.index.date).cumsum() / (df['Volume'].groupby(df.index.date).cumsum() + 1e-9)
    
    # 14-period RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 20-period Volume Average (5 hours of 15m candles)
    df['AvgVol_20'] = df['Volume'].rolling(20).mean()
    
    return df

def scan_for_breakouts(tickers):
    """Downloads live intraday data and filters for math-based momentum setups."""
    kolkata_time = datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%H:%M:%S')
    print(f"[{kolkata_time}] Fetching 15m candles for {len(tickers)} stocks...")
    
    # Download 2 days of 15m data to have enough history for MAs/RSI
    data = yf.download(tickers, period="2d", interval="15m", group_by="ticker", auto_adjust=False, progress=False)
    
    breakouts = []
    
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                df = data.copy()
            else:
                if ticker not in data.columns.levels[0]:
                    continue
                df = data[ticker].dropna(subset=['Close']).copy()
                
            df = compute_intraday_indicators(df)
            if df is None: continue
            
            last = df.iloc[-1]
            
            # --- THE BREAKOUT CRITERIA ---
            # 1. Price is trading above the daily VWAP
            above_vwap = last['Close'] > last['VWAP']
            
            # 2. Sudden volume surge (current candle volume is 1.5x the rolling average)
            vol_surge = last['Volume'] > (last['AvgVol_20'] * 1.5)
            
            # 3. Momentum is bullish but not hopelessly overbought
            bullish_rsi = 55 < last['RSI'] < 85
            
            # 4. Closing near the highs of the day (within 1%)
            day_high = df.loc[str(df.index[-1].date())]['High'].max()
            near_high = last['Close'] >= (day_high * 0.99)
            
            if above_vwap and vol_surge and bullish_rsi and near_high:
                breakouts.append({
                    'ticker': ticker,
                    'price': last['Close'],
                    'vwap': last['VWAP'],
                    'rsi': last['RSI'],
                    'volume_ratio': last['Volume'] / (last['AvgVol_20'] + 1e-9)
                })
        except Exception as e:
            pass
            
    return breakouts

def analyze_breakout_with_llm(breakout_data):
    """Sends the mathematically verified breakout to NVIDIA NIM for qualitative validation."""
    ticker = breakout_data['ticker']
    print(f"  🚀 BREAKOUT DETECTED: {ticker} at {breakout_data['price']:.2f} (Vol Surge: {breakout_data['volume_ratio']:.1f}x)")
    print(f"  🧠 Asking LLM for real-time catalyst check...")
    
    try:
        news = yf.Ticker(ticker).news
        headlines = [n['title'] for n in news[:5]] if news else ["No major recent news."]
    except:
        headlines = ["No major recent news."]
        
    prompt = f"""
    Stock: {ticker}
    Live Price: {breakout_data['price']:.2f} 
    Technical Setup: Breaking out above VWAP on {breakout_data['volume_ratio']:.1f}x average volume with bullish RSI.
    
    Recent Headlines:
    {json.dumps(headlines, indent=2)}
    
    You are an elite high-frequency day trading AI. Look at the headlines. Does this technical volume breakout have a legitimate news catalyst behind it, or is it just noise?
    Provide a rapid 2-3 sentence verdict. Conclude your response with exactly: 'VERDICT: BUY' or 'VERDICT: PASS'.
    """
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta/llama-3.2-11b-vision-instruct", 
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 150
    }
    
    try:
        res = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            text = res.json()['choices'][0]['message']['content'].strip()
            print(f"\n[{ticker} LLM VERDICT]")
            print(f"{text}\n")
            print("-" * 50)
        else:
            print(f"  [LLM Error: {res.status_code}]\n")
    except Exception as e:
        print(f"  [LLM Request Failed: {e}]\n")

def main():
    print("==================================================")
    print(" ⚡ LIVE INTRADAY SCREENER (NSE) ⚡ ")
    print("==================================================")
    
    if not NVIDIA_API_KEY:
        print("WARNING: NVIDIA_API_KEY not found in .env. LLM validation will fail.")
        
    tickers = get_indian_universe()
    print(f"Loaded {len(tickers)} Indian stocks for live scanning.")
    print("Engine started. Polling every 5 minutes... (Press Ctrl+C to stop)\n")
    
    # Just run once for demonstration and exit (so it doesn't loop infinitely in the background for now)
    breakouts = scan_for_breakouts(tickers)
    
    if not breakouts:
        print(f"  No fresh volume breakouts found this cycle.")
    else:
        for b in breakouts:
            analyze_breakout_with_llm(b)
            
    print("\nScan complete.")

if __name__ == "__main__":
    main()
