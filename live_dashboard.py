import os
import sys
import json
import time
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# Page Config
st.set_page_config(page_title="Live AI Screener (NSE)", page_icon="⚡", layout="wide")

# Load Env
load_dotenv()
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')

@st.cache_data(ttl=300) 
def get_indian_universe():
    sys.path.append(os.path.join(os.path.dirname(__file__), 'market-tools'))
    try:
        from india_screener import UNIVERSE, WATCHLIST
        return list(set(UNIVERSE + WATCHLIST))
    except Exception:
        return ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']

def compute_intraday_indicators(df):
    if df.empty or len(df) < 20: return None
    df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP'] = (df['Typical'] * df['Volume']).groupby(df.index.date).cumsum() / (df['Volume'].groupby(df.index.date).cumsum() + 1e-9)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['AvgVol_20'] = df['Volume'].rolling(20).mean()
    return df

@st.cache_data(ttl=300, show_spinner=False)
def scan_market(_tickers):
    data = yf.download(_tickers, period="2d", interval="15m", group_by="ticker", auto_adjust=False, progress=False)
    breakouts = []
    
    for ticker in _tickers:
        try:
            if len(_tickers) == 1: 
                df = data.copy()
            else: 
                if ticker not in data.columns.levels[0]: continue
                df = data[ticker].dropna(subset=['Close']).copy()
                
            df = compute_intraday_indicators(df)
            if df is None: continue
            
            last = df.iloc[-1]
            
            # The mathematical breakout conditions
            if last['Close'] > last['VWAP'] and last['Volume'] > (last['AvgVol_20'] * 1.5) and (55 < last['RSI'] < 85):
                day_high = df.loc[str(df.index[-1].date())]['High'].max()
                if last['Close'] >= (day_high * 0.99):
                    breakouts.append({
                        'Ticker': ticker,
                        'Price': round(last['Close'], 2),
                        'VWAP': round(last['VWAP'], 2),
                        'RSI': round(last['RSI'], 2),
                        'Vol_Surge': round(last['Volume'] / (last['AvgVol_20'] + 1e-9), 1)
                    })
        except: 
            pass
    return breakouts

@st.cache_data(ttl=300, show_spinner=False)
def analyze_breakout_with_llm(ticker, price, vol_surge):
    try:
        news = yf.Ticker(ticker).news
        headlines = [n['title'] for n in news[:5]] if news else ["No major recent news."]
    except: 
        headlines = ["No major recent news."]
        
    prompt = f"Stock: {ticker}\nLive Price: {price}\nSetup: Breakout above VWAP on {vol_surge}x vol.\nHeadlines: {json.dumps(headlines)}\n\nYou are a day trading AI. Does this technical breakout have a news catalyst? Provide a rapid 2-3 sentence verdict. Conclude with exactly: 'VERDICT: BUY' or 'VERDICT: PASS'."
    
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "meta/llama-3.2-11b-vision-instruct", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 150}
    
    try:
        res = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content'].strip()
        return f"API Error: {res.status_code}"
    except Exception as e:
        return f"Request Failed: {e}"

# UI Layout
st.title("⚡ Live AI Intraday Screener (NSE)")
st.markdown("Fast Math Filters (VWAP, RSI, Vol) + Smart AI Verification (NVIDIA LLM)")

with st.sidebar:
    st.header("⚙️ Controls")
    if not NVIDIA_API_KEY:
        st.error("NVIDIA_API_KEY missing from .env! AI analysis will fail.")
    else:
        st.success("✅ NVIDIA LLM Connected")
    
    if st.button("🚀 Force Rescan Now"):
        st.cache_data.clear()
        st.rerun()
        
    st.info("Market data and AI responses are automatically cached for 5 minutes to prevent hitting API rate limits. Click 'Force Rescan' to bypass.")

# Main Execution
tickers = get_indian_universe()

with st.spinner(f"Fetching live 15m candles and scanning {len(tickers)} stocks..."):
    breakouts = scan_market(tickers)

if not breakouts:
    st.warning(f"No fresh breakouts found at {datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%H:%M:%S')} IST. (All {len(tickers)} stocks were scanned but none met the criteria).")
else:
    st.success(f"Successfully scanned {len(tickers)} stocks. Found {len(breakouts)} active breakouts!")
    
    # Display Results
    st.subheader(f"🔥 Active Breakouts (Filtered from {len(tickers)} stocks)")
    st.markdown("These are the **only** stocks out of your entire universe that are currently surging above VWAP on high volume.")
    df_results = pd.DataFrame(breakouts)
    st.dataframe(df_results, use_container_width=True)
    
    st.subheader("🧠 LLM Catalyst Verification")
    for b in breakouts:
        with st.expander(f"{b['Ticker']} (Price: {b['Price']} | Vol Surge: {b['Vol_Surge']}x)"):
            with st.spinner("Asking AI to analyze headlines..."):
                verdict = analyze_breakout_with_llm(b['Ticker'], b['Price'], b['Vol_Surge'])
                
                # Color code the result based on verdict
                if "VERDICT: BUY" in verdict:
                    st.success(verdict)
                elif "VERDICT: PASS" in verdict:
                    st.error(verdict)
                else:
                    st.info(verdict)
