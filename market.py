import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date  # 正確寫法
import os

st.set_page_config(page_title="市場熱力圖", layout="wide")
st.title("每日市場熱力圖（yfinance 版）")
st.markdown("資料來源：Yahoo Finance，自動生成主要資產漲跌熱力表。")

# ======== 資產清單 ========
ASSETS = {
    "TESLA": "TSLA","APPLE": "AAPL","NVIDIA": "NVDA","GOOGLE": "GOOGL","META": "META","MSFT": "MSFT","AMAZON": "AMZN","NIO": "NIO","XPEV": "XPEV",
    "美元指數": "DX-Y.NYB", "2年美債收益率": "^IRX", "10年美債收益率": "^TNX",
    "TLT(長期美債ETF)": "TLT", "S&P500": "^GSPC", "那斯達克": "^IXIC",
    "道瓊工業": "^DJI", "羅素2000": "^RUT", "VIX(恐慌指數)": "^VIX",
    "黃金期貨": "GC=F", "WTI原油": "CL=F", "REITs指數ETF": "VNQ",
    "科技ETF": "XLK", "醫療ETF": "XLV", "金融ETF": "XLF", "能源ETF": "XLE",
    "非必需消費ETF": "XLY", "公用事業ETF": "XLU", "必需消費ETF": "XLP",
    "標普成長ETF": "SPYG", "標普價值ETF": "SPYV", "標普500 ETF": "SPY",
    "那斯達克100(大型科技)": "QQQ"
}

# ======== 側欄設定 ========
st.sidebar.header("設定")
start_of_fetch = st.sidebar.date_input(
    "最早抓取日期", value=date.today() - timedelta(days=400),
    min_value=date(2000, 1, 1), max_value=date.today()
)
save_folder = st.sidebar.text_input("CSV 存放資料夾", value="data")
os.makedirs(save_folder, exist_ok=True)

# ======== 時區工具 ========
NY_TZ = 'America/New_York'

def to_ny_aware(dt):
    if dt is None:
        return None
    try:
        ts = pd.to_datetime(dt)
        if ts.tz is None:
            return ts.tz_localize(NY_TZ)
        else:
            return ts.tz_convert(NY_TZ)
    except Exception as e:
        st.warning(f"時間轉換錯誤: {e}")
        return None

# ======== 資料抓取 ========
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(tickers, start_date):
    data = {}
    start_ny = to_ny_aware(start_date)
    end_ny = to_ny_aware(date.today() + timedelta(days=1))

    for tk in tickers:
        try:
            hist = yf.Ticker(tk).history(
                start=start_ny.date() if start_ny else None,
                end=end_ny.date() if end_ny else None,
                raise_errors=False
            )
            if not hist.empty:
                if hist.index.tz is None:
                    hist.index = hist.index.tz_localize(NY_TZ)
                else:
                    hist.index = hist.index.tz_convert(NY_TZ)
            data[tk] = hist
        except Exception as e:
            st.warning(f"抓取 {tk} 失敗：{e}")
            data[tk] = pd.DataFrame()
    return data

# ======== 最近價格 ========
def nearest_price(hist_df, target_date):
    if hist_df is None or hist_df.empty:
        return np.nan
    target = to_ny_aware(target_date)
    if target is None:
        return np.nan
    idx = hist_df.index
    if idx.tz is None:
        idx = idx.tz_localize(NY_TZ)
    else:
        idx = idx.tz_convert(NY_TZ)
    past = idx[idx <= target]
    if past.empty:
        return np.nan
    return hist_df.loc[past.max(), "Close"]

# ======== 季度起點 ========
def quarter_start(today):
    q = (today.month - 1) // 3 + 1
    return date(today.year, 3*(q-1)+1, 1)

# ======== 計算漲跌 ========
def compute_changes(hist_map):
    rows = []
    today = date.today()

    targets = {
        "1日": today - timedelta(days=1),
        "1週": today - timedelta(days=7),
        "1月": today - timedelta(days=30),
        "1年": today - timedelta(days=365),
        "QTD": quarter_start(today),
        "YTD": date(today.year, 1, 1)
    }

    for name, tk in ASSETS.items():
        hist = hist_map.get(tk, pd.DataFrame())
        latest = hist["Close"].iloc[-1] if not hist.empty and "Close" in hist.columns else np.nan

        changes = {}
        for label, base_date in targets.items():
            prev = nearest_price(hist, base_date)
            changes[label] = round((latest / prev - 1) * 100, 2) if pd.notna(latest) and pd.notna(prev) and prev != 0 else np.nan

        rows.append({
            "資產": name, "Ticker": tk, "收盤": round(latest, 4) if pd.notna(latest) else np.nan,
            "1日": changes["1日"], "1週": changes["1週"], "1月": changes["1月"],
            "1年": changes["1年"], "QTD": changes["QTD"], "YTD": changes["YTD"]
        })

    return pd.DataFrame(rows).set_index("資產")

# ======== 顏色 ========
def colorize(val):
    if pd.isna(val):
        return "background-color: #f0f0f0; color: #999;"
    v = float(val)
    if v > 0:
        intensity = min(abs(v)/15, 1)
        g = int(180 + 75*intensity)
        return f"background-color: rgba(0,{g},0,{0.2+0.3*intensity}); color: green; font-weight: bold;"
    elif v < 0:
        intensity = min(abs(v)/15, 1)
        r = int(255 - 55*intensity)
        return f"background-color: rgba({r},100,100,{0.2+0.3*intensity}); color: red; font-weight: bold;"
    else:
        return "background-color: #fff; color: #333;"

# ======== 主程式 ========
if st.button("生成市場熱力圖", type="primary"):
    with st.spinner("抓取資料中..."):
        hist_map = fetch_history(list(ASSETS.values()), start_of_fetch)
        df = compute_changes(hist_map)

    style_cols = ["1日", "1週", "1月", "1年", "QTD", "YTD"]
    styled = df.style\
        .map(colorize, subset=style_cols)\
        .format({"收盤": "{:.4f}", **{col: "{:.2f}%" for col in style_cols}}, na_rep="N/A")\
        .set_properties(**{'text-align': 'center'}, subset=style_cols)

    st.subheader(f"市場表現 {date.today()}")
    st.dataframe(styled, use_container_width=True)

    csv_file = f"market_heatmap_{date.today().isoformat()}.csv"
    csv_path = os.path.join(save_folder, csv_file)
    df.to_csv(csv_path, encoding="utf-8-sig", float_format="%.4f")
    st.success(f"已儲存：`{csv_path}`")

    with open(csv_path, "rb") as f:
        st.download_button("下載 CSV", f, csv_file, "text/csv")
else:
    st.info("點擊按鈕生成熱力圖")

st.markdown("---\n**綠漲紅跌**｜資料：Yahoo Finance｜時間：美東時間")
