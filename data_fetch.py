import yfinance as yf
import pandas as pd

def get_bist100_list(csv_file: str = "bist100.csv") -> list[str]:
    df = pd.read_csv(csv_file)
    # boş satır / boşluk temizliği
    tickers = [str(x).strip() for x in df["Ticker"].tolist() if str(x).strip()]
    return tickers

def get_data_from_yfinance(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame | None:
    """
    Returns OHLCV daily data with DatetimeIndex.
    """
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        print(f"[WARN] {ticker}: veri alınamadı (empty).")
        return None

    # YFinance bazen kolonları MultiIndex yapabiliyor; düzleştir
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(df.columns)):
        print(f"[WARN] {ticker}: kolonlar eksik -> {df.columns}")
        return None

    df = df[list(needed)].copy()
    df.dropna(inplace=True)
    return df