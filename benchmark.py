import numpy as np
import pandas as pd
import yfinance as yf


def get_xu100(period="2y", interval="1d") -> pd.DataFrame:
    for symbol in ["XU100.IS", "^XU100"]:
        df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
        if df is not None and not df.empty:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            # MultiIndex kolon gelirse düzelt
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            return df
    raise RuntimeError("XU100 verisi çekilemedi (XU100.IS ve ^XU100 denendi).")


def buy_and_hold_equity(index_df: pd.DataFrame, initial_cash: float) -> pd.Series:
    """
    Close serisine göre buy&hold equity eğrisi (Series) döndürür.
    """
    if index_df is None or index_df.empty:
        raise ValueError("Benchmark dataframe boş.")

    close = index_df["Close"]
    # close bazen DataFrame gibi gelebilir -> Series'e indir
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.astype(float).dropna()
    if close.empty:
        raise ValueError("Benchmark Close serisi boş.")

    first_price = float(close.iloc[0])  # FutureWarning fix
    shares = initial_cash / first_price
    equity = shares * close
    equity.name = "XU100_BH"
    return equity


def compute_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    rets = equity.pct_change().dropna()

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1)

    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.25 if days > 0 else 0.0
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else np.nan

    if len(rets) > 10 and rets.std() > 0:
        sharpe = float((rets.mean() / rets.std()) * np.sqrt(252))
    else:
        sharpe = np.nan

    return {
        "final_equity": float(equity.iloc[-1]),
        "total_return": total_return,
        "max_dd": max_dd,
        "cagr": cagr,
        "sharpe": sharpe,
    }


def align_equities(system_eq: pd.Series, bench_eq: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    İkisini de Series'e zorlar, ortak tarihlere indirger.
    """
    # System equity bazen DataFrame gelirse Series'e indir
    if isinstance(system_eq, pd.DataFrame):
        system_eq = system_eq.iloc[:, 0]
    if isinstance(bench_eq, pd.DataFrame):
        bench_eq = bench_eq.iloc[:, 0]

    system_eq = pd.Series(system_eq).dropna()
    bench_eq = pd.Series(bench_eq).dropna()

    system_eq.name = "SYS"
    bench_eq.name = "BCH"

    df = pd.concat([system_eq, bench_eq], axis=1).dropna()
    return df["SYS"], df["BCH"]


def print_comparison(system_eq: pd.Series, bench_eq: pd.Series):
    sys_m = compute_metrics(system_eq)
    bch_m = compute_metrics(bench_eq)

    alpha_total = sys_m["total_return"] - bch_m["total_return"]
    alpha_cagr = sys_m["cagr"] - bch_m["cagr"] if (not np.isnan(sys_m["cagr"]) and not np.isnan(bch_m["cagr"])) else np.nan

    print("\n==== BENCHMARK COMPARISON (System vs XU100 Buy&Hold) ====")
    print(f"System final:   {sys_m['final_equity']:,.2f}")
    print(f"XU100 final:    {bch_m['final_equity']:,.2f}")
    print(f"System return:  {sys_m['total_return']*100:.2f}%")
    print(f"XU100 return:   {bch_m['total_return']*100:.2f}%")
    print(f"ALPHA (total):  {alpha_total*100:.2f}%")
    print(f"System MaxDD:   {sys_m['max_dd']*100:.2f}%")
    print(f"XU100 MaxDD:    {bch_m['max_dd']*100:.2f}%")
    if not np.isnan(sys_m["cagr"]) and not np.isnan(bch_m["cagr"]):
        print(f"System CAGR:    {sys_m['cagr']*100:.2f}%")
        print(f"XU100 CAGR:     {bch_m['cagr']*100:.2f}%")
        print(f"ALPHA (CAGR):   {alpha_cagr*100:.2f}%")
    print(f"System Sharpe:  {sys_m['sharpe']:.3f}")
    print(f"XU100 Sharpe:   {bch_m['sharpe']:.3f}")