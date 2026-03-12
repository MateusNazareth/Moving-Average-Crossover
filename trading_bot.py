"""
Multi-Confirmation Trading Bot
Strategy: EMA Crossover + ADX + ATR + Multi-Timeframe Confirmation
Supports: Binance, Kraken, Forex (OANDA)

Configure via environment variables (see .env.example)
"""

import os
import time
import logging
import warnings
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ─────────────────────────────────────────────
#  CONFIGURATION  (edit here or via .env file)
# ─────────────────────────────────────────────
class Config:
    # ── Broker selection ──────────────────────
    # Set BROKER to: "binance" | "kraken" | "forex"
    BROKER         = os.getenv("BROKER", "binance")

    # ── Asset ─────────────────────────────────
    # Binance:  "BTC/USDT", "ETH/USDT", etc.
    # Kraken:   "BTC/USD",  "ETH/USD",  etc.
    # Forex:    "EUR_USD",  "GBP_USD",  etc.
    SYMBOL         = os.getenv("SYMBOL", "BTC/USDT")

    # ── Timeframes ────────────────────────────
    # Options: "1m","5m","15m","30m","1h","4h","1d"
    LTF            = os.getenv("LTF", "1h")     # Trading timeframe
    HTF            = os.getenv("HTF", "4h")     # Higher timeframe for confirmation

    # ── EMA periods ───────────────────────────
    EMA_FAST       = int(os.getenv("EMA_FAST", "9"))
    EMA_SLOW       = int(os.getenv("EMA_SLOW", "21"))

    # ── ADX settings ─────────────────────────
    ADX_PERIOD     = int(os.getenv("ADX_PERIOD", "14"))
    ADX_THRESHOLD  = float(os.getenv("ADX_THRESHOLD", "25"))

    # ── ATR settings ─────────────────────────
    ATR_PERIOD     = int(os.getenv("ATR_PERIOD", "14"))
    ATR_SL_MULT    = float(os.getenv("ATR_SL_MULT", "1.5"))   # Stop loss = 1.5x ATR
    ATR_TP_MULT    = float(os.getenv("ATR_TP_MULT", "2.5"))   # Take profit = 2.5x ATR

    # ── Risk management ───────────────────────
    RISK_PCT       = float(os.getenv("RISK_PCT", "1.0"))      # % of equity per trade
    INITIAL_CAPITAL= float(os.getenv("INITIAL_CAPITAL", "10000"))

    # ── Backtest period ───────────────────────
    BACKTEST_DAYS  = int(os.getenv("BACKTEST_DAYS", "365"))

    # ── API credentials ───────────────────────
    # Binance
    BINANCE_API_KEY    = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    # Kraken
    KRAKEN_API_KEY     = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET  = os.getenv("KRAKEN_API_SECRET", "")

    # Forex / OANDA
    OANDA_API_KEY      = os.getenv("OANDA_API_KEY", "")
    OANDA_ACCOUNT_ID   = os.getenv("OANDA_ACCOUNT_ID", "")
    OANDA_ENVIRONMENT  = os.getenv("OANDA_ENVIRONMENT", "practice")  # "practice" | "live"

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  DATA FETCHERS  (one per broker)
# ─────────────────────────────────────────────

def fetch_binance(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    try:
        import ccxt
        exchange = ccxt.binance({
            "apiKey": Config.BINANCE_API_KEY,
            "secret": Config.BINANCE_API_SECRET,
        })
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.set_index("timestamp")
    except ImportError:
        log.error("ccxt not installed. Run: pip install ccxt")
        raise


def fetch_kraken(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    try:
        import ccxt
        exchange = ccxt.kraken({
            "apiKey": Config.KRAKEN_API_KEY,
            "secret": Config.KRAKEN_API_SECRET,
        })
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.set_index("timestamp")
    except ImportError:
        log.error("ccxt not installed. Run: pip install ccxt")
        raise


def fetch_forex(symbol: str, timeframe: str, limit: int = 1000) -> pd.DataFrame:
    try:
        import oandapyV20
        import oandapyV20.endpoints.instruments as instruments

        tf_map = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
                  "1h": "H1", "4h": "H4", "1d": "D"}
        granularity = tf_map.get(timeframe, "H1")

        client = oandapyV20.API(
            access_token=Config.OANDA_API_KEY,
            environment=Config.OANDA_ENVIRONMENT,
        )
        params = {"granularity": granularity, "count": limit}
        r = instruments.InstrumentsCandles(instrument=symbol, params=params)
        client.request(r)

        rows = []
        for candle in r.response["candles"]:
            if candle["complete"]:
                rows.append({
                    "timestamp": pd.to_datetime(candle["time"]),
                    "open":  float(candle["mid"]["o"]),
                    "high":  float(candle["mid"]["h"]),
                    "low":   float(candle["mid"]["l"]),
                    "close": float(candle["mid"]["c"]),
                    "volume": int(candle["volume"]),
                })
        df = pd.DataFrame(rows).set_index("timestamp")
        return df
    except ImportError:
        log.error("oandapyV20 not installed. Run: pip install oandapyV20")
        raise


def fetch_demo_data(symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
    """
    Generates realistic synthetic OHLCV data for backtesting
    when no API credentials are configured.
    """
    tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
                  "1h": 60, "4h": 240, "1d": 1440}
    mins = tf_minutes.get(timeframe, 60)
    n = (days * 1440) // mins

    np.random.seed(42)
    log_returns = np.random.normal(0.0001, 0.002, n)

    # Inject trend regimes
    for i in range(0, n, n // 6):
        direction = np.random.choice([-1, 1])
        length = n // 6
        log_returns[i:i+length] += direction * 0.0003

    prices = 100 * np.exp(np.cumsum(log_returns))

    high   = prices * (1 + np.abs(np.random.normal(0, 0.003, n)))
    low    = prices * (1 - np.abs(np.random.normal(0, 0.003, n)))
    open_  = np.roll(prices, 1); open_[0] = prices[0]
    volume = np.random.randint(1000, 50000, n).astype(float)

    end   = datetime.now()
    start = end - timedelta(minutes=mins * n)
    idx   = pd.date_range(start=start, periods=n, freq=f"{mins}min")

    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": prices, "volume": volume,
    }, index=idx)


def get_data(symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
    limit = max(500, (days * 1440) // {"1m":1,"5m":5,"15m":15,"30m":30,
                                        "1h":60,"4h":240,"1d":1440}.get(timeframe, 60))
    broker = Config.BROKER.lower()

    has_creds = {
        "binance": bool(Config.BINANCE_API_KEY),
        "kraken":  bool(Config.KRAKEN_API_KEY),
        "forex":   bool(Config.OANDA_API_KEY),
    }

    if not has_creds.get(broker, False):
        log.warning(f"No API credentials found for {broker}. Using synthetic demo data.")
        return fetch_demo_data(symbol, timeframe, days)

    fetchers = {"binance": fetch_binance, "kraken": fetch_kraken, "forex": fetch_forex}
    return fetchers[broker](symbol, timeframe, limit)


# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["high"] - df["low"]
    hc  = (df["high"] - df["close"].shift()).abs()
    lc  = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up   = df["high"].diff()
    down = -df["low"].diff()

    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr_val = atr(df, period)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=period, adjust=False).mean() / tr_val
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / tr_val

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(span=period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ema(df["close"], Config.EMA_FAST)
    df["ema_slow"] = ema(df["close"], Config.EMA_SLOW)
    df["atr"]      = atr(df, Config.ATR_PERIOD)
    df["atr_ma"]   = df["atr"].rolling(Config.ATR_PERIOD * 2).mean()
    df["adx"]      = adx(df, Config.ADX_PERIOD)
    return df.dropna()


# ─────────────────────────────────────────────
#  SIGNAL GENERATION
# ─────────────────────────────────────────────

def generate_signals(ltf_df: pd.DataFrame, htf_df: pd.DataFrame) -> pd.DataFrame:
    df = ltf_df.copy()

    # Resample HTF trend to LTF index
    htf_trend = (htf_df["ema_fast"] > htf_df["ema_slow"]).astype(int)
    htf_trend = htf_trend.reindex(df.index, method="ffill")

    # EMA crossover
    cross_up   = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift() <= df["ema_slow"].shift())
    cross_down = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift() >= df["ema_slow"].shift())

    # All filters
    trend_ok      = df["adx"] > Config.ADX_THRESHOLD
    volatility_ok = df["atr"] > df["atr_ma"]
    htf_bull      = htf_trend == 1
    htf_bear      = htf_trend == 0

    df["signal"] = 0
    df.loc[cross_up   & trend_ok & volatility_ok & htf_bull, "signal"] =  1   # BUY
    df.loc[cross_down & trend_ok & volatility_ok & htf_bear, "signal"] = -1   # SELL

    # Stop loss and take profit levels
    df["sl"] = np.where(
        df["signal"] ==  1, df["close"] - Config.ATR_SL_MULT * df["atr"],
        np.where(df["signal"] == -1, df["close"] + Config.ATR_SL_MULT * df["atr"], np.nan)
    )
    df["tp"] = np.where(
        df["signal"] ==  1, df["close"] + Config.ATR_TP_MULT * df["atr"],
        np.where(df["signal"] == -1, df["close"] - Config.ATR_TP_MULT * df["atr"], np.nan)
    )
    return df


# ─────────────────────────────────────────────
#  BACKTESTER
# ─────────────────────────────────────────────

class Backtester:
    def __init__(self, df: pd.DataFrame):
        self.df     = df
        self.equity = Config.INITIAL_CAPITAL
        self.trades = []
        self.equity_curve = []

    def run(self) -> dict:
        position   = None
        entry_price = sl = tp = 0.0

        for i, (ts, row) in enumerate(self.df.iterrows()):
            self.equity_curve.append({"timestamp": ts, "equity": self.equity})

            # Check exit on open position
            if position is not None:
                hit_sl = (position ==  1 and row["low"]  <= sl) or \
                         (position == -1 and row["high"] >= sl)
                hit_tp = (position ==  1 and row["high"] >= tp) or \
                         (position == -1 and row["low"]  <= tp)

                if hit_sl or hit_tp:
                    exit_price = sl if hit_sl else tp
                    pnl_pct    = (exit_price - entry_price) / entry_price * position
                    pnl_dollar = self.equity * Config.RISK_PCT / 100 * pnl_pct / (Config.ATR_SL_MULT * row["atr"] / entry_price)
                    self.equity += pnl_dollar

                    self.trades.append({
                        "entry_time":  entry_time,
                        "exit_time":   ts,
                        "direction":   "LONG" if position == 1 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price":  exit_price,
                        "sl":          sl,
                        "tp":          tp,
                        "pnl":         pnl_dollar,
                        "result":      "WIN" if (hit_tp) else "LOSS",
                    })
                    position = None

            # New entry signal
            if position is None and row["signal"] != 0:
                position    = row["signal"]
                entry_price = row["close"]
                entry_time  = ts
                sl          = row["sl"]
                tp          = row["tp"]

        return self._stats()

    def _stats(self) -> dict:
        if not self.trades:
            return {"error": "No trades generated"}

        trades_df = pd.DataFrame(self.trades)
        wins      = trades_df[trades_df["result"] == "WIN"]
        losses    = trades_df[trades_df["result"] == "LOSS"]

        equity_df  = pd.DataFrame(self.equity_curve).set_index("timestamp")
        peak       = equity_df["equity"].cummax()
        drawdown   = (equity_df["equity"] - peak) / peak * 100
        max_dd     = drawdown.min()

        returns    = trades_df["pnl"] / Config.INITIAL_CAPITAL
        sharpe     = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        return {
            "total_trades":    len(trades_df),
            "win_rate":        round(len(wins) / len(trades_df) * 100, 2),
            "total_return":    round((self.equity - Config.INITIAL_CAPITAL) / Config.INITIAL_CAPITAL * 100, 2),
            "final_equity":    round(self.equity, 2),
            "sharpe_ratio":    round(sharpe, 2),
            "max_drawdown":    round(max_dd, 2),
            "avg_win":         round(wins["pnl"].mean(), 2)   if len(wins)   > 0 else 0,
            "avg_loss":        round(losses["pnl"].mean(), 2) if len(losses) > 0 else 0,
            "profit_factor":   round(wins["pnl"].sum() / abs(losses["pnl"].sum()), 2) if len(losses) > 0 else float("inf"),
            "trades_df":       trades_df,
            "equity_curve":    pd.DataFrame(self.equity_curve).set_index("timestamp"),
        }


# ─────────────────────────────────────────────
#  CHART
# ─────────────────────────────────────────────

def plot_results(df: pd.DataFrame, stats: dict, symbol: str):
    trades     = stats["trades_df"]
    equity     = stats["equity_curve"]

    fig = plt.figure(figsize=(16, 12), facecolor="#0f0f0f")
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.08,
                            height_ratios=[3, 1, 1, 1.5])

    ax1 = fig.add_subplot(gs[0])  # Price + EMAs + signals
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # ADX
    ax3 = fig.add_subplot(gs[2], sharex=ax1)  # ATR
    ax4 = fig.add_subplot(gs[3])               # Equity curve

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="#888", labelsize=8)
        ax.spines[:].set_color("#333")
        ax.yaxis.label.set_color("#888")

    # ── Price panel ───────────────────────────
    ax1.plot(df.index, df["close"],    color="#555",  lw=0.8, label="Price")
    ax1.plot(df.index, df["ema_fast"], color="#378ADD", lw=1.2, label=f"EMA {Config.EMA_FAST}")
    ax1.plot(df.index, df["ema_slow"], color="#E85D24", lw=1.2, label=f"EMA {Config.EMA_SLOW}")

    # Plot trade entries/exits
    for _, t in trades.iterrows():
        color  = "#1D9E75" if t["direction"] == "LONG" else "#D85A30"
        marker = "^" if t["direction"] == "LONG" else "v"
        ax1.scatter(t["entry_time"], t["entry_price"], color=color,
                    marker=marker, s=60, zorder=5)
        ax1.scatter(t["exit_time"],  t["exit_price"],  color="#aaa",
                    marker="x", s=40, zorder=5)

    ax1.set_ylabel("Price", color="#888")
    ax1.legend(loc="upper left", fontsize=8, facecolor="#1a1a1a",
               labelcolor="white", framealpha=0.7)

    title = (f"{symbol} — {Config.LTF} | "
             f"Return: {stats['total_return']}%  "
             f"Win Rate: {stats['win_rate']}%  "
             f"Sharpe: {stats['sharpe_ratio']}  "
             f"Max DD: {stats['max_drawdown']}%")
    ax1.set_title(title, color="white", fontsize=10, pad=8)

    # ── ADX panel ─────────────────────────────
    ax2.plot(df.index, df["adx"], color="#9F7FD4", lw=1)
    ax2.axhline(Config.ADX_THRESHOLD, color="#555", lw=0.8, linestyle="--")
    ax2.set_ylabel("ADX", color="#888")
    ax2.set_ylim(0, 60)

    # ── ATR panel ─────────────────────────────
    ax3.plot(df.index, df["atr"],    color="#E8B84B", lw=1,   label="ATR")
    ax3.plot(df.index, df["atr_ma"], color="#555",    lw=0.8, linestyle="--", label="ATR MA")
    ax3.set_ylabel("ATR", color="#888")
    ax3.legend(loc="upper left", fontsize=7, facecolor="#1a1a1a",
               labelcolor="white", framealpha=0.7)

    # ── Equity curve ──────────────────────────
    ax4.plot(equity.index, equity["equity"], color="#1D9E75", lw=1.5)
    ax4.fill_between(equity.index, equity["equity"],
                     Config.INITIAL_CAPITAL, alpha=0.15,
                     color="#1D9E75" if stats["total_return"] >= 0 else "#D85A30")
    ax4.axhline(Config.INITIAL_CAPITAL, color="#555", lw=0.8, linestyle="--")
    ax4.set_ylabel("Equity ($)", color="#888")
    ax4.set_xlabel("Date", color="#888")

    plt.tight_layout()
    out = f"/mnt/user-data/outputs/backtest_{symbol.replace('/', '_')}_{Config.LTF}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    log.info(f"Chart saved → {out}")
    plt.close()
    return out


# ─────────────────────────────────────────────
#  LIVE TRADING (skeleton)
# ─────────────────────────────────────────────

class LiveTrader:
    """
    Live trading loop. Polls for new candles and fires orders
    when all confirmation layers align.
    """
    def __init__(self):
        self.position = None

    def place_order(self, direction: str, entry: float, sl: float, tp: float):
        """
        Override this method with your broker's order API.
        Binance:  ccxt exchange.create_order(...)
        Kraken:   ccxt exchange.create_order(...)
        Forex:    oandapyV20 orders endpoint
        """
        log.info(f"ORDER → {direction}  entry={entry:.5f}  SL={sl:.5f}  TP={tp:.5f}")

    def run(self):
        log.info(f"Live trader started — {Config.BROKER} | {Config.SYMBOL} | {Config.LTF}")
        while True:
            try:
                ltf_df  = get_data(Config.SYMBOL, Config.LTF,  days=60)
                htf_df  = get_data(Config.SYMBOL, Config.HTF,  days=120)
                ltf_ind = compute_indicators(ltf_df)
                htf_ind = compute_indicators(htf_df)
                sig_df  = generate_signals(ltf_ind, htf_ind)

                latest = sig_df.iloc[-1]
                if latest["signal"] != 0 and self.position is None:
                    direction = "LONG" if latest["signal"] == 1 else "SHORT"
                    self.place_order(direction, latest["close"], latest["sl"], latest["tp"])
                    self.position = latest["signal"]

            except Exception as e:
                log.error(f"Live loop error: {e}")

            # Sleep until next candle
            tf_seconds = {"1m":60,"5m":300,"15m":900,"30m":1800,
                          "1h":3600,"4h":14400,"1d":86400}
            time.sleep(tf_seconds.get(Config.LTF, 3600))


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def run_backtest():
    log.info(f"══ Backtest ══  broker={Config.BROKER}  symbol={Config.SYMBOL}  "
             f"LTF={Config.LTF}  HTF={Config.HTF}")

    ltf_df  = get_data(Config.SYMBOL, Config.LTF,  days=Config.BACKTEST_DAYS)
    htf_df  = get_data(Config.SYMBOL, Config.HTF,  days=Config.BACKTEST_DAYS * 2)

    ltf_ind = compute_indicators(ltf_df)
    htf_ind = compute_indicators(htf_df)

    sig_df  = generate_signals(ltf_ind, htf_ind)

    bt      = Backtester(sig_df)
    stats   = bt.run()

    if "error" in stats:
        log.warning(stats["error"])
        return

    log.info("─" * 50)
    log.info(f"  Total trades   : {stats['total_trades']}")
    log.info(f"  Win rate       : {stats['win_rate']}%")
    log.info(f"  Total return   : {stats['total_return']}%")
    log.info(f"  Final equity   : ${stats['final_equity']:,.2f}")
    log.info(f"  Sharpe ratio   : {stats['sharpe_ratio']}")
    log.info(f"  Max drawdown   : {stats['max_drawdown']}%")
    log.info(f"  Profit factor  : {stats['profit_factor']}")
    log.info(f"  Avg win        : ${stats['avg_win']:,.2f}")
    log.info(f"  Avg loss       : ${stats['avg_loss']:,.2f}")
    log.info("─" * 50)

    chart = plot_results(sig_df, stats, Config.SYMBOL)
    return stats, chart


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "backtest"
    if mode == "live":
        LiveTrader().run()
    else:
        run_backtest()
