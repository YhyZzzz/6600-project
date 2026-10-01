import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="SPY")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default="2026-12-31")
    p.add_argument("--minutes", type=int, default=15)
    p.add_argument("--feed", default="sip", choices=["sip", "iex"])
    p.add_argument("--adjustment", default="all", choices=["raw", "split", "dividend", "all"])
    p.add_argument("--out", default=None)
    return p.parse_args()


def fetch_year(client, args, start, end):
    req = StockBarsRequest(
        symbol_or_symbols=args.symbol,
        timeframe=TimeFrame(args.minutes, TimeFrameUnit.Minute),
        start=start,
        end=end,
        feed=DataFeed(args.feed),
        adjustment=Adjustment(args.adjustment),
    )
    return client.get_stock_bars(req).df


def main():
    args = parse_args()
    load_dotenv(ROOT / ".env")
    key, secret = os.getenv("APCA_API_KEY_ID"), os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in env or .env")

    client = StockHistoricalDataClient(key, secret)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) + timedelta(days=1)
    # free plan cannot query SIP data from the latest 15 minutes
    end = min(end, datetime.now(timezone.utc) - timedelta(minutes=16))

    frames = []
    for year in range(start.year, end.year + 1):
        s = max(start, datetime(year, 1, 1, tzinfo=timezone.utc))
        e = min(end, datetime(year + 1, 1, 1, tzinfo=timezone.utc))
        df = fetch_year(client, args, s, e)
        print(f"{year}: {len(df)} bars")
        if not df.empty:
            frames.append(df)

    df = pd.concat(frames).reset_index()
    df = df.drop(columns="symbol").rename(columns={"timestamp": "ts_utc"})
    df["ts_et"] = df["ts_utc"].dt.tz_convert("America/New_York")
    df = df.drop_duplicates("ts_utc").sort_values("ts_utc").reset_index(drop=True)
    df = df[["ts_utc", "ts_et", "open", "high", "low", "close", "volume", "trade_count", "vwap"]]

    out = Path(args.out) if args.out else ROOT / "data" / f"{args.symbol.lower()}_{args.minutes}min_{args.start[:4]}_{args.end[:4]}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved {len(df)} rows -> {out} ({df.ts_et.min()} .. {df.ts_et.max()})")


if __name__ == "__main__":
    main()
