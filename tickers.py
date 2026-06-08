from pathlib import Path
import csv

tickers = [
    row["ticker"].strip().upper()
    for row in csv.DictReader(Path(__file__).with_name("tickers.csv").open())
    if row["ticker"].strip()
]
