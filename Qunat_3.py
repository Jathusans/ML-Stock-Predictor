import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

ticker = input("Enter ticker (AAPL, TSLA, MSFT): ").upper().strip()

if ticker == "":
    raise ValueError("Invalid ticker")

alpha_key = "YOUR_ALPHA_VANTAGE_KEY"
forecast_days = 7
num_simulations = 500
trading_days = 252

cache_file = f"{ticker}_data.csv"

# ---------------------------
# CLEAN DATA (FIXED)
# ---------------------------
def clean_dataframe(df):

    df.columns = [c.strip() for c in df.columns]

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # FIXED

    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise Exception("No price column found")

    df = df.dropna(subset=[price_col])

    return df, price_col

# ---------------------------
# ALPHA FETCH (FIXED)
# ---------------------------
def fetch_alpha(symbol):
    if alpha_key == "YOUR_ALPHA_VANTAGE_KEY":
        return None

    print("Trying Alpha Vantage...")

    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={symbol}"
        f"&outputsize=full"
        f"&apikey={alpha_key}"
    )

    try:
        r = requests.get(url, timeout=20)
        data = r.json()

        if "Time Series (Daily)" not in data:
            return None

        df = pd.DataFrame.from_dict(
            data["Time Series (Daily)"], orient="index"
        )

        df = df.rename(columns={"5. adjusted close": "Adj Close"})
        df.index = pd.to_datetime(df.index)

        return df.sort_index()

    except:
        return None

# ---------------------------
# YAHOO FETCH (FIXED)
# ---------------------------
def fetch_yahoo(symbol):
    print("Downloading from Yahoo...")

    for i in range(5):
        try:
            data = yf.download(symbol, period="5y", progress=False)

            if data is not None and len(data) > 100:  # FIXED
                return data

        except:
            pass

        wait = 5 + i * 5
        print(f"Retrying in {wait}s...")
        time.sleep(wait)

    return None

# ---------------------------
# LOAD DATA
# ---------------------------
df = None

if os.path.exists(cache_file):
    try:
        print("Loading cached data...")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        df, price_col = clean_dataframe(df)
    except:
        print("Bad cache — deleting.")
        os.remove(cache_file)
        df = None

if df is None:
    df = fetch_alpha(ticker)

    if df is None:
        df = fetch_yahoo(ticker)

    if df is None:
        raise Exception("Could not download data.")

    df.to_csv(cache_file)
    df, price_col = clean_dataframe(df)

print("Data shape:", df.shape)

# ---------------------------
# RETURNS (FIXED SCALING)
# ---------------------------
prices = df[price_col]

log_returns = np.log(prices / prices.shift(1)).dropna()

mu = log_returns.mean()
sigma = log_returns.std()

dt = 1 / trading_days  # FIXED

print("Drift (daily):", mu)
print("Volatility (daily):", sigma)

# ---------------------------
# MONTE CARLO (FIXED)
# ---------------------------
S0 = prices.iloc[-1]

simulations = np.zeros((forecast_days, num_simulations))

for i in range(num_simulations):

    path = [S0]

    for _ in range(forecast_days):

        shock = np.random.normal(
            (mu - 0.5 * sigma**2) * dt,   # FIXED
            sigma * np.sqrt(dt)           # FIXED
        )

        next_price = path[-1] * np.exp(shock)
        path.append(next_price)

    simulations[:, i] = path[1:]

# ---------------------------
# STATS
# ---------------------------
mean_path = simulations.mean(axis=1)
lower = np.percentile(simulations, 5, axis=1)
upper = np.percentile(simulations, 95, axis=1)

# ---------------------------
# OUTPUT
# ---------------------------
print("\nForecast:")

for i in range(forecast_days):
    print(
        f"Day {i+1}: "
        f"Expected={mean_path[i]:.2f}, "
        f"Low={lower[i]:.2f}, "
        f"High={upper[i]:.2f}"
    )

# ---------------------------
# PLOT
# ---------------------------
plt.figure(figsize=(10, 5))

for i in range(50):
    plt.plot(simulations[:, i], alpha=0.1)

plt.plot(mean_path, linewidth=3, label="Expected")
plt.fill_between(range(forecast_days), lower, upper, alpha=0.3)

plt.title(f"{ticker} Forecast")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid()

plt.show()