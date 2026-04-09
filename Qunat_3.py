import os
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

ticker = input("Enter ticker (AAPL, TSLA, MSFT): ").upper()

# Optional Alpha Vantage key
alpha_key = "YOUR_ALPHA_VANTAGE_KEY"  # leave or replace
forecast_days = 7
num_simulations = 500

cache_file = f"{ticker}_data.csv"

def clean_dataframe(df):
    """Ensure numeric columns + correct price column"""

    df.columns = [c.strip() for c in df.columns]

    # convert everything numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # detect price column automatically
    if "Adj Close" in df.columns:
        price_col = "Adj Close"
    elif "Close" in df.columns:
        price_col = "Close"
    else:
        raise Exception("No price column found")

    return df, price_col

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

        df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
        df = df.rename(columns={"5. adjusted close": "Adj Close"})
        df.index = pd.to_datetime(df.index)

        return df.sort_index()

    except:
        return None

def fetch_yahoo(symbol):
    print("Downloading from Yahoo...")

    for i in range(5):
        try:
            data = yf.download(symbol, period="5y", progress=False)
            if len(data) > 100:
                return data
        except:
            pass

        wait = 10 + i * 10
        print(f"Retrying in {wait}s...")
        time.sleep(wait)

    return None

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

prices = df[price_col].dropna()

log_returns = np.log(prices / prices.shift(1)).dropna()

mu = log_returns.mean()
sigma = log_returns.std()

print("Drift:", mu)
print("Volatility:", sigma)

S0 = prices.iloc[-1]

simulations = np.zeros((forecast_days, num_simulations))

for i in range(num_simulations):
    path = [S0]

    for _ in range(forecast_days):
        shock = np.random.normal(
            (mu - 0.5 * sigma**2),
            sigma
        )
        path.append(path[-1] * np.exp(shock))

    simulations[:, i] = path[1:]

mean_path = simulations.mean(axis=1)
lower = np.percentile(simulations, 5, axis=1)
upper = np.percentile(simulations, 95, axis=1)

print("\nForecast:")
for i in range(forecast_days):
    print(
        f"Day {i+1}: "
        f"Expected={mean_path[i]:.2f}, "
        f"Low={lower[i]:.2f}, "
        f"High={upper[i]:.2f}"
    )

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