import yfinance as yf
import warnings
import logging

logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Downloading the daily close price off yfinance
def get_prices(tickers, start, end):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = yf.download(tickers, start, end, progress=False)
    close = data["Close"]
    return close

# Computing price-weighted allocation
def get_weights(close):
    weight = close.div(close.sum(axis=1), axis=0)
    return weight

if __name__ == "__main__":
    start = "2020-01-01"
    end = "2024-01-01"
    weight = get_weights(get_prices(['LMT', 'BA', 'GLD', 'SPY'], start, end))
    print(weight)
    print(weight.sum(axis=1))