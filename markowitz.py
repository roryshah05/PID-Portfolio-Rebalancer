import numpy as np
import pandas as pd
from scipy.optimize import minimize
from data_pipeline import get_prices

def compute_inputs(returns):
    expected_returns = returns.mean()
    covariance_matrix = returns.cov()
    
    return expected_returns, covariance_matrix

def optimize_weights(mu, Sigma, prev_weights=None, lambda_turnover=0.1):
    N = len(mu)
    x0 = np.ones(N) / N
    bounds = [(0,1)] * N
    def objective_function(w):
        if prev_weights is not None:
            fun = (w.T @ mu / np.sqrt(w.T @ Sigma @ w)) - lambda_turnover * np.sum(np.abs(w - prev_weights))
        else:
            fun = (w.T @ mu / np.sqrt(w.T @ Sigma @ w))
        return -fun
    result = minimize(objective_function, x0, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    weights = result.x
    weights[weights < 1e-6] = 0
    weights = weights / weights.sum()
    return weights

def get_optimal_weights(close, window, rebalance_freq):
    returns = close.pct_change().dropna()
    rebalance_dates = returns.resample(rebalance_freq).first().index
    result = pd.DataFrame(columns=close.columns)
    prev_weights = None

    for date in rebalance_dates:
        window_returns = returns.loc[:date].tail(window)
        if len(window_returns) < window:
            continue
        mu, Sigma = compute_inputs(window_returns)
        result.loc[date] = optimize_weights(mu, Sigma, prev_weights)
        prev_weights = result.loc[date].values
    
    return result

if __name__ == "__main__":
    tickers = ['SPY', 'EFA', 'TLT', 'TIP', 'GLD', 'VNQ']
    start = "2012-01-01"
    end = "2026-03-01"
    close = get_prices(tickers, start, end)
    returns = close.pct_change()
    values = get_optimal_weights(close, window=252, rebalance_freq='MS')
    print(values[:5])
