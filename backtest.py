from pid_controller import PIDController
from data_pipeline import get_prices
from markowitz import get_optimal_weights
import numpy as np

# Initializing parameters
target_weight = np.array([0.25, 0.15, 0.20, 0.15, 0.15, 0.10])
Kp = 0.5
Ki = 0.01
Kd = 0.1
minintegral = -1.0
maxintegral = 1.0
dt = 1
transaction_cost = 0.001 # 0.1% Transaction Cost
start = "2012-01-01"
end = "2026-03-01"
tickers = ['SPY', 'EFA', 'TLT', 'TIP', 'GLD', 'VNQ']

# Backtest for PID controller
def run_backtest(target_weight=None, Kp=0.5, Ki=0.01, Kd=0.1, minintegral=-1.0, maxintegral=1.0, dt=1, transaction_cost=0.001, returns=None, close=None, window=252, rebalance_freq='MS'):
    portfolio_value = 1.0
    u_tracker = []
    portfolio_value_tracker = []
    dates = []
    weights_tracker = []

    if target_weight is not None:
        current_weights = target_weight.copy()
    else:
        current_weights = np.ones(len(returns.columns)) / len(returns.columns)

    pid = PIDController(Kp, Ki, Kd, minintegral, maxintegral)

    if close is not None:
        optimal_weights = get_optimal_weights(close, window, rebalance_freq)

    # Main loop
    returns_array = returns.values
    dates_index = returns.index

    for i in range(len(returns_array)):
        daily_returns = returns_array[i]
        date = dates_index[i]

        if close is not None:
            valid_dates = optimal_weights.index[optimal_weights.index <= date]
            if len(valid_dates) == 0:
                continue
            latest_date = valid_dates.max()
            target_weight = optimal_weights.loc[latest_date].values

        if np.any(np.isnan(daily_returns)):
            continue
        current_weights = (current_weights * (1 + daily_returns)) / np.sum((current_weights * (1 + daily_returns))) # Update the current weights based on returns
        error = target_weight - current_weights # Computing initial error
        u = pid.step(error, dt) # Compute total u
        portfolio_return = np.dot(daily_returns, current_weights) # Compute the entire portfolio return
        portfolio_value = portfolio_value * (1 + portfolio_return) - (transaction_cost * np.sum(np.abs(u))) # Update the value of the portfolio given transaction costs
        current_weights = current_weights + u
        current_weights = current_weights / np.sum(current_weights)
        u_tracker.append(u) # Store the updated u value for analysis
        portfolio_value_tracker.append(portfolio_value) # Store the updated portfolio value
        dates.append(date) # Store the updated date
        weights_tracker.append(current_weights.copy()) # Store the updated weights
    return(u_tracker, dates, portfolio_value_tracker, weights_tracker)

# Buy and Hold strategy comparison
def run_buy_and_hold(start, end, tickers, target_weight):
    portfolio_value = 1.0
    portfolio_value_tracker = []
    dates = []
    weights_tracker = []

    close = (get_prices(tickers, start, end))
    current_weights = target_weight

    returns = close.pct_change()

    for date, daily_returns in returns.iterrows():
        if np.any(np.isnan(daily_returns)):
            continue
        current_weights = (current_weights * (1 + daily_returns)) / np.sum((current_weights * (1 + daily_returns)))
        portfolio_return = np.dot(daily_returns, current_weights)
        portfolio_value = portfolio_value * (1 + portfolio_return)
        portfolio_value_tracker.append(portfolio_value)
        dates.append(date)
        weights_tracker.append(current_weights.copy())
    return(dates, portfolio_value_tracker, weights_tracker)


# 5% Threshold Strategy Comparison
threshold = 0.05
def run_threshold(target_weight, transaction_cost, start, end, tickers, threshold):
    portfolio_value = 1.0
    u_tracker = []
    portfolio_value_tracker = []
    dates = []
    weights_tracker = []

    close = (get_prices(tickers, start, end))
    current_weights = target_weight

    returns = close.pct_change()

    # Main loop
    for date, daily_returns in returns.iterrows():
        if np.any(np.isnan(daily_returns)):
            continue
        current_weights = (current_weights * (1 + daily_returns)) / np.sum((current_weights * (1 + daily_returns)))
        error = target_weight - current_weights
        u =  np.where(np.abs(error) > threshold, error, 0)
        portfolio_return = np.dot(daily_returns, current_weights) # Compute the entire portfolio return
        portfolio_value = portfolio_value * (1 + portfolio_return) - (transaction_cost * np.sum(np.abs(u))) # Update the value of the portfolio given transaction costs
        current_weights = current_weights + u
        current_weights = current_weights / np.sum(current_weights)
        u_tracker.append(u) # Store the updated u value for analysis
        portfolio_value_tracker.append(portfolio_value) # Store the updated portfolio value
        dates.append(date) # Store the updated date
        weights_tracker.append(current_weights.copy()) # Store the updated weights
    return(u_tracker, dates, portfolio_value_tracker, weights_tracker)

# Test Calls
if __name__ == "__main__":
    close = get_prices(tickers, start, end)
    returns = close.pct_change()
    u, dates, values, weights = run_backtest(Kp=Kp, Ki=Ki, Kd=Kd, minintegral=minintegral, maxintegral=maxintegral, dt=dt, transaction_cost=transaction_cost, returns=returns, close=close)
    #dates, values, weights = run_buy_and_hold(start, end, tickers)
    #dates, values, weights = run_threshold(target_weight, transaction_cost, start, end, tickers, threshold)
    print(values[:5])