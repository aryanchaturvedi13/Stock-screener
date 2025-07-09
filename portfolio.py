
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

t=int(input('How many companies are you willing to include in your portfolio '))
tickers = [input (f'Enter the companies you want to invest ').upper() for i in range(t)]
start_date = input("Enter the start date as (YYYY-MM-DD)  ")
end_date = input("Enter the end date (YYYY-MM-DD)  ")

trades=[]
buy_price=0
selling_price=0

# Initial capital per stock
total_cash = float(input('How much money do you want to include in your portfolio '))
pct_per_trade=float(input('How much percent of your available cash are you willing to use per trade '))

# Dictionary to store each stock's portfolio
portfolios = {}

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    close = data['Close']
    close=close.squeeze()

    data['SMA20'] = close.rolling(20).mean()
    data['std'] = close.rolling(20).std()
    data['lower_bb'] = data['SMA20'] - 2 * data['std']
    data['upper_bb'] = data['SMA20'] + 2 * data['std']
    data['zscore'] = (close - data['SMA20']) / data['std']

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Entry and Exit Rules
    data['Buy'] = (data['zscore'] < -1) & (close < data['lower_bb']) & (data['RSI'] < 30)
    data['Sell'] = (data['zscore'] > 0) | (close > data['upper_bb']) | (data['RSI'] > 70)

    # Backtest
    cash_per_trade = total_cash*pct_per_trade/100
    shares = 0
    position = 0
    portfolio = []

    for i in range(len(data)):
        if data['Buy'].iloc[i] and position == 0:
            shares = cash_per_trade // close.iloc[i]
            total_cash -= shares * close.iloc[i]
            buy_price=close.iloc[i]
            position = 1
        elif data['Sell'].iloc[i] and position == 1:
            total_cash += shares * close.iloc[i]
            selling_price=close.iloc[i]
            profit=(selling_price-buy_price)*shares
            trades.append(profit)
            shares = 0
            position = 0
        portfolio_value = total_cash + shares * close.iloc[i]
        portfolio.append(portfolio_value)

    data['Portfolio'] = portfolio
    portfolios[ticker] = data[['Portfolio']]

# Merge all portfolios into one DataFrame
merged = pd.concat(portfolios.values(), axis=1)
merged.columns = tickers
merged['Total'] = merged.sum(axis=1) / len(tickers)  # Equal-weighted

# Plot combined portfolio
plt.figure(figsize=(14, 6))
plt.plot(merged.index, merged['Total'], label='Combined Portfolio', linewidth=2)
for ticker in tickers:
    plt.plot(merged.index, merged[ticker], label=ticker, alpha=0.4)
plt.title("Performance of portfolio using Mean Reversion Strategy")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Use merged total portfolio
returns = merged['Total'].pct_change().dropna()
portfolio_value = merged['Total'].iloc[-1]
cumulative_return = (merged['Total'].iloc[-1] / merged['Total'].iloc[0]) - 1
annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
max_drawdown = ((merged['Total'].cummax() - merged['Total']) / merged['Total'].cummax()).max()
print("hello")
rf = 0
downside_returns = returns[returns < rf]
expected_returns = returns.mean()
downside_std = downside_returns.std()
sortino_ratio = ((expected_returns - rf) / downside_std) * np.sqrt(252) if downside_std != 0 else np.nan
trades = pd.Series(trades)
win_trades = trades[trades > 0]
loss_trades = trades[trades < 0]

win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
avg_profit = win_trades.mean() if not win_trades.empty else 0
avg_loss = loss_trades.mean() if not loss_trades.empty else 0
max_profit = trades.max() if not trades.empty else 0
max_loss = trades.min() if not trades.empty else 0

print("Performance Metrics")
print("-" * 30)
print(f'Total Portfolio value: {portfolio_value}')
print(f'Cumulative Return:   {cumulative_return:.2%}')
print(f'Annualized Return:   {annualized_return:.2%}')
print(f'Sharpe Ratio:        {sharpe_ratio:.2f}')
print(f'Max Drawdown:        {max_drawdown:.2%}')
print(f'Sortino Ratio: {sortino_ratio}')
print(f'Win rate : {win_rate}')
print(f'Avg profit: {avg_profit}')
print(f'Avg loss: {avg_loss}')
print(f'Max profit: {max_profit}')
print(f'Max loss: {max_loss}')
