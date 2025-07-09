from fileinput import close

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

ticker = 'ATGL.NS'
data = yf.download(ticker, start='2020-01-01', end='2025-01-01')
close=data['Close']
close=close.squeeze()

data['SMA20']= close.rolling(20).mean()
data['std']= close.rolling(20).std()
data['upper_bb']= data['SMA20']+2*data['std']
data['lower_bb']=data['SMA20']-2*data['std']

data['z_score']=(close-data['SMA20'])/data['std']

delta=close.diff()
gain=delta.clip(lower=0)
loss=-delta.clip(upper=0)
avg_gain=gain.rolling(14).mean()
avg_loss=loss.rolling(14).mean()
rs=avg_gain/avg_loss
data['RSI']=100-(100/1+rs)

data['buy']=(data['z_score']<-1) & (data['RSI']<30) & (close<data['lower_bb'])
data['sell']=(data['z_score']>0) | (data['RSI']>70) | (close>data['upper_bb'])

position=0
cash=100000
cash_per_trade=0.1*cash
shares=0
portfolio=[]

for i in range(len(data)):
    if data['buy'].iloc[i] and position==0:
        shares=cash_per_trade//close.iloc[i]
        cash-=shares*close.iloc[i]
        position=1
    elif data['sell'].iloc[i] and position==1:
        cash=cash+shares*close.iloc[i]
        shares=0
        position=0
    portfolio_value=cash+shares*close.iloc[i]
    portfolio.append(portfolio_value)

data['Portfolio']=portfolio

plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Portfolio'], label='Portfolio Value', linewidth=2)
plt.plot(data.index, data['Close'], label='Stock Price', alpha=0.5)
plt.title(f'Mean Reversion Strategy on {ticker}')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


returns = pd.Series(data['Portfolio']).pct_change().dropna()
cumulative_return = (data['Portfolio'].iloc[-1] / data['Portfolio'].iloc[0]) - 1
annualized_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
max_drawdown = ((data['Portfolio'].cummax() - data['Portfolio']) / data['Portfolio'].cummax()).max()
rf=0
downside_returns=returns[returns<rf]
expected_returns=returns.mean()
downside_std=downside_returns.std()

sortino_ratio= ((expected_returns-rf)/downside_std)*np.sqrt(252)

print("Performance Metrics")
print("-" * 30)
print(f'Cumulative Return:   {cumulative_return:.2%}')
print(f'Annualized Return:   {annualized_return:.2%}')
print(f'Sharpe Ratio:        {sharpe_ratio:.2f}')
print(f'Max Drawdown:        {max_drawdown:.2%}')
print(f'Sortino Ratio: {sortino_ratio}')

