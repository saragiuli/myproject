
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# I need the ticker of the ISIN, to download from yfinance
#Ticker del mio portafoglio del 18/04/2025


'''
pesi attuali di Robo4 Revoulut

'500X.AS' -- SPDR S&P 500 ESG Leaders ETF--50%

'XDWT.MI' -- Xtrackers MSCI World Information Technology ETF-- 24,86%

'UIMM.DE'-- UBS MSCI World Socially Responsible -- 10,15 %

'MIVB.DE' -- Amundi Index MSCI Europe ARI PAB ETF -- 9,96 %

'UIMT.DE' -- UBS MSCI Pacific Socially Responsible (Dist) ETF -- 2,98%

'''
tickers = ['500X.AS', 'XDWT.MI', 'UIMM.DE', 'MIVB.DE', 'UIMT.DE']

#tickers = ['^GSPC','GC=F']


df = yf.download(tickers,start='2010-01-01', threads=False)['Close']

print(df.head(10))


ret_df = np.log(df/df.shift(1))
ret_df.cumsum().plot()
plt.show()


# compute the mean and the standard deviation --> risk

print("return_df_mean\n", ret_df.mean())
print("return_df_std\n", ret_df.std())


W = np.ones(len(ret_df.columns))/ np.ones(len(ret_df.columns)).sum()

print("\n ptf equally weighted\n", W)

# return of ptf


ret_1 = (W * ret_df.mean()).sum()
print("\n ret of ptf \n", ret_1)


def sharpe_pf(W, returns):
    pf_risk = (W.T.dot(returns.cov()).dot(W)) ** 0.5
    SR = W.T.dot(returns.mean()) / pf_risk
    return -SR




print("\n Sharpe of PTF\n", sharpe_pf(W, ret_df))

cons = ({"type":"eq", 
         "fun": lambda x: np.sum(x)-1})


# Bounds: No short selling, weights must be between 0 and 1
bounds = [(0, 1)] * len(W)

res = minimize(sharpe_pf, W, ret_df, bounds=bounds, constraints = cons)
print(res)
opt_W = res['x']

print("\n optimal weights \n", opt_W)

# Compare with the equally weighted



ptf_1 = ret_df.dot(opt_W).cumsum()
ptf_2 = ret_df.dot(W).cumsum()

ptf_1.plot()
ptf_2.plot()

plt.legend()
plt.show()

print("\n Optimal weighted portfolio\n", ptf_1)
print("\n Equally weighted portfolio\n", ptf_2)



# iterative approach


sharpes = []

for w in np.arange(0,1,0.01):
    weights = w, 1-w
    weights = np.array(weights)
    print(weights)
    sharpes.append(sharpe_pf(weights, ret_df))
   






