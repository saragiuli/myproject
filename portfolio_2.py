
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import functions as f

# Microsoft
# Coca Cola
# MMM

tickers = ["MSFT","MMM", "KO"]
df = yf.download(tickers,start='2015-01-01', auto_adjust=False)['Adj Close']
#df.xs('AAPL', axis = 1, level = 'Ticker')


print(df.head(10))


ret_df = df.pct_change()
ret_df.dropna(inplace=True)
print("ret_df", ret_df)

# Comparing how the stocks performed since 2015

cum_ret = (ret_df + 1).cumprod() -1
print(cum_ret)


cum_ret.plot()
plt.title("Return of stocks from 2015 until now")
plt.show()

# mean --> daily returns

print(ret_df.mean())

# standard deviation --> risk 

print(ret_df.std())

# covariance matrix --> direction of relation through assets

print(ret_df.cov())

# correlation matrix --> strenght of relation through assets

print(ret_df.corr())

# weights vector for the portfolio; scalable formula

W = np.ones(len(ret_df.columns))/len(ret_df.columns)

print("weights:", W)

# calculate the PTF expected return and risk

ptf_exp = W.dot(ret_df.mean())
ptf_std = (W.dot(ret_df.cov().dot(W)))**(1/2)

print("ptf_exp:{}, & ptf_std: {}".format(ptf_exp, ptf_std))

# total return of ptf

ret_df.mean(axis = 1)
ret_ptf = (ret_df.mean(axis = 1) +1).cumprod() -1

print("ret_ptf:", ret_ptf)

ret_ptf.plot()
plt.title("Total return of ptf")
plt.show()



############ SECOND PART ############

# EFFICIENT FRONTIER AND MINIMUM VARIANCE PORTFOLIO

# now we use the same stocks but we need to establish which are the best weights 
# for a minimun variance optimization portfolio

# 1) compute random weights --> use the function that computes normalized random weights

print(f.give_weights(ret_df))


# 2) construct the ptf and its weights --> from a random weights, we compute over 2000 portfolio with the formulas used before

weights, ptf_exp_1, ptf_std_1 = f.efficient_frontier(df, ret_df)

# 3) create the dataframe of ptf 

tog =pd.DataFrame({'ptf_exp':ptf_exp_1,
                   'ptf_stds': ptf_std_1,
                   'weights': weights})

print(tog)

# 4) Compute the sharpes ratio of each portfolios

# Which one is the optimal portfolio ? 

sharpes = f.sharpe_pf(tog)

print("sharpes of all portfolios computed:", sharpes)

# 5) apply the largest function on the sharpe ratio to find the best 
#    sharpe ratio (that is on the efficient frontier)

# find the index where the sharpes is largest, and than loc the portfolio there

ptf_opt = tog.loc[sharpes.nlargest(1).index]

print("the ptf opt:{}, \n and the weights of respective (Coca Cola, MMM and Microsoft) are:{}".format(ptf_opt, ptf_opt.weights))

# 6) let's find the minimum variance portfolio 

# from the tog Dataframe, we just check the n smallest value and pass the column standard deviation (the minimum risk)
# with this, we find the minimum variance portfolio

ptf_min_var = tog.nsmallest(1, 'ptf_stds')

print("\n The minimum standard deviation is:", ptf_min_var)


# -------- plotting of all the portfolio computed

tog.plot(x='ptf_stds', y='ptf_exp', kind = 'scatter')
plt.title("Ptf -- Max Sharpe & min Risk evidence")
plt.scatter(ptf_opt['ptf_stds'], ptf_opt['ptf_exp'], color = 'red', s=100, label = 'Ptf - Max Sharpe', edgecolors = 'black')
plt.scatter(ptf_opt['ptf_stds'],ptf_min_var['ptf_exp'], color = 'yellow', s=100, label = 'Ptf - Min Risk', edgecolors = 'black')
plt.legend()
plt.show()


