import numpy as np

# function that, from a df in input, it returns random normalized weights

def give_weights(df):
    rand = np.random.random(len(df.columns))
    rand /= rand.sum() # normalizing the array
    return rand

# function that computs all the portfolio under the efficient frontier

def efficient_frontier(df, ret_df, weights = [], ptf_exp_1 =[], ptf_std_1 = []):
    for i in range(2000):
        W_1 = give_weights(df)
        weights.append(W_1)
        ptf_exp_1.append(W_1.dot(ret_df.mean())*252) # annualizing the return
        ptf_std_1.append(W_1.dot(ret_df.cov().dot(W_1))*252)
    return weights, ptf_exp_1, ptf_std_1



# function of sharpe ratio: from a df that contains the ptf return and the ptf risk on the first two columns
# compute the sharpe ratio
def sharpe_pf(df):
    # first colum of df is the return, second one is the risk
    SR = df.iloc[:,0]/ df.iloc[:,1]
    return SR

