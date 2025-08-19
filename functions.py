import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

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



# function of lagged returns --> input df and number of lagged provides

def lagit(df, lags):
    for i in range(1, lags+1):
        df['Lag_'+str(i)] = df['ret'].shift(i)
    return ['Lag_'+str(i) for i in range(1, lags+1)]
 


 # function of model of regression 

 
# IF WE WANT TO EXPLAIN BETTER THE MODEL....


# we can compare the prediction with the direction of the market
 # 7) Compute the return based on the strategy of the model

# 8) plot the comparison




def Logistic_Model(df, X_train, y_train, X_test, y_test):
    
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)

    X_test['prediction_LR'] = model.predict(X_test)
    
    X_test['intraret'] = df.intraret[X_test.index[0]:]
    X_test['ret'] = df.ret[X_test.index[0]:]


    X_test['strat'] = X_test['prediction_LR'] * X_test.intraret
    print(X_test)
    print((X_test[['strat', 'ret']]+1).cumprod()-1)


    (X_test[['strat', 'ret']]+1).cumprod().plot()
    plt.show()


