
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import functions as f
from sklearn.model_selection import train_test_split

# Machine Learning portfolio which outperformed the S&P500
#
#
# input of the model: previous day returns (t-1, t-2, ...)
#
#
# Model predict the market movement. Logistic regression model
# the independent variables are: lagged returns (t-1,..)
# the dependent variables are: UP (+1), Down (0)
# probabilistic model --> predicted probability is above 0.5 --> model label +1 (UP)
#                     --> predicted probability is below 0.5 --> model label 0 (Down)   
#
#
# test --> if we follow those predictions, how overall performance would look like
#
#

# 1) download the price of S&P 500

tickers = ["^GSPC"]
df = yf.download(tickers,start='2010-01-01', auto_adjust=False)
df = df.xs('^GSPC', axis = 1, level = 'Ticker')

#print(df.head(10))

# 2) Add a column and compute the relative price change

df['ret'] = df.Close.pct_change()

df['intraret'] = df.Close / df.Open - 1


print(df)

# 3) Create a function that computes the lagged returns (previous day before)

# 4) Transform return in binary, transorm positive returns in 1 and negative returns in 0

df['direction'] = np.where(df.ret > 0, 1, 0)


# 5) Storing the independent variables

features = f.lagit(df, 3)
df.dropna(inplace=True)

print(df)

# 6) Starting the regression: storing the features (so the lagged returns on the independent variables, and the direction
# on the dependent variables)

X = df[features]
y = df['direction']

print(X, y)


f.Logistic_Model(df, X, y, X, y)


# IF WE WANT TO EXPLAIN BETTER THE MODEL....


'''
model = LogisticRegression(class_weight='balanced')
model.fit(X, y)


# we can compare the prediction with the direction of the market

df['prediction_LR'] = model.predict(X)
print(df)

# 7) Compute the return based on the strategy of the model

df['strat'] = df['prediction_LR'] * df.intraret

print((df[['strat', 'ret']]+1).cumprod()-1)

# 8) plot the comparison

(df[['strat', 'ret']]+1).cumprod().plot()
plt.show()
'''

############ STEP 2 --> NOT USING THE OVERFITTING, BUT SPLIT THE DATASET

# we've used the overfitting data set (we used all the dtaset to train), now we want to split the train data set and create an x train variable with the independent 
# and the independent variable testing data, and the same for the dependent variable
# test_size=0.3 means that 30% is for testing and 70% is for training
# shuffle = False --> we want to have subsequence data (with time series data)


# we've just prepared the data set splitting through first part of traing and second part of testing

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, shuffle = False)

print(X_test)

# Take the training data and train the model and than test that on my testing data and see if i'm going better or worst results


#X_test['intraret'] = df.intraret[X_test.index[0]:]
#X_test['ret'] = df.ret[X_test.index[0]:]

f.Logistic_Model(X_test, X_train, y_train, X_test, y_test)


'''
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

X_test['prediction_LR'] = model.predict(X_test)


# create a new column in the test set just to add the relative returns on the data (not the lagged ones)
# we need it to calculate the strategy (same as before with all the dataframe)

X_test['intraret'] = df.intraret[X_test.index[0]:]
X_test['ret'] = df.ret[X_test.index[0]:]


X_test['strat'] = X_test['prediction_LR'] * X_test.intraret

print(X_test)


print((X_test[['strat', 'ret']]+1).cumprod()-1)

# 8) plot the comparison

(X_test[['strat', 'ret']]+1).cumprod().plot()
plt.show()
'''

