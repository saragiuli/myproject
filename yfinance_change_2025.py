import yfinance as yf

print(yf.__version__)

df_ = yf.download('AAPL', start = "2024-01-01")
print(df_.xs('AAPL', axis = 1, level = 'Ticker'))



df_2 = yf.download('500X.AS', start = "2024-01-01", threads=False)
print(df_2.xs('500X.AS', axis = 1, level = 'Ticker'))