import yfinance as yf

print(yf.__version__)

#df_ = yf.download('500X.AS', start = "2024-01-01")
#print(df_.xs('500X.AS', axis = 1, level = 'Ticker'))

# use the cross section function to reduce a level

df_2 = yf.download(['UIMT.DE', 'TSLA'], start = "2024-01-01", threads=False)
print(df_2.xs('UIMT.DE', axis = 1, level = 'Ticker'))

print(df_2)

# what if i need the adj close ? 

df = yf.download('AAPL', start = "2024-01-01", auto_adjust=False)
