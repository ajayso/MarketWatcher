# Imports
#Importing dependencies 
import pandas as pd
import numpy as np
import talib as ta
import math
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10) #change this if you want to reduce the plot images
import locale
from locale import atof
locale.setlocale(locale.LC_NUMERIC, '') 

#Quandl dependency with API key
import quandl
quandl.ApiConfig.api_key = "f7_JWui3ztp2Yxh_xddT"

scriptcode = "ASHOKLEY"
df = quandl.get("NSE/" + scriptcode)
data = pd.DataFrame(df, columns = ['Date', 'Open','High' ,'Low', 'Last','Close', 'Total Trade Quantity' ,'Turnover (Lacs)'])
data['Date'] = data['Date'].apply(pd.to_datetime)
data= data.set_index("Date")
data = data.dropna(axis=0)

from MainProcess import DataManager
dm = DataManager()
datacopy = dm.Load(data,5000, "Close", 0.9)