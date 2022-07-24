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
from sqlalchemy import create_engine
import mysql.connector
engine = create_engine("mysql+mysqlconnector://Savinay:savinay@35.185.177.170/Currencies")
print("SQL Connected with Currencies")

class CurrencyAnalyzer:
    
    def __init__(self,dataframe):
        #loads dataframe
        self.data = dataframe
        
        
        
    #Class method to return CorrMatrix, Indicator Values and Summary
        
    #indicator accepted values : USDINR,GBPINR,JPYINR,EURUSD,USDJPY,EURINR,GBPUSD
    def getCurrencyIndicators(self,indicator):
                
        #USD to Pound Sterling
        if(indicator == 'GBPUSD'):
            #gbpusd = pd.read_csv("Currencies/GBPUSD.csv")
            gbpusd = pd.read_sql("SELECT * FROM GBPUSD",con=engine)
            gbpusd = gbpusd.set_index("Date")
            gbpusd = gbpusd.rename(columns={'Open':'GBPUSD_Open','Low':'GBPUSD_Low','High':'GBPUSD_High'
                                            ,'Close':'GBPUSD_Close','Volume':'GBPUSD_Volume'})
            gbpusd_df = gbpusd.join(self.data,how='inner') #inner join with dataframe
            gbpusd_df = gbpusd_df.dropna(axis=0)#drop NaNs
            gbpusd_df_corr = gbpusd_df.corr()#Corr Matix
            return gbpusd_df_corr.style.background_gradient(cmap='coolwarm'), gbpusd_df,gbpusd_df.describe()
        
        #USD to INR
        elif(indicator == 'USDINR'):
            #usdinr = pd.read_csv("Currencies/USDINR.csv")
            usdinr = pd.read_sql("SELECT * FROM USDINR",con=engine)
            usdinr = usdinr.set_index("Date")
            usdinr = usdinr.rename(columns={'Open':'USDINR_Open','Low':'USDINR_Low','High':'USDINR_High'
                                            ,'Close':'USDINR_Close','Volume':'USDINR_Volume'})
            usdinr_df = usdinr.join(self.data,how='inner')
            usdinr_df = usdinr_df.dropna(axis=0)
            usdinr_df_corr = usdinr_df.corr()
            return usdinr_df_corr.style.background_gradient(cmap='coolwarm'),usdinr_df,usdinr_df.describe()
         
        #USD to JPY
        elif(indicator == 'USDJPY'):
            #usdjpy = pd.read_csv("Currencies/USDJPY.csv")
            usdjpy = pd.read_sql("SELECT * FROM USDJPY",con=engine)
            usdjpy = usdjpy.set_index("Date")
            usdjpy = usdjpy.rename(columns={'Open':'USDJPY_Open','Low':'USDJPY_Low','High':'USDJPY_High'
                                            ,'Close':'USDJPY_Close','Volume':'USDJPY_Volume'})
            usdjpy_df = usdjpy.join(self.data,how='inner')
            usdjpy_df = usdjpy_df.dropna(axis=0)
            usdjpy_df_corr = usdjpy_df.corr()
            return usdjpy_df_corr.style.background_gradient(cmap='coolwarm'),usdjpy_df,usdjpy_df.describe()
         
        #GBPINR
        elif(indicator == 'GBPINR'):
            #gbpinr = pd.read_csv("Currencies/GBPINR.csv")
            gbpinr = pd.read_sql("SELECT * FROM GBPINR",con=engine)
            gbpinr = gbpinr.set_index("Date")
            gbpinr = gbpinr.rename(columns={'Open':'GBPINR_Open','Low':'GBPINR_Low','High':'GBPINR_High'
                                            ,'Close':'GBPINR_Close','Volume':'GBPINR_Volume'})
            gbpinr_df = gbpinr.join(self.data,how='inner')
            gbpinr_df = gbpinr_df.dropna(axis=0)
            gbpinr_df_corr = gbpinr_df.corr()
            return gbpinr_df_corr.style.background_gradient(cmap='coolwarm'),gbpinr_df,gbpinr_df.corr()

        #JPYINR
        elif(indicator == 'JPYINR'):
            #jpyinr = pd.read_csv("Currencies/JPYINR.csv")
            jpyinr = pd.read_sql("SELECT * FROM JPYINR",con=engine)
            jpyinr = jpyinr.set_index("Date")
            jpyinr = jpyinr.rename(columns={'Open':'JPYINR_Open','Low':'JPYINR_Low','High':'JPYINR_High'
                                            ,'Close':'JPYINR_Close','Volume':'JPYINR_Volume'})
            jpyinr_df = jpyinr.join(self.data,how='inner')
            jpyinr_df = jpyinr_df.dropna(axis=0)
            jpyinr_df_corr = jpyinr_df.corr()
            return jpyinr_df_corr.style.background_gradient(cmap='coolwarm'),jpyinr_df,jpyinr_df.corr()

        #EURINR
        elif(indicator == 'EURINR'):
            #eurinr = pd.read_csv("Currencies/EURINR.csv")
            eurinr = pd.read_sql("SELECT * FROM EURINR",con=engine)
            eurinr = eurinr.set_index("Date")
            eurinr = eurinr.rename(columns={'Open':'EURINR_Open','Low':'EURINR_Low','High':'EURINR_High'
                                            ,'Close':'EURINR_Close','Volume':'EURINR_Volume'})
            eurinr_df = eurinr.join(self.data,how='inner')
            eurinr_df = eurinr_df.dropna(axis=0)
            eurinr_df_corr = eurinr_df.corr()
            return eurinr_df_corr.style.background_gradient(cmap='coolwarm'),eurinr_df,eurinr_df.corr()

        #EURUSD
        elif(indicator == 'EURUSD'):
            #eurusd = pd.read_csv("Currencies/EURUSD.csv")
            eurusd = pd.read_sql("SELECT * FROM EURUSD",con=engine)
            eurusd = eurusd.set_index("Date")
            eurusd = eurusd.rename(columns={'Open':'EURUSD_Open','Low':'EURUSD_Low','High':'EURUSD_High'
                                            ,'Close':'EURUSD_Close','Volume':'EURUSD_Volume'})
            eurusd_df = eurusd.join(self.data,how='inner')
            eurusd_df = eurusd_df.dropna(axis=0)
            eurusd_df_corr = eurusd_df.corr()
            return eurusd_df_corr.style.background_gradient(cmap='coolwarm'),eurusd_df,eurusd_df.corr()
        else:
            raise ValueError("Invalid Currency Indicator-->",indicator)