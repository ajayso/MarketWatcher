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
import yfinance as yf
from rutils.YahooData import Puller
#Quandl dependency with API key
import quandl
quandl.ApiConfig.api_key = "RB48Rib76iwBPxtzeVs2"

class CurrencyAnalyzer:
    
    def __init__(self,dataframe):
        #loads dataframe
        self.data = dataframe
        
        
        
    #Class method to return CorrMatrix, Indicator Values and Summary
    #Plot by default is False
    #If plot == True, A figure for the indicator shall be plotted with their dependent OHLC values.
    #X parameters must be of type String or None
    #Y parameters need to be type int/float or None
    #All Currency data taken from Quandl
        
    #indicator accepted values : USDINR, GBPUSD,USDJPY and DTWEXM
    def getCurrencyIndicators(self,indicator,plot=False, xmin=None, xmax=None, ymin=None, ymax=None):
        
        if(xmin != None):
            if(type(xmin)!=str):
                raise TypeError("X parameters must be of type -> str")
        
        if(xmax != None):
            if(type(xmax)!=str):
                raise TypeError("X parameters must be of type -> str")
                
        if(ymin != None):
            if(not isinstance(ymin, (int, float))):
                raise TypeError("Y parameters must be of type -> Int or Float")
                                 
        if(ymax != None):
            if(not isinstance(ymax, (int,float))):
                raise TypeError("Y parameters must be of type -> Int or Float")
                
        #USD to Pound Sterling
        if(indicator == 'GBPUSD'):
            gbpusd = yf.download('GBPUSD=X')
            indicator = indicator.replace(" ", "_")
            gbpusd = gbpusd.rename(columns={"Low":indicator + "_Low","High":indicator + "_High","Open":indicator + "_Open","Close":indicator + "_Close","Last":indicator + "_Last","Volume":indicator + "_Volume"})
            #gbpusd = quandl.get('GBPUSD=X') #Import Currency Data from Quandl, Change this to import personal data
            gbpusd_df = gbpusd.join(self.data,how='inner') #inner join with dataframe
            gbpusd_df = gbpusd_df.dropna(axis=0)#drop NaNs
            gbpusd_df_corr = gbpusd_df.corr()#Corr Matix
            if(plot == True):
                gbpusd_df_corr.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return gbpusd_df_corr.style.background_gradient(cmap='coolwarm'), gbpusd_df,gbpusd_df.describe()
            else:
                return gbpusd_df_corr.style.background_gradient(cmap='coolwarm'), gbpusd_df,gbpusd_df.describe()
        
        #USD to INR
        elif(indicator == 'USDINR'):
            usdinr = yf.download('INR=X')
            indicator = indicator.replace(" ", "_")
            usdinr = usdinr.rename(columns={"Low":indicator + "_Low","High":indicator + "_High","Open":indicator + "_Open","Close":indicator + "_Close","Last":indicator + "_Last","Volume":indicator + "_Volume"})
            usdinr_df = usdinr.join(self.data,how='inner')
            usdinr_df = usdinr_df.rename(columns={'RATE':'USD2INR'}) #Renaming columns
            usdinr_df = usdinr_df.dropna(axis=0)
            usdinr_df_corr = usdinr_df.corr()
            if(plot == True):
                usdinr_df_corr.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return usdinr_df_corr.style.background_gradient(cmap='coolwarm'),usdinr_df,usdinr_df.describe()
            else:
                return usdinr_df_corr.style.background_gradient(cmap='coolwarm'),usdinr_df,usdinr_df.describe()
         
        #USD to JPY
        elif(indicator == 'USDJPY'):
            usdjpy = yf.download('JPY=X')
            indicator = indicator.replace(" ", "_")
            usdjpy = usdjpy.rename(columns={"Low":indicator + "_Low","High":indicator + "_High","Open":indicator + "_Open","Close":indicator + "_Close","Last":indicator + "_Last","Volume":indicator + "_Volume"})
            usdjpy_df = usdjpy.join(self.data,how='inner')
            usdjpy_df = usdjpy_df.rename(columns={"RATE":"USD2JPY"})
            usdjpy_df = usdjpy_df.dropna(axis=0)
            usdjpy_df_corr = usdjpy_df.corr()
            if(plot == True):
                usdjpy_df_corr.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return usdjpy_df_corr.style.background_gradient(cmap='coolwarm'),usdjpy_df,usdjpy_df.describe()
            else:
                return usdjpy_df_corr.style.background_gradient(cmap='coolwarm'),usdjpy_df,usdjpy_df.describe()
         
        #Major Weighted Currency
        elif(indicator == 'DTWEXM'):
            wmc = quandl.get("FRED/DTWEXM")
            wmc_df = wmc.join(self.data,how='inner')
            wmc_df = wmc_df.dropna(axis=0)
            wmc_df_corr = wmc_df.corr()
            if(plot == True):
                wmc_df_corr.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return wmc_df_corr.style.background_gradient(cmap='coolwarm'),wmc_df,wmc_df.corr()
            else:
                return wmc_df_corr.style.background_gradient(cmap='coolwarm'),wmc_df,wmc_df.corr()
        else:
            raise ValueError("Invalid Currency Indicator-->",indicator)