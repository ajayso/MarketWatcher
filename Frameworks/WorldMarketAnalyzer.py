#Importing dependencies 
import pandas as pd
import numpy as np
import talib as ta
import math
import matplotlib.pyplot as plt
import os
plt.rcParams["figure.figsize"] = (20,10) #change this if you want to reduce the plot images
import locale
from locale import atof
import yfinance as yf
locale.setlocale(locale.LC_NUMERIC, '') 

#Quandl dependency with API key
import quandl
quandl.ApiConfig.api_key = "V8v63CRpQLfH8YKfsAmu"

# Analyzer Imports
import sys
sys.path.insert(0,'\rutils/')
from rutils.YahooData import Puller

class WorldMarketAnalyzer:
    
    def __init__(self,dataframe):
        self.data = dataframe
        #print("Loaded Dataframe for Analysis",self.data.describe())
        
        
    #Class method to return CorrMatrix, Indicator Values and Summary
    #Plot by default is False
    #If plot == True, A figure for the indicator shall be plotted with their dependent OHLC values.
    #X parameters must be of type String or None
    #Y parameters need to be type int/float or None
    
    def getWorldMarketIndicators(self,indicator, plot=True, xmin=None, xmax=None, ymin=None, ymax=None):
        print("Indicator is {}".format(indicator))
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
        pull = Puller("Y")
        #Bombay Stock Echange
        if(indicator == 'BSE'):
            #BSE = pd.read_csv("Data/BSE.csv")    #Read the Dataset from Data Directory
            bse_data = yf.download("^BSESN")
            bse_data = bse_data.rename(columns={"Low":indicator + "_Low","High":indicator + "_High","Open":indicator + "_Open","Close":indicator + "_Close","Last":indicator + "_Last","Volume":indicator + "_Volume"})
            BSE = pull.get_history("BSE","^BSESN") #pd.read_csv("Data/BSE.csv") 
            #print(BSE)
            #BSE = bse_data.history(period="max")
            #BSE = BSE.set_index("Date") #Set "Date" Column as the index
            BSE_df = BSE.join(self.data, how="inner")  #inner join with the maruti data set
            print(BSE_df.shape)
            BSE_df = BSE_df.dropna(axis=0) #Drop NaNs
            BSE_df_corr = BSE_df.corr() #Corr Matrix
            if(plot==True):
                BSE_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return BSE_df_corr.style.background_gradient(cmap="coolwarm"),BSE_df,BSE_df.describe()
            else:
                return BSE_df_corr.style.background_gradient(cmap="coolwarm"),BSE_df,BSE_df.describe()
        
        #All Ordinaries Austrailia
        elif(indicator == 'AORD'):
            AORD = pull.get_history("AORD","^AORD") # pd.read_csv("Data/AORD.csv")
            #aord_data = yf.Ticker("^AORD")
            #AORD = aord_data.history(period="max")
            #AORD = AORD.set_index("Date")
            AORD_df = AORD.join(self.data,how="inner")
            AORD_df = AORD_df.dropna(axis=0)
            AORD_df_corr = AORD_df.corr()
            if(plot==True):
                AORD_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return AORD_df_corr.style.background_gradient(cmap='coolwarm'),AORD_df,AORD_df.describe()
            else:
                return AORD_df_corr.style.background_gradient(cmap='coolwarm'),AORD_df,AORD_df.describe()
         
        #Dow Jones Industrial Average -- United States
        elif(indicator == 'DJI'):
            #DJI = pd.read_csv("Data/DJI.csv")
            DJI = pull.get_history("DJI","^DJI")
            #DJI = DJI.set_index("Date")
            DJI_df = DJI.join(self.data,how='inner')
            DJI_df = DJI_df.dropna(axis=0)
            DJI_df_corr = DJI_df.corr()
            if(plot == True):
                DJI_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return DJI_df_corr.style.background_gradient(cmap='coolwarm'),DJI_df,DJI_df.describe()
            else:
                return DJI_df_corr.style.background_gradient(cmap='coolwarm'),DJI_df,DJI_df.describe()
         
        #Franklin High Income
        elif(indicator == 'FCHIX'):
            FCHIX = pull.get_history("FCHIX","FCHIX") #pd.read_csv("Data/FCHIX.csv")
            #FCHIX = FCHIX.set_index("Date")
            FCHIX_df = FCHIX.iloc[:,:5] #FCHIX volume is zero hence dropped it ( producing NaNs in corr)
            FCHIX_df = FCHIX_df.join(self.data,how='inner')
            FCHIX_df = FCHIX_df.dropna(axis=0)
            FCHIX_df_corr = FCHIX_df.corr()
            if(plot == True):
                FCHIX_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return FCHIX_df_corr.style.background_gradient(cmap='coolwarm'),FCHIX_df,FCHIX_df.describe()
            else:
                return FCHIX_df_corr.style.background_gradient(cmap='coolwarm'),FCHIX_df,FCHIX_df.describe()
        
        #United Kingdom - The Financial Times Stock Exchange
        elif(indicator == 'FTSE'):
            FTSE = pull.get_history("FTSE","^FTSE") # pd.read_csv("Data/FTSE.csv")
            #FTSE['Date']=pd.to_datetime(FTSE['Date']) #convert the Date to DateTime dtype
            #FTSE = FTSE.set_index("Date") #Set index to Date column
            FTSE_df = FTSE.sort_index()
            FTSE_df = FTSE_df.drop(FTSE_df[FTSE_df['FTSE_Volume']=="-"].index) #Remove "-" from the dataset
            FTSE_df = FTSE_df.dropna(axis=0)            
            #FTSE_df = FTSE_df.applymap(atof) #Change strings to floats in the dataset
            FTSE_df = FTSE_df.join(self.data,how='inner') #inner join with the data set
            FTSE_df = FTSE_df.dropna(axis=0) #drop nans
            FTSE_df_corr = FTSE_df.corr()
            if(plot=='True'):
                FTSE_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return FTSE_df_corr.style.background_gradient(cmap='coolwarm'),FTSE_df,FTSE_df.describe()
            else:
                return FTSE_df_corr.style.background_gradient(cmap='coolwarm'),FTSE_df,FTSE_df.describe()
        
        #DAX - Europe
        elif(indicator == 'GDAXI'):
            GDAXI = pull.get_history("GDAXI","^GDAXI") #pd.read_csv("Data/GDAXI.csv")
            #GDAXI = GDAXI.set_index("Date")
            GDAXI_df = GDAXI.join(self.data,how='inner')
            GDAXI_df = GDAXI_df.dropna(axis=0)
            GDAXI_df_corr = GDAXI_df.corr()
            if(plot==True):
                GDAXI_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return GDAXI_df_corr.style.background_gradient(cmap='coolwarm'),GDAXI_df,GDAXI_df.describe()
            else:
                return GDAXI_df_corr.style.background_gradient(cmap='coolwarm'),GDAXI_df,GDAXI_df.describe()
         
        #S&P 500
        elif(indicator == 'GSPC'):
            GSPC = pull.get_history("GSPC","^GSPC")  #pd.read_csv("Data/GSPC.csv")
            #GSPC = GSPC.set_index("Date")
            GSPC_df = GSPC.join(self.data,how='inner')
            GSPC_df = GSPC_df.dropna(axis=0)
            GSPC_df_corr=GSPC_df.corr()
            if(plot == True):
                GSPC_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return GSPC_df_corr.style.background_gradient(cmap='coolwarm'),GSPC_df,GSPC_df.describe()
            else:
                return GSPC_df_corr.style.background_gradient(cmap='coolwarm'),GSPC_df,GSPC_df.describe()
         
        #Hang Seng Index -- Hong Kong
        elif(indicator == 'HSI'):
            HSI = pull.get_history("HSI","^HSI")   #pd.read_csv("Data/HSI.csv")
            #HSI = HSI.set_index("Date")
            HSI_df = HSI.join(self.data,how='inner')
            HSI_df = HSI_df.dropna(axis=0)
            HSI_df_corr = HSI_df.corr()
            if(plot == True):
                HSI_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return HSI_df_corr.style.background_gradient(cmap='coolwarm'),HSI_df,HSI_df.describe()
            else:
                return HSI_df_corr.style.background_gradient(cmap='coolwarm'),HSI_df,HSI_df.describe()
         
        #Korea
        elif(indicator == 'KS11'):
            KS11 =  pull.get_history("KS11","^KS11")  #pd.read_csv("Data/KS11.csv")
            #KS11 = KS11.set_index("Date")
            KS11_df = KS11.join(self.data,how='inner')
            KS11_df = KS11_df.dropna(axis=0)
            KS11_df_corr = KS11_df.corr()
            if(plot == True):
                KS11_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return KS11_df_corr.style.background_gradient(cmap='coolwarm'),KS11_df,KS11_df.describe()
            else:
                return KS11_df_corr.style.background_gradient(cmap='coolwarm'),KS11_df,KS11_df.describe()
        
        #Malaysia
        elif(indicator == 'KLSE'):
            KLSE = pull.get_history("KLSE","^KLSE")  #pd.read_csv("Data/KSLE.csv")
            #KLSE = KLSE.set_index("Date")
            KLSE_df = KLSE.sort_index()
            KLSE_df = KLSE_df.drop(KLSE_df[KLSE_df['KLSE_Volume']=="-"].index)
            KLSE_df = KLSE_df.dropna(axis=0)            
            #KLSE_df = KLSE_df.applymap(atof)
            KLSE_df = KLSE_df.join(self.data,how='inner')
            KLSE_df = KLSE_df.dropna(axis=0)
            KLSE_df_corr = KLSE_df.corr()
            if(plot == True):
                KLSE_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return KLSE_df_corr.style.background_gradient(cmap='coolwarm'),KLSE_df,KLSE_df.describe()
            else:
                return KLSE_df_corr.style.background_gradient(cmap='coolwarm'),KLSE_df,KLSE_df.describe()
          
        #Philliphines
        elif(indicator == 'PSEI'):
            PSEI = pull.get_history("PSEIPS","PSEI.PS")  #pd.read_csv("Data/PSEI.csv")
            #PSEI = PSEI.set_index("Date")
            PSEI_df = PSEI.join(self.data,how='inner')
            PSEI_df = PSEI_df.dropna(axis=0)
            PSEI_df_corr = PSEI_df.corr()
            if(plot == True):
                PSEI_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return PSEI_df_corr.style.background_gradient(cmap='coolwarm'),PSEI_df,PSEI_df.describe()
            else:
                return PSEI_df_corr.style.background_gradient(cmap='coolwarm'),PSEI_df,PSEI_df.describe()
         
        #Taiwan Weighted
        elif(indicator == 'TWII'):
            TWII = pull.get_history("TWII","^TWII")
            #TWII = TWII.set_index("Date")
            TWII_df = TWII.join(self.data,how='inner')
            TWII_df = TWII_df.dropna(axis=0)
            TWII_df_corr = TWII_df.corr()
            if(plot  == True):
                TWII_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return TWII_df_corr.style.background_gradient(cmap='coolwarm'),TWII_df,TWII_df.describe()
            else:
                return TWII_df_corr.style.background_gradient(cmap='coolwarm'),TWII_df,TWII_df.describe()
         
        #Japan
        elif(indicator == 'NIKKEI225'):
            NK225 = pull.get_history("NIKKEI225","^N225")
            #NK225 = NK225.set_index("Date")
            #Rename columns for convinience
            #NK225 = NK225.rename(columns={"Open":"NK_Open","High":"NK_High","Low":"NK_Low","Close":"NK_Close",
            #                    "Adj Close":"NK_AdjClose","Volume":"NK_Volume"})
            NK225_df = NK225.join(self.data,how='inner')  #inner join with the dataset
            NK225_df = NK225_df.dropna(axis=0)
            NK225_df_corr = NK225_df.corr()
            if(plot == True):
                NK225_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return NK225_df_corr.style.background_gradient(cmap='coolwarm'),NK225_df,NK225_df.describe()
            else:
                return NK225_df_corr.style.background_gradient(cmap='coolwarm'),NK225_df,NK225_df.describe()
         
        #NIFTY -  National Stock Exchange of India
        elif(indicator == 'NIFTY50'):
            NIFTY50 = pull.get_history("NSEI","^NSEI") # pd.read_csv("Data/^NSEI.csv")
            #NIFTY50 = NIFTY50.set_index("Date")
            NIFTY50 = NIFTY50.rename(columns={"Open":"N50_Open","High":"N50_High","Low":"N50_Low","Close":"N50_Close",
                                 "Volume":"N50_Volume",'Adj Close':'N50_AdjClose'})
            NIFTY50_df = NIFTY50.join(self.data,how='inner')
            #NIFTY50_df = NIFTY50_df.dropna(axis=0)
            NIFTY50_df_corr = NIFTY50_df.corr()
            if(plot == True):
                NIFTY50_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return NIFTY50_df_corr.style.background_gradient(cmap='coolwarm'),NIFTY50_df,NIFTY50_df.describe()
            else:
                return NIFTY50_df_corr.style.background_gradient(cmap='coolwarm'),NIFTY50_df,NIFTY50_df.describe()
         
        #NIFTY - Auto
        elif(indicator == 'NIFTYAUTO'):
            NIFTYAuto = quandl.get("NSE/NIFTY_AUTO")
            NIFTYAuto = NIFTYAuto.rename(columns={"Open":"NA_Open","High":"NA_High","Low":"NA_Low","Close":"NA_Close",
                                 "Shares Traded":"NA_Shares_Traded","Turnover (Rs. Cr)":"NA_Turnover(Cr)"})
            NIFTYAuto_df = NIFTYAuto.join(self.data,how='inner')
            NIFTYAuto_df = NIFTYAuto_df.dropna(axis=0)
            NIFTYAuto_df_corr = NIFTYAuto_df.corr()
            if(plot==True):
                NIFTYAuto_df.plot(ylim=(ymin,ymax),xlim=(xmin,xmax))
                return NIFTYAuto_df_corr.style.background_gradient(cmap='coolwarm'),NIFTYAuto_df,NIFTYAuto_df.describe()
            else:
                return NIFTYAuto_df_corr.style.background_gradient(cmap='coolwarm'),NIFTYAuto_df,NIFTYAuto_df.describe()
            
        else:
            raise ValueError("Invalid Indicator Symbol")
        
