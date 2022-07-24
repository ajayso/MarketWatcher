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
#from sqlalchemy import create_engine
#import mysql.connector

#Quandl dependency with API key
import quandl
quandl.ApiConfig.api_key = "RB48Rib76iwBPxtzeVs2"
#SQL Connector for WorldMarket Database

#engine = create_engine("mysql+mysqlconnector://Savinay:savinay@35.185.177.170/WorldMarkets")
#print("SQL Connected to World Markets") 

class WorldMarketAnalyzer:
    def __init__(self,dataframe):
        self.data = dataframe
        #print("Loaded Dataframe for Analysis",self.data.describe()) 

    #Class method to return CorrMatrix, Indicator Values and Summary
    
    def getWorldMarketIndicators(self,indicator):
        print("I am inside get WorkMarketIndicator{}".format(indicator))
        print(indicator)
        #Bombay Stock Echange
        if(indicator == 'BSE'):
            BSE = pd.read_csv("Data/BSE.csv")    #Read the Dataset from Data Directory
            print(BSE)
            #BSE = pd.read_sql('SELECT * FROM BSE',con=engine)
            BSE = BSE.set_index("Date") #Set "Date" Column as the index
            BSE = BSE.drop(columns={'Close'})
            BSE = BSE.rename(columns={'Open':'BSE_Open','Adj Close':'BSE_Close','High':'BSE_High','Low':'BSE_Low',
                                        'Volume':'BSE_Volume'})
            BSE_df = BSE.join(self.data, how="inner")  #inner join with the maruti data set
            BSE_df = BSE_df.dropna(axis=0) #Drop NaNs
            BSE_df_corr = BSE_df.corr() #Corr Matrix
            return BSE_df_corr.style.background_gradient(cmap="coolwarm"),BSE_df,BSE_df.describe()
        
        #All Ordinaries Austrailia
        elif(indicator == 'AORD'):
            AORD = pd.read_csv("Data/AORD.csv")
            #AORD = pd.read_sql('SELECT * FROM AORD',con=engine)
            AORD = AORD.set_index("Date")
            AORD = AORD.drop(columns={'Close'})
            AORD = AORD.rename(columns={'Open':'AORD_Open','Adj Close':'AORD_Close','High':'AORD_High','Low':'AORD_Low',
                                        'Volume':'AORD_Volume'})
            AORD_df = AORD.join(self.data,how="inner")
            AORD_df = AORD_df.dropna(axis=0)
            AORD_df_corr = AORD_df.corr()
            return AORD_df_corr.style.background_gradient(cmap='coolwarm'),AORD_df,AORD_df.describe()
         
        #Dow Jones Industrial Average -- United States
        elif(indicator == 'DJI'):
            DJI = pd.read_csv("Data/DJI.csv")
            #DJI = pd.read_sql('SELECT * FROM DJI',con=engine)
            DJI = DJI.set_index("Date")
            DJI = DJI.drop(columns={'Close'})
            DJI = DJI.rename(columns={'Open':'DJI_Open','Adj Close':'DJI_Close','High':'DJI_High','Low':'DJI_Low',
                                        'Volume':'DJI_Volume'})
            DJI_df = DJI.join(self.data,how='inner')
            DJI_df = DJI_df.dropna(axis=0)
            DJI_df_corr = DJI_df.corr()
            return DJI_df_corr.style.background_gradient(cmap='coolwarm'),DJI_df,DJI_df.describe()
         
        #Franklin High Income
        elif(indicator == 'FCHIX'):
            FCHIX = pd.read_csv("Data/FCHIX.csv")
            #FCHIX = pd.read_sql('SELECT * FROM FCHIX',con=engine)
            FCHIX = FCHIX.set_index("Date")
            FCHIX_df = FCHIX.iloc[:,:5] #FCHIX volume is zero hence dropped it ( producing NaNs in corr)
            FCHIX_df = FCHIX_df.join(self.data,how='inner')
            FCHIX_df = FCHIX_df.dropna(axis=0)
            FCHIX_df_corr = FCHIX_df.corr()
            return FCHIX_df_corr.style.background_gradient(cmap='coolwarm'),FCHIX_df,FCHIX_df.describe()
        
        #United Kingdom - The Financial Times Stock Exchange
        elif(indicator == 'FTSE'):
            FTSE = pd.read_csv("Data/FTSE.csv")
            #FTSE =pd.read_sql('SELECT * FROM FTSE',con=engine)
            #FTSE['Date']=pd.to_datetime(FTSE['Date']) #convert the Date to DateTime dtype
            FTSE_df = FTSE.set_index("Date") #Set index to Date column
            """FTSE_df = FTSE.sort_index()
            FTSE_df = FTSE_df.drop(FTSE_df[FTSE_df['FTSE_Volume']=="-"].index) #Remove "-" from the dataset
            FTSE_df = FTSE_df.dropna(axis=0)            
            FTSE_df = FTSE_df.applymap(atof) #Change strings to floats in the dataset"""
            FTSE_df = FTSE_df.rename(columns={'Open':'FTSE_Open','Close':'FTSE_Close','High':'FTSE_High','Low':'FTSE_Low','Volume':'FTSE_Volume'})
            FTSE_df = FTSE_df.drop(columns={'OpenInt'})
            FTSE_df = FTSE_df.join(self.data,how='inner') #inner join with the data set
            FTSE_df = FTSE_df.dropna(axis=0) #drop nans
            FTSE_df_corr = FTSE_df.corr()
            return FTSE_df_corr.style.background_gradient(cmap='coolwarm'),FTSE_df,FTSE_df.describe()
        
        #DAX - Europe
        elif(indicator == 'GDAXI'):
            GDAXI = pd.read_csv("Data/GDAXI.csv")
            #GDAXI =pd.read_sql('SELECT * FROM GDAXI',con=engine)
            GDAXI = GDAXI.set_index("Date")
            GDAXI = GDAXI.drop(columns={'Close'})
            GDAXI = GDAXI.rename(columns={'Open':'GDAX_Open','Adj Close':'GDAX_Close','High':'GDAX_High','Low':'GDAX_Low',
                                        'Volume':'GDAX_Volume'})
            GDAXI_df = GDAXI.join(self.data,how='inner')
            GDAXI_df = GDAXI_df.dropna(axis=0)
            GDAXI_df_corr = GDAXI_df.corr()
            return GDAXI_df_corr.style.background_gradient(cmap='coolwarm'),GDAXI_df,GDAXI_df.describe()
         
        #S&P 500
        elif(indicator == 'GSPC'):
            GSPC = pd.read_csv("Data/GSPC.csv")
            #GSPC =pd.read_sql('SELECT * FROM GSPC',con=engine)
            GSPC = GSPC.set_index("Date")
            GSPC = GSPC.drop(columns={'Close'})
            GSPC = GSPC.rename(columns={'Open':'GSPC_Open','Adj Close':'GSPC_Close','High':'GSPC_High','Low':'GSPC_Low',
                                        'Volume':'GSPC_Volume'})
            GSPC_df = GSPC.join(self.data,how='inner')
            GSPC_df = GSPC_df.dropna(axis=0)
            GSPC_df_corr=GSPC_df.corr()
            return GSPC_df_corr.style.background_gradient(cmap='coolwarm'),GSPC_df,GSPC_df.describe()
         
        #Hang Seng Index -- Hong Kong
        elif(indicator == 'HSI'):
            HSI = pd.read_csv("Data/HSI.csv")
            #HSI =pd.read_sql('SELECT * FROM HSI',con=engine)
            HSI = HSI.set_index("Date")
            HSI = HSI.drop(columns={'Close'})
            HSI = HSI.rename(columns={'Open':'HIS_Open','Adj Close':'HIS_Close','High':'HIS_High','Low':'HIS_Low',
                                        'Volume':'HIS_Volume'})
            HSI_df = HSI.join(self.data,how='inner')
            HSI_df = HSI_df.dropna(axis=0)
            HSI_df_corr = HSI_df.corr()
            return HSI_df_corr.style.background_gradient(cmap='coolwarm'),HSI_df,HSI_df.describe()
         
        #Korea
        elif(indicator == 'KS11'):
            KS11 = pd.read_csv("Data/KS11.csv")
            #KS11 =pd.read_sql('SELECT * FROM KS11',con=engine)
            KS11 = KS11.set_index("Date")
            KS11 = KS11.drop(columns={'Close'})
            KS11 = KS11.rename(columns={'Open':'KOSPI_Open','Adj Close':'KOSPI_Close','High':'KOSPI_High','Low':'KOSPI_Low',
                                        'Volume':'KOSPI_Volume'})
            KS11_df = KS11.join(self.data,how='inner')
            KS11_df = KS11_df.dropna(axis=0)
            KS11_df_corr = KS11_df.corr()
            return KS11_df_corr.style.background_gradient(cmap='coolwarm'),KS11_df,KS11_df.describe()
        
        #Malaysia
        elif(indicator == 'KLSE'):
            KLSE = pd.read_csv("Data/KSLE.csv")
            #KLSE =pd.read_sql('SELECT * FROM KLSE',con=engine)
            KLSE_df = KLSE.set_index("Date")
            """KLSE_df = KLSE.sort_index()
            KLSE_df = KLSE_df.drop(KLSE_df[KLSE_df['KLSE_Volume']=="-"].index)
            KLSE_df = KLSE_df.dropna(axis=0)            
            KLSE_df = KLSE_df.applymap(atof)"""
            KLSE_df = KLSE_df.rename(columns={'Open':'KLSE_Open','Close':'KLSE_Close','High':'KLSE_High','Low':'KLSE_Low',
                                        'Volume':'KLSE_Volume'})
            KLSE_df = KLSE_df.join(self.data,how='inner')
            KLSE_df = KLSE_df.dropna(axis=0)
            KLSE_df_corr = KLSE_df.corr()
            return KLSE_df_corr.style.background_gradient(cmap='coolwarm'),KLSE_df,KLSE_df.describe()
          
        #Philliphines
        elif(indicator == 'PSEI'):
            PSEI = pd.read_csv("Data/PSEI.csv")
            #PSEI =pd.read_sql('SELECT * FROM PSEI',con=engine)
            PSEI = PSEI.set_index("Date")
            PSEI = PSEI.drop(columns={'Close'})
            PSEI = PSEI.rename(columns={'Open':'PSEIPS_Open','Adj Close':'PSEIPS_Close','High':'PSEIPS_High','Low':'PSEIPS_Low',
                                        'Volume':'PSEIPS_Volume'})
            PSEI_df = PSEI.join(self.data,how='inner')
            PSEI_df = PSEI_df.dropna(axis=0)
            PSEI_df_corr = PSEI_df.corr()
            return PSEI_df_corr.style.background_gradient(cmap='coolwarm'),PSEI_df,PSEI_df.describe()
         
        #Taiwan Weighted
        elif(indicator == 'TWII'):
            TWII = pd.read_csv("Data/TWII.csv")
            #TWII =pd.read_sql('SELECT * FROM TWII',con=engine)
            TWII = TWII.set_index("Date")
            TWII = TWII.drop(columns={'Close'})
            TWII = TWII.rename(columns={'Open':'TWII_Open','Adj Close':'TWII_Close','High':'TWII_High','Low':'TWII_Low',
                                        'Volume':'TWII_Volume'})
            TWII_df = TWII.join(self.data,how='inner')
            TWII_df = TWII_df.dropna(axis=0)
            TWII_df_corr = TWII_df.corr()
            return TWII_df_corr.style.background_gradient(cmap='coolwarm'),TWII_df,TWII_df.describe()
         
        #Japan
        elif(indicator == 'NIKKEI225'):
            NK225 = pd.read_csv("Data/N225.csv")
            #NK225 =pd.read_sql('SELECT * FROM N225',con=engine)
            NK225 = NK225.set_index("Date")
            #Rename columns for convinience
            NK225 = NK225.rename(columns={"Open":"NK_Open","High":"NK_High","Low":"NK_Low","Close":"NK_Close",
                                 "Adj Close":"NK_AdjClose","Volume":"NK_Volume"})
            NK225_df = NK225.join(self.data,how='inner')  #inner join with the dataset
            NK225_df = NK225_df.dropna(axis=0)
            NK225_df_corr = NK225_df.corr()
            return NK225_df_corr.style.background_gradient(cmap='coolwarm'),NK225_df,NK225_df.describe()
         
        #NIFTY -  National Stock Exchange of India
        elif(indicator == 'NIFTY50'):
            NIFTY50 = pd.read_csv("Data/^NSEI.csv")
            #NIFTY50 =pd.read_sql('SELECT * FROM NSEI',con=engine)
            NIFTY50 = NIFTY50.set_index("Date")
            NIFTY50 = NIFTY50.rename(columns={"Open":"N50_Open","High":"N50_High","Low":"N50_Low","Close":"N50_Close",
                                 "Volume":"N50_Volume",'Adj Close':'N50_AdjClose'})
            NIFTY50_df = NIFTY50.join(self.data,how='inner')
            #NIFTY50_df = NIFTY50_df.dropna(axis=0)
            NIFTY50_df_corr = NIFTY50_df.corr()
            return NIFTY50_df_corr.style.background_gradient(cmap='coolwarm'),NIFTY50_df,NIFTY50_df.describe()
         
        #NIFTY - Auto
        elif(indicator == '$NIFTYAUTO'): # skip this
            NIFTYAuto = quandl.get("NSE/NIFTY_AUTO")
            NIFTYAuto = NIFTYAuto.rename(columns={"Open":"NA_Open","High":"NA_High","Low":"NA_Low","Close":"NA_Close",
                                 "Shares Traded":"NA_Shares_Traded","Turnover (Rs. Cr)":"NA_Turnover(Cr)"})
            NIFTYAuto_df = NIFTYAuto.join(self.data,how='inner')
            NIFTYAuto_df = NIFTYAuto_df.dropna(axis=0)
            NIFTYAuto_df_corr = NIFTYAuto_df.corr()
            return NIFTYAuto_df_corr.style.background_gradient(cmap='coolwarm'),NIFTYAuto_df,NIFTYAuto_df.describe()
            
        else:
            raise ValueError("Invalid Indicator Symbol")
        