#Importing dependencies 
import pandas as pd
import numpy as np
import talib as ta
import math
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)

#Quandl dependency with API key
import quandl
quandl.ApiConfig.api_key = "f7_JWui3ztp2Yxh_xddT"

class Tech_IndicatorEvaluator:
    
    #Init method for the class
    #This method extracts OHLCV details from the dataframe
    
    def __init__(self, dataframe):

        self.high = dataframe.loc[:,'High']
        self.low = dataframe.loc[:,'Low']
        self.close = dataframe.loc[:,"Close"]
        if('Volume' in dataframe.columns):
            self.volume = dataframe.loc[:,'Volume']
            self.data = dataframe[['Open','High','Low','Close','Volume']] 
        else:
            self.volume = dataframe.loc[:,'Total Trade Quantity']
            self.data = dataframe[['Open','High','Low','Close','Total Trade Quantity']]
        
    #Class method to return CorrMatrix, Indicator Values and Summary
    
    def getIndicator(self,indicator):
        
        """
        Bollinger Bands® are a technical analysis tool developed by John Bollinger.
        There are three lines that compose Bollinger Bands: A simple moving average (middle band) and an upper and lower band.
        The upper and lower bands are typically 2 standard deviations +/- from a 20-day simple moving average, but can be modified
        
        Many traders believe the closer the prices move to the upper band, the more overbought the market,
        and the closer the prices move to the lower band, the more oversold the market.
        
        """
        
        if(indicator == 'BBANDS'):

            BB = ta.BBANDS(self.close)  #talib function)
            BB_df = pd.DataFrame(list(BB)).transpose() #converting to Dataframe
            BB_df = BB_df.rename(columns={0:'Upperband',1:"Middleband",2:"Lowerband"}) #renaming columns
            BB_df = BB_df.join(self.data , how='inner') #inner join with the OHLCV dataframe(on common timestamps)
            BB_df = BB_df.dropna(axis=0) #drop NaNs
            BB_df_corr=BB_df.corr(method='pearson') #corr matrix
            return BB_df_corr.style.background_gradient(cmap='coolwarm'),BB_df,BB_df.describe()
            
            """
            The Arron indicator is composed of two lines. 
            An up line which measures the number of periods since a High, 
            and a down line which measures the number of periods since a Low.
            
            When the Aroon Up is above the Aroon Down, it indicates bullish price behavior.
            When the Aroon Down is above the Aroon Up, it signals bearish price behavior.
            
            For example, when Aroon Up crosses above Aroon Down it may mean a new uptrend is starting.
            """

        elif(indicator == 'AROON'):
            aroondf = ta.AROON(self.high,self.low,timeperiod=20)
            aroondf= pd.DataFrame(list(aroondf)).transpose()
            aroondf = aroondf.rename(columns={0:"AroonDown",1:"AroonUp"})
            aroondf = aroondf.join(self.data,how='inner')
            aroondf = aroondf.dropna(axis=0)
            aroon_corr = aroondf.corr()
            return aroon_corr.style.background_gradient(cmap='coolwarm'),aroondf,aroondf.describe()
            
        elif(indicator == 'CCI'):
            cci = list(ta.CCI(self.high, self.low, self.close)) #Talib function
            cci_df = self.data.copy() #creating a copy of OHLCV DataFramee
            cci_df['Real'] = pd.Series(cci).values  #converting to a series and appending to the  prev Dataframe
            cci_df = cci_df.dropna(axis=0)
            cci_corr= cci_df.corr()
            return cci_corr.style.background_gradient(cmap='coolwarm'),cci_df,cci_df.describe()
        
        elif(indicator == 'ATR'):
            atr = ta.ATR(self.high,self.low,self.close,timeperiod=14)
            atrdf= pd.DataFrame(atr)
            atrdf = atrdf.rename(columns={0:"ATR_real"})
            atrdf = atrdf.join(self.data,how='inner')
            atrdf = atrdf.dropna(axis=0)
            atr_corr = atrdf.corr()

            return atr_corr.style.background_gradient(cmap='coolwarm'),atrdf,atrdf.describe()
        
        elif(indicator == 'CHAIKINAD'):
            chaikinad = ta.AD(self.high, self.low, self.close, self.volume)
            chaikinad_df = pd.DataFrame(chaikinad)
            chaikinad_df = chaikinad_df.rename(columns={0:"ChaikinAD"})
            chaikinad_df = chaikinad_df.join(self.data,how='inner')
            chaikinad_df = chaikinad_df.dropna(axis=0)
            chaikinad_corr = chaikinad_df.corr()
            return chaikinad_corr.style.background_gradient(cmap='coolwarm'),chaikinad_df,chaikinad_df.describe()
            
        elif(indicator == 'CHAIKINVOLATILITY'):
            chaikin_os = ta.ADOSC(self.high,self.low,self.close,self.volume)
            chaikin_os_df = pd.DataFrame(chaikin_os)
            chaikin_os_df = chaikin_os_df.rename(columns={0:"ChaikinOSC"})
            chaikin_os_df = chaikin_os_df.join(self.data,how='inner')
            chaikin_os_df = chaikin_os_df.dropna(axis=0)
            chaikin_os_corr= chaikin_os_df.corr()
            return chaikin_os_corr.style.background_gradient(cmap='coolwarm'),chaikin_os_df,chaikin_os_df.describe()
            
        elif(indicator == 'CMO'):
            cmo = ta.CMO(self.close, timeperiod=14)
            cmo_df = pd.DataFrame(cmo)
            cmo_df = cmo_df.rename(columns={0:"CMO"})
            cmo_df = cmo_df.join(self.data,how='inner')
            cmo_df = cmo_df.dropna(axis=0)
            cmo_corr = cmo_df.corr()
            
            return cmo_corr.style.background_gradient(cmap='coolwarm'),cmo_df,cmo_df.describe()
        
        elif(indicator == 'MACD'):
            macd = ta.MACD(self.close, fastperiod=6, slowperiod=13, signalperiod=5)
            macd_df = pd.DataFrame(macd).transpose()
            macd_df = macd_df.rename(columns={0:"macd",1:"MACDSignal",2:"MACDHist"})
            macd_df = macd_df.join(self.data,how='inner')
            macd_df = macd_df.dropna(axis=0)
            macd_df_corr = macd_df.corr()
            return macd_df_corr.style.background_gradient(cmap='coolwarm'),macd_df,macd_df.describe()
        
        elif(indicator == 'ROC'):
            roc = ta.ROC(self.close, timeperiod=14)
            roc_df = pd.DataFrame(roc)
            roc_df = roc_df.rename(columns={0:"ROC_value"})
            roc_df = roc_df.join(self.data,how='inner')
            roc_df_corr= roc_df.corr()
            return roc_df_corr.style.background_gradient(cmap='coolwarm'),roc_df,roc_df.describe()
        
        elif(indicator == 'MOMENTUM'):
            mom = ta.MOM(self.close , timeperiod=14)
            mom_df = pd.DataFrame(mom)
            mom_df = mom_df.rename(columns={0:'Momentum'})
            mom_df = mom_df.join(self.data,how='inner')
            mom_df_corr = mom_df.corr()
            
            return mom_df_corr.style.background_gradient(cmap='coolwarm'),mom_df,mom_df.describe()
        
        elif(indicator == 'RSI'):
            rsi = ta.RSI(self.close, timeperiod=14)
            rsi_df = pd.DataFrame(rsi)
            rsi_df = rsi_df.rename(columns={0:"RSI"})
            rsi_df = rsi_df.join(self.data,how="inner")
            rsi_df = rsi_df.dropna(axis=0)
            rsi_df_corr = rsi_df.corr()
            return rsi_df_corr.style.background_gradient(cmap='coolwarm'),rsi_df,rsi_df.describe()
            
        elif(indicator == 'MFI'):
            mfi = ta.MFI(self.high, self.low, self.close, self.volume, timeperiod=14)
            mfi_df = pd.DataFrame(mfi)
            mfi_df = mfi_df.rename(columns={0:"MFI"})
            mfi_df = mfi_df.join(self.data,how='inner')
            mfi_df = mfi_df.dropna(axis=0)
            mfi_df_corr= mfi_df.corr()
            return mfi_df_corr.style.background_gradient(cmap='coolwarm'),mfi_df,mfi_df.describe()
        
        elif(indicator == 'OBV'):
            obv = ta.OBV(self.close, self.volume)
            obv_df = pd.DataFrame(obv)
            obv_df = obv_df.rename(columns={0:'OBV'})
            obv_df = obv_df.join(self.data,how='inner')
            obv_df_corr = obv_df.corr()
            return obv_df_corr.style.background_gradient(cmap='coolwarm'),obv_df,obv_df.describe()
            
        elif(indicator == 'SAR'):
            sar = ta.SAR(self.high, self.low, acceleration=1, maximum=1)
            sar_df = pd.DataFrame(sar)
            sar_df = sar_df.rename(columns={0:'SAR'})
            sar_df = sar_df.join(self.data,how='inner')
            sar_df = sar_df.dropna(axis=0)
            sar_df_corr = sar_df.corr()
            return sar_df_corr.style.background_gradient(cmap='coolwarm'),sar_df,sar_df.describe()
            
        elif(indicator == 'STOCHASTIC'):
            stoch = ta.STOCH(self.high, self.low, self.close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            stoch_df = pd.DataFrame(list(stoch)).transpose()
            stoch_df = stoch_df.rename(columns={0:"SlowK",1:"SlowD"})
            stoch_df = stoch_df.join(self.data,how='inner')
            stoch_df = stoch_df.dropna(axis=0)
            stoch_df_corr = stoch_df.corr()
            return stoch_df_corr.style.background_gradient(cmap='coolwarm'),stoch_df,stoch_df.describe()
            
        elif(indicator == 'TRIX'):
            trix = ta.TRIX(self.close, timeperiod=20)
            trix_df = pd.DataFrame(trix)
            trix_df = trix_df.rename(columns={0:'Trix'})
            trix_df= trix_df.join(self.data, how='inner')
            trix_df = trix_df.dropna(axis=0)
            trix_df_corr = trix_df.corr()
            return trix_df_corr.style.background_gradient(cmap='coolwarm'),trix_df,trix_df.describe()
            
        elif(indicator == 'ultimateOscillator'):
            ultiosc = ta.ULTOSC(self.high, self.low, self.close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
            ultiosc_df = pd.DataFrame(ultiosc)
            ultiosc_df = ultiosc_df.rename(columns={0:"ULTI_OSC"})
            ultiosc_df = ultiosc_df.join(self.data,how='inner')
            ultiosc_df = ultiosc_df.dropna(axis=0)
            ultiosc_df_corr = ultiosc_df.corr()
            return ultiosc_df_corr.style.background_gradient(cmap='coolwarm'),ultiosc_df,ultiosc_df.describe()
        
        elif(indicator == 'CLV'):
            clvlist = [] #creating an empty list 
            for i in range(0, self.data.shape[0]):
                if((self.data.iloc[i,1] - self.data.iloc[i,2]) == 0):
                    clvlist.append(1)
                else:
                    #CLV = ((Close - Low) - (High - Close))/(High - Low)
                    clv = ((self.data.iloc[i,3] - self.data.iloc[i,2]) - 
                        (self.data.iloc[i,1] - self.data.iloc[i,3]))/(self.data.iloc[i,1] - self.data.iloc[i,2])
                    clvlist.append(clv) #appending values to list
            clv_df = pd.DataFrame() #creating an empty Dataframe
            clv_df = self.data.copy() #creating a copy of OHLCV Dataframe
            clv_df['CLV'] = clvlist #appending a column of values from clvlist
            clv_df_corr = clv_df.corr() #corr matrix
            return clv_df_corr.style.background_gradient(cmap='coolwarm'),clv_df,clv_df.describe()
        
        elif(indicator == 'CMF'):
            temp =[] #empty list
            for i in range(0,self.data.shape[0]):
                if(self.data.iloc[i,4] == 0):
                    temp.append(0)
                elif((self.data.iloc[i,1] - self.data.iloc[i,2])==0):
                    mfm = 1
                    cmf = mfm*self.data.iloc[i,4]
                    temp.append(cmf)
                else:
                    #MoneyFlowMultiplier = ((Close - Low) - (High - Close))/(High - Low)
                    mfm = ((self.data.iloc[i,3] - self.data.iloc[i,2]) - (self.data.iloc[i,1] - self.data.iloc[i,3]))/(self.data.iloc[i,1] - self.data.iloc[i,2])
                    #MoneyFlowMultipler x Volume of that period = MoneyFlow Volume
                    cmf = mfm*self.data.iloc[i,4]
                    temp.append(cmf)

            cmf_df = pd.DataFrame() #Create an empty dataframe
            cmf_df['MFVolume'] = temp #append list in Dataframe
            cmf_df['Volume'] = list(self.volume) #Add Volume data in dataframe
            cmf_df = cmf_df.rolling(20).sum() #Rolling sum for timeperiod = 20 ----> can be changed
            cmf_df['Ratio'] = cmf_df['MFVolume']/cmf_df['Volume'] #Taking raio
            cmflist = list(cmf_df.iloc[:,2]) #creating a list of the ratios obtained
            cmf_df = pd.DataFrame() #appending it to the empty DataFrame
            cmf_df = self.data.copy()
            cmf_df['CMF'] = cmflist
            cmf_df = cmf_df.dropna(axis=0) #Dropping NaN
            cmf_df_corr = cmf_df.corr() #Corr Matrix
            return cmf_df_corr.style.background_gradient(cmap='coolwarm'),cmf_df,cmf_df.describe()
            
        elif(indicator == 'Donchain'):
            temp =[]
            highest = pd.DataFrame(self.data.iloc[:,1].rolling(20).max()) #Highest 20 day high
            lowest = pd.DataFrame(self.data.iloc[:,2].rolling(20).min()) #Lowest 20 day Low
            combined = highest.join(lowest,how='inner') #comibining on inner join
            for i in range(0,combined.shape[0]):
                if(math.isnan(combined.iloc[i,1]) == True or math.isnan(combined.iloc[i,0] == True)):
                    temp.append(None)
                else:
                    temp.append((combined.iloc[i,0] + combined.iloc[i,1])/2) #The middle = (Highest+Lowest)/2
            combined['Mid'] = temp
            combined = combined.rename(columns={'High':"Upper","Low":'Lower',"Mid":"Middle"}) #Renaming Columns
            donchain_df = combined.join(self.data, how='inner') #joining with OHLCV data
            donchain_df.dropna(axis=0) #Dropping NaNs
            donchain_df_corr = donchain_df.corr()
            return donchain_df_corr.style.background_gradient(cmap='coolwarm'),donchain_df,donchain_df.describe()
            
        elif(indicator == 'DPO'):
            time = 20
            dpo = ta.SMA(self.close, timeperiod=time) #Calculate Simple Moving Average
            dpo_df = pd.DataFrame(dpo)
            dpo_df['DPO'] = np.nan #Create a column filled with NaNs
            
            #Start from i = timeperiod 
            for i in range(time-1,dpo_df.shape[0]):
                #(Price of (N/2 + 1) periods ago) - (N Period SMA) = DPO
                dpo_df.iloc[i,1] = self.data.iloc[(i//2+1),3] - dpo_df.iloc[i,0]
                
            dpo_df = dpo_df.join(self.data,how='inner') #inner join
            dpo_df = dpo_df.dropna(axis=0) #dropping nans
            dpo_df = dpo_df.iloc[:,1:]
            dpo_df_corr = dpo_df.corr() #corr matrix
            return dpo_df_corr.style.background_gradient(cmap='coolwarm'),dpo_df,dpo_df.describe()
        
        elif(indicator == 'ADX'):
            adx = ta.ADX(self.high, self.low, self.close, timeperiod=14)
            adx_df = pd.DataFrame(adx)
            adx_df = adx_df.rename(columns={0:'ADX'})
            adx_df = adx_df.join(self.data,how='inner')
            adx_df = adx_df.dropna(axis=0)
            adx_df_corr = adx_df.corr()
            return adx_df_corr.style.background_gradient(cmap='coolwarm'),adx_df,adx_df.describe()
        
        else:
            raise ValueError("Invalid Indicator Symbol")