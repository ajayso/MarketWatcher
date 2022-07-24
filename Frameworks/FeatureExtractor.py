#Importing dependencies 
from pickle import TRUE
import pandas as pd
import numpy as np
import talib as ta
import math
import sys
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10) #change this if you want to reduce the plot images
import locale
from locale import atof
locale.setlocale(locale.LC_NUMERIC, '') 

#Quandl dependency with API key
import quandl
quandl.ApiConfig.api_key = "RB48Rib76iwBPxtzeVs2"
sys.path.insert(0,'\Frameworks/')
from Frameworks.TechnicalAnalyzer import Tech_IndicatorEvaluator
from Frameworks.CommodityAnalyzer import CommodityAnalyzer
from Frameworks.CurrencyAnalyzer import CurrencyAnalyzer
from Frameworks.WorldMarketAnalyzer import WorldMarketAnalyzer

class FeatureExtractor:
    
    def __init__ (self, dataframe):
        self.data = dataframe
        self.tech_class = Tech_IndicatorEvaluator(self.data)
        self.world_class = WorldMarketAnalyzer(self.data)
        self.commodity_class = CommodityAnalyzer(self.data)
        self.currency_class = CurrencyAnalyzer(self.data)
        
    def TechnicalFeatureExtractor(self,target, p_thresh , n_thresh):
        commandlist = ['ADX', 'DPO','Donchain','CMF','CLV',"BBANDS","AROON","CCI","ATR","CHAIKINAD",
               "CHAIKINVOLATILITY","CMO","ROC","MOMENTUM","RSI","MFI","OBV","SAR",
               "STOCHASTIC","TRIX","ultimateOscillator"]
        techlist =[]
        for command in commandlist :
            a,b,c = self.tech_class.getIndicator(command,TRUE,None,None,None,None)
            bdf = b.corr()
            bdf = bdf.loc[target]
            #print("Dependent Indexes for Command",command,"are:\n")
            for i,row in bdf.iteritems():
                if(row >= p_thresh or row <= n_thresh):
                    if(i == 'Open' or i == 'Close' or i == 'High'or i=='Low' or i=='Last' or i == 'Total Traded Quantity' or i == 'Turnover (Lacs)'
                        or i == 'Volume' or i == 'Adj Close'):
                        continue
                    else:
                        print("Index:",i,"Value:",row)
                        print(command,"is dependent on",target,"through",i,'indexes with value',row)
                        if( command not in techlist):
                            techlist.append(command)
                else:
                    continue
        #print("\n")
        print("Final Technical Feature Set is:\n",techlist)
        return techlist
        
    def WorldMarketExtractor(self,target,p_thresh,n_thresh):
        
        worldmarketcommand = ["BSE","AORD","DJI","FCHIX","FTSE","GDAXI","GSPC","HSI"] #,"KS11","KLSE","PSEI","TWII"]
        worldmarket =[]

        for command in worldmarketcommand :
            a,b,c = self.world_class.getWorldMarketIndicators(command,True,None,None,None,None)
            print(command)
            print(b.shape)
            bdf = b.corr()
            print(bdf)
            #print(bdf)
            bdf = bdf.loc[target]
            #print("p_thresh {}, n_thresh {}".format(p_thresh,n_thresh))
            #print("Dependent Indexes for Command",command,"are:\n")
            for i,row in bdf.iteritems():
                #print("i {} --row {}, p_thresh {}, n_thresh {}".format(i, row, p_thresh,n_thresh))
                if(row > p_thresh or row < n_thresh):
                    if(i == 'Open' or i == 'Close' or i == 'High'or i=='Low' or i=='Last' or i == 'Total Traded Quantity' or i == 'Volume' or i == 'Adj Close' or i == 'Turnover (Lacs)'):
                        continue
                    else:
                        #print("Index:",i,"Value:",row)
                        #print(command,"is dependent on",target,"through",i,'indexes')
                        if( command not in worldmarket):
                            worldmarket.append(command)
        #print("\n")
        print("Final World Market Feature Set is:\n",worldmarket)
        return worldmarket
        
    def CommodityDataExtractor(self,target,p_thresh,n_thresh):
        #'Crude Oil',
        commoditycommand = ['Crude Oil', 'Brent Crude','Natural Gas','Gasoline','Gold','Silver','Aluminium',
                           "Platinum","Palladium","Copper","Lead","Rhodium","Tin","Zinc","Nickel",
                           "Corn","Rice","Soyabeans","Wheat","Coal","Coffee","Cocoa","Sugar",
                           "Cotton"]
        commoditydata =[]
        for command in commoditycommand :
            a,b,c = self.commodity_class.getCommodityIndicators(command,True,None,None,None,None)
            bdf = b.corr()
            bdf = bdf.loc[target]
            #print("Dependent Indexes for Command",command,"are:\n")
            for i,row in bdf.iteritems():
                if(row > p_thresh or row < n_thresh):
                    if(i == 'Open' or i == 'Close' or i == 'High'or i=='Low' or i=='Last' or i == 'Adj Close' or i == 'Total Traded Quantity' or i =='Volume' or i == 'Turnover (Lacs)'):
                        continue
                    else:
                        #print("Index:",i,"Value:",row)
                        #print(command,"is dependent on",target,"through",i,'indexes')
                        if( command not in commoditydata):
                            commoditydata.append(command)
        #print("\n")
        print("Final Commodity Market Feature Set is:\n",commoditydata)
        return commoditydata
        
    def CurrencyDataExtractor(self,target,p_thresh,n_thresh):
        currencycommand = ['GBPUSD','USDINR','USDJPY',]
        currencydata =[]
        for command in currencycommand :
            a,b,c = self.currency_class.getCurrencyIndicators(command,True,None,None,None,None)
            bdf = b.corr()
            bdf = bdf.loc[target]
            #print("Dependent Indexes for Command",command,"are:\n")
            for i,row in bdf.iteritems():
                if(row > p_thresh or row < n_thresh):
                    if(i == 'Open' or i == 'Close' or i == 'High'or i == 'Adj Close' or i=='Low' or i=='Last' or i == 'Volume' or i == 'Total Traded Quantity' or i == 'Turnover (Lacs)'):
                        continue
                    else:
                        #print("Index:",i,"Value:",row)
                        #print(command,"is dependent on",target,"through",i,'indexes')
                        if( command not in currencydata):
                            currencydata.append(command)
        #print("\n")
        print("Final Curency data Feature Set is:\n",currencydata)
        return currencydata