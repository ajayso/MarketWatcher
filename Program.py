#Generic Imports
import pandas as pd
import numpy as np
import math
import locale
from datetime import date
from locale import atof
locale.setlocale(locale.LC_NUMERIC, '') 
from nsepy import get_history

# Analyzer Imports
import sys
sys.path.insert(0,'\Frameworks/')
from Frameworks.FeatureExtractor import FeatureExtractor
from Frameworks.TechnicalAnalyzer import Tech_IndicatorEvaluator
from Frameworks.CommodityAnalyzer import CommodityAnalyzer
from Frameworks.CurrencyAnalyzer import CurrencyAnalyzer
from Frameworks.WorldMarketAnalyzer import WorldMarketAnalyzer


sys.path.insert(0, '\Engine')
sys.path.insert(0, '\ModelsGenerators')
from ModelsGenerators.ModelSelectorTarget import ModelManager
from ModelsGenerators.LearnerTarget import ModelBuilder

# support for index pull
import yfinance as yf
sys.path.insert(0,'\rutils/')
from rutils.YahooData import Puller

#Monitoring on NR 
import newrelic.agent
application = newrelic.agent.application()

class Main:
        @newrelic.agent.background_task(name='Main-init', group='Task')
        def __init__(self,scriptcode,Threshold,Corr_Thresh,Target,split,timesteps,modelpath, index=0,read_from_file=0,forecast=0):
                self.Threshold = Threshold
                self.Corr_Thresh=Corr_Thresh
                self.Target=Target
                self.split= split
                self.timesteps=timesteps
                self.scriptcode = scriptcode
                if (forecast==0):
                        # Just forward the read data
                    if (read_from_file==0):
                        if (index==0):
                            df= get_history(symbol=scriptcode, start=date(2020,1,1), end=date.today())
                            df = df.rename(columns = {"No. of Shares": "Volume"})
                            df = df.rename(columns = {"Last": "Adj Close"})
                            df = df.dropna(axis=0)
                        else:
                            pull = Puller("Y")
                            df = pull.get_history(scriptcode,"^"+scriptcode)
                            str_replace = scriptcode+"_"
                            df = df.rename(columns=lambda x: x.replace(str_replace, ''))
                            print(df.columns)
                    else:
                        df = pd.read_csv(scriptcode + "Scrapped.csv")
                        df = df.drop("Date", axis='columns')
                        print(df.columns)
                        self.analyzed_data= df

                    #df = df[["Open","High","Close"]]
                    data = df
                    self.analyzed_data= df
                    print("Size of the dataset {}",format(df.shape))
                    self.data = data
                    self.modelpath = modelpath
                else:
                    df = pd.read_csv("Forecast-"+ scriptcode + ".csv")
                    df = df.drop("Date", axis='columns')
                    data = df
                    print("Size of the forecasted--dataset {}",format(data.shape))
                    self.data = data
                    self.modelpath = modelpath



        def _buildModels(self):
                manager = ModelManager()
                learner = ModelBuilder()	
                token = manager.Selector(self.scriptcode,self.analyzed_data,self.Threshold,self.Target,self.Corr_Thresh,self.split,self.timesteps,self.modelpath)
                self.model = learner.Trainer(token,self.analyzed_data,self.Threshold,self.Target,self.Corr_Thresh,self.timesteps)
                print("The best model is " + token)

        def _forecast_data(self,modelpath,name):
                manager = ModelManager()
                manager.Forecast(self.scriptcode,self.data ,7,name,modelpath)

        def buildAnalyzers(self):
                #self.Threshold=0.8
                #self.Corr_Thresh=0.7
                FE = FeatureExtractor(self.data)
                Target='Close'
                """
                Extracts Features Sets for Technical , Commodities, Currencies and World Markets.
                Imported from different python sources codes from Root Directory.
                """
                FE = FeatureExtractor(self.data)

                print("Commencing Feature Extraction ...\n")
                tech_feat = FE.TechnicalFeatureExtractor(self.Target,self.Corr_Thresh, -self.Corr_Thresh)
                world_feat = FE.WorldMarketExtractor(self.Target,self.Corr_Thresh, -self.Corr_Thresh)
                comm_feat = FE.CommodityDataExtractor(self.Target,self.Corr_Thresh, -self.Corr_Thresh)
                curr_feat = FE.CurrencyDataExtractor(self.Target,self.Corr_Thresh, -self.Corr_Thresh)

                tech_class = Tech_IndicatorEvaluator(self.data)
                world_class = WorldMarketAnalyzer(self.data)
                commodity_class = CommodityAnalyzer(self.data)
                currency_class = CurrencyAnalyzer(self.data)

                print("\n")
                print("Appending Feature List into the Target data set .... \n")

                data_copy = self.data.copy()
                print(data_copy.shape)
                excludelist = ['Open','High','Low','Close','Volume','Adj Close','Last','Total Trade Quantity','Turnover (Lacs)']

                for command in tech_feat:
                    print(command)
                    a,b,c = tech_class.getIndicator(command)
                    cols = [col for col in b.columns if col not in excludelist]
                    b = b[cols]
                    data_new = data_copy.join(b,how='inner')
                    data_new = data_new.dropna(axis=0)
                    if(data_new.shape[0] < self.Threshold) :
                        print(command,"is running short on common Data Points with",data_new.shape[0],"rows in common, omitting....\n")
                        data_copy = data_copy
                    else:
                        print(data_new.shape[0],"common with",command)
                        data_copy = data_new

                    for command in curr_feat:
                        a,b,c = currency_class.getCurrencyIndicators(command)
                        b = pd.DataFrame(b.iloc[:,3])
                        #b= pd.DataFrame(b.iloc[:,3])
                        data_new = pd.concat([data_copy,b],axis=1)
                        data_new = data_new.dropna(axis=0)
                        if(data_new.shape[0] < self.Threshold) :
                            print(command,"is running short on common Data Points with",data_new.shape[0],"rows in common, omitting....\n")
                            data_copy = data_copy
                        else:
                            print(data_new.shape[0],"common with",command)
                            data_copy = data_new

                    for command in comm_feat:
                        a,b,c = commodity_class.getCommodityIndicators(command)
                        b = pd.DataFrame(b.iloc[:,3])
                        data_new = pd.concat([data_copy,b],axis=1)
                        data_new = data_new.dropna(axis=0)
                        if(data_new.shape[0] < self.Threshold) :
                            print(command,"is running short on common Data Points with",data_new.shape[0],"rows in common, omitting....\n")
                            data_copy = data_copy
                        else:
                            print(data_new.shape[0],"common with",command)
                            data_copy = data_new

                    for command in world_feat:
                        a,b,c = world_class.getWorldMarketIndicators(command)
                        b = pd.DataFrame(b.iloc[:,3])
                        data_new = pd.concat([data_copy,b],axis=1)
                        data_new = data_new.dropna(axis=0)
                        if(data_new.shape[0] < self.Threshold) :
                            print(command,"is running short on common Data Points with",data_new.shape[0],"rows in common, omitting....\n")
                            data_copy = data_copy
                        else:
                            print(data_new.shape[0],"common with",command)
                            data_copy = data_new

                    data_copy = data_copy.dropna(axis=0)
                    
                    print("\n")
                    print("Final Data Set contains",data_copy.shape[0],"rows and",data_copy.shape[1],"columns\n")
                    print("The Column List is : \n")
                    print(data_copy.columns)
                    self.analyzed_data = data_copy
                    return data_copy    
                




