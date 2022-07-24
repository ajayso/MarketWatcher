# Imports
#Importing dependencies 
import pandas as pd
import numpy as np
import talib as ta
import math
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10) #change this if you want to reduce the plot images
import locale
from datetime import date
from locale import atof
locale.setlocale(locale.LC_NUMERIC, '') 
#Quandl dependency with API key
import quandl
quandl.ApiConfig.api_key = "V8v63CRpQLfH8YKfsAmu"

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

class Program:

        def __init__(self,scriptcode,Threshold,Corr_Thresh,Target,split,timesteps):
                self.Threshold = Threshold
                self.Corr_Thresh=Corr_Thresh
                self.Target=Target
                self.split= split
                self.timesteps=timesteps
                quandl.ApiConfig.api_key = "V8v63CRpQLfH8YKfsAmu"
                df = quandl.get("NSE/" + scriptcode,  start_date="2002-07-01", end_date=date.today())
                df = df.rename(columns = {"Total Trade Quantity": "Volume"})
                df = df.rename(columns = {"Last": "Adj Close"})
                df = df.dropna(axis=0)
                data =df [['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
                print ("Data Shape is {}".format(df.shape))
                self.data = df

        def _buildModels(self):
                manager = ModelManager()
                learner = ModelBuilder()	
                token = manager.Selector(self.analyzed_data,self.Threshold,self.Target,self.Corr_Thresh,self.split,self.timesteps)
                self.model = learner.Trainer(token,self.analyzed_data,self.Threshold,self.Target,self.Corr_Thresh,self.timesteps)
                print("The best model is " + token)


        def buildAnalyzers(self):
                self.Threshold=0.6
                self.Corr_Thresh=0.6
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
                excludelist = ['Open','High','Low','Close','Volume','Adj Close','Last','Total Trade Quantity','Turnover (Lacs)']

                for command in tech_feat:
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
                        if(data_new.shape[0] < Threshold) :
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
                        if(data_new.shape[0] < Threshold) :
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
                


scriptcode = "RELIANCE"
Threshold=0.6
Corr_Thresh=0.6
Target='Close'
split=0.8
timesteps=7
p = Program(scriptcode,Threshold,Corr_Thresh,Target,split,timesteps)
dataset = p.buildAnalyzers()
dataset.to_csv(scriptcode + ".csv")
print(dataset)
p._buildModels()

