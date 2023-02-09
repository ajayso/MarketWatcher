# MarketWatcher Main Start
# Generic Imports
import pandas as pd
import numpy as np
import os
import locale
from datetime import date
from locale import atof
locale.setlocale(locale.LC_NUMERIC, '') 


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

from Program import Main

scriptcode = "GSPC"
print(scriptcode)
Threshold=0.7
Corr_Thresh=0.7
Target=['Open','High','Close']
split=0.7
timesteps=7
modelpath = os.getcwd() + "\Models" 
p = Main(scriptcode,Threshold,Corr_Thresh,Target,split,timesteps,modelpath,1,read_from_file=1,forecast=0)
#dataset = p.buildAnalyzers()
#dataset.to_csv(scriptcode + "Scrapped.csv")
#print(dataset)
while(True):
    try:
        p._buildModels()
    except:
        print("Exception occured but continuing....")
#p._forecast_data(modelpath,"LSTM")


