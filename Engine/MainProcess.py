from TechnicalAnalyzer import Tech_IndicatorEvaluator
from CommodityAnalyzer import CommodityAnalyzer
from CurrencyAnalyzer import CurrencyAnalyzer
from WorldMarketAnalyzer import WorldMarketAnalyzer
from FeatureExtractor import FeatureExtractor

import pandas as pd
import numpy as np
import mysql

"""
Main Process appends the feature lists into the final dataset.
However it will ommit indicators which do not have sufficient data points on common timestamps.
It will ommit the features which will yield data set below the Data Point Threshold.

data ---> Dataframe
Threshold -> Int only , defines number of rows.
target ---> String , the target index
Corr_Thress --> Float or int , the minimum Correlation Threshold (includes positive and negative)

"""
class DataManager:
	def Load(self, data, Threshold, Target, Corr_Thresh ):
		"""
		Extracts Features Sets for Technical , Commodities, Currencies and World Markets.
		Imported from different python sources codes from Root Directory.
		"""
		FE = FeatureExtractor(data)
		print("Commencing Feature Extraction ...\n")
		tech_feat = FE.TechnicalFeatureExtractor(Target,Corr_Thresh, -Corr_Thresh)
		print("TechnicalFeatureExtractor done. ...\n")
		world_feat = FE.WorldMarketExtractor(Target,Corr_Thresh, -Corr_Thresh)
		print("WorldMarketExtractor done. ...\n")
		comm_feat = FE.CommodityDataExtractor(Target,Corr_Thresh, -Corr_Thresh)
		print("CommodityDataExtractor done. ...\n")
		curr_feat = FE.CurrencyDataExtractor(Target,Corr_Thresh, -Corr_Thresh)
		print("CurrencyDataExtractor done. ...\n")

		tech_class = Tech_IndicatorEvaluator(data)
		world_class = WorldMarketAnalyzer(data)
		commodity_class = CommodityAnalyzer(data)
		currency_class = CurrencyAnalyzer(data)

		print("\n")
		print("Appending Feature List into the Target data set .... \n")

		data_copy = data.copy()
		excludelist = ['Open','High','Low','Close','Volume','Adj Close','Last','Total Trade Quantity','Turnover (Lacs)']

		for command in tech_feat:
			a,b,c = tech_class.getIndicator(command)
			cols = [col for col in b.columns if col not in excludelist]
			b = b[cols]
			data_new = data_copy.join(b,how='inner')
			data_new = data_new.dropna(axis=0)
			if(data_new.shape[0] < Threshold) :
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
			if(data_new.shape[0] < Threshold) :
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
		return data_copy


	
	