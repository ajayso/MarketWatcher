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
from sqlalchemy import create_engine
import mysql.connector

engine = create_engine("mysql+mysqlconnector://Savinay:savinay@35.185.177.170/Commodities")
print("SQL Connected with Commodities")

class CommodityAnalyzer:
    def __init__(self, dataframe):
        #sets "data" as the imported dataframe
        self.data = dataframe
        
    #Method to return Commodity Indicators against a OHLC of a company...
    #Can add more indicators if followed the same if block code structure.
    def getCommodityIndicators(self,indicator):
        
        #CRUDE OIL
        if(indicator == 'Crude Oil'):
            #crudeoil = pd.read_csv("Commodities/CrudeOil.csv")
            crudeoil = pd.read_sql("SELECT * FROM CrudeOil",con=engine)
            crudeoil = crudeoil.set_index("Date")
            crudeoil = crudeoil.iloc[:,:5]
            crudeoil = crudeoil.rename(columns={'Open':'Crude_Open','Low':"Crude_Low",'High':'Crude_High','Close':'Crude_Close','Volume':'Crude_Vol'})
            crudeoil_df = crudeoil.join(self.data,how='inner')#inner join with the dataframe
            crudeoil_df = crudeoil_df.dropna(axis=0)#dropping NaNs
            crudeoil_df_corr = crudeoil_df.corr()#Correlation Matrix
            return crudeoil_df_corr.style.background_gradient(cmap='coolwarm'),crudeoil_df,crudeoil_df.describe()
            
        elif(indicator == 'Brent Crude'):
            #Bcrudeoil = pd.read_csv("Commodities/BrentCrude.csv")
            Bcrudeoil = pd.read_sql("SELECT * FROM BrentCrude",con=engine)
            Bcrudeoil = Bcrudeoil.set_index("Date")
            Bcrudeoil = Bcrudeoil.iloc[:,:5]
            Bcrudeoil = Bcrudeoil.rename(columns={'Open':'Brent_Open','Low':"Brent_Low",'High':'Brent_High','Close':'Brent_Close','Volume':'Brent_Vol'})
            Bcrudeoil_df = Bcrudeoil.join(self.data,how='inner')
            Bcrudeoil_df = Bcrudeoil_df.dropna(axis=0)
            Bcrudeoil_df_corr = Bcrudeoil_df.corr()
            return Bcrudeoil_df_corr.style.background_gradient(cmap='coolwarm'),Bcrudeoil_df,Bcrudeoil_df.describe()
            
        elif(indicator == 'Natural Gas'):
            #naturalgas = pd.read_csv("Commodities/NaturalGas.csv")
            naturalgas = pd.read_sql("SELECT * FROM NaturalGas",con=engine)
            naturalgas = naturalgas.set_index("Date")
            naturalgas = naturalgas.iloc[:,:5]
            #renaming columns to remove ambiguous columns.
            naturalgas = naturalgas.rename(columns={'Low':"Gas_Low","High":"Gas_High","Open":"Gas_Open",
                                                "Close":"Gas_Close","Volume":"Gas_Volume"})
            naturalgas_df = naturalgas.join(self.data,how='inner')
            naturalgas_df = naturalgas_df.dropna(axis=0)
            naturalgas_df_corr = naturalgas_df.corr()
            return naturalgas_df_corr.style.background_gradient(cmap='coolwarm'),naturalgas_df,naturalgas_df.describe()
            
        elif(indicator == 'Gasoline'):
            #fredcurs = pd.read_csv("Commodities/Gasoline.csv")
            fredcurs = pd.read_sql("SELECT * FROM Gasoline",con=engine)
            fredcurs = fredcurs.set_index("Date")
            fredcurs = fredcurs.iloc[:,:5]
            fredcurs = fredcurs.rename(columns={'Low':"Gasoline_Low","High":"Gasoline_High","Open":"Gasoline_Open",
                                                "Close":"Gasoline_Close","Volume":"Gasoline_Volume"})
            fredcurs_df = fredcurs.join(self.data,how='inner')
            fredcurs_df = fredcurs_df.dropna(axis=0)
            fredcurs_df_corr = fredcurs_df.corr()
            return fredcurs_df_corr.style.background_gradient(cmap='coolwarm'),fredcurs_df,fredcurs_df.describe()
            
        elif(indicator == 'Gold'):
            #gold = pd.read_csv("Commodities/Gold.csv")
            gold = pd.read_sql("SELECT * FROM Gold",con=engine)
            gold = gold.set_index("Date")
            gold = gold.iloc[:,:5]
            gold = gold.rename(columns={'Low':"Gold_Low","High":"Gold_High","Open":"Gold_Open",
                                                "Close":"Gold_Close","Volume":"Gold_Volume"})
            gold_df = gold.join(self.data,how='inner')
            gold_df= gold_df.dropna(axis=0)
            gold_df_corr = gold_df.corr()
            return gold_df_corr.style.background_gradient(cmap='coolwarm'),gold_df,gold_df.describe()
        
        elif(indicator == 'Silver'):
            #silver = pd.read_csv("Commodities/Silver.csv")
            silver = pd.read_sql("SELECT * FROM Silver",con=engine)
            silver = silver.set_index("Date")
            silver = silver.iloc[:,:5]
            silver = silver.rename(columns={'Low':"Silver_Low","High":"Silver_High","Open":"Silver_Open",
                                                "Close":"Silver_Close","Volume":"Silver_Volume"})
            silver_df = silver.join(self.data,how='inner')
            silver_df= silver_df.dropna(axis=0)
            silver_df_corr = silver_df.corr()
            return silver_df_corr.style.background_gradient(cmap='coolwarm'),silver_df,silver_df.describe()
            
        elif(indicator == 'Aluminium'):
           # al = pd.read_csv("Commodities/Aluminium.csv")
            al = pd.read_sql("SELECT * FROM Aluminium",con=engine)
            al = al.set_index("Date")
            al = al.iloc[:,:5]
            al = al.rename(columns={'Low':"Al_Low","High":"Al_High","Open":"Al_Open",
                                                "Close":"Al_Close","Volume":"Al_Volume"})
            al_df = al.join(self.data,how='inner')
            al_df = al_df.dropna(axis=0)
            al_df_corr = al_df.corr()
            return al_df_corr.style.background_gradient(cmap='coolwarm'),al_df,al_df.describe()
        
        elif(indicator == 'Platinum'):
            #platdf = pd.read_csv("Commodities/Platinum.csv")
            platdf = pd.read_sql("SELECT * FROM Platinum",con=engine)
            platdf = platdf.set_index("Date")
            platdf = platdf.iloc[:,:5]
            platdf = platdf.rename(columns={"Open":"Plat_Open","High":"Plat_High",
                                           "Low":"Plat_Low","Close":"Plat_Close","Volume":"Plat_Volume"})

            platdf = platdf.join(self.data,how='inner')
            platdf = platdf.dropna(axis=0)
            platdf_corr = platdf.corr()
            return platdf.corr().style.background_gradient(cmap='coolwarm'),platdf,platdf.describe()
        
        elif(indicator =='Palladium'):
            #pldm = pd.read_csv("Commodities/Palladium.csv")
            pldm = pd.read_sql("SELECT * FROM Palladium",con=engine)
            pldm = pldm.set_index("Date")
            pldm = pldm.iloc[:,:5]
            pldm = pldm.rename(columns={"Open":"Pldm_Open","High":"PLdm_High","Low":"Pldm_Low",
                                       "Close":"Pldm_Close","Volume":"Pldm_Volume"})
            pldm = pldm.join(self.data,how='inner')
            pldm = pldm.dropna(axis=0)
            return pldm.corr().style.background_gradient(cmap='coolwarm'),pldm,pldm.describe()
            
        elif(indicator == 'Copper'):
            #copp = pd.read_csv("Commodities/Copper.csv")
            copp = pd.read_sql("SELECT * FROM Copper",con=engine)
            copp = copp.set_index("Date")
            copp = copp.iloc[:,:5]
            copp = copp.rename(columns={"Open":"Copp_Open","High":"Copp_High","Low":"Copp_Low",
                                       "Close":"Copp_Close","Volume":"Copp_Volume"})
            copp = copp.join(self.data,how='inner')
            copp = copp.dropna(axis=0)
            return copp.corr().style.background_gradient(cmap='coolwarm'),copp,copp.describe()
        
        elif(indicator == 'Lead'):
            #ld = pd.read_csv("Commodities/Lead.csv")
            ld = pd.read_sql("SELECT * FROM Lead",con=engine)
            ld = ld.set_index("Date")
            ld = ld.iloc[:,:5]
            ld = ld.rename(columns={"Open":"Ld_Open","High":"Ld_High","Low":"Ld_Low",
                                       "Close":"Ld_Close","Volume":"Ld_Volume"})
            ld = ld.join(self.data,how="inner")
            ld = ld.dropna(axis=0)
            return ld.corr().style.background_gradient(cmap='coolwarm'),ld,ld.describe()
        
        elif(indicator == 'Tin'):
            #tin = pd.read_csv("Commodities/Tin.csv")
            tin = pd.read_sql("SELECT * FROM Tin",con=engine)
            tin = tin.set_index("Date")
            tin = tin.iloc[:,:5]
            tin = tin.rename(columns={'Low':"Tin_Low","High":"Tin_High","Open":"Tin_Open",
                                                "Close":"Tin_Close","Volume":"Tin_Volume"})
            tin = tin.join(self.data,how='inner')
            tin = tin.dropna(axis=0)
            return tin.corr().style.background_gradient(cmap='coolwarm'),tin,tin.describe()
            
        elif(indicator == 'Zinc'):
            zinc = pd.read_sql("SELECT * FROM Zinc",con=engine)
            zinc = pd.read_csv("Commodities/Zinc.csv")
            zinc = zinc.set_index("Date")
            zinc = zinc.iloc[:,:5]
            zinc = zinc.rename(columns={'Low':"Zinc_Low","High":"Zinc_High","Open":"Zinc_Open",
                                                "Close":"Zinc_Close","Volume":"Zinc_Volume"})
            zinc = zinc.join(self.data,how='inner')
            zinc = zinc.dropna(axis=0)
            return zinc.corr().style.background_gradient(cmap='coolwarm'),zinc,zinc.describe()
            
        elif(indicator == 'Nickel'):
            #nick = pd.read_csv("Commodities/Nickel.csv")
            nick = pd.read_sql("SELECT * FROM Nickel",con=engine)
            nick = nick.set_index("Date")
            nick = nick.iloc[:,:5]
            nick = nick.rename(columns={'Low':"Nick_Low","High":"Nick_High","Open":"Nick_Open",
                                                "Close":"Nick_Close","Volume":"Nick_Volume"})
            nick = nick.join(self.data, how='inner')
            nick = nick.dropna(axis=0)
            return nick.corr().style.background_gradient(cmap='coolwarm'),nick,nick.describe()
                 
        elif(indicator == 'Corn'):
            #corn = pd.read_csv("Commodities/Corn.csv")
            corn = pd.read_sql("SELECT * FROM Corn",con=engine)
            corn = corn.set_index("Date")
            corn = corn.iloc[:,:5]
            corn = corn.rename(columns={"Open":"Corn_Open","High":"Corn_High","Low":"Corn_Low",
                                       'Close':"Corn_Close","Volume":"Corn_Volume"})
            corn = corn.join(self.data,how='inner')
            corn = corn.dropna(axis=0)
            return corn.corr().style.background_gradient(cmap='coolwarm'),corn,corn.describe()
    
        elif(indicator == 'Soyabeans'):
            #soya = pd.read_csv("Commodities/Soyabean.csv")
            soya = pd.read_sql("SELECT * FROM Soyabean",con=engine)
            soya = soya.set_index("Date")
            soya = soya.iloc[:,:5]
            soya = soya.rename(columns={"Open":"Soya_Open","High":"Soya_High","Low":"Soya_Low",
                                       "Close":"Soya_Close","Volume":"Soya_Volume"})
            soya = soya.join(self.data,how='inner')
            soya = soya.dropna(axis=0)
            return soya.corr().style.background_gradient(cmap='coolwarm'),soya,soya.describe()
        
        elif(indicator == "Wheat"):
           # wheat = pd.read_csv("Commodities/Wheat.csv")
            wheat = pd.read_sql("SELECT * FROM Wheat",con=engine)
            wheat = wheat.set_index("Date")
            wheat = wheat.iloc[:,:5]
            wheat = wheat.rename(columns={"Open":"Wheat_Open","High":"Wheat_High","Low":"Wheat_Low",
                                       "Close":"Wheat_Close","Volume":"Wheat_Volume"})
            wheat = wheat.join(self.data,how='inner')
            wheat = wheat.dropna(axis=0)
            return wheat.corr().style.background_gradient(cmap='coolwarm'),wheat,wheat.describe()
            
        elif(indicator == 'Coffee'):
            #coffee = pd.read_csv("Commodities/Coffee.csv")
            coffee = pd.read_sql("SELECT * FROM Coffee",con=engine)
            coffee = coffee.set_index("Date")
            coffee = coffee.iloc[:,:5]
            coffee = coffee.rename(columns={"Open":"Coffee_Open","High":"Coffee_High","Low":"Coffee_Low",
                                       "Close":"Coffee_Close","Volume":"Coffee_Volume"})
            coffee = coffee.join(self.data,how='inner')
            coffee = coffee.dropna(axis=0)
            return coffee.corr().style.background_gradient(cmap='coolwarm'),coffee,coffee.describe()
            
        elif(indicator == 'Cocoa'):
            #cocoa = pd.read_csv("Commodities/Cocoa.csv")
            cocoa = pd.read_sql("SELECT * FROM Cocoa",con=engine)
            cocoa = cocoa.set_index("Date")
            cocoa = cocoa.iloc[:,:5]
            cocoa = cocoa.rename(columns={"Open":"Cocoa_Open","High":"Cocoa_High","Low":"Cocoa_Low",
                                            "Close":"Cocoa_Close","Volume":"Cocoa_Volume"})
            cocoa = cocoa.join(self.data,how='inner')
            cocoa = cocoa.dropna(axis=0)
            return cocoa.corr().style.background_gradient(cmap='coolwarm'),cocoa,cocoa.describe()
            
        elif(indicator=='Sugar'):
            #sugar = pd.read_csv("Commodities/Sugar.csv")
            sugar = pd.read_sql("SELECT * FROM Sugar",con=engine)
            sugar = sugar.set_index("Date")
            sugar = sugar.iloc[:,:5]
            sugar = sugar.rename(columns={"Open":"Sugar_Open","High":"Sugar_High","Low":"Sugar_Low",
                                       "Close":"Sugar_Close","Volume":"Sugar_Volume"})
            sugar = sugar.join(self.data, how='inner')
            sugar = sugar.dropna(axis=0)
            return sugar.corr().style.background_gradient(cmap='coolwarm'),sugar,sugar.describe()
            
        elif(indicator =='Cotton'):
            #cotton = pd.read_csv("Commodities/Cotton.csv")
            cotton = pd.read_sql("SELECT * FROM Cotton",con=engine)
            cotton = cotton.set_index("Date")
            cotton = cotton.iloc[:,:5]
            cotton = cotton.rename(columns={"Open":"Cotton_Open","High":"Cotton_High","Low":"Cotton_Low",
                                       "Close":"Cotton_Close","Volume":"Cotton_Volume"})
            cotton = cotton.join(self.data,how='inner')
            cotton = cotton.dropna(axis=0)
            return cotton.corr().style.background_gradient(cmap='coolwarm'),cotton,cotton.describe()
        else:
            raise ValueError("Invalid Commodity Symbol-->",indicator)