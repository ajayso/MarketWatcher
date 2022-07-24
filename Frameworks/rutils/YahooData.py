import yfinance as yf
import pandas as pd


class Puller:
    def __init__(self, store_data):
        self.store_data = store_data
    def get_history(self,script_code, fetch_script):
        data = yf.Ticker(fetch_script)
        historical_data = data.history(period="max")
        names = historical_data.columns.tolist()
        for name in names:
            if name is not "Date":
                names[names.index(name)] = script_code + "_" + name
        historical_data.columns = names
        return historical_data


