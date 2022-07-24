# -*- coding: utf-8 -*-
"""
Created on Thu May 14 18:12:45 2020

@author: Ajay Solanki
"""


import quandl
api_key = "HHCBs8CFrnTXyu__s7xv"
scriptcode="RELIANCE"
df = quandl.get("NSE/" + scriptcode, api_key =api_key)

df