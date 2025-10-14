import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder



# 1- Charges des donnes:
def charge_donnes():
    data =  pd.read_csv('data.csv')
    return data


def pretraitement_donnes(data):

# 1- Remplaser les valeurs manquantes:
    mean = data['Courier_Experience_yrs'].mean()
    data['Courier_Experience_yrs'] = data['Courier_Experience_yrs'].fillna(mean)

# 2- Supprimer les colonnes inutiles:
    data = data.drop(columns=['Order_ID', 'Courier_Experience_yrs', 'Vehicle_Type', 'Time_of_Day'])

# 3- Encoder les variables categprielles:
    ca_columns = ['Weather', 'Traffic_Level']
    for col in ca_columns:
        