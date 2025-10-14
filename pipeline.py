import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split




# 1- Charges des donnes:
def charge_donnes():
    data =  pd.read_csv('data.csv')
    return data
data = charge_donnes()


def pretraitement_donnes(data):

# 1- Remplaser les valeurs manquantes:
    mean = data['Courier_Experience_yrs'].mean()
    data['Courier_Experience_yrs'] = data['Courier_Experience_yrs'].fillna(mean)

# 2- Supprimer les colonnes inutiles:
    data = data.drop(columns=['Order_ID', 'Courier_Experience_yrs', 'Vehicle_Type', 'Time_of_Day'])

# 3- Encoder les variables categprielles:
    ca_columns = ['Weather', 'Traffic_Level']
    encoder = OneHotEncoder()
    encoder_columns = encoder.fit_transform(data[['Weather', 'Traffic_Level']])
    return data
clean_data = pretraitement_donnes(data)


# 1- concat clean_data et num_columns:
num_columns = data[['Distance_km', 'Preparation_Time_min', 'Delivery_Time_min']]
clean_columns = pd.concat([clean_data, num_columns], axis=0)
print(clean_columns)


def split_features(data):
# 1- Separation de features et la cible(Delivery_Time_min):
    x = data.drop('Delivery_Time_min', axis=1)
    y = data['Delivery_Time_min']
    return(x,y)
x,y = split_features(clean_data)

def split_data(x,y):
# 1- train_test_split:
    x_train,x_test,y_tain,y_test = train_test_split(x,y, test_siz=0.2)
    return(x_train,x_test,y_tain,y_test)
x_train,x_test,y_tain,y_test = split_data(x,y)



