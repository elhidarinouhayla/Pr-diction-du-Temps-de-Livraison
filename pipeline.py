import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler



# 1- Charges des donnes:
def charge_donnes():
    data =  pd.read_csv('data.csv')
    return data
data = charge_donnes()
# print(data)

def pretraitement_donnes(data):

# 1- Remplaser les valeurs manquantes:
    mean = data['Courier_Experience_yrs'].mean()
    data['Courier_Experience_yrs'] = data['Courier_Experience_yrs'].fillna(mean)

    for column in data:
        if data[column].dtypes == 'object':
            data[column] = data[column].fillna(data[column].mode()[0])



# 2- Supprimer les colonnes inutiles:
    data = data.drop(columns=['Order_ID', 'Courier_Experience_yrs', 'Vehicle_Type', 'Time_of_Day'])

# 3- Encoder les variables categprielles:
    ca_columns = ['Weather', 'Traffic_Level']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder_columns = encoder.fit_transform(data[ca_columns])
    encoder_f=pd.DataFrame(encoder_columns,columns=encoder.get_feature_names_out(ca_columns))
    data = pd.concat([data.drop(columns= ca_columns),encoder_f],axis=1)
    return data
clean_data = pretraitement_donnes(data)


# # 1- concat clean_data et num_columns:
# num_columns = data[['Distance_km', 'Preparation_Time_min', 'Delivery_Time_min']]
# clean_columns = pd.concat([clean_data, num_columns], axis=1)
# print(clean_columns)


def split_features(data):
# 1- Separation de features et la cible(Delivery_Time_min):
    x = data.drop('Delivery_Time_min', axis=1)
    y = data['Delivery_Time_min']
    return x,y
x,y = split_features(clean_data)


def split_data(x,y):
# 1- train_test_split:
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
    return x_train,x_test,y_train,y_test
x_train,x_test,y_train,y_test = split_data(x,y)


def normalized_data(x_train,x_test):
# 1- normalisation :
    scaler = StandardScaler()
    x_train_scaler = scaler.fit_transform(x_train)
    x_test_scaler = scaler.transform(x_test)
    return x_train_scaler,x_test_scaler
x_train_scaled, x_test_scaled = normalized_data(x_train, x_test)


def selectKBest_features(x_train,y_train):
# 1-  les 5 meilleures features:
    selector =  SelectKBest(score_func=f_regression, k=5)
    x_new = selector.fit_transform(x_train, y_train)
    selected_features = np.array(x_train.columns)[selector.get_support()]
    return x_new

x_train,x_test,y_train,y_test = split_data(x,y)

x_new = selectKBest_features(x_train, y_train)






