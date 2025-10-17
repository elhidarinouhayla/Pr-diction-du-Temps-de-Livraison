import pandas as pd
import numpy as np
import pytest
from pipeline import pretraitement_donnes, split_features, split_data
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


@pytest.fixture
def charge_data():
    data = pd.read_csv('data.csv')
    return data

# 1- Verification des dimensions des données :
def test_len(charge_data):
    data = charge_data
    data_clean =pretraitement_donnes(data)
    x,y = split_features(data_clean)
    x_test,x_train,y_test,y_train = split_data(x,y)
   
    assert len(x_test) == len(y_test)
    assert len(x_train) == len(y_train)
    


# 2- Vérification que la MAE maximale ne dépasse pas un seuil défini(RandomForestRegressor) :
def test_mae_max_RandomForestRegressor(charge_data):
    data = charge_data
    data_clean = pretraitement_donnes(charge_data)
    x,y = split_features(data_clean)
    x_test,y_test,x_train,y_train = split_data(x,y)

    model = RandomForestRegressor(random_state=42)
    model.fit(x_train,y_train)
    y_pred_model = model.predict(x_test)

    mae_model = mean_absolute_error(y_test,y_pred_model)
    assert mae_model < 7 





# 3- Vérification que la MAE maximale ne dépasse pas un seuil défini(SVR) :
def test_mae_max_SVR(charge_data):
    data = charge_data
    data_clean = pretraitement_donnes(charge_data)
    x,y = split_features(data_clean)
    x_test,y_test,x_train,y_train = split_data(x,y)

    svr = SVR()
    svr.fit(x_train,y_train)
    y_pred_svr = svr.predict(x_test)


    mae_svr = mean_absolute_error(y_test,y_pred_svr)
    assert mae_svr < 7 


