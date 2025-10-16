import pandas as pd
import numpy as np
import pytest
from pipeline import pretraitement_donnes, split_features, split_data

@pytest.fixture
def charge_data():
    data = pd.read_csv('data.csv')
    return data

def test(charge_data):
    data = charge_data
    data_clean =pretraitement_donnes(charge_data)
    x,y = split_features(data_clean)
    x_test,x_train,y_test,y_train = split_data(x,y)
   
    assert len(x_test) == len(y_test)
    assert len(x_train) == len(y_train)
    


