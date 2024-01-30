import os
import pickle

from sklearn.ensemble import RandomForestRegressor

from src.data_access.data_ops import get_file_dataset
from src.preprocessing.preprocess_dataset import create_train_test_sets, drop_empty_cols, convert_features_to_num, \
    limit_units, convert_target_to_num, set_lower_bound_for_year, set_bound_for_target, feature_engineering
from variables import UNIV_NUM_COLS
import numpy as np

BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    errors = np.abs((actual - predicted) / (actual+1))
    mape = np.mean(errors) * 100
    return mape

def prepare_data():
    df = get_file_dataset()


    df = drop_empty_cols(df)
    df = convert_features_to_num(df)

    df = limit_units(df)
    df = set_lower_bound_for_year(df)

    df = feature_engineering(df)

    df_train, df_test = create_train_test_sets(df)
    df_train = convert_target_to_num(df_train)
    df_train = set_bound_for_target(df_train)

    df_train = df_train.dropna(axis=0)
    df_test = df_test.dropna(axis=0)

    print(df_train.shape, df_test.shape)
    return df_train, df_test

def train_model(df_train):
    bivar_cols = UNIV_NUM_COLS + ["building_class", "tax_class", "age"]

    x = df_train[bivar_cols].drop(["YEAR BUILT", "TAX CLASS AT TIME OF SALE", "TOTAL UNITS"], axis=1)
    y = df_train["SALE PRICE"]

    model = RandomForestRegressor(n_estimators=250, max_depth=10, random_state=42)
    model.fit(x, y)

    model_filename = os.path.join(BASEDIR, "models", "rf_model.sav")
    with open(model_filename, 'wb') as pickle_file:
        pickle.dump(model, pickle_file)


