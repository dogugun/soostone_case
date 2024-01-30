import numpy as np
import re

from variables import CUR_YEAR


def create_train_test_sets(df):
    df_test = df[df["SALE PRICE"].str.contains("-")]
    df_train = df[df["SALE PRICE"].str.contains("-") == False]
    return df_train, df_test

def drop_empty_cols(df):
    df.drop(["EASE-MENT"], axis=1, inplace=True)
    return df

def convert_features_to_num(df):
    df.loc[df["LAND SQUARE FEET"].str.contains("-"), "LAND SQUARE FEET"] = np.nan
    df.loc[df["GROSS SQUARE FEET"].str.contains("-"), "GROSS SQUARE FEET"] = np.nan

    df["LAND SQUARE FEET"] = df["LAND SQUARE FEET"].astype("float64")
    df["GROSS SQUARE FEET"] = df["GROSS SQUARE FEET"].astype("float64")

    return df

def limit_units(df, limit=1000):
    df = df[df["RESIDENTIAL UNITS"] <= 1000]
    df = df[df["COMMERCIAL UNITS"] <= 1000]
    return df

def set_lower_bound_for_year(df):
    df.loc[df["YEAR BUILT"] < 1900, "YEAR BUILT"] = 1900
    return df

def feature_engineering(df):
    df["building_class"] = df["BUILDING CLASS CATEGORY"].apply(lambda x: ''.join(re.findall(r'\d+', x))).astype(float)
    df["tax_class"] = df["TAX CLASS AT PRESENT"].apply(lambda x: ''.join(re.findall(r'\d+', x))).replace("", "1").astype(float)
    df.drop(['BUILDING CLASS AT PRESENT'], axis=1, inplace=True)

    df["age"] = CUR_YEAR - df["YEAR BUILT"]
    return df


# Target specific preprocessing
def set_bound_for_target(df):
    df = df[df["SALE PRICE"]<10000000]
    df = df[df["SALE PRICE"] > 10000]
    return df

def convert_target_to_num(df):
    """
    run only for df_train
    :param df:
    :return:
    """
    df["SALE PRICE"] = df["SALE PRICE"].astype(float)
    return df