import pandas as pd
import os

BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


def get_file_dataset():
    filepath = os.path.join(BASEDIR, "data", "nyc-rolling-sales.csv")
    df = pd.read_csv(filepath)
    return df