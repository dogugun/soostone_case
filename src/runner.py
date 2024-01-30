from model.price_model import prepare_data, train_model
from scoring.score import load_model
import os

from variables import UNIV_NUM_COLS

BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

df_train, df_test = prepare_data()

train_model(df_train)
model = load_model()
bivar_cols = UNIV_NUM_COLS + ["building_class", "tax_class", "age"]
df_score = df_test[bivar_cols].drop(["YEAR BUILT", "TAX CLASS AT TIME OF SALE", "TOTAL UNITS"], axis=1)
pred = model.predict(df_score)
df_test["SALE PRICE"] = pred

save_path = os.path.join(BASEDIR, "data", "output", "result.csv")
df_test.to_csv(save_path, index=False)