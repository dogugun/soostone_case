import pickle
import os

BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

def load_model():
    model_filename = os.path.join(BASEDIR, "models", "rf_model.sav")

    model = pickle.load(open(model_filename, 'rb'))
    # with open(model_filename, 'wb') as pickle_file:
    #     model = pickle.load(open(model_filename, 'rb'))
    return model


def score(x, model):
    y_pred = model.score(x)
    return y_pred
