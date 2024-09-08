import pandas as pd
import numpy as np
import re
import yaml
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

with open("config.yaml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    lower_case = cfg['Preprocessing']['lower_case']

def load_data(input_data_file):
    df = pd.read_csv(input_data_file, delimiter="\t")
    df["Title"] = df["Title"].fillna("")
    df["Abstract"] = df["Abstract"].fillna("")
    df["Abstract"] = df["Title"] + ' ' + df["Abstract"]
    X = list(df["Abstract"])
    # X = [preprocess_text(elem) for elem in X]
    X = [re.sub(r"[\W]+", " ", elem) for elem in X]
    X = [re.sub(r"[\n\r\t ]+", " ", elem) for elem in X]
    if lower_case:
        X = [elem.lower() for elem in X]
    X = np.asarray(X)

    y = list(df["Label"])
    y = np.asarray(y)

    return X, y


def load_pico_file(input_data_file):
    df = pd.read_csv(input_data_file)
    # could this only be used replaced by
    df['pico_text'] = df_pico= df['pico_text'] .fillna(df['Original_text'])
    #   np.where(df['pico_text'] == '', df['Original_text'])
    X = list(df["pico_text"])
    # X = [preprocess_text(elem) for elem in X]
    X = [re.sub(r"[\W]+", " ", elem) for elem in X]
    X = [re.sub(r"[\n\r\t ]+", " ", elem) for elem in X]
    X = [elem.lower() for elem in X]
    X = np.asarray(X)
    return X


def split_train_test_data(X, y, test_size=0.5, seed=18):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        stratify=y,
                                                        test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test
