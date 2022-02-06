import pandas as pd

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

features = ["Tm_QBRating", "Opp_QBRating", "Home"]

def split_data(df):
    return df.copy()[df.training == 1], df.copy()[df.training == 0]

def split_features(df, feat=features):
    return df["Spread"], df[feat]

def base_model(df):
    train, test = split_data(df)
    y_train, X_train = split_features(train)
    y_test, X_test = split_features(test)

    sf = StandardScaler()
    sf.fit(X_train)

    X_train = sm.add_constant(sf.transform(X_train))

    model = sm.OLS(y_train, X_train)
    results = model.fit()
    print(results.params_)