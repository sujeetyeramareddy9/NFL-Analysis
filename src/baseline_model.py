import numpy as np

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

features = ["Home", "Tm_QBRating", "Opp_QBRating", "Tm_RshY/A", "Opp_RshY/A"]

def split_data(df):
    return df.copy()[df.training == 1], df.copy()[df.training == 0]

def split_features(df, feat=features):
    return df["Spread"], df[feat]

def mae(pred, actual):
    return np.abs(pred - actual).mean()

def build_model(train, test):
    y_train, X_train = split_features(train)
    y_test, X_test = split_features(test)

    sf = StandardScaler()
    sf.fit(X_train)

    X_train = sm.add_constant(sf.transform(X_train))
    X_test = sm.add_constant(sf.transform(X_test))

    model = sm.OLS(y_train, X_train)
    mod_fit = model.fit()
    res = mod_fit.resid
    print(mod_fit.params)
    print(mod_fit.summary())
    print("MAE: ", mae(mod_fit.predict(X_test), list(y_test)))
    sm.qqplot(res).set_size_inches(4,4)
    plt.show()
    print()
    return mod_fit