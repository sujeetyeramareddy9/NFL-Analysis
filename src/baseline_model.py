import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# list of features to use in baseline model
features = ["Home", "Tm_QBRating", "Opp_QBRating", "Tm_RshTD", "Tm_Temperature", "Tm_Pass1stD"]

def split_data(df):
    """
    Split dataset into training and testing (2021 Season)

    Parameters
    ----------
    df: pandas.DataFrame
        our final dataset of all NFL games from 2000

    Returns
    -------
    pandas.DataFrame
        train dataset
    pandas.DataFrame
        test dataset
    """
    # split df into training and testing based on column created
    return df.copy()[df.training == 1], df.copy()[df.training == 0]

def split_features(df, feat=features):
    """
    Split train/test data into covariate matrix and outcome vector

    Parameters
    ----------
    df: pandas.DataFrame
        train/test data of NFL games

    Returns
    -------
    pandas.Series
        Spread of NFL games (outcome)
    pandas.DataFrame
        covariate matrix based on features listed above
    """
    # split covariate and outcome
    return df["Spread"], df[feat]

def mae(pred, actual):
    """
    Calculate MAE of model predictions

    Parameters
    ----------
    pred: pandas.Series
        vector of predictions
    actual: pandas.Sries
        vector of actual observed outcomes

    Returns
    -------
    int
        mean absolute error of our predictions
    """
    # mean absolute error
    return np.abs(pred - actual).mean()

def build_model(train, test):
    """
    Builds OLS model after standardizing coefficients as a baseline

    Parameters
    ----------
    train: pandas.DataFrame
        train data, 2000-2020 NFL Seasons
    test: pandas.DataFrame
        test data, 2021 NFL Season

    Returns
    -------
    statsmodels.regression.linear_model.RegressionResultsWrapper
        model that can contains the coefficients of our features
    """
    # split entire dataset into covariates and outcomes 
    y_train, X_train = split_features(train)
    y_test, X_test = split_features(test)
    
    # fit a standard scaler to scale covariates to align with assumptions
    sf = StandardScaler()
    sf.fit(X_train)

    # add a bias term and transform the covariate matrices of both train and test sets
    X_train = sm.add_constant(sf.transform(X_train))
    X_test = sm.add_constant(sf.transform(X_test))

    # create and fit the OLS model
    model = sm.OLS(y_train, X_train)
    mod_fit = model.fit()

    # calculate residuals and parameters for analysis
    res = mod_fit.resid
    print(mod_fit.summary())

    # calculate MAE of model predictions
    print("Linear Regression Baseline MAE: ", mae(mod_fit.predict(X_test), list(y_test)))

    # qqplot to analyze if our model as a whole follows linear regression assumptions
    sm.qqplot(res).set_size_inches(6,6)
    plt.title("Linear Regression Q-Q Plot")
    plt.savefig("./src/plots/qqplot.png")
    plt.clf()
    # shows that the model is approximately normal and significant

    # return model to allow for predictions
    return y_test, mod_fit.predict(X_test)
