import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


def standardize_x(X_train, X_test, scaler=StandardScaler()):
    """
    Use the training data to have a standardizer and transform both train
    and test set.

    Parameters
    ----------
    X_train: pandas.DataFrame
        training data
    X_test: pandas.DataFrame
        test data
    scaler: sklearn.preprocessing.StandardScaler
        used to standardize train and test data column-wise 

    Returns
    -------
    X_train: pandas.DataFrame
        standardized training data
    X_test: pandas.DataFrame
        standardized test data
    """
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def get_data_ready_for_nn(train ,test):
    """
    use the train and test data to create model ready data variables

    Parameters
    ----------
    train: pandas.DataFrame
        training data passed by run.py script
    test: pandas.DataFrame
        test data passed by run.py script

    Returns
    -------
    X_train: pandas.DataFrame
        includes features for training data
    X_test: pandas.DataFrame
        includes features for test data
    y_train: pandas.Series
        outcomes for training data
    y_test: pandas.Series  
        outcomes for test data
    cn: list
        column names for our features
    """
    training_cols = ['Home', 'Tm_1stD', 'Tm_Rsh1stD', 'Tm_Pass1stD', 'Tm_Pen1stD', 'Tm_3D%', 'Comb_Pen', 'Comb_Yds', 'Opp_1stD', 'Opp_Rush1stD', 'Opp_Pass1stD', 'Opp_Pen1stD', 'Opp_PassCmp%', 'Opp_PassYds', 'Opp_PassTD', 'Opp_Int', 'Opp_Sk',
       'Opp_SkYds', 'Opp_QBRating', 'Opp_RshY/A', 'Opp_RshTD', 'Tm_Temperature', 'Tm_RshY/A', 'Tm_RshTD', 'Tm_PassCmp%', 'Tm_PassYds', 'Tm_PassTD', 'Tm_INT', 'Tm_Sk', 'Tm_SkYds', 'Tm_QBRating', 'Tm_TOP']
    X_train = train.copy()[training_cols]
    X_test = test.copy()[training_cols]

    y_train = train.copy()["Spread"]
    y_test = test.copy()["Spread"]


    X_train = feature_select(X_train)
    X_test = feature_select(X_test)
    cn = X_train.columns
    X_train, X_test = standardize_x(X_train.to_numpy(),X_test.to_numpy())

    return X_train, X_test, np.array(y_train), np.array(y_test),cn


def feature_select(covariates):
    """
    Uses our entire feature matrix to reduce the number of features by 
    subtracting Opp statistics by Tm statistics

    Parameters
    ----------
    covariates: pandas.DataFrame
        entire matrix of variables

    Returns
    -------
    pandas.DataFrame
        new feature matrix with reduced features
    """
    x = pd.DataFrame()
    x["Home"] = covariates["Home"]
    x["QBRating"]= covariates["Tm_QBRating"]-covariates["Opp_QBRating"]
    x["1stD"] = covariates["Tm_1stD"]-covariates["Opp_1stD"]
    x["Rsh1stD"] = covariates["Tm_Rsh1stD"]-covariates["Opp_Rush1stD"]
    x["Pass1stD"] = covariates["Tm_Pass1stD"]-covariates["Opp_Pass1stD"]
    x["PenaltyYds"] = covariates["Comb_Yds"]
    x["SkYds"] = covariates["Tm_SkYds"] - covariates["Opp_SkYds"]
    x["Sk"] = covariates["Tm_Sk"] - covariates["Opp_Sk"]
    x["Int"] = covariates["Tm_INT"] - covariates["Opp_Int"]
    x["PassTD"] = covariates["Tm_PassTD"] - covariates["Opp_PassTD"]
    x["PassYds"] = covariates["Tm_PassYds"] - covariates["Opp_PassYds"]
    x["PassCmp%"] = covariates["Tm_PassCmp%"] - covariates["Opp_PassCmp%"]
    x["RushY/A"] = covariates["Tm_RshY/A"] - covariates["Opp_RshY/A"]
    x["RushTD"] = covariates["Tm_RshTD"] - covariates["Opp_RshTD"]
    x["TOP"] = covariates["Tm_TOP"]
    x["Temperature"] = covariates["Tm_Temperature"]
    return x
   

def importance(clf, X, y, cn):
    """
    Uses our model and feature matrix and outcomes to determine which features provide most 
    value to our model.

    Method: permutate each individual feature separately and see how much model performance decreases 
    models with a great reduction in performance means feature is very important 

    Parameters
    ----------
    clf: sklearn.neural_network.MLPRegressor
        model used to determine permutation importance
    X: pandas.DataFrame
        feature matrix of model data
    y: pandas.Series
        observed outcomes for data
    cn: List
        column names used to label barplot

    Returns
    -------
    sns.barplot
        plots the feature importance in descending order
    """
    # use sklearn permutation importance function to get the information
    imp = permutation_importance(
        clf, X, y, scoring="neg_mean_squared_error", n_repeats=10, random_state=1234
    )

    # prepare dataframe to plot from sklearn method output
    data = pd.DataFrame(imp.importances.T)
    data.columns = cn
    order = data.agg("mean").sort_values(ascending=False).index

    # plot the information contained by the dataframe created
    fig = sns.barplot(
        x="value", y="variable", color="slateblue", data=pd.melt(data[order])
    )
    fig.set(title="Permutation Importances", xlabel=None, ylabel=None)

    return fig


def train_nn(X_train, X_test, y_train, y_test, cn):
    """
    Trains our Multi-layer Perceptron NN model using training data and outputs
    related plots as well as metrics on accuracy.

    Parameters
    ----------
    X_train: pandas.DataFrame
        train data for neural network model
    X_test: pandas.DataFrame
        test data for neural network model, last 3 NFL seasons
    y_train: pandas.Series
        observed outcomes in the training set
    y_test: pandas.Series
        observed outcomes in the test set

    Returns
    -------
    sklearn.neural_network.MLPRegressor
        our final neural network model used to predict spread
    """
    # model creation with optimal parameters for our data
    model = MLPRegressor(activation="logistic", solver="adam", early_stopping=True, validation_fraction = 0.2,
    learning_rate="adaptive", max_iter=200,alpha=0.001,hidden_layer_sizes = (32,64,64,128),random_state = 453)
 
    # fit model to training data
    model.fit(X_train, y_train)

    # print model MSE on training set
    print("MLP Regressor NN Training Set MSE: ", model.loss_)
    
    # plot model loss curve based on epochs, used early stopping
    plt.plot(model.loss_curve_)
    plt.xlabel("Epoch")
    plt.ylabel("Loss, MSE")
    plt.title("Training Loss Curve with Early Stopping")
    plt.savefig("./src/plots/training_losscurve.png")
    plt.clf()

    # code for gridsearch, commented because of large runtime (used in development)
    # param_grid = {"hidden_layer_sizes": [(5,10), (7, 5)], "alpha": [1e-3, 1e-4]}
    # clf = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=5)
    # clf.fit(X_train, np.array(y_train))
    # print(clf.best_params_)
    
    # print model MAE on test set to determine what our average error is on predictions
    print("MLP Regressor NN Test Set MAE: ", np.mean(np.abs((model.predict(X_test)) - np.array(y_test))))

    # use importance function to plot permutation importances of features in our model
    plt.figure(figsize=(10, 10))
    fig = importance(
        model, X_train, np.array(y_train), cn
    )
    plt.savefig("./src/plots/Permutation_Importances.png")
    plt.clf()

    # return nn model to use for future predictions such as superbowl
    return model
    