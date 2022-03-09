import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split



def standardize_x(X_train, X_test, scaler=StandardScaler()):
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def get_data_ready_for_nn(train ,test):
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
    imp = permutation_importance(
        clf, X, y, scoring="neg_mean_squared_error", n_repeats=10, random_state=1234
    )

    data = pd.DataFrame(imp.importances.T)
    data.columns = cn
    order = data.agg("mean").sort_values(ascending=False).index
    fig = sns.barplot(
        x="value", y="variable", color="slateblue", data=pd.melt(data[order])
    )
    fig.set(title="Permutation Importances", xlabel=None, ylabel=None)
    return fig


def train_nn(X_train, X_test, Y_train, y_test, cn):
    model = MLPRegressor(activation="logistic", solver="adam", early_stopping=True, validation_fraction = 0.2,
    learning_rate="adaptive", max_iter=200,alpha=0.001,hidden_layer_sizes = (32,64,64,128),random_state = 453)
 
    model.fit(X_train, Y_train)

    print("MLP Regressor NN Training Set MSE: ", model.loss_)
    
    plt.plot(model.loss_curve_)
    plt.xlabel("epoch")
    plt.ylabel("current loss")
    plt.savefig("./src/plots/training_losscurve.png")

    # param_grid = {"hidden_layer_sizes": [(5,10), (7, 5)], "alpha": [1e-3, 1e-4]}
    # clf = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=5)
    # clf.fit(X_train, np.array(y_train))
    # print(clf.best_params_)
    
    print("MLP Regressor NN Test Set MAE: ", np.mean(np.abs((model.predict(X_test)) - np.array(y_test))))

    fig = importance(
        model, X_train, np.array(Y_train), cn
    )
    plt.savefig("./src/plots/Permutation_Importances.png")

    return model
    
            

    