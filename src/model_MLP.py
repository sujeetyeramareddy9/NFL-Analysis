import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

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

    scaling_features = ['Tm_1stD', 'Tm_Rsh1stD', 'Tm_Pass1stD', 'Tm_Pen1stD', 'Tm_3D%', 'Comb_Pen', 'Comb_Yds', 'Opp_1stD', 'Opp_Rush1stD', 'Opp_Pass1stD', 'Opp_Pen1stD', 'Opp_PassCmp%', 'Opp_PassYds', 'Opp_PassTD', 'Opp_Int', 'Opp_Sk',
       'Opp_SkYds', 'Opp_QBRating', 'Opp_RshY/A', 'Opp_RshTD', 'Tm_Temperature', 'Tm_RshY/A', 'Tm_RshTD', 'Tm_PassCmp%', 'Tm_PassYds', 'Tm_PassTD', 'Tm_INT', 'Tm_Sk', 'Tm_SkYds', 'Tm_QBRating', 'Tm_TOP']
    for scaling_col in scaling_features:
        mu = X_train[scaling_col].mean()
        sigma = X_train[scaling_col].std()
        X_train[scaling_col] = (X_train[scaling_col] - mu) / sigma
        X_test[scaling_col] = (X_test[scaling_col] - mu) / sigma

    # mu = y_train.mean()
    # sigma = y_train.std()
    # print("Std. Dev. of Spread: ", sigma)
    # y_train = (y_train - mu) / sigma
    # y_test = (y_test - mu) / sigma

    return X_train.to_numpy(), X_test.to_numpy(), np.array(y_train), np.array(y_test), X_train.columns


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


def train_nn(X_train, X_test, y_train, y_test, cn):
    model = MLPRegressor(activation="relu", solver="sgd", early_stopping=True, learning_rate="adaptive", max_iter=750)
    model.fit(X_train, np.array(y_train))

    print("Training Set MSE: ", model.loss_)
    
    param_grid = {"hidden_layer_sizes": [(5,10), (7, 5)], "alpha": [1e-3, 1e-4]}

    clf = GridSearchCV(model, param_grid, scoring="neg_mean_squared_error", cv=5)
    clf.fit(X_train, np.array(y_train))
    print(clf.best_params_)
    
    print("Test Set MAE: ", np.mean(np.abs((clf.predict(X_test)) - np.array(y_test))))

    fig = importance(
        clf, X_train, np.array(y_train), cn
    )
    plt.savefig("./src/plots/Permutation_Importances.png")

    return clf
    
            

    