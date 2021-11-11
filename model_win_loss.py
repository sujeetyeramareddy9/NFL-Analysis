import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import pandas as pd

def get_model_for_win_loss(training_schedule, test_schedule, teams, week):
    X_train = []
    y_train = []
    features_to_include = teams.columns
    training = training_schedule[['Winner/tie', 'Loser/tie', 'HomeWin']]

    for row in training.iterrows():
        w = row[1][0]
        l = row[1][1]
        # w_vec = list(teams.loc[w, features_to_include])
        # l_vec = list(teams.loc[l, features_to_include])
        home_win = row[1][2]
        if home_win == 1:
            w_vec = list(teams.loc[w, features_to_include]) + [1]
            l_vec = list(teams.loc[l, features_to_include]) + [0]
        if home_win == 0:
            w_vec = list(teams.loc[w, features_to_include]) + [0]
            l_vec = list(teams.loc[l, features_to_include]) + [1]

        X_train.append([1] + w_vec + l_vec)
        X_train.append([1] + l_vec + w_vec)
        y_train.append(1)
        y_train.append(0)

    model = LogisticRegression(C=5, max_iter=25000)
    model.fit(X_train, y_train)

    X_test = []
    y_test = []
    features_to_include = teams.columns
    testing = test_schedule[test_schedule['Week']<week][['Winner/tie', 'Loser/tie', 'HomeWin']]
    
    for row in testing.iterrows():
        w = row[1][0]
        l = row[1][1]
        # w_vec = list(teams.loc[w, features_to_include])
        # l_vec = list(teams.loc[l, features_to_include])
        home_win = row[1][2]
        if home_win == 1:
            w_vec = list(teams.loc[w, features_to_include]) + [1]
            l_vec = list(teams.loc[l, features_to_include]) + [0]
        if home_win == 0:
            w_vec = list(teams.loc[w, features_to_include]) + [0]
            l_vec = list(teams.loc[l, features_to_include]) + [1]
        X_test.append([1] + w_vec + l_vec)
        X_test.append([1] + l_vec + w_vec)
        y_test.append(1)
        y_test.append(0)

    y_pred = model.predict(X_test)
    print("Test Accuracy: " + str(np.mean(y_pred == y_test)))
    return model


def predict_win_loss(test_schedule, teams, week, model):
    X_val = []
    features_to_include = teams.columns
    validation = test_schedule[test_schedule['Week']==week][['Winner/tie', 'Loser/tie']]

    for row in validation.iterrows():
        a = row[1][0]
        h = row[1][1]
        # a_vec = list(teams.loc[a, features_to_include])
        # h_vec = list(teams.loc[h, features_to_include])
        a_vec = list(teams.loc[a, features_to_include]) + [0]
        h_vec = list(teams.loc[h, features_to_include]) + [1]
        X_val.append([1] + a_vec + h_vec)
    
    y_val = model.predict(X_val)
    validation['prediction'] = y_val
    print(validation)