import tensorflow as tf
import numpy as np

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
    # y_train = (y_train - mu) / sigma
    # y_test = (y_test - mu) / sigma

    return tf.convert_to_tensor(X_train.to_numpy()), tf.convert_to_tensor(X_test.to_numpy()), tf.convert_to_tensor(np.array(y_train)), tf.convert_to_tensor(np.array(y_test))


def train_nn(x_train, x_test, y_train, y_test):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(x_train.shape[1])))
    model.add(tf.keras.layers.Dense(50, activation="relu"))
    model.add(tf.keras.layers.Dense(25, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="relu"))
    model.compile(optimizer="adam", loss="MSE", metrics=["mae"])

    model.fit(x_train, y_train, batch_size=64, epochs=30)

    model.evaluate(x_test, y_test)
    
    
            

    