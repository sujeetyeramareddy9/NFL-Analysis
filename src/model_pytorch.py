import torch
import torch.nn.functional as F
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

    return torch.FloatTensor(X_train.to_numpy()), torch.FloatTensor(X_test.to_numpy()), torch.FloatTensor(np.array(y_train)), torch.FloatTensor(np.array(y_test))


class Net(torch.nn.Module):
    def __init__(self , input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, 15)
        self.fc3 = torch.nn.Linear(15, 1)

    def forward(self, x):
        x = F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x))))))
        return x


def train_nn(x_train, x_test, y_train, y_test):
    input_size = x_train.size()[1]
    hidden_size = 50
    model = Net(input_size, hidden_size)
    criterion = torch.nn.MSELoss()
    # without momentum parameter
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3) # adam # lr # lr_shceduler # sigmoid

    model.train()
    epochs = 500
    errors = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred.squeeze(), y_train)
        errors.append(loss.item())
        print(f"Epoch {epoch}: Train Loss: {loss}")
        loss.backward()
        optimizer.step()

    model.eval()
    y_pred = model(x_test)
    after_train = criterion(y_pred.squeeze(), y_test)
    print('Test loss after Training' , after_train.item())