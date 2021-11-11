from get_data import *
from validation_correction import *
from model_win_loss import *
from model_over_under import *
from model_spread import *
from model_pytorch import *
import os


def main():
    pytorch = False
    predict = 'O-U'

    if pytorch:
        practice()
    else:
        if predict == 'W-L':
            years = [2020]
            teams = get_team_stat_df()
            train_schedule, test_schedule = get_schedule(years)
            model = get_model_for_win_loss(train_schedule, test_schedule, teams, 10)
            predict_win_loss(test_schedule, teams, 10, model)
        elif predict == 'O-U':
            years = [2019, 2020]
            teams = get_team_stat_df()
            train_schedule, test_schedule = get_schedule(years)
            inferential_model, relationship_model = get_model_for_over_under(train_schedule, test_schedule, teams, 10)
            X_val, y_val, validation = predict_over_under(test_schedule, teams, 10, inferential_model)
            beta_bs = post_prediction_correction(X_val, y_val, relationship_model)
            validation['y_val'] = correct_validation(beta_bs, X_val).astype(int)
            print(validation)
        elif predict == 'spread':
            years = [2019, 2020]
            teams = get_team_stat_df()
            train_schedule, test_schedule = get_schedule(years)
            inferential_model, relationship_model = get_model_for_spread(train_schedule, test_schedule, teams, 9)
            X_val, y_val, validation = predict_spread(test_schedule, teams, 9, inferential_model)
            beta_bs = post_prediction_correction(X_val, y_val, relationship_model)
            validation['y_val'] = correct_validation(beta_bs, X_val).astype(int)
            print(validation)

if __name__ == '__main__':
    main()
