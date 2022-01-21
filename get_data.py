from turtle import pen
import pandas as pd
import numpy as np

game_columns = ['Tm', 'Year', 'Date', 'Time', 'LTime', 'Unnamed: 6', 'Opp', 'Week', 'G#', 'Day', 'Result']


def get_individual_data_files(input_dir):
    opp_first_downs = pd.read_csv(input_dir+"opp_first_downs.csv") 
    first_downs = pd.read_csv(input_dir+"first_downs.csv")
    penalties = pd.read_csv(input_dir+"penalties.csv")

    first_downs = first_downs.drop(columns=["Rk", "OT"])
    first_downs.columns = game_columns + ['Tm_1stD', 'Tm_Rsh1stD', 'Tm_Pass1stD', 'Tm_Pen1stD', 'Tm_3DAtt', 'Tm_3DConv', 'Tm_3D%', 'Tm_4DAtt', 'Tm_4DConv', 'Tm_4D%']

    opp_first_downs = opp_first_downs.drop(columns=["Rk", "OT"])
    opp_first_downs.columns = game_columns + ['Opp_1stD', 'Opp_Rush1stD', 'Opp_Pass1stD', 'Opp_Pen1stD']

    penalties = penalties.drop(columns=["Rk", "OT"])
    penalties.columns = game_columns + ["Tm_Pen", "Tm_Yds", "Opp_Pen", "Opp_Yds", "Comb_Pen", "Comb_Yds"]

    #df_all = first_downs.merge(opp_first_downs, how="outer", on=game_columns)
    print(first_downs.merge(penalties, how="left", on=game_columns))
    # print(first_downs.info())
    # print(opp_first_downs.info())
