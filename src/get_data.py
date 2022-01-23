from datetime import datetime
import pandas as pd
import numpy as np


dtypes = {"Date": str}
parse_dates = list(dtypes.keys())

def preprocess_game_columns(df):
    df["Date"] = df["Date"] + " " +  df["Time"]
    df.drop(columns=["Time", "LTime"], inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def get_individual_data_files(input_dir):
    opp_first_downs = pd.read_csv(input_dir+"opp_first_downs.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})
    first_downs = pd.read_csv(input_dir+"tm_first_downs.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})
    penalties = pd.read_csv(input_dir+"penalties.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})
    
    game_columns = ['Tm', 'Year', 'Date', 'Home', 'Opp', 'Week', 'G#', 'Day', 'Result']
    first_downs = preprocess_game_columns(first_downs)
    opp_first_downs = preprocess_game_columns(opp_first_downs)
    penalties = preprocess_game_columns(penalties)

    first_downs.columns = game_columns + ['Tm_1stD', 'Tm_Rsh1stD', 'Tm_Pass1stD', 'Tm_Pen1stD', 'Tm_3DAtt', 'Tm_3DConv', 'Tm_3D%', 'Tm_4DAtt', 'Tm_4DConv', 'Tm_4D%']
    opp_first_downs.columns = game_columns + ['Opp_1stD', 'Opp_Rush1stD', 'Opp_Pass1stD', 'Opp_Pen1stD']
    penalties.columns = game_columns + ["Tm_Pen", "Tm_Yds", "Opp_Pen", "Opp_Yds", "Comb_Pen", "Comb_Yds"]

    df_first_third = first_downs.merge(penalties, how="left", on=game_columns).merge(opp_first_downs, how="left", on=game_columns)
    print(df_first_third)
    