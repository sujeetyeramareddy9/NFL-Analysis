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
    # first third of our data files
    opp_first_downs = pd.read_csv(input_dir+"opp_first_downs.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})
    tm_first_downs = pd.read_csv(input_dir+"tm_first_downs.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})
    penalties = pd.read_csv(input_dir+"penalties.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})

    # second third of our data files
    tm_pass_comp = pd.read_csv(input_dir+"tm_pass_comp.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})
    tm_rush_yds = pd.read_csv(input_dir+"tm_rush_yds.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})
    tm_tot_yds = pd.read_csv(input_dir+"tm_tot_yds.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})

    # final third of our data files
    opp_pass_comp = pd.read_csv(input_dir+"opp_pass_comp.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})
    opp_rush_yds = pd.read_csv(input_dir+"opp_rush_yds.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})
    opp_tot_yds = pd.read_csv(input_dir+"opp_total_yds.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})
    punts_temperature = pd.read_csv(input_dir+"punts_temperature.csv").drop(columns=["Rk", "OT"]).rename(columns={"Unnamed: 6": "Home"})

    tm_first_downs = preprocess_game_columns(tm_first_downs)
    opp_first_downs = preprocess_game_columns(opp_first_downs)

    tm_pass_comp = preprocess_game_columns(tm_pass_comp)
    opp_pass_comp = preprocess_game_columns(opp_pass_comp)

    tm_rush_yds = preprocess_game_columns(tm_rush_yds)
    opp_rush_yds = preprocess_game_columns(opp_rush_yds)

    tm_tot_yds = preprocess_game_columns(tm_tot_yds)
    opp_tot_yds = preprocess_game_columns(opp_tot_yds)

    penalties = preprocess_game_columns(penalties)
    punts_temperature = preprocess_game_columns(punts_temperature)

    game_columns = ['Tm', 'Year', 'Date', 'Home', 'Opp', 'Week', 'G#', 'Day', 'Result']
    tm_first_downs.columns = game_columns + ['Tm_1stD', 'Tm_Rsh1stD', 'Tm_Pass1stD', 'Tm_Pen1stD', 'Tm_3DAtt', 'Tm_3DConv', 'Tm_3D%', 'Tm_4DAtt', 'Tm_4DConv', 'Tm_4D%']
    opp_first_downs.columns = game_columns + ['Opp_1stD', 'Opp_Rush1stD', 'Opp_Pass1stD', 'Opp_Pen1stD']
    penalties.columns = game_columns + ["Tm_Pen", "Tm_Yds", "Opp_Pen", "Opp_Yds", "Comb_Pen", "Comb_Yds"]

    opp_pass_comp.columns = game_columns + ["Opp_PassCmp","Opp_PassAtt","Opp_PassCmp%","Opp_PassYds","Opp_PassTD","Opp_Int","Opp_Sk","Opp_SkYds","Opp_QBRating"]
    opp_rush_yds.columns = game_columns + ["Opp_RshAtt","Opp_RshYds","Opp_RshY/A","Opp_RshTD"]
    punts_temperature.columns = ["Tm_Pnt","Tm_PntYds","Tm_Y/P","Tm_Surface","Tm_Roof","Tm_Temperature"]

    tm_rush_yds.columns = game_columns + ["Tm_RshAtt", "Tm_RshYds", "Tm_RshY/A", "Tm_RshTD"]
    tm_pass_comp.columns = game_columns + ["Tm_PassAtt", "Tm_PassCmp%", "Tm_PassYds", "Tm_PassTD", "Tm_INT","Tm_Sk","Tm_SkYds","Tm_QBRating"]
    tm_tot_yds.columns = game_columns + ["Tm_TotYds","Tm_Plys","Tm_Y/P","Tm_DPlys","Tm_DY/P", "Tm_TO","Tm_TOP","Tm_Gametime"]

    first_third = tm_first_downs.merge(penalties, how="left", on=game_columns).merge(opp_first_downs, how="left", on=game_columns)

    print(first_third)
    