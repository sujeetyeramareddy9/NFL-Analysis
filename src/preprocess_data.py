import numpy as np
import pandas as pd

def fix_home_col(home_col_to_fix):
    return home_col_to_fix.fillna(1).replace({"@": 0})

def fix_score_col(score):
    tm_opp_result = np.array(score.split()[1].split("-")).astype(int)
    return tm_opp_result[0] - tm_opp_result[1]

def fix_percent_col(percent):
    if type(percent) == str:
        return float(percent.replace("%", ""))
    return percent

def preprocess_dataframe(df):
    df = df.assign(
        Home = fix_home_col(df.Home),
        Spread = df.Result.apply(fix_score_col)
    )

    df["Tm_3D%"] = df["Tm_3D%"].apply(fix_percent_col)
    df["Opp_PassCmp%"] = df["Opp_PassCmp%"].apply(fix_percent_col)

    columns_to_drop = [
        "Year", "Result", "Tm_3DAtt", "Tm_3DConv", "Tm_4DAtt", "Tm_4DConv", 
        "Tm_4D%", "Tm_Pen", "Tm_Yds", "Opp_Pen", "Opp_Yds", "Opp_PassCmp", "Opp_PassAtt", "Opp_RshAtt", "Opp_RshYds",
        "Tm_Y/P_x", "Tm_Roof", "Tm_Surface", "Tm_RshAtt", "Tm_RshYds", "Tm_PassAtt", "Tm_cmp", "Tm_TotYds", "Tm_Plys", 
        "Tm_Y/P_y", "Tm_DPlys", "Tm_DY/P", "Tm_TO", "Tm_Gametime", "Tm_Pnt", "Tm_PntYds"
    ]

    df = df.drop(columns = columns_to_drop)
    df = df.assign(Tm_TOP = df["Tm_TOP"].str[:2].astype(float))
    df = impute_missing(df)
    df = remove_outliers(df)
    split_train_test(df)

    return df

def randomly_impute(df, col):
    imputed_col = df[col].copy()
    rnd_sample = imputed_col[~imputed_col.isnull()].sample(imputed_col.isnull().sum())
    num_missing = rnd_sample.size
    print(f"{col} has {num_missing} missing values that are being replaced with random samples from the column.")
    rnd_sample.index = df[(df[col].isnull())].index
    imputed_col[imputed_col.isnull()] = rnd_sample
    return imputed_col

def impute_missing(df):
    cols_to_impute = ['Tm_RshY/A', 'Tm_RshTD','Tm_PassCmp%','Tm_PassYds','Tm_PassTD','Tm_INT','Tm_Sk','Tm_SkYds','Tm_QBRating','Tm_TOP', 'Tm_Temperature']
    for col in cols_to_impute:
        df[col] = randomly_impute(df, col)

    return df

def remove_outliers(df):
    iqr = df.Spread.describe()["75%"] - df.Spread.describe()["25%"]
    tmp = df.copy()
    tmp["outlier"] = (df.Spread > df.Spread.describe()["75%"]+1.5*iqr) | (df.Spread < df.Spread.describe()["25%"]-1.5*iqr)
    num_outliers = tmp["outlier"].sum()
    print(f"{num_outliers} data points have been removed from the dataset due to them being outliers.")
    return tmp[~tmp["outlier"]].drop(columns=["outlier"])


def split_train_test(df):
    df["training"] = (df["Date"].dt.year <= 2020).astype(float)
