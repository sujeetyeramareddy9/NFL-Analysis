import numpy as np

def fix_home_col(home_col_to_fix):
   
    """
    replaces the @ and null values with a 0 or 1 to represent a home game vs away game respectively.

    

    Parameters
    ----------
    home_col_to_fix: str

    Returns
    -------
    int
        either a 0 (home game) or 1 (away game)

    """ 
    return home_col_to_fix.fillna(1).replace({"@": 0})

def fix_score_col(score):
    """
    fixes the score column.

    scores were originally written as "W15-4" or "L9-20", changed to now be the result of team's points - opponent's points

    Parameters
    ----------
    score : str
        contains the scores
    

    Returns
    -------
    int
        team score - opponent score
    """ 
    tm_opp_result = np.array(score.split()[1].split("-")).astype(int)
   
    return tm_opp_result[0] - tm_opp_result[1]

def fix_percent_col(percent):
    """
    converts the percent column from a string (due to "15%" etc) to 15.0

    Parameters
    ----------
    percent : str
        percent column
    

    Returns
    -------
    float
        the percent converted from string to float
    """ 
    if type(percent) == str:
        return float(percent.replace("%", ""))
    return percent

def preprocess_dataframe(df):
    """
    preprocesses the dataframe
    drops uncessary columns, uses column fixing functions, imputes the data and removes outliers

    Parameters
    ----------
    df : dataframe
        dataframe to be revised
   

    Returns
    -------
    dataframe
        the same dataframe just preprocessed
    """ 
    df = df.assign(
        Home = fix_home_col(df.Home),
        Spread = df.Result.apply(fix_score_col)
        #fixes the home and spread column, saves the spread as the refined score column
    )

    df["Tm_3D%"] = df["Tm_3D%"].apply(fix_percent_col)
    #applies the fix percent to both of the percent columns
    df["Opp_PassCmp%"] = df["Opp_PassCmp%"].apply(fix_percent_col)

    columns_to_drop = [
        "Year", "Result", "Tm_3DAtt", "Tm_3DConv", "Tm_4DAtt", "Tm_4DConv", 
        #list of unecessary columns
        "Tm_4D%", "Tm_Pen", "Tm_Yds", "Opp_Pen", "Opp_Yds", "Opp_PassCmp", "Opp_PassAtt", "Opp_RshAtt", "Opp_RshYds",
        "Tm_Y/P_x", "Tm_Roof", "Tm_Surface", "Tm_RshAtt", "Tm_RshYds", "Tm_PassAtt", "Tm_cmp", "Tm_TotYds", "Tm_Plys", 
        "Tm_Y/P_y", "Tm_DPlys", "Tm_DY/P", "Tm_TO", "Tm_Gametime", "Tm_Pnt", "Tm_PntYds"
    ]

    df = df.drop(columns = columns_to_drop)
    #drops unecessary columns
    df = df.assign(Tm_TOP = df["Tm_TOP"].str[:2].astype(float))
    df = impute_missing(df)
    #calls the impute_missing function to impute any missing values
    df = remove_outliers(df)
    #removes outliters
    split_train_test(df)
    #create train and test datasets

    return df

def randomly_impute(df, col):
    """
    performs imputation based on the columns that it's called on in impute_missing

    Parameters
    ----------
    df : dataframe
        the dataframe in question
    col : object (can be done on multiple datatypes)
        column to be imputed on

    Returns
    -------
    object (can be done on multiple datatyoes)
        randomly imputed column

    """ 
    imputed_col = df[col].copy()
    rnd_sample = imputed_col[~imputed_col.isnull()].sample(imputed_col.isnull().sum())
    #takes a sample of non-null rows of equal size to the # of null rows
    num_missing = rnd_sample.size
    #gives the # of missing values
    print(f"{col} has {num_missing} missing values that are being replaced with random samples from the column.")
    rnd_sample.index = df[(df[col].isnull())].index
    imputed_col[imputed_col.isnull()] = rnd_sample
    #replaces null rows with rows from the random sample
    return imputed_col

def impute_missing(df):
    """
    calls random impute on columns with missing values

    

    Parameters
    ----------
    df : dataframe
        the dataframe in question
   

    Returns
    -------
    dataframe
        refined dataframe with null values imputed

    """ 
    cols_to_impute = ['Tm_RshY/A', 'Tm_RshTD','Tm_PassCmp%','Tm_PassYds','Tm_PassTD','Tm_INT','Tm_Sk','Tm_SkYds','Tm_QBRating','Tm_TOP', 'Tm_Temperature']
    for col in cols_to_impute:
        #calls for random imputation on selected columns
        df[col] = randomly_impute(df, col)

    return df

def remove_outliers(df):
    """
    removes the outliers based on the Spread column

    

    Parameters
    ----------
    df : dataframe
        dataframe in question
    

    Returns
    -------
    dataframe
        a copy of the original dataframe with the outlier values removed
    """ 
    iqr = df.Spread.describe()["75%"] - df.Spread.describe()["25%"]
    #calculates the interquartile range
    tmp = df.copy()
    tmp["outlier"] = (df.Spread > df.Spread.describe()["75%"]+1.5*iqr) | (df.Spread < df.Spread.describe()["25%"]-1.5*iqr)
    #creates an extra column where the spread is labebled as an outlier
    num_outliers = tmp["outlier"].sum()
    #calculates the number of outliers
    print(f"{num_outliers} data points have been removed from the dataset due to them being outliers.")
    return tmp[~tmp["outlier"]].drop(columns=["outlier"])


def export_data(train, test):
    """
    exports the train and test data as csv files

    Parameters
    ----------
    train : dataframe
        train data
    test : dataframe
        test data

    Returns
    -------
    exports train/test.csv to allow for use in other py files
        
    """ 
    output_dir = "src/final_data/"
    train.to_csv(output_dir + "train.csv", index=False)
    test.to_csv(output_dir + "test.csv", index=False)


def split_train_test(df):
    """
    splits into train and test data

    every game from before 2021 (2000-2020) becomes part of the training data, games from the year 2021 become test data

    Parameters
    ----------
    df: dataframe
      
    

    Returns
    -------
    exports two dataframes based on whether or not the training label ==1 or ==0 (test data)
        
    """ 
    df["training"] = (df["Date"].dt.year <= 2019).astype(float)
    #any game before 2021 becomes train data
    export_data(df[df.training==1], df[df.training==0])

