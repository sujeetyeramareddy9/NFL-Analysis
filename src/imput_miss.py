
import pandas as pd
import numpy as np



def fillby_last7(tempdata):
    for i in range(tempdata):
        if np.isnan(temp[i]):
            if i <= 6:
                tempdata[i] = round(np.nanmean(tempdata[i:i+7]))
            else:
                tempdata[i] = round(np.nanmean(tempdata[i-7:i]))



def impute_missing(path):
    df = pd.read_csv(path) # read the data for checking missing
    df=df.dropna(subset=['Tm_RshY/A', 'Tm_RshTD','Tm_PassCmp%','Tm_PassYds','Tm_PassTD','Tm_INT','Tm_Sk','Tm_SkYds','Tm_QBRating','Tm_TOP']) # drop minor missing value


    df = df.sort_values(by=['Date'])

    temp=list(df.Tm_Temperature)
    fillby_last7(temp)
    df.Tm_Temperature = temp

    return df


