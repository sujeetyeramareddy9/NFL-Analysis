import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup

def get_team_stat_df(year=2021, team_stats_URL='https://www.pro-football-reference.com/years/'):
    team_stats_URL = team_stats_URL + str(year) + '/'
    team_page = requests.get(team_stats_URL)
    soup = BeautifulSoup(team_page.content, 'html.parser')

    afc = pd.read_html(str(soup.find('table', {'id': 'AFC'})))[0]
    afc = afc.drop(labels=[0,5,10,15])
    nfc = pd.DataFrame(pd.read_html(str(soup.find('table', {'id': 'NFC'})))[0])
    nfc = nfc.drop(labels=[0,5,10,15])

    def fix_playoff_marking(tm):
        tm_ret = tm.replace('*', '')
        tm_ret = tm_ret.replace('+', '')
        return tm_ret
    afc['Tm'] = afc['Tm'].apply(fix_playoff_marking)
    nfc['Tm'] = nfc['Tm'].apply(fix_playoff_marking)
    team = pd.concat([afc, nfc])
    team = team.drop(columns=['W', 'L', 'PF', 'PA'])
    team = team.set_index('Tm')
    team['W-L%'] = team['W-L%'].astype(float)
    team['PD'] = team['PD'].astype(int)
    team['MoV'] = team['MoV'].astype(float)
    team['SoS'] = team['SoS'].astype(float)
    team['SRS'] = team['SRS'].astype(float)
    team['OSRS'] = team['OSRS'].astype(float)
    team['DSRS'] = team['DSRS'].astype(float)

    return team


def get_schedule(years, schedule_URL='https://www.pro-football-reference.com/years/'):
    training_schedule = pd.DataFrame()
    for year in years:
        train_schedule_URL = schedule_URL + str(year) + '/games.htm'
        train_schedule_page = requests.get(train_schedule_URL)
        soup = BeautifulSoup(train_schedule_page.content, 'html.parser')

        train_schedule = pd.DataFrame(pd.read_html(str(soup.find('table', {'id': 'games'})))[0])
        train_schedule['HomeWin'] = train_schedule['Unnamed: 5'].replace({np.NaN: 1, '@': 0})
        train_schedule = train_schedule.drop(columns=['Unnamed: 5'])
        def remove_playoffs(wk):
            try:
                return int(wk)
            except:
                return
        train_schedule['Week'] = train_schedule['Week'].apply(remove_playoffs)
        train_schedule = train_schedule.dropna()
        train_schedule['PtsW'] = train_schedule['PtsW'].astype(float)
        train_schedule['PtsL'] = train_schedule['PtsL'].astype(float)
        training_schedule = pd.concat([training_schedule, train_schedule], ignore_index=True)

    test_schedule_URL = schedule_URL + str(2021) + '/games.htm'
    test_schedule_page = requests.get(test_schedule_URL)
    soup = BeautifulSoup(test_schedule_page.content, 'html.parser')

    test_schedule = pd.DataFrame(pd.read_html(str(soup.find('table', {'id': 'games'})))[0])
    test_schedule['HomeWin'] = test_schedule['Unnamed: 5'].replace({np.NaN: 1, '@': 0})
    test_schedule = test_schedule.drop(columns=['Unnamed: 5'])

    old_teams = {
        "Washington Redskins": "Washington Football Team", 'Oakland Raiders': 'Las Vegas Raiders', 
        'San Diego Chargers': 'Los Angeles Chargers', 'St. Louis Rams': 'Los Angeles Rams'
    }
    training_schedule['Winner/tie'] = training_schedule['Winner/tie'].replace(old_teams)
    training_schedule['Loser/tie'] = training_schedule['Loser/tie'].replace(old_teams)
    test_schedule['Winner/tie'] = test_schedule['Winner/tie'].replace(old_teams)
    test_schedule['Loser/tie'] = test_schedule['Loser/tie'].replace(old_teams)
    test_schedule['Week'] = test_schedule['Week'].apply(remove_playoffs)
    test_schedule = test_schedule[test_schedule['PtsW']!='PtsW']
    test_schedule['PtsW'] = test_schedule['PtsW'].astype(float)
    test_schedule['PtsL'] = test_schedule['PtsL'].astype(float)

    return training_schedule, test_schedule