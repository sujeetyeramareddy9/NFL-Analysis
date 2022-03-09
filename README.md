# <p align="center"> NFL-Analysis </p> 

#### <p align="center"> By: Sujeet Yeramareddy, Jonathan Langley , Yong Liu </p>

<hr />

After researching about a new inference correction approach called post-prediction inference, we chose to 
apply it to sports analysis based on NFL games. We designed a model that can predict the Spread
of a football game, such as which team will win and what the margin of their victory will be. We then analyzed the most/least
important features so that we can accurately correct inference for these variables in order to more accurately understand
their impact on our response variable, Spread.

## Usage Instructions
To use this code, please execute the following command in a terminal from the working directory of the project: `python run.py`

To use specific test target, please execute the following command: `python run.py test`

## Project Structure

```text
├── src
│   ├── data
│       ├── final_data.csv
│       ├── opp_first_downs.csv
│       ├── opp_pass_comp.csv
│       ├── opp_rush_yds.csv
│       ├── opp_total_yds.csv
│       ├── penalties.csv
│       ├── punts_temperature.csv
│       ├── tm_first_downs.csv
│       ├── tm_passing_comp.csv
│       ├── tm_rushing_yards.csv
│       ├── tm_total_yds.csv
│   ├── hextri_data
│       ├── estimates.csv
│       ├── pvals.csv
│       ├── ses.csv
│       ├── tstats.csv
│       ├── plots.R
│       ├── hextri_plots.Rproj
│   ├── plots
│       ├── histograms
│       ├── scatter
│       ├── Permutation_Importances.png
│       ├── p-value_plot.png
│       ├── postpi_Fig2.png
│       ├── postpi_Fig3.png
│       ├── postpi_Fig4.png
│       ├── qqplot.png
│   ├── test
│       ├── test.csv
│       ├── test.txt
│   ├── get_data.py
│   ├── preprocess_data.py
│   ├── baseline_model.py
│   ├── model_MLP.py
│   └── postpi.py
├── .DS_Store
├── .gitignore
├── Dockerfile
├── README.md
└── run.py
```

## Description of Files
Our project consists of these individual files:

`root`
  - `run.py`: Python file containing main method to run the project code below

`src`
  - `get_data.py`: uses separate data files and merges them together to return entire 10k row dataset
  - `preprocess_data.py`: uses final dataframe to complete preprocessing steps and split training and testing data
  - `baseline_model.py`: creates a baseline linear regression model to improve on
  - `model_MLP.py`: creates a neural network MLPregressor from sklearn library
  - `postpi.py`: uses model predictions to conduct post-prediction inference

<hr />

Data collected from: https://stathead.com/football/

GitHub Website: https://jonlangley2022.github.io/
