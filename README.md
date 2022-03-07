# NFL analysis 
## By: Sujeet Yeramareddy, Jonathan Langley , Yong Liu


After researching about a new inference correction approach called post-prediction inference, we chose to 
apply it to sports analysis based on NFL games. We designed a model that can predict the Spread
of a football game, such as which team will win and what the margin of their victory will be. We then analyzed the most/least
important features so that we can accurately correct inference for these variables in order to more accurately understand
their impact on our response variable, Spread.


Data collected from: https://stathead.com/football/

Website: https://jonlangley2022.github.io/



```text
DIRECTORY STRUCTURE

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
