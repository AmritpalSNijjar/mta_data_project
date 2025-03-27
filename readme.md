The notebook file (mta_hourly_ridership_analysis.ipynb) gives a walkthrough of my analysis, performed on data obtained from https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-2025/5wq4-mkjj/about_data, looking at just the month of January 2025.

If you would like to perform the same analysis on a different month or month(s), and generate the same plots for a specific weekend date, obtain the respective MTA Subway Hourly Ridership .csv file from *data.ny.gov*, and run the following line in the terminal:

`python model.py -f filename.csv`

The file **MUST** be an official 'MTA Subway Hourly Ridership' .csv file, because the analysis depends on the data being structured as such.

When prompted for a specific weekend date for which to generate the maps, enter the weekend in the following format:

`mm-dd-yyyy`

Thank you!
