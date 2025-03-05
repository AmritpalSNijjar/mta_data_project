import sys
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None


#FIGURE OUT A WAY TO SUPRESS ERRORS/WARNINGS!


# this file should be called using the following format
# python preprocess_hourly_data.py -f file_name.csv

# read in filename
if sys.argv[1] == "-f":
    data_file_loc = sys.argv[2]
    if sys.argv[2][-4:] != ".csv":
        data_file_loc += ".csv"

print("\nLoading Dataset.....\n")
# read in file to DataFrame
df = pd.read_csv(data_file_loc, low_memory=False, nrows=10000)
print("\nDataset Loaded!\n")

### TO ADD: CHECK TO MAKE SURE INPUTTED DATASET HAS THE NECESSARY COLUMNS ###

# This dataset includes data for the subway, staten island railway, and roosevelt island tram.
df = df[df['transit_mode'] == 'subway']

# DataFrame for station iformation
station_info_df = df[["station_complex", "borough", "latitude", "longitude", "Georeference"]].drop_duplicates()

# Remove columns which are not necessary in analysis df
df = df.drop(columns=['latitude', 'longitude', 'Georeference', 'station_complex_id', 'transit_mode', 'fare_class_category'])

# convert to datetime and sort by datetime
df['transit_timestamp'] = pd.to_datetime(df['transit_timestamp'])
df = df.sort_values(by = 'transit_timestamp')
df = df.reset_index(drop = True)

# Aggregate ridership by timestamp/station
df = df.groupby(['transit_timestamp', 'station_complex'], as_index = False).agg({'ridership': 'sum'})

df.to_csv('df_cache.csv', index=False)

# Get unique weekend dates in the dataset
dates = pd.DataFrame({"date": df['transit_timestamp'].dt.strftime('%m-%d-%Y'), "day_of_week": df['transit_timestamp'].dt.weekday})
dates = dates[dates["day_of_week"] >= 5]

dates = dates.drop_duplicates()
dates["day_of_week"] = dates["day_of_week"].apply(lambda x: "Saturday" if x == 5 else "Sunday")

print("\n************************************************")
print("Weekend dates found in this dataset:\n")

for _, row in dates.iterrows():
    print(f"{row['date']} ({row['day_of_week']:^8})")

print("\n")

print("Which weekend dates would you like to look at? You may enter:\n")

print("           mm-dd-yyyy: for the specified weekend")

print("mm-dd-yyyy mm-dd-yyyy: for the specified weekends (separate dates with a space)") # implement this

print("  the name of a month: for all weekends in the specified month") # implement this
print("                  all: for all weekends in the dataset")         # implement this

print("\nOr enter EXIT to quit the script.\n")


















