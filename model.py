import sys
import datetime
import numpy as np
import pandas as pd
import seaborn as sns

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib.patches as mpatches

from scipy.fft import fft, ifft
from scipy import stats

#from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA

from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

# this file should be called using the following format
# python preprocess_hourly_data.py -f file_name.csv

# read in filename
if sys.argv[1] == "-f":
    data_file_loc = sys.argv[2]
    if sys.argv[2][-4:] != ".csv":
        data_file_loc += ".csv"
        
# read in file to DataFrame

print("\nLoading Dataset.....")
df = pd.read_csv(data_file_loc, low_memory=False)
print("\nDataset Loaded!")

print("Transforming data.....")

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



df["transit_hour"] = df["transit_timestamp"].dt.hour
df["transit_day"] = df.apply(lambda x: x["transit_timestamp"].strftime('%m-%d-%Y'), axis = 1)

# Keep weekends, drop weekdays
df["is_weekends"] = df.apply(lambda x: x["transit_timestamp"].weekday(), axis = 1)
df = df[df["is_weekends"] >= 5]

# Get unique weekend dates in the dataset
weekend_dates = pd.DataFrame({"date": df['transit_timestamp'].dt.strftime('%m-%d-%Y'), "day_of_week": df['transit_timestamp'].dt.weekday})
weekend_dates = weekend_dates[weekend_dates["day_of_week"] >= 5]

weekend_dates = weekend_dates.drop_duplicates()
weekend_dates["day_of_week"] = weekend_dates["day_of_week"].apply(lambda x: "Saturday" if x == 5 else "Sunday")

def make_day_station_col(day, station):
    result = "day_" + str(day)+"_station_" + station
    result = result.replace(" ", "_")
    result = result.replace("(", "")
    result = result.replace(")", "")
    result = result.replace(",", "")
    return result

def row_to_fft_normalized(hours):
    ffts = np.real(fft(hours))[1:12]
    fft_min = np.min(ffts)
    fft_max = np.max(ffts)
    
    # if daily_ridership is zero all day, fft will also be zero
    if (fft_max - fft_min) == 0:
        return ffts
    
    ffts = (ffts - fft_min)/(fft_max - fft_min)
    return ffts


# need "day_at_station" column to be able to pivot table later
df["day_at_station"] = df.apply(lambda x: make_day_station_col(x['transit_day'], x['station_complex']), axis = 1)


# need this key to translate "day_at_station" back to "transit_day" and "station_complex" columnds after pivoting
day_at_station_key = df[["day_at_station", "transit_day", "station_complex"]]
day_at_station_key.drop_duplicates(inplace=True)

df = df[["day_at_station", "transit_hour", "ridership"]]


# DataFrame pivoted
# Columns = day_at_station, hour_0, hour_1, ...., hour_23
df = df.pivot(index = "day_at_station", columns = "transit_hour", values = "ridership").reset_index().rename_axis(None, axis=1)

hour_columns = ["hour_" + str(i) for i in range(24)]
df.rename(columns = {i : hour_columns[i] for i in range(24)}, inplace=True)

# Replace missing values (original dataset does not enter data whenever/wherever there may be zero ridership)
df.fillna(0., inplace = True)

# Do I use tot_ridership anywhere ? I don't think so.... double check!

# Add column for total ridership for given day/station
df["tot_ridership"] = df[hour_columns].sum(axis=1)

# And norm'd
df["tot_ridership_norm"] = (df["tot_ridership"] - df["tot_ridership"].min())/(df["tot_ridership"].max() - df["tot_ridership"].min())

# MIN-MAX normalize ridership data across all hours
mins = df[hour_columns].min(axis=1)
maxs = df[hour_columns].max(axis=1)
df[hour_columns] = df[hour_columns].apply(lambda x: (x - mins[x.name])/(maxs[x.name] - mins[x.name]), axis = 1)

# reintroduce station_complex column
df = df.merge(day_at_station_key, how="inner", on="day_at_station")

df = df[["transit_day", "station_complex", "tot_ridership", "tot_ridership_norm"] + hour_columns]


# add fft'd values for each row, calculated from the hour_columns
fft_columns = [f"fft_{i}" for i in range(0, 11)]

df[fft_columns] = df.apply(lambda x: row_to_fft_normalized(x[hour_columns]), axis = 1, result_type = 'expand')


# exclude outliers, days with too many or too little riders
# tossed values will not be used to generate labels... but will be readded to the df later 
ridership_min_cutoff = 500
ridership_max_cutoff = 200000 

df_tossed = df[(df["tot_ridership"] >= ridership_max_cutoff) | (df["tot_ridership"] <= ridership_min_cutoff)]
df        = df[(df["tot_ridership"] < ridership_max_cutoff) & (df["tot_ridership"] > ridership_min_cutoff)]

df_tossed.reset_index(drop = True, inplace=True)
df.reset_index(drop = True, inplace=True) 

df_tossed[hour_columns] = normalize(df_tossed[hour_columns], axis = 1)
df[hour_columns]        = normalize(df[hour_columns], axis = 1)

print("Data transformed!")


print("Running PCA.....")
# PCA for visualization
pca_columns = ["PCA_1", "PCA_2", "PCA_3"]

pcas = PCA(n_components = 3, random_state = 0)

df[pca_columns] = pcas.fit_transform(df[hour_columns + fft_columns])

df_tossed[pca_columns] = pcas.transform(df_tossed[hour_columns + fft_columns])
print("PCA Done!")

print("Running Unsupervised Learning Model.....")
# Unsupervised learning
labels = AgglomerativeClustering(n_clusters = 3).fit_predict(df[hour_columns + fft_columns])

df = pd.concat([df, pd.DataFrame(labels).rename(columns={0:"label"})], axis = 1)

df_tossed["label"] = -1

df = pd.concat([df, df_tossed], axis=0)

print("Labels generated!")

# rearranging label names to be in ascending order of mean value in PCA_1

mean_0 = df[df["label"] == 0]["PCA_1"].mean()
mean_1 = df[df["label"] == 1]["PCA_1"].mean()
mean_2 = df[df["label"] == 2]["PCA_1"].mean()

rearrange_to = np.argsort(np.argsort([mean_0, mean_1, mean_2])).tolist()
rearrange_to.append(-1)

df["label"] = df["label"].apply(lambda x: rearrange_to[x])

label_names=["Workday Like", "Weekend - Going Out", "Weekend - Out Late", "Outlier"]

df["label"] = df["label"].apply(lambda x: label_names[x])

# adding in station location info for plotting
df = df.merge(station_info_df, how="inner", on="station_complex")

df.to_csv('df_cache.csv', index = False)

print(df.head())

# write plots to file

print("Generating plots.....")

sns.pairplot(df[pca_columns + ["borough"]], hue="borough", corner=True, plot_kws={'s': 10})

plt.savefig("borough_PCA_visualization.png")
print("Plot borough_PCA_visualization.png created!")

sns.pairplot(df[df["label"] != "Outlier"][pca_columns + ["borough"]], hue="borough", corner=True, plot_kws={'s': 10})

plt.savefig("borough_PCA_visualization_no_outliers.png")
print("Plot borough_PCA_visualization_no_outliers.png created!")


sns.pairplot(df[pca_columns + ["label"]], hue="label", corner=True, plot_kws={'s': 10})

plt.savefig("label_PCA_visualization.png")
print("Plot label_PCA_visualization.png created!")

sns.pairplot(df[df["label"] != "Outlier"][pca_columns + ["label"]], hue="label", corner=True, plot_kws={'s': 10})

plt.savefig("label_PCA_visualization_no_outliers.png")
print("Plot label_PCA_visualization_no_outliers.png created!")

print("Plots generated!\n\n\n\n")

print("Weekend dates found in this dataset:\n")

for _, row in weekend_dates.iterrows():
    print(f"{row['date']} ({row['day_of_week']:^8})")

print("\n")

print("Which weekend dates would you like to look at? You may enter:\n")

print("mm-dd-yyyy")

## take in input.....







