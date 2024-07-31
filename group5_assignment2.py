# -*- coding: utf-8 -*-
"""
Created on Wed Apr  18 12:00:00 2024

"""

# %% Libraries and options 

# Libraries:

import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import branca.colormap as cm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os

# ML and classifiers:

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Install if necessary:
# pip install scikit-learn

# Set working directory:

working_directory = '/Users/sarabcidf/Desktop/ASDS/Machine Learning/HW2' # Set
os.chdir(working_directory) # Change
print("Current working directory:", os.getcwd()) # Double-check

# %% Loading and preprocessing data

# Loading
dataset = pd.read_csv("dublinbikes_20180701_20181001.csv")

# Converting time column to datetime and keeping it as date_time: 
dataset['DATE_TIME'] = pd.to_datetime(dataset['TIME'], format="%Y-%m-%d %H:%M:%S")

# Filtering for opening hours: 
# Getting rid of obs. beteween 12:30 and 5 am
# And also anything after 8 pm as this is past the second rush hour
filtered_dataset = dataset[((dataset['DATE_TIME'].dt.hour > 5) |
                            ((dataset['DATE_TIME'].dt.hour == 5) & (dataset['DATE_TIME'].dt.minute >= 0)) |
                            ((dataset['DATE_TIME'].dt.hour == 20) & (dataset['DATE_TIME'].dt.minute <= 0)))]

filtered_dataset = filtered_dataset.copy()

# Creating separate date and time columns: 
filtered_dataset['DATE'] = filtered_dataset['DATE_TIME'].dt.date
filtered_dataset['TIME'] = filtered_dataset['DATE_TIME'].dt.time
filtered_dataset['HOUR'] = filtered_dataset['DATE_TIME'].dt.hour

# Creating day of the week column and a dummy for weekdays/weekends:
filtered_dataset['DAY_OF_WEEK'] = filtered_dataset['DATE_TIME'].dt.day_name()

# Function to determine if it's a weekend:
def is_weekend(day):
    return 1 if day in ['Saturday', 'Sunday'] else 0

# Applying function to determine weekdays/weekends:
filtered_dataset['IS_WEEKEND'] = filtered_dataset['DAY_OF_WEEK'].apply(is_weekend)

# Function to determine time of day category:
def get_time_of_day(hour):
    if 5 <= hour < 10:
        return 'morning'
    elif 10 <= hour < 16:
        return 'afternoon'
    elif 16 <= hour < 20:
        return 'evening'

# Applying function to determine time of day category:
filtered_dataset['TIME_OF_DAY'] = filtered_dataset['DATE_TIME'].dt.hour.apply(get_time_of_day)

# Removing weekends from our working dataset as well:
filtered_dataset = filtered_dataset[filtered_dataset['IS_WEEKEND'] == 0]
filtered_dataset['IS_WEEKEND'].value_counts()

# Creating subsets for morning, afternoon and evening to study commuting patterns: 
am_data = filtered_dataset[filtered_dataset['TIME_OF_DAY'] == 'morning']
aft_data = filtered_dataset[filtered_dataset['TIME_OF_DAY'] == 'afternoon']
pm_data = filtered_dataset[filtered_dataset['TIME_OF_DAY'] == 'evening']

# Checking:

am_data
aft_data
pm_data

# Calling our filtered dataset something more accesible: 

dataset = filtered_dataset
dataset.columns

# %% Availability ratio variability analysis

# Creating availability ratio: 
dataset['availability_ratio'] = dataset['AVAILABLE BIKES'] / dataset['BIKE STANDS']
mean_ratio_all_day = dataset.groupby('STATION ID')['availability_ratio'].mean()

# Calculating the variance in the ratio by station for each day:
# (on each day, how much does the ratio vary hour to hour?)
ratio_var_by_day = dataset.groupby(['STATION ID', 'DATE'])['availability_ratio'].var()
print(ratio_var_by_day)

# Adding the daily ratio variance for all days in the station:
# (higher values mean that that station has the most variance each day)
ratio_var_by_stat = ratio_var_by_day.groupby('STATION ID').sum()
ratio_var_by_stat

ratio_df = ratio_var_by_stat.to_frame()
ratio_df

# Renaming column
ratio_df = ratio_df.rename(columns={'availability_ratio': 'ratio_variability'})

# Merging the two DataFrames based owith station id: 
dataset = pd.merge(dataset, ratio_df, left_on='STATION ID', right_index=True)

# Checking:

print(dataset)
print(dataset.columns)

ratio_var_data = dataset[['ratio_variability', 'STATION ID', 'LONGITUDE', 'LATITUDE']]
ratio_var_data

ratio_var_data = ratio_var_data.drop_duplicates()
ratio_var_data

# Plotting the ratios variances on map: 

# Initializing  map
ratio_variabilities = folium.Map(location=[53.349805, -6.26031], zoom_start=12)

# Adding heatmap layer
HeatMap(data=ratio_var_data[['LATITUDE', 'LONGITUDE', 'ratio_variability']], radius=10).add_to(ratio_variabilities)

# Defining our color scale
linear = cm.LinearColormap(['green', 'yellow', 'red'], vmin=ratio_var_data['ratio_variability'].min(), vmax=ratio_var_data['ratio_variability'].max())

# Adding color bar for scale
linear.add_to(ratio_variabilities)

# Determining marker color based on variance: 
def get_color(variance):
    return linear(variance)

for index, row in ratio_var_data.iterrows():
    color = get_color(row['ratio_variability'])
    station_id = row['STATION ID']
    station_name = ratio_var_data.loc[ratio_var_data['STATION ID'] == station_id].values[0] 
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=f"Station Name: {station_name}<br>Ratio by Station: {row['ratio_variability']}"
    ).add_to(ratio_variabilities)

# Saving the map
ratio_variabilities.save('ratio_variabilities.html')

# %% Availability ratios by time of day

# Creating availability ratios by time
# The am captures the morning rush hour
# In the afternoon, people are at work
# And in the evening, it is the second rush hour
am_data['availability_ratio'] = am_data['AVAILABLE BIKES'] / am_data['BIKE STANDS']
aft_data['availability_ratio'] = aft_data['AVAILABLE BIKES'] / aft_data['BIKE STANDS']
pm_data['availability_ratio'] = pm_data['AVAILABLE BIKES'] / pm_data['BIKE STANDS']

# Calculating the mean availibility
am_data = am_data.copy()
aft_data = aft_data.copy()
pm_data = pm_data.copy()

mean_ratio_am = am_data.groupby('STATION ID')['availability_ratio'].mean()
mean_ratio_am

mean_ratio_aft = aft_data.groupby('STATION ID')['availability_ratio'].mean()
mean_ratio_aft

mean_ratio_pm = pm_data.groupby('STATION ID')['availability_ratio'].mean()
mean_ratio_pm

# Converting mean ratios to dataframes:
mean_ratio_am_df = pd.DataFrame({'STATION ID': mean_ratio_am.index, 'mean_availability_ratio': mean_ratio_am.values})
mean_ratio_aft_df = pd.DataFrame({'STATION ID': mean_ratio_aft.index, 'mean_availability_ratio': mean_ratio_aft.values})
mean_ratio_pm_df = pd.DataFrame({'STATION ID': mean_ratio_pm.index, 'mean_availability_ratio': mean_ratio_pm.values})

# Merging mean ratios back to the respective datasets:
am_data = pd.merge(am_data, mean_ratio_am_df, on='STATION ID', how='left')
aft_data = pd.merge(aft_data, mean_ratio_aft_df, on='STATION ID', how='left')
pm_data = pd.merge(pm_data, mean_ratio_pm_df, on='STATION ID', how='left')

# Subsetting with just name id time of day time and mean ratio:
am_data_subset = am_data[['NAME', 'STATION ID', 'TIME_OF_DAY', 'mean_availability_ratio']]
aft_data_subset = aft_data[['NAME', 'STATION ID', 'TIME_OF_DAY', 'mean_availability_ratio']]
pm_data_subset = pm_data[['NAME', 'STATION ID', 'TIME_OF_DAY', 'mean_availability_ratio']]

# Dropping duplicates
am_data_subset = am_data_subset.drop_duplicates()
aft_data_subset = aft_data_subset.drop_duplicates()
pm_data_subset = pm_data_subset.drop_duplicates()

# Checking
am_data_subset
am_data_subset.columns

## Fixing the afternoon dataset (which we'll focus on) so that it also includes variability

# Subseetting so it runs faster when we merge: 
new_dataset = dataset[['STATION ID', 'ratio_variability']].drop_duplicates()

# Merging the ratio variability column from dataset to aft_data based on station id
aft_data = aft_data.merge(new_dataset[['STATION ID', 'ratio_variability']], on='STATION ID', how='left')

# Seeing  updated aft_data df
print(aft_data.columns)

# Sorting each dataset by mean_availability_ratio in descending order:
#am_data_sorted = am_data_subset.sort_values(by='mean_availability_ratio', ascending=False)
aft_data_sorted = aft_data_subset.sort_values(by='mean_availability_ratio', ascending=False)
#pm_data_sorted = pm_data_subset.sort_values(by='mean_availability_ratio', ascending=False)

# Selecting the top 5 rows from each sorted dataset:
#top_5_am = am_data_sorted.head(5)
top_15_aft = aft_data_sorted.head(15)
#top_5_pm = pm_data_sorted.head(5)

# And the bottom 5 (we have now selected those with the highest and lowest ratios):
#bottom_5_am = am_data_sorted.tail(5)
bottom_15_aft = aft_data_sorted.tail(15)
#bottom_5_pm = pm_data_sorted.tail(5)

# Concatenating the selected rows into a single DataFrame:
topbot_30_mean_ratios = pd.concat([top_15_aft, bottom_15_aft], ignore_index=True)

# Seeing results:
print(topbot_30_mean_ratios)

# Filtering and checking: 
topbot_30_mean_ratios_subset = topbot_30_mean_ratios[['STATION ID', 'TIME_OF_DAY', 'mean_availability_ratio']]
print(topbot_30_mean_ratios_subset)

# %% Putting ratio availiabiliy and its variabililty together

# Merging: 
merged_subset = pd.merge(topbot_30_mean_ratios_subset, dataset[['STATION ID', 'ratio_variability']], on='STATION ID', how='left')

# Checking: 
print(merged_subset)

# Getting rid of duplicates: 
merged_subset = merged_subset.drop_duplicates()

# %% Making morning, afternoon and evening maps 

# Subsets:
am_subset = am_data[['mean_availability_ratio', 'STATION ID', 'NAME', 'LONGITUDE', 'LATITUDE']]
am_subset = am_subset.drop_duplicates()
am_subset

aft_subset = aft_data[['mean_availability_ratio', 'STATION ID', 'NAME', 'LONGITUDE', 'LATITUDE']]
aft_subset = aft_subset.drop_duplicates()
aft_subset

pm_subset = pm_data[['mean_availability_ratio', 'STATION ID', 'NAME', 'LONGITUDE', 'LATITUDE']]
pm_subset = pm_subset.drop_duplicates()

# Function to create the maps:
def create_availability_map(data, name):
    
    # Initializing the map
    availability_map = folium.Map(location=[53.349805, -6.26031], zoom_start=12)

    # Adding a heatmap layer
    HeatMap(data=data[['LATITUDE', 'LONGITUDE', 'mean_availability_ratio']], radius=10).add_to(availability_map)

    # Defining our color scale
    linear = cm.LinearColormap(['green', 'yellow', 'red'], vmin=data['mean_availability_ratio'].min(), vmax=data['mean_availability_ratio'].max())

    # Adding color bar
    linear.add_to(availability_map)

    # Function to pick marker color based on availability ratio
    def get_color(variance):
        return linear(variance)

    # Iterating over each dataset and adding the markers to the map
    for index, row in data.iterrows():
        color = get_color(row['mean_availability_ratio'])
        station_id = row['STATION ID']
        station_name = data.loc[data['STATION ID'] == station_id, 'NAME'].values[0]  
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=f"Station Name: {station_name}<br>{name} Availability Ratio: {row['mean_availability_ratio']}"
        ).add_to(availability_map)

    # Saving
    availability_map.save(f'availability_map_{name}.html')

# Creating maps for each of the subsets
create_availability_map(am_subset, 'AM')
create_availability_map(aft_subset, 'AFT')
create_availability_map(pm_subset, 'PM')

# %% Clustering by latitude and longitude to get feature

# Making x the lat and long
X = aft_data[['LATITUDE', 'LONGITUDE']]

# Picking 20 clusters
k = 20

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)

# Adding cluster labels to the dataset
aft_data['cluster'] = kmeans.labels_
aft_data[['STATION ID', 'cluster']]

# Plotting the clusters to examine against the maps:
# The results are very good, the cluster will be a good "proxy" for location to use this as a feature.
plt.figure(figsize=(10, 6))
sns.scatterplot(data=dataset, x='LONGITUDE', y='LATITUDE', hue='cluster', palette='tab20', legend='full')
plt.title('K-Means Clustering of Bike Stations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()

# %% Exploring bike stations expansions/reductions cutoffs

print(merged_subset)

merged_sorted = merged_subset.sort_values(by='ratio_variability', ascending=False)

print(merged_sorted)
len(merged_sorted)

# %% Creating model target

aft_data.columns

# Creating "full" five-category indicator of problems:

def create_problem_indicator(availability_ratio, ratio_variability):
    if availability_ratio < 0.4 and ratio_variability < 4:
        return "problematic low low"
    elif availability_ratio < 0.4 and ratio_variability > 7:
        return "problematic low high"
    elif availability_ratio > 0.6 and ratio_variability < 4:
        return "problematic high low"
    elif availability_ratio > 0.6 and ratio_variability > 7:
        return "problematic high high"
    else:
        return "non-problematic"

aft_data['problem_indicator'] = aft_data.apply(lambda row: create_problem_indicator(row['availability_ratio'], row['ratio_variability']), axis=1)

# Now, creating our simplified binary indicator (simply problematic/non-problematic):

def simplify_problem_indicator(problem_indicator):
    if "problematic low low" in problem_indicator or \
       "problematic low high" in problem_indicator or \
       "problematic high low" in problem_indicator or \
       "problematic high high" in problem_indicator:
        return "problematic"
    else:
        return "non-problematic"

aft_data['simplified_problem_indicator'] = aft_data['problem_indicator'].apply(simplify_problem_indicator)

print(aft_data['simplified_problem_indicator'].value_counts())

# %% Model (KNN)

# Now fitting the classifier: 

model_data = aft_data[['cluster', 'HOUR', 'simplified_problem_indicator']]

X = model_data.drop('simplified_problem_indicator', axis=1)  # Features
y = model_data['simplified_problem_indicator']  # Target

# Splitting:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model: 
    
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Measures: 

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)

# Predicting new values based on the train dataset AND the test dataset:
pred_ytrain = model.predict(X_train)
pred_ytest = model.predict(X_test)

#Evaluating:
# Score: Correctly classified over total number of points
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print('Training set score: ', train_score, '|Testing set score: ', test_score)

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize = 'true')

# We get 0.65 and 0.65 on the train and test

# Convert predicted y_test values to a DataFrame
pred_ytest_df = pd.DataFrame({'predicted_class': pred_ytest})

# Seeing the DataFrame
print(pred_ytest_df)

# This is what we will be comparing to the simulated model. 
# This value should decrease in the simulation (have less problematic stations).
pred_ytest_df.value_counts() 

# %% Creating the simulation

# Creating a deep copy of the dataset - deep to remove
#errors in adjusting it later on
simulation_data = aft_data.copy(deep=True)

# Applying expansion and reduction factors to the bike stand capacities
expansion_factor = 1.5  # Increase capacity by 50%
reduction_factor = 0.5  # Reduce capacity by 50%

# Defining stations to expand and reduce
stations_to_expand = [93, 47, 86, 57, 69, 84, 25, 26, 63, 82]
stations_to_reduce = [50, 101, 70, 49, 96, 75, 107, 108, 106, 105]

# Updating bike stand capacities for stations to expand
for station_id in stations_to_expand:
    simulation_data.loc[simulation_data['STATION ID'] == station_id, 'BIKE STANDS'] *= expansion_factor

# Updating bike stand capacities for stations to reduce
for station_id in stations_to_reduce:
    simulation_data.loc[simulation_data['STATION ID'] == station_id, 'BIKE STANDS'] *= reduction_factor

# Recalculating availability ratio based on updated bike stand capacities
simulation_data['availability_ratio'] = simulation_data['AVAILABLE BIKES'] / simulation_data['BIKE STANDS']

simulation_data['problem_indicator'] = simulation_data.apply(lambda row: create_problem_indicator(row['availability_ratio'], row['ratio_variability']), axis=1)

# Now, creating our simplified binary indicator: 
simulation_data['simplified_problem_indicator'] = simulation_data['problem_indicator'].apply(simplify_problem_indicator)

# Generating features used in the model (cluster, hour, simplified problem indicator):
simulation_model_data = simulation_data[['cluster', 'HOUR', 'simplified_problem_indicator']]

print(simulation_model_data.head())
print(simulation_model_data.columns)

# Re-running model on new dzata: 

model_data['simplified_problem_indicator'].value_counts()

simulation_model_data['simplified_problem_indicator'].value_counts()

# %% Model on simulation data

# Defining features and target again for simulation:
X2 = simulation_model_data.drop('simplified_problem_indicator', axis=1)  # Features
y2 = simulation_model_data['simplified_problem_indicator']  # Target

y.value_counts()
y2.value_counts()

# Splitting:
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

y2_test.value_counts()
y_test.value_counts()

# Model: 
model2 = KNeighborsClassifier(n_neighbors=3) #stays the same
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

# Measures: 
accuracy2 = accuracy_score(y2_test, y2_pred)
conf_matrix2 = confusion_matrix(y2_test, y2_pred)

print(f"Accuracy: {accuracy2}")
print("Confusion Matrix:")
print(conf_matrix2)

#Predicting new values based on the train dataset AND the test dataset
pred_ytrain2 = model2.predict(X2_train)
pred_ytest2 = model2.predict(X2_test)

# Evaluating:
# Score: Correctly classified over total number of points
train_score2 = model2.score(X2_train, y2_train)
test_score2 = model2.score(X2_test, y2_test)

print('Training set score: ', train_score2, '|Testing set score: ', test_score2)

ConfusionMatrixDisplay.from_estimator(model, X2_test, y2_test, normalize = 'true')

# We get around 0.65 and 0.65 on the train and test

# Converting predicted y_test values to a DataFrame
pred_ytest2_dataframe = pd.DataFrame({'predicted_class': pred_ytest2})

# Seeing the DataFrame
print(pred_ytest2_dataframe)

# Seeing problematic counts
pred_ytest2_dataframe.value_counts()

# %% Comparing the predicted problematic obs. for original vs. proposal

# Comparing 
pred_ytest_df.value_counts() # Original
pred_ytest2_dataframe.value_counts() # Simulation (proposal)

# There is an decrease in problematic cases!

# %% Creating visuals for report

# 1. Table for problematic observations for morning: 

#aft_subset = merged_subset[merged_subset['TIME_OF_DAY'] == 'afternoon']
#aft_subset = aft_subset.drop(columns=['TIME_OF_DAY'])

# Rounding
aft_subset['mean_availability_ratio'] = aft_subset['mean_availability_ratio'].round(2)
aft_subset['ratio_variability'] = aft_subset['ratio_variability'].round(2)

# Renaming 
aft_subset = aft_subset.rename(columns={
    'mean_availability_ratio': 'Mean availability ratio',
    'ratio_variability': 'Ratio variability',
    'STATION ID': 'Station ID'
})

print(aft_subset)

# Putting in Latex for report: 
    
latex_code = """
\\begin{table}[h]
\\centering
""" + aft_subset.to_latex(index=False, float_format="%.2f") + """
\\end{table}
"""

print(latex_code)

# 2. Table showing improvement: *** THIS IS MISSING ***

# Original 
original_value_counts = pred_ytest_df.value_counts()

# Simulation 
simulation_value_counts = pred_ytest2_dataframe.value_counts()

# Proportion
original_proportions = original_value_counts / original_value_counts.sum()
simulation_proportions = simulation_value_counts / simulation_value_counts.sum()

# DF
proportions_df = pd.DataFrame({'Original': original_proportions, 'Simulation': simulation_proportions})

# Formatting
proportions_df = proportions_df.applymap(lambda x: f'{x:.2%}')

proportions_df = proportions_df.rename(index={'non-problematic': 'Non problematic', 
                                              'problematic': 'Problematic'})
proportions_df = proportions_df.rename(columns={'predicted_class': 'Predicted class'})

print(proportions_df)

# Putting in Latex for reporting: 

latex_code2 = """
\\begin{table}[h]
\\centering
""" + proportions_df.to_latex(float_format="%.2f") + """
\\end{table}
"""

print(latex_code2)

# 3. Maps

# We chose to show the morning ratio map and the variability map. 
# Improving them: 

## First the variability one:     

# Normalizing (to be comparable with the other map)
scaler_variability = MinMaxScaler()
ratio_var_data['normalized_variability'] = scaler_variability.fit_transform(ratio_var_data[['ratio_variability']])

# Initializing the map for variability
variability_map = folium.Map(location=[53.349805, -6.26031], zoom_start=12)

# Adding heatmap layer for variability
HeatMap(data=ratio_var_data[['LATITUDE', 'LONGITUDE', 'normalized_variability']], radius=10).add_to(variability_map)

# Defining color scale for variability with plasma colormap
cmap_variability = sns.color_palette("plasma", as_cmap=True)
linear_variability = cm.LinearColormap([cmap_variability(0), cmap_variability(0.5), cmap_variability(1)], vmin=0, vmax=1)
linear_variability.add_to(variability_map)

# Function to pick marker color based on normalized variability
def get_color_variability(variance):
    return linear_variability(variance)

# Adding markers for variability map
for index, row in ratio_var_data.iterrows():
    color = get_color_variability(row['normalized_variability'])
    station_id = row['STATION ID']
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=f"Station ID: {station_id}<br>Normalized Variability: {row['normalized_variability']}"
    ).add_to(variability_map)

# Saving the variability map
variability_map.save('variability_map_FFF.html')

## Now the availability one:

# Normalizing (to be comparable with the other map)
scaler_ratio = MinMaxScaler()
am_subset['normalized_availability_ratio'] = scaler_ratio.fit_transform(am_subset[['mean_availability_ratio']])

# Initializing the map for availability ratio
availability_map = folium.Map(location=[53.349805, -6.26031], zoom_start=12)

# Adding heatmap layer for availability ratio
HeatMap(data=am_subset[['LATITUDE', 'LONGITUDE', 'normalized_availability_ratio']], radius=10).add_to(availability_map)

# Defining color scale for availability ratio with plasma colormap
cmap_availability = sns.color_palette("plasma", as_cmap=True)
linear_availability = cm.LinearColormap([cmap_availability(0), cmap_availability(0.5), cmap_availability(1)], vmin=0, vmax=1)
linear_availability.add_to(availability_map)

# Function to pick marker color based on normalized availability ratio
def get_color_availability(variance):
    return linear_availability(variance)

# Adding markers for availability ratio map
for index, row in am_subset.iterrows():
    color = get_color_availability(row['normalized_availability_ratio'])
    station_id = row['STATION ID']
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=5,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        popup=f"Station ID: {station_id}<br>Normalized Availability Ratio: {row['normalized_availability_ratio']}"
    ).add_to(availability_map)

# Saving the availability ratio map
availability_map.save('availability_map_AM_FFF.html')

# 5. Barplot for report

# Adding the predictions back to test data:
X_test_with_pred = X_test.copy()
X_test_with_pred['predicted_class'] = pred_ytest

X2_test_with_pred = X2_test.copy()
X2_test_with_pred['predicted_class'] = pred_ytest2

# Defining function to calculate the proportion of problematic predictions
def calculate_problematic_proportion(data):
    grouped = data.groupby('cluster')['predicted_class'].apply(lambda x: (x == 'problematic').mean())
    return grouped.reset_index(name='Proportion Problematic')

# Calculate proportions for both datasets
problematic_proportion_original = calculate_problematic_proportion(X_test_with_pred)
problematic_proportion_simulation = calculate_problematic_proportion(X2_test_with_pred)

# Preparing data for plotting
problematic_proportion_original['Type'] = 'Original'
problematic_proportion_simulation['Type'] = 'Simulation'
combined_data = pd.concat([problematic_proportion_original, problematic_proportion_simulation])

# Picking some clusters to show barplot with change: 
filtered_data = combined_data[combined_data['cluster'].isin([9, 16])]

plt.figure(figsize=(10, 5))

# Using the plasma color palette to match the other plot:
palette = plt.get_cmap('plasma')

# Getting unique type counts to generate distinct colors from the plasma palette
type_count = filtered_data['Type'].nunique()
colors = [palette(i / type_count) for i in range(type_count)]

# Creating barplot for the filtered data
ax = sns.barplot(x='cluster', y='Proportion Problematic', hue='Type', data=filtered_data, palette=colors)
plt.xticks(rotation=90)
plt.xlabel('Cluster')
plt.ylabel('Proportion of Problematic Hours')
plt.legend(title='Dataset')

ax.spines['top'].set_visible(False)  
ax.spines['right'].set_visible(False) 

# Looping through the bars in the plot to add text annotations and make the proportions visible
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'),  
                (p.get_x() + p.get_width() / 2., p.get_height()),  
                ha='center', va='center',  
                xytext=(0, 9),  
                textcoords='offset points')

plt.tight_layout()

# Display the plot
plt.show()

# Save the plot
plt.savefig('Proportion_of_Problematic_Hours.png')


