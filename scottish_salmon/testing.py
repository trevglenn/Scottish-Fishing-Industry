import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("scottish_salmon\SalmonandSeaTroutNets1952-2022.csv")

"""
1.) For a better idea of catch method we'll construct a pie chart to look at the total ratio of the 4 catch methods used.

2.) For Unique Regions: we want to look at total fish caught in each region, type of fish most caught in each region, total weight, method most used, and/or total + avg netting effort for each.
This would be better accomplished mainly with bar charts, some pie graphs, and if we want time-series data we can look at regplot, histogram, or other to visualize change over time

3.) Besides our exploratory analysis for the 2 models we are using, we can also do some additional statistical analysis to build key insights into the fishing industry. 

"""

orkney = df[df['Region'] == 'Orkney']

print(orkney.info())
print(orkney.head())

#The region of Orkney only has 3 recorded catches, for a lot of our EDA we can drop this but make sure to include it as a possibility for our regional ML model

indices_to_drop = [22834, 22835, 22836]
df = df.drop(indices_to_drop)

print(df['Region'].nunique())


region_counts = df.groupby("Region").count()

print(region_counts.head())

############################ Not Important ! #####################################
wild_salmon = ['Wild MSW number', 'Wild 1SW number']

trout = ['Sea trout number', 'Finnock number']

farmed_salmon = ['Farmed MSW number', 'Farmed 1SW number']

total_ws = df[wild_salmon].sum(axis=1)

total_trout = df[trout].sum(axis=1)

total_fs = df[farmed_salmon].sum(axis=1)

############################ Not Important ! ######################################

ws_by_region = df.groupby('Region')[wild_salmon].sum()

trout_by_region = df.groupby('Region')[trout].sum()

fs_by_region = df.groupby('Region')[farmed_salmon].sum()

print(ws_by_region.head())



##########################################################################################

"""Basis for next function - summing all of our fish types. """

sum_wildmsw = df["Wild MSW number"].sum()
print(sum_wildmsw)
sum_wildmsw_weight = df["Wild MSW weight (kg)"].sum()
#print(sum_wildmsw_weight)
avg_wildmsw_weight = (sum_wildmsw_weight / sum_wildmsw)
print(avg_wildmsw_weight)
## AVG wild msw weight is about 4.85 kg

sum_wild1sw = df["Wild 1SW number"].sum()
#print(sum_wild1sw)
sum_wild1sw_weight = df["Wild 1SW weight (kg)"].sum()
#print(sum_wild1sw_weight)
avg_wild1sw_weight = (sum_wild1sw_weight / sum_wild1sw)
print(avg_wild1sw_weight)

sea_trout_sum = df["Sea trout number"].sum()
sea_trout_weight = df["Sea trout weight (kg)"].sum()
avg_seatrout_weight = (sea_trout_weight / sea_trout_sum)
print(avg_seatrout_weight)

finnock_sum = df["Finnock number"].sum()
finnockweight_sum = df["Finnock weight (kg)"].sum()
avg_finnock_weight = (finnockweight_sum / finnock_sum)
print(avg_finnock_weight)

farmed_msw_sum = df["Farmed MSW number"].sum()
farmed_msw_weightsum = df["Farmed MSW weight (kg)"].sum()
avg_farmedmsw_weight = (farmed_msw_weightsum / farmed_msw_sum)
print(avg_farmedmsw_weight)

farmed_1sw_sum = df["Farmed 1SW number"].sum()
farmed_1sw_weightsum = df["Farmed 1SW weight (kg)"].sum()
avg_farmed1sw_weight = (farmed_1sw_weightsum / farmed_1sw_sum)
print(avg_farmed1sw_weight)

####################################################################################

""" Calculating Average weight of each fish """


def calculate_average_weight(df, category, number_column, weight_column):
    sum_number = df[number_column].sum()
    sum_weight = df[weight_column].sum()
    if sum_number == 0:
        print(f"No {category} data available.")
        return None
    else:
        avg_weight = sum_weight / sum_number
        print(f"Average {category} weight is about {avg_weight:.2f} kg")
        return avg_weight

categories = [
    ("Wild MSW", "Wild MSW number", "Wild MSW weight (kg)"),
    ("Wild 1SW", "Wild 1SW number", "Wild 1SW weight (kg)"),
    ("Sea trout", "Sea trout number", "Sea trout weight (kg)"),
    ("Finnock", "Finnock number", "Finnock weight (kg)"),
    ("Farmed MSW", "Farmed MSW number", "Farmed MSW weight (kg)"),
    ("Farmed 1SW", "Farmed 1SW number", "Farmed 1SW weight (kg)")
]

categories_data = []
avg_weights = []

for category, number_col, weight_col in categories:
    avg_weight = calculate_average_weight(df, category, number_col, weight_col)
    if avg_weight is not None:
        categories_data.append(category)
        avg_weights.append(avg_weight)

fig1 = plt.figure()
plt.bar(categories_data, avg_weights, color='red')
plt.xlabel("Fish")
plt.ylabel("Average Weight")
plt.title("Average Weight of Fish")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()

##########################################################################################

""" Calculating average netting effort --- needs work this is not accurate yet. Need to first do this proportionally (use results from below)
as well as iterate through each fish type to recognize any major changes related to fish type changing netting effort. """

def calculate_average_netting_effort(df, categories, total_netting_effort_column, fish_netting_effort_column):
    total_netting_effort = df[total_netting_effort_column].sum()
    average_netting_efforts = {}
    print("Total Netting Effort: ", total_netting_effort)

    for category, number_col, weight_col in categories:
        fish_netting_effort_sum = df[number_col].sum()
        fish_proportion = fish_netting_effort_sum / total_netting_effort
        average_netting_efforts[category] = fish_proportion * total_netting_effort
        
    return average_netting_efforts

# Define the categories list and column names
categories = [
    ("Wild MSW", "Wild MSW number", "Wild MSW weight (kg)"),
    ("Wild 1SW", "Wild 1SW number", "Wild 1SW weight (kg)"),
    ("Sea trout", "Sea trout number", "Sea trout weight (kg)"),
    ("Finnock", "Finnock number", "Finnock weight (kg)"),
    ("Farmed MSW", "Farmed MSW number", "Farmed MSW weight (kg)"),
    ("Farmed 1SW", "Farmed 1SW number", "Farmed 1SW weight (kg)")
]
total_netting_effort_column = "Netting effort"
fish_netting_effort_column = "Netting effort"

average_netting_efforts = calculate_average_netting_effort(df, categories, total_netting_effort_column, fish_netting_effort_column)
print("Average Netting Effort for Each Fish Category:")
for fish_category, avg_netting_effort in average_netting_efforts.items():
    print(f"{fish_category}: {avg_netting_effort:.2f}")


#############################################################################################

""" Calculates our total fish number """


def calculate_total_fish(df, categories):
    total_fish = 0
    for category, number_col, weight_col in categories:
        total_fish += df[number_col].sum()
    return total_fish

# Define the categories list
categories = [
    ("Wild MSW", "Wild MSW number", "Wild MSW weight (kg)"),
    ("Wild 1SW", "Wild 1SW number", "Wild 1SW weight (kg)"),
    ("Sea trout", "Sea trout number", "Sea trout weight (kg)"),
    ("Finnock", "Finnock number", "Finnock weight (kg)"),
    ("Farmed MSW", "Farmed MSW number", "Farmed MSW weight (kg)"),
    ("Farmed 1SW", "Farmed 1SW number", "Farmed 1SW weight (kg)")
]

total_fish = calculate_total_fish(df, categories)
print("Total number of all fish in the dataset:", total_fish)
print("Total effort: ", df["Netting effort"].sum())

#################################################################################################

""" Calculates and graphs our total Fish proportions (sum of each fish divided by total fish) and then plots them """


def calculate_fish_proportion(df, categories):
    total_fish = sum(df[number_col].sum() for _, number_col, _ in categories)
    fish_proportions = {}
    
    other_proportion = 0
    for category, number_col, _ in categories:
        if category in ["Finnock", "Farmed MSW", "Farmed 1SW"]:
            other_proportion += df[number_col].sum()
        else:
            fish_count = df[number_col].sum()
            fish_proportion = fish_count / total_fish
            fish_proportions[category] = fish_proportion
    
    if other_proportion > 0:
        fish_proportions["Other"] = other_proportion / total_fish

    return fish_proportions

# Define the categories list
categories = [
    ("Wild MSW", "Wild MSW number", "Wild MSW weight (kg)"),
    ("Wild 1SW", "Wild 1SW number", "Wild 1SW weight (kg)"),
    ("Sea trout", "Sea trout number", "Sea trout weight (kg)"),
    ("Finnock", "Finnock number", "Finnock weight (kg)"),
    ("Farmed MSW", "Farmed MSW number", "Farmed MSW weight (kg)"),
    ("Farmed 1SW", "Farmed 1SW number", "Farmed 1SW weight (kg)")
]

fish_proportions = calculate_fish_proportion(df, categories)
print("Proportion of each fish category compared to the total of all fish:")
for category, proportion in fish_proportions.items():
    print(f"{category}: {proportion:.5f}")


labels = list(fish_proportions.keys())
sizes = list(fish_proportions.values())

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Proportion of Each Fish Category')
plt.show()
plt.clf()

plt.barh(labels, sizes, color='skyblue')
plt.xlabel('Proportion')
plt.ylabel('Fish Category')
plt.title('Proportion of Each Fish Category (Bar Graph)')

plt.tight_layout()
plt.show()
plt.clf()

################################################################################################


print(df['Year'].min())
print(df['Year'].max())