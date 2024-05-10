import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("scottish_salmon\SalmonandSeaTroutNets1952-2022.csv")

#df.info()
#print(df.head())

""" 
20 columns, 16 are dtype float64 the other 4 are categorical object dtype. Our objects are District, Region, Method, and Month.

There are 6 different species/type of salmon and sea trout that are caught: Wild MSW, Wild 1SW, Sea trout, Finnock, Farmed MSW, Farmed 1SW.

Measured by total number caught, total weight(kg), and total netting effort.

Month number is also provided, can be very useful but should beware of the duplicated data of Month and Month Number.

Method column includes two different data points: the method for fishing as well as if the fish was retained or released. Can be converted to string and separated at ':'.

All released reports are at the bottom of the dataset, there are 38 in total of the 23,140 reports. Starts at 23,104.
    - These can be separated from the dataset when we look at retained catches, and because there are so few they can be dropped or separated entirely for most of our EDA.
    - However, for a truly predictive model we will need to separate them into a new column to differentiate and be able to account for potentially released catches, even if it is rare it is still a possibility.
    - Will also be able to get insights into which fish are released and compare that to our retained data. 
    - Using those insights will fine-tune our predictive model, either by adding rules, adjusting the weights of our model, or both for better retained/released accuracy. 



Insights we can look for:
Avg fish caught, weight, netting effort for each District and Region.
Can do the same for Month and Year, Effort, etc for each species.
Then we can look at avg market price for the species and make economic analysis for the Scottish Salmon and Sea Trout industry.

Then we can build a simple predictive model for future analysis, and potentially another model that uses our industry insights in combination with our data to predict the market.

"""


## We can make some pie charts, histograms, regplot, and bar charts to compare/contrast some of our features
## Our target will more than likely be a self-created variable 
## Can build either regression or categorical model based on chosen target variable.

print(df.head())

df.info()


## 97 Unique Districts - P; too many unique values
## 10 Unique Regions - Y; We can build a predictive model for region
## 4 Unique methods - Y+; We can build a predictive model for catch method

## Could build predictive models for District, Region, and Method - all object Dtypes so we would need a categorical processing model, likely best accomplished by SGD Classifier or a Neural Network

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

######################################

wild_salmon = ['Wild MSW number', 'Wild 1SW number']

trout = ['Sea trout number', 'Finnock number']

farmed_salmon = ['Farmed MSW number', 'Farmed 1SW number']

total_ws = df[wild_salmon].sum(axis=1)

total_trout = df[trout].sum(axis=1)

total_fs = df[farmed_salmon].sum(axis=1)

########################################
ws_by_region = df.groupby('Region')[wild_salmon].sum()

trout_by_region = df.groupby('Region')[trout].sum()

fs_by_region = df.groupby('Region')[farmed_salmon].sum()

print(ws_by_region.head())

fish_type = ['Sea trout number', 'Finnock number', 'Wild MSW number', 'Wild 1SW number', 'Farmed MSW number', 'Farmed 1SW number']

fish_weight = ['Sea trout weight (kg)', 'Finnock weight (kg)', 'Wild MSW weight (kg)', 'Wild 1SW weight (kg)', 'Farmed MSW weight (kg)', 'Farmed 1SW weight (kg)']


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



# ------------------------------------------------------
# FIGURES
# Pie Charts:
fig1 = plt.figure()

plt.pie(region_counts['Netting effort'], labels=region_counts.index, autopct='%1.1f%%', labeldistance=1.05)
plt.axis('equal')
plt.title("Netting Effort by Region")
plt.legend(title='Region', loc='upper left')
plt.show()
plt.clf()



# Bar Graphs:
plt.bar(categories_data, avg_weights, color='red')
plt.xlabel("Fish")
plt.ylabel("Average Weight")
plt.title("Average Weight of Fish")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()


ws_by_region.sum(axis=1).plot(kind='bar', color='skyblue')
plt.title("Wild Salmon Caught by Region")
plt.xlabel("Region")
plt.ylabel("Wild Salmon")
plt.tight_layout()
plt.show()
plt.clf()

trout_by_region.sum(axis=1).plot(kind='bar', color='skyblue')
plt.title("Trout Caught by Region")
plt.xlabel("Region")
plt.ylabel("Trout")
plt.tight_layout()
plt.show()
plt.clf()

fs_by_region.sum(axis=1).plot(kind='bar', color='skyblue')
plt.title("Farmed Salmon Caught by Region")
plt.xlabel("Region")
plt.ylabel("Farmed Salmon")
plt.tight_layout()
plt.show()
plt.clf()

######################################

method_by_region = df.groupby(['Region', 'Method']).size().unstack(fill_value=0)

method_by_region.plot(kind='bar', stacked=True)
plt.title('Catching Method by Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
plt.clf()

#######################################
# Lineplots:

sns.lineplot(data=df, x='Year', y='Wild MSW number', hue='Region', palette='husl')
plt.title('Multi-sea Winter Salmon Caught Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Multi-sea Winter Salmon Caught')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()


sns.lineplot(data=df, x='Year', y='Wild 1SW number', hue='Region', palette='husl')
plt.title('One-sea Winter Salmon Caught Over Time by Region')
plt.xlabel('Date')
plt.ylabel('One-sea Winter Salmon Caught')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()

sns.lineplot(data=df, x='Year', y='Sea trout number', hue='Region', palette='husl')
plt.title('Sea Trout Caught Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Sea Trout Caught')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()


sns.lineplot(data=df, x='Year', y='Finnock number', hue='Region', palette='husl')
plt.title('Finnock Caught Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Finnock Caught')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()

sns.lineplot(data=df, x='Year', y='Farmed MSW number', hue='Region', palette='husl')
plt.title('Farmed Multi-sea Winter Salmon Caught Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Farmed MSW Caught')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()

sns.lineplot(data=df, x='Year', y='Farmed 1SW number', hue='Region', palette='husl')
plt.title('Farmed One-sea Winter Salmon Caught Over Time by Region')
plt.xlabel('Date')
plt.ylabel('Farmed 1SW Caught')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()
