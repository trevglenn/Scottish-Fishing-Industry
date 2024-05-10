import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("scottish_salmon\SalmonandSeaTroutNets1952-2022.csv")


#----------------------------------------------------------------
""" Dropping unneccesary data and columns"""

indices_to_drop = [22834, 22835, 22836, 23140, 23141, 23142]
df = df.drop(indices_to_drop)

df.info()
df.drop(['Month'], axis=1, inplace=True)


#-----------------------------------------------------------------
""" Finding our null values (this was used for our previous dropping of indices) and
the checking to make sure zero null values remain. """

null_values = df.isnull().sum()

print("Null values in each column:")
print(null_values)

null_rows = df[df.isnull().any(axis=1)]

null_rows_indices = null_rows.index

print("Indices of rows with null values:")
print(null_rows_indices)

#-----------------------------------------------------------
"""Assembling our features and labels (x and y)."""

y = df['Region']
x = df.drop(['Region'], axis=1)

print(x.head())
print(y.head())

#----------------------------------------------------------
""" Feature Engineering. """

cat_cols = ['District', 'Method']
num_cols = ['District ID', 'Report order', 'Year', 'Month number', 'Wild MSW number', 'Wild MSW weight (kg)', 'Wild 1SW number', 'Wild 1SW weight (kg)', 'Sea trout number', 'Sea trout weight (kg)', 'Finnock number', 'Finnock weight (kg)', 'Farmed MSW number', 'Farmed MSW weight (kg)', 'Farmed 1SW number', 'Farmed 1SW weight (kg)', 'Netting effort']

x = pd.get_dummies(x, columns=cat_cols)

sc = StandardScaler()

x[num_cols] = sc.fit_transform(x[num_cols])
#------------------------------------------------------------
""" Assigning training and testing sets to our data. """

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

#------------------------------------------------------------
""" Building and testing our Model. """

clf = SGDClassifier(loss="hinge", max_iter=1000)

clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

score = clf.score(x_test, y_test)
print("Model Score: ", score)

# Model Score = 99.7%

#---------------------------------------------------------------
""" Using Grid search to find our best model parameters, score, and 
tuning hyperparameters."""

param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'l1_ratio': [0.15, 0.5, 0.85],
    'penalty': ['l1', 'l2', 'elasticnet']
}

clf = SGDClassifier(loss="hinge", max_iter=1000)

grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

## Our Best Parameters are: {'alpha':0.001, 'l1_ratio':0.5, 'penalty':'12'}
## Our Best Score: 0.998271

#----------------------------------------------------------------
