# doing a train/test split on linear regression...
# importing pandas as pd...
import pandas as pd

# importing LinearRegression from sklearn.linear_model...
from sklearn.linear_model import LinearRegression

# importing train_test_split from sklearn.linear_model...
from sklearn.model_selection import train_test_split

# Create the dataset
# https://bit.ly/3cIH97A
df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "y": [5, 10, 10, 15, 14, 15, 19, 18, 25, 23]
})

# Extract input variables
X = df.values[:, :-1]

# Extract output variable
Y = df.values[:, -1]

# Separate training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=1/3
)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, Y_train)

# Test the model
result = model.score(X_test, Y_test)

# Print R-squared
print("r^2: %.3f" % result)

# using three-fold cross-validation for a linear regression...
# importing pandas as pd...
import pandas as pd

# importing LinearRegression from sklearn.linear_model...
from sklearn.linear_model import LinearRegression

# importing KFold, cross_val_score from sklearn.model_selection...
from sklearn.model_selection import KFold, cross_val_score

# create the dataset...
# https://bit.ly/3cIH97A
df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "y": [5, 10, 10, 15, 14, 15, 19, 18, 25, 23]
})

# extract input variables...
X = df.values[:, :-1]

# extract output variable...
Y = df.values[:, -1]

# p...erform three-fold cross-validation...
kfold = KFold(n_splits=3, random_state=7, shuffle=True)

# creates an object of Scikit-learn's Linear Regression algorithm...
model = LinearRegression()

# cross_val_score() = performs cross-validation and calculates a score for each fold
# model =  the Linear Regression model you created
# X = input data/features
# Y = output/target data
# cv=kfold =use your 3-fold KFold method
# resuls =  store all the scores in results
results = cross_val_score(model, X, Y, cv=kfold)

# prints score stored in results...
print(results)

# prints the average score and standard deviation of those results...
print("MSE: mean=%.3f (stdev=%.3f)" % (results.mean(), results.std()))

# using a random-fold validation for a linear regression... 
# importing pandas as pd...
import pandas as pd

# importing LinearRegression from sklearn.linear_model...
from sklearn.linear_model import LinearRegression

# importing cross_val_score, ShuffleSplit from sklearn.model_selection...
from sklearn.model_selection import cross_val_score, ShuffleSplit

# create the dataset...
df = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "y": [5, 10, 10, 15, 14, 15, 19, 18, 25, 23]
})

# extract input variables..
X = df.values[:, :-1]

# extract output variables...
Y = df.values[:, -1]

# perform random-fold validation...
kfold = ShuffleSplit(
    n_splits=10,
    test_size=0.33,
    random_state=7
)

# creates an object of Scikit-learn's Linear Regression algorithm...
model = LinearRegression()

# cross_val_score() = performs cross-validation and calculates a score for each fold
# model =  the Linear Regression model you created
# X = input data/features
# Y = output/target data
# cv=kfold =use your 3-fold KFold method
# resuls =  store all the scores in results
results = cross_val_score(model, X, Y, cv=kfold)

# prints results contains the scores from each random train/test split...
print(results)

# calculate the average score and standard deviation, then print them rounded to 3 decimal places...
print("mean=%.3f (stdev=%.3f)" %
      (results.mean(), results.std()))