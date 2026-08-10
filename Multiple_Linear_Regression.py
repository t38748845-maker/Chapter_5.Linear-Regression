# a linear regression with two input variables...
# importing pandas as pd...
import pandas as pd

# importing LinearRegression from sklearn.linear_model...
from sklearn.linear_model import LinearRegression

# create the dataset...
# https://bit.ly/2X1HWH7
df = pd.DataFrame({
    "x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "x2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
    "y": [5, 9, 13, 17, 21, 25, 29, 33, 37, 41]
})

# extract input variables...
X = df.values[:, :-1]

# extract output variable...
Y = df.values[:, -1]

# training...
fit = LinearRegression().fit(X, Y)

# prints coefficients...
print("Coefficients = {}".format(fit.coef_))

# prints intercept...
print("Intercept = {}".format(fit.intercept_))

# formulates and then print it...
print("z = {} + ({})x1 + ({})x2".format(
    fit.intercept_,
    fit.coef_[0],
    fit.coef_[1]
))