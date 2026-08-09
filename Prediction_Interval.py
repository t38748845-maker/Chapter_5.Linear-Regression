# calculating prediction interval of vets for a dog that's 8.5 years old... 
# importing pandas as pd... 
import pandas as pd 

# importing t from scipy.stats... 
from scipy.stats import t

# importing sqrt from math... 
from math import sqrt

# Create the dataset...
# https://bit.ly/2KF29Bd
df = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y": [5,10,10,15,14,15,19,18,25,23]
})

# load the data...
points = list(df.itertuples(index=False))

# n for length of points...
n = len(points)

# ression line...
m = 1.939
b = 4.733

# calculating prediction interval for x = 8.5...
x_0 = 8.5

x_mean = sum(p. x for p in points) / len(points)

# formulate for  t_value... 
t_value = t(n - 2).ppf(0.975)

# formulate for standard_error... 
standard_error = sqrt(sum((p.y - (m * p.x + b)) ** 2 for p in points) / (n-2))

# formulate for margime_of_errors...
margin_of_error = t_value * standard_error * \
    sqrt(1 + (1 / n) + (n * (x_0 - x_mean) ** 2) / \
    (n * sum(p.x ** 2 for p in points) - \
    sum(p.x for p in points) ** 2))
    
# formulate for predicted_y...
predicted_y = m * x_0 + b

# calculating prediction interval... 
print(predicted_y - margin_of_error, predicted_y + margin_of_error)