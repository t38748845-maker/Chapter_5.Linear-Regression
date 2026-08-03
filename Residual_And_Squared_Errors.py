# y = 1.93939x + 4.73333
# calculating the residuals for a given line ad data... 
# importing pandas as pd... 
import pandas as pd

# Create the dataset...
# https://bit.ly/3go0Ant
df = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y": [5,10,10,15,14,15,19,18,25,23]
})

# load data..
points = df.itertuples()

# test with given line...
m = 1.93939
b = 4.73333

# calculating the residuals... 
for p in points:
    y_actual = p.y
    y_predict = m * p.x + b
    
# formulate for residual...     
    residual = y_actual - y_predict 
   
# prints residual...
    print(residual)

# calculating the sum of square of a given line and data... 

# y = 1.93939x + 4.73333
# calculating the residuals for a given line ad data... 
# importing pandas as pd... 
import pandas as pd

# Create the dataset...
# https://bit.ly/2KF29Bd
df = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y": [5,10,10,15,14,15,19,18,25,23]
})

# load data...
points = df.itertuples()

# test with given line...
m = 1.93939
b = 4.73333

# define sum_of_squares...
sum_of_squares = 0.0

# calculating the residuals... 
for p in points:
    y_actual = p.y
    y_predict = m * p.x + b

# formulate for residual_squared    
    residual_squared = (y_actual - y_predict)**2

# applies condition...        
    sum_of_squares +=  residual_squared
    
# prints residual...
print("sum of squares = {}".format(sum_of_squares))    