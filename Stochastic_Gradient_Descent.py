# performing stotastic gradient descent for a linear regression...
# importing pandas as pd...
import pandas as pd 

# importing numpy as np...
import numpy as np 

# Create the dataset...
# https://bit.ly/2KF29Bd
data = pd.DataFrame({
    "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "y": [5, 10, 10, 15, 14, 15, 19, 18, 25, 23]
})


X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

# shows rows...
n = data.shape[0]

# building the model...
m = 0.0
b = 0.0

# sample size...
sample_size = 1

# learning rate...
L = 0.0001

# the number of iterations to perform gradient descent...
epochs = 1_000_000

# performing stochastic gradient descent.
# starts for-loop...
for i in range(epochs):
    idx = np.random.choice(n, sample_size, replace=False)
    x_sample = X[idx]
    y_sample = Y[idx]

# the current predict value of Y...
    Y_pred = m * x_sample + b

# d/dm derivative of loss function...
    D_m = (-2 / sample_size) * sum(
        x_sample * (y_sample - Y_pred))

# d/db derivative of loss function...  
    D_b = (-2 / sample_size) * sum(
        y_sample - Y_pred)

# updates m...
    m = m - L * D_m
    
# updates b...    
    b = b - L * D_b

# prints progress...
    if i % 10000 == 0:
        print(i, m, b)

# print final result...
print("y = {0}x + {1}".format(m, b))