# using inverse and transposed matrices to fit a linear regression...
# importing pandas as pd...
import pandas as pd

# importing inv from numpy.linalg...
from numpy.linalg import inv

# importing numpy as np...
import numpy as np

# create the dataset...
# https://bit.ly/3go0Ant
df = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y": [5,10,10,15,14,15,19,18,25,23]
})

# Extract input variable
X = df.values[:, :-1]

# Add a column of 1s for the intercept
X_1 = np.hstack((X, np.ones((len(X), 1))))

# Extract output variable
Y = df.values[:, -1]

# Calculate coefficients
b = inv(X_1.T @ X_1) @ (X_1.T @ Y)

# prints slope of m...
print("Slope (m) =", b[0])

# rints the intercept of c...
print("Intercept (c) =", b[1])

# Predict y values
y_predict = X_1 @ b

# prints the predict value of y...
print("\nPredicted y values:\n",y_predict)

# using QR decomposition to perform a linear regression...
# importing pandas as pd...
import pandas as pd

# importing qr, inv from numpy.linalg...
from numpy.linalg import qr, inv

# importing numpy as np...
import numpy as np

# create the dataset...
# https://bit.ly/3go0Ant
df = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y": [5,10,10,15,14,15,19,18,25,23]
})

# Extract input variable
X = df.values[:, :-1]

# Add a column of 1s for the intercept
X_1 = np.hstack((X, np.ones((len(X), 1))))

# Extract output variable
Y = df.values[:, -1]

# calculate coefficient for slope and intercept...
# using QR decomposition...
Q, R = qr(X_1)

# formulate...
b = inv(R).dot(Q. transpose()).dot(Y)

# prints the coefficient for slope and intercept...
print("\nCoefficient for slope and intercept ",b)