# using gradient descent descent to find the minimum of a parabola...
# importing random...
import random

# declare function for f(x)...
def f(x):
    return (x - 3) ** 2 + 4

# declare function for dx_f(x)...
def dx_f(x):
    return 2 * (x - 3)

# the learning rate...
L = 0.001

# the number of iterations...
iterations = 100_000

# start at a random x...
x = random.randint(-15, 15)

# gradient descent...
for i in range(iterations):
    
# get slope...
    d_x = dx_f(x)

# update x by substracting the (learning rate) * (slope)...
    x -= L * d_x

# prints the value of x...
print(x, f(x))

# performing gradient descent for a linear regression... 
# importing pandas as pd... 
import pandas as pd 

# Create the dataset...
# https://bit.ly/2KF29Bd
df = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y": [5,10,10,15,14,15,19,18,25,23]
})

# load data...
# here use list operator...
points = list(df.itertuples())

# buliding the model... 
# m and b are both the parameters...
m = 0.0
b = 0.0

# the learning rate...
L = 0.001

# the number of iterations...
iterations = 100_000

# number of elements in x... 
n = float(len(points))

# perform gradient descent... 
for i in range(iterations):
    
# slope with respect to m...
    D_m = sum(2 * p.x * ((m * p.x + b) - p.y) for p in points)
    
# slope with respect to b...
    D_b = sum(2 * ((m * p.x + b) - p.y) for p in points)    
    
# update m and b... 
    m -= L * D_m
    b -= L * D_b

# prints value in from of y = {0}x + {1}.. 
print("y = {0}x + {1}".format(m, b))  

# calculating the partial derivatives for m and b... 
# from sympy importing *... 
from sympy import *

# declear (m, b i, n) symbols...
m, b, i, n = symbols('m b i n')

# is used in SymPy to create symbolic functions, not ordinary variables...
x, y = symbols('x y', cls=Function)

# formulate for sum_of_squares...
sum_of_squares = Sum((m*x(i) + b - y(i)) ** 2, (i, 0, n))

# declearing for m...
d_m = diff(sum_of_squares, m)

# declearing for b...
d_b = diff(sum_of_squares, b)

# prints partial derivatives for m...
print(d_m)

# prints partial derivative for b...
print(d_b)

# solving linear regression using sympy... 
# importing pandas as pd...
import pandas as pd 

# from sympy importing *... 
from sympy import *

# Create the dataset...
# https://bit.ly/2KF29Bd
df = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y": [5,10,10,15,14,15,19,18,25,23]
})

# load data...
# here use list operator...
points = list(df.itertuples())

# formulate for sum_of_squares...
sum_of_squares = Sum((m*x(i) + b - y(i)) ** 2, (i, 0, n))

# Partial derivative with respect to m...
d_m = (
    diff(sum_of_squares, m)
    .subs(n, len(points) - 1)
    .doit()
    .replace(x, lambda j: points[j].x)
    .replace(y, lambda j: points[j].y)
)

# Partial derivative with respect to b...
d_b = (
    diff(sum_of_squares, b)
    .subs(n, len(points) - 1)
    .doit()
    .replace(x, lambda j: points[j].x)
    .replace(y, lambda j: points[j].y)
)

# compile using lambdaify for faster computation... 
d_m = lambdify([m, b], d_m)
d_b = lambdify([m, b], d_b)

# buliding the model... 
# m and b are both the parameters...
m = 0.0
b = 0.0

# the learning rate...
L = 0.001

# the number of iterations...
iterations = 100_000

# perform gradient descent...
for i in range(iterations):
    
# update m by substracting d_m(m,b) * L...
    m -= d_m(m,b) * L
    
# update m by substracting d_b(m,b) * L... 
    b -= d_b(m,b) * L
    
# prints value in from of y = {0}x + {1}.. 
print("y = {0}x + {1}".format(m, b))  

# plotting the loss function for linear regression...
# from sympy importing *... 
from sympy import *

# importing plot3d from sympy.plotting...
from sympy.plotting import plot3d

# importing pandas as pd...
import pandas as pd 

# Create the dataset...
# https://bit.ly/2KF29Bd
points = pd.DataFrame({
    "x": [1,2,3,4,5,6,7,8,9,10],
    "y": [5,10,10,15,14,15,19,18,25,23]
})

# load data...
# here use list operator...
points = list(points.itertuples())

# declear (m, b i, n) symbols...
m, b, i, n = symbols('m b i n')

# is used in SymPy to create symbolic functions, not ordinary variables...
x, y = symbols('x y', cls=Function)

# build the symbolic sum of squares...
sum_of_squares = (
    Sum((m*x(i) + b - y(i))**2, (i, 0, n))
    .subs(n, len(points) - 1)
    .doit()
    .replace(x, lambda j: points[j].x)
    .replace(y, lambda j: points[j].y)
)

# Plot the loss function
plot3d(sum_of_squares)