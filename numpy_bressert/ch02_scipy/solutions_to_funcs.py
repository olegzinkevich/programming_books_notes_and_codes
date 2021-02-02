# With data modeling and fitting under our belts, we can move on to finding solutions,
# such as “What is the root of a function?” or “Where do two functions intersect?” SciPy
# provides an arsenal of tools to do this in the optimize module.

# Let’s start simply, by solving for the root of an equation (see Figure 3-4). Here we will
# use scipy.optimize.fsolve

from scipy.optimize import fsolve
import numpy as np

# Find the roots of a function.
line = lambda x: x + 3

solution = fsolve(line, -2)
print(solution)

# Finding the intersection points between two equations is nearly as simple.3

# Defining function to simplify intersection solution
def findIntersection(func1, func2, x0):
    return fsolve(lambda x : func1(x) - func2(x), x0)

# Defining functions that will intersect
funky = lambda x : np.cos(x / 5) * np.sin(x / 2)
line = lambda x : 0.01 * x - 0.5

# Defining range and getting solutions on intersection points
x = np.linspace(0,45,10000)

result = findIntersection(funky, line, [15, 20, 30, 35, 40, 45])
# Printing out results for x and y
print(result, line(result))