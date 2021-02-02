# The optimization package in SciPy allows us to solve minimization problems easily and
# quickly. But wait: what is minimization and how can it help you with your work? Some
# classic examples are performing linear regression, finding a function’s minimum and
# maximum values, determining the root of a function, and finding where two functions
# intersect

# The optimization and minimization tools thatNumPy and SciPy provide
# are great, but they do not have Markov Chain Monte Carlo (MCMC)
# capabilities—in otherwords, Bayesian analysis.
#
# There are several ways to fit data with a linear regression. In this section we will use
# curve_fit, which is a χ2-based method (in other words, a best-fit method). In the
# example below, we generate data from a known function with noise, and then fit the
# noisy data with curve_fit. The function we will model in the example is a simple linear
# equation, f (x) = ax + b.
import numpy as np

from scipy.optimize import curve_fit

# Creating a function to model and create data
def func(x, a, b):
    return a * x + b

# Generating clean data
x = np.linspace(0, 10, 100)
print('clean data', x)
y = func(x, 1, 2)
print('func result', y)

# Adding noise to the data
yn = y + 0.9 * np.random.normal(size=len(x))
print(yn)

# Executing curve_fit on noisy data
popt, pcov = curve_fit(func, x, yn)

# popt returns the best fit values for parameters of
# the given model (func). # The values from popt, if a good fit, should be close to the values for the y assignment.
# # You can check the quality of the fit with pcov, where the diagonal elements are the
# # variances for each parameter.
print(popt)

# Taking this a step further,we can do a least-squares fit to a Gaussian profile
# Creating a function to model and create data
def func(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))
# Generating clean data
x = np.linspace(0, 10, 100)
y = func(x, 1, 5, 2)
# Adding noise to the data
yn = y + 0.2 * np.random.normal(size=len(x))
# Executing curve_fit on noisy data
popt, pcov = curve_fit(func, x, yn)