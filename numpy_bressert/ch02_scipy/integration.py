# Integration is a crucial tool in math and science, as differentiation and integration are
# # the two key components of calculus. Given a curve from a function or a dataset, we
# # can calculate the area below it. In the traditional classroom setting we would integrate
# # a function analytically, but data in the research setting is rarely given in this form, and
# # we need to approximate its definite integral.

import numpy as np
from scipy.integrate import quad
# Defining function to integrate
func = lambda x: np.cos(np.exp(x)) ** 2
# Integrating function with upper and lower
# limits of 0 and 3, respectively
solution = quad(func, 0, 3)
print(solution)
# The first element is the desired value
# and the second is the error.
# (1.296467785724373, 1.397797186265988e-09)

# Let’s move on to a problem where we are given data instead of some known equation
# and numerical integration is needed. Figure 3-11 illustrates what type of data sample
# can be used to approximate acceptable indefinite integrals.
from scipy.integrate import quad, trapz

# Setting up fake data
x = np.sort(np.random.randn(150) * 4 + 4).clip(0,5)
func = lambda x: np.sin(x) * np.cos(x ** 2) + 1
y = func(x)

# Integrating function with upper and lower
# limits of 0 and 5, respectively
fsolution = quad(func, 0, 5)
dsolution = trapz(y, x=x)

print('fsolution = ' + str(fsolution[0]))
print('dsolution = ' + str(dsolution))
print('The difference is ' + str(np.abs(fsolution[0] - dsolution)))
# fsolution = 5.10034506754
# dsolution = 5.04201628314
# The difference is 0.0583287843989.