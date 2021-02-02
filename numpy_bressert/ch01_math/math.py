import numpy as np

# Python comes with its own math module that works on Python native objects. Unfortunately,
# if you try to use math.cos on a NumPy array, it will not work, as the math
# functions are meant to operate on elements and not on lists or arrays. Hence, NumPy
# comes with its own set of math tools

# NumPy arrays do not behave like matrices in linear algebra by default. Instead, the
# operations are mapped from each element in one array onto the next. This is quite
# a useful feature, as loop operations can be done away with for efficiency. But what
# about when transposing or a dot multiplication are needed? Without invoking other
# classes, you can use the built-in numpy.dot and numpy.transpose to do such operations
#
# 3x + 6y − 5z = 12
# x − 3y + 2z = −2
# 5x − y + 4z = 10

# Now let us represent the matrix system as AX = B, and solve for the variables. This
#     means we should try to obtain X = A−1B. Here is howwe would do this withNumPy.
# import numpy as np
# Defining the matrices

A = np.matrix([[3, 6, -5],
               [1, -3, 2],
               [5, -1, 4]])

B = np.matrix([[12],
               [-2],
               [10]])

# Solving for the variables, where we invert A
X = A ** (-1) * B
print(X)
# matrix([[ 1.75],
# [ 1.75],
# [ 0.75]])

# The solutions for the variables are x = 1.75, y = 1.75, and z = 0.75. You can easily check
# this by executing AX, which should produce the same elements defined in B

# Now that we understand howNumPy matrices work, we can show how to do the same
# operations without specifically using the numpy.matrix subclass. (The numpy.matrix
# subclass is contained within the numpy.array class, which means that we can do the
# same example as that above without directly invoking the numpy.matrix class.) First, the NumPy array is the standard for using nearly anything in
# the scientific Python environment, so bugs pertaining to the linear algebra operations
# will be less frequent than with numpy.matrix operations. Sticking with
# one data structure will lead to fewer headaches and less worry than switching between
# matrices and arrays. It is advisable, then, to use numpy.array whenever possible.

a = np.array([[3, 6, -5],
              [1, -3, 2],
              [5, -1, 4]])

# Defining the array
b = np.array([12, -2, 10])

# Solving for the variables, where we invert A
x = np.linalg.inv(a).dot(b)

print(x)
# array([ 1.75, 1.75, 0.75])