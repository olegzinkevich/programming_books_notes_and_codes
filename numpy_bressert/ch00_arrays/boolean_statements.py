import numpy as np
#
# Boolean statements are commonly used in combination with the and operator and the
#     or operator. These operators are useful when comparing single boolean values to one
# another, but when using NumPy arrays, you can only use & and | as this allows fast
# comparisons of boolean values

# Creating an image
img1 = np.zeros((20, 20)) + 3
img1[4:-4, 4:-4] = 6
img1[7:-7, 7:-7] = 9
print(img1)

# Let's filter out all values larger than 2 and less than 6.
index1 = img1 > 2
index2 = img1 < 6
compound_index = index1 & index2
print(compound_index)

# The compound statement can alternatively be written as
compound_index = (img1 > 3) & (img1 < 7)
img2 = np.copy(img1)
img2[compound_index] = 0
print(img2)

# Making the boolean arrays even more complex
index3 = img1 == 9
index4 = (index1 & index2) | index3
img3 = np.copy(img1)
img3[index4] = 0
# See Plot C.
print(img3)



import numpy.random as rand

# Creating a 100-element array with random values
# from a standard normal distribution or, in other
# words, a Gaussian distribution.
# The sigma is 1 and the mean is 0.
a = rand.randn(100)
print(a)

# Here we generate an index for filtering
# out undesired elements.
index = a > 0.2
b = a[index]
print(b)

# We execute some operation on the desired elements.
b = b ** 2 - 2
print(b)

# Then we put the modified elements back into the
# original array.
a[index] = b
print(a)