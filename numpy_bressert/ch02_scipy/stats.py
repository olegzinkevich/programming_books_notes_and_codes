# InNumPy there are basic statistical functions like mean, std, median, argmax, and argmin.
# Moreover, the numpy.arrays have built-in methods that allow us to use most of the
# NumPy statistics easily.
import numpy as np

# Constructing a random array with 1000 elements
x = np.random.randn(1000)
# Calculating several of the built-in methods
# that numpy.array has
mean = x.mean()

print(mean)
# standart deviation
std = x.std()
print(std)
# Returns the variance of the array elements, along given axis.
var = x.var()
print(var)


# h stats functions

import numpy as np
from scipy import stats
# Generating a normal distribution sample
# with 100 elements
sample = np.random.randn(100)
# normaltest tests the null hypothesis.
out = stats.normaltest(sample)
print('normaltest output')
print('Z-score = ' + str(out[0]))
print('P-value = ' + str(out[1]))
# kstest is the Kolmogorov-Smirnov test for goodness of fit.
# Here its sample is being tested against the normal distribution.
# D is the KS statistic and the closer it is to 0 the better.
out = stats.kstest(sample, 'norm')
print('\nkstest output for the Normal distribution')
print('D = ' + str(out[0]))
print('P-value = ' + str(out[1]))
# Similarly, this can be easily tested against other distributions,
# like the Wald distribution.
out = stats.kstest(sample, 'wald')
print('\nkstest output for the Wald distribution')
print('D = ' + str(out[0]))
print('P-value = ' + str(out[1]))

# Researchers commonly use descriptive functions for statistics. Some descriptive functions
# that are available in the stats package include the geometric mean (gmean), the
# skewness of a sample (skew), and the frequency of values in a sample (itemfreq). Using
# these functions is simple and does not require much input. A few examples follow.


# Generating a normal distribution sample
# with 100 elements
sample = np.random.randn(100)


# The harmonic mean: Sample values have to
# be greater than 0.
out = stats.hmean(sample[sample > 0])

print('Harmonic mean = ' + str(out))

# The mean, where values below -1 and above 1 are
# removed for the mean calculation
out = stats.tmean(sample, limits=(-1, 1))
print('\nTrimmed mean = ' + str(out))

# Calculating the skewness of the sample
out = stats.skew(sample)
print('\nSkewness = ' + str(out))
# Additionally, there is a handy summary function called
# describe, which gives a quick look at the data.
out = stats.describe(sample)
print('\nSize = ' + str(out[0]))
print('Min = ' + str(out[1][0]))
print('Max = ' + str(out[1][1]))
print('Mean = ' + str(out[2]))
print('Variance = ' + str(out[3]))
print('Skewness = ' + str(out[4]))
print('Kurtosis = ' + str(out[5]))