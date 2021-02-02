# There are roughly 80 continuous distributions and over 10 discrete distributions.
# Twenty of the continuous functions are shown in Figure 3-12 as probability density
# functions (PDFs) to give a visual impression of what the scipy.stats package provides.
# These distributions are useful as random number generators, similar to the functions
# found in numpy.random. Yet the rich variety of functions SciPy provides stands in contrast
# to the numpy.random functions, which are limited to uniform and Gaussian-like
# distributions

import numpy as np
from scipy.stats import norm

# Set up the sample range
x = np.linspace(-5,5,1000)

# Here set up the parameters for the normal distribution,
# where loc is the mean and scale is the standard deviation.
dist = norm(loc=0, scale=1)

# Retrieving norm's PDF and CDF
pdf = dist.pdf(x)
cdf = dist.cdf(x)

# Here we draw out 500 random values from the norm.
sample = dist.rvs(500)
print(sample)


# Here set up the parameters for the geometric distribution.
p = 0.5
dist = geom(p)
# Set up the sample range.
x = np.linspace(0, 5, 1000)
# Retrieving geom's PMF and CDF
pmf = dist.pmf(x)
cdf = dist.cdf(x)
# Here we draw out 500 random values.
sample = dist.rvs(500)